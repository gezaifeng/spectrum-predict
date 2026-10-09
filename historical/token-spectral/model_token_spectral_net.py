# -*- coding: utf-8 -*-
"""
TokenSpectralNet
----------------
针对“RGB色块矩阵 -> 吸光度光谱”任务的优化网络：
1) 不再依赖卷积的局部平移先验，而是把每个色块视为一个 token；
2) 使用全局 token mixer 建模跨色块协同关系；
3) 使用“波长查询”从 token 集合中逐波长解码；
4) 额外保留一个平滑基底分支，让模型更符合吸光度光谱的连续/平滑特性；
5) 输出既可以 linear，也可以 softplus 保证非负。

适用输入：x.shape = [B, 3, H, W]，例如 [B, 3, 4, 6]
输出：y.shape = [B, D]，例如 D=76
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class TokenMixerBlock(nn.Module):
    """标准 Transformer encoder block，用于建模跨色块全局关系。"""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionDecoderBlock(nn.Module):
    """波长 query 对 token memory 做交叉注意力。"""
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, q, mem):
        qn = self.norm_q(q)
        mn = self.norm_mem(mem)
        h, _ = self.cross_attn(qn, mn, mn, need_weights=False)
        q = q + h
        q = q + self.mlp(self.norm2(q))
        return q


class SmoothBasisHead(nn.Module):
    """
    平滑基底分支：
    用少量全局系数拟合“主形状”，再由 query 分支补细节。
    基底默认使用低频 DCT，天然偏平滑。
    """
    def __init__(self, in_dim: int, out_dim: int, num_basis: int = 12, nonnegative_coef: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.num_basis = num_basis
        self.nonnegative_coef = nonnegative_coef
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, num_basis),
        )
        basis = self._make_dct_basis(out_dim, num_basis)
        self.register_buffer("basis", basis, persistent=True)

    @staticmethod
    def _make_dct_basis(n_wave: int, k: int):
        B = torch.zeros(n_wave, k)
        for i in range(n_wave):
            for j in range(k):
                alpha = (1.0 / n_wave) ** 0.5 if j == 0 else (2.0 / n_wave) ** 0.5
                B[i, j] = alpha * math.cos(math.pi * (i + 0.5) * j / n_wave)
        return B

    def forward(self, g):
        coef = self.fc(g)
        if self.nonnegative_coef:
            coef = F.softplus(coef)
        base = coef @ self.basis.t()  # [B, D]
        return base, coef


class TokenSpectralNet(nn.Module):
    """
    推荐主模型。

    核心思想：
    - 输入不是图像语义，不适合强卷积先验；
    - 每个色块更像“一个测量通道”，所以转成 token 更合理；
    - 波长输出天然是序列，因此用 wavelength query 逐波长解码；
    - 再加一个平滑基底分支，专门负责光谱的大体形状。
    """
    def __init__(
        self,
        in_ch: int = 3,
        grid_h: int = 4,
        grid_w: int = 6,
        out_dim: int = 76,
        d_model: int = 128,
        depth: int = 4,
        decoder_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        num_basis: int = 12,
        out_activation: str = "linear",   # linear | softplus | relu
        nonnegative_coef: bool = False,
        use_patch_confidence: bool = True,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_tokens = grid_h * grid_w
        self.out_dim = out_dim
        self.d_model = d_model
        self.use_patch_confidence = use_patch_confidence

        # 1) patch/token embedding: 对每个色块的RGB做通道升维
        self.token_embed = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, stride=1, padding=0, bias=True),
        )

        # 2) 显式位置编码：因为色块位置固定，不需要卷积的平移不变性，而需要“记住谁在什么位置”
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 可选：给每个 token 学一个置信度门控，允许模型自动压低信息差/噪声大的色块
        if use_patch_confidence:
            self.patch_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.patch_gate = None

        # 3) token 全局关系建模
        self.encoder = nn.ModuleList([
            TokenMixerBlock(d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # 4) 波长 query：每个波长一个查询向量 + 连续波长标量嵌入
        wave = torch.linspace(0.0, 1.0, out_dim).view(1, out_dim, 1)
        self.register_buffer("wave_coord", wave, persistent=True)
        self.wave_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.wave_query = nn.Parameter(torch.zeros(1, out_dim, d_model))
        nn.init.trunc_normal_(self.wave_query, std=0.02)

        self.decoder = nn.ModuleList([
            CrossAttentionDecoderBlock(d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # 5) 平滑主形状分支 + 细节残差分支
        self.base_head = SmoothBasisHead(
            in_dim=d_model,
            out_dim=out_dim,
            num_basis=num_basis,
            nonnegative_coef=nonnegative_coef,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, 1),
        )

        # 6) 最终输出激活
        oa = out_activation.lower()
        if oa == "linear":
            self.out_act = nn.Identity()
        elif oa == "softplus":
            self.out_act = nn.Softplus(beta=1.0, threshold=20.0)
        elif oa == "relu":
            self.out_act = nn.ReLU(inplace=True)
        else:
            raise ValueError("out_activation must be one of ['linear', 'softplus', 'relu']")

    def encode_tokens(self, x):
        # x: [B,3,H,W]
        feat = self.token_embed(x)                # [B,d,H,W]
        B, C, H, W = feat.shape
        assert H == self.grid_h and W == self.grid_w, \
            f"Input grid must be ({self.grid_h},{self.grid_w}), but got ({H},{W})"

        tokens = feat.flatten(2).transpose(1, 2) # [B,N,d]
        tokens = tokens + self.pos_embed

        if self.patch_gate is not None:
            gate = self.patch_gate(tokens)        # [B,N,1]
            tokens = tokens * gate
        else:
            gate = None

        for blk in self.encoder:
            tokens = blk(tokens)
        tokens = self.encoder_norm(tokens)

        g = tokens.mean(dim=1)                    # [B,d]
        return tokens, g, gate

    def decode_spectrum(self, tokens, g):
        B = tokens.size(0)
        q = self.wave_query + self.wave_embed(self.wave_coord)   # [1,D,d]
        q = q.expand(B, -1, -1)                                  # [B,D,d]

        for blk in self.decoder:
            q = blk(q, tokens)
        q = self.decoder_norm(q)                                 # [B,D,d]

        base, coef = self.base_head(g)                           # [B,D], [B,K]
        residual = self.residual_head(q).squeeze(-1)             # [B,D]
        pred = self.out_act(base + residual)
        return pred, base, residual, coef

    def forward(self, x, return_dict: bool = False):
        tokens, g, gate = self.encode_tokens(x)
        pred, base, residual, coef = self.decode_spectrum(tokens, g)

        if return_dict:
            return {
                "spectrum": pred,
                "base": base,
                "residual": residual,
                "coef": coef,
                "token_gate": gate,
                "tokens": tokens,
            }
        return pred


def build_token_spectral_model(cfg=None):
    if cfg is None:
        return TokenSpectralNet()
    if isinstance(cfg, dict):
        return TokenSpectralNet(
            in_ch=cfg.get("in_ch", 3),
            grid_h=cfg.get("grid_h", 4),
            grid_w=cfg.get("grid_w", 6),
            out_dim=cfg.get("out_dim", 76),
            d_model=cfg.get("d_model", 128),
            depth=cfg.get("depth", 4),
            decoder_depth=cfg.get("decoder_depth", 2),
            num_heads=cfg.get("num_heads", 4),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            drop=cfg.get("drop", 0.1),
            num_basis=cfg.get("num_basis", 12),
            out_activation=cfg.get("out_activation", "linear"),
            nonnegative_coef=cfg.get("nonnegative_coef", False),
            use_patch_confidence=cfg.get("use_patch_confidence", True),
        )
    return TokenSpectralNet(
        in_ch=getattr(cfg, "in_ch", 3),
        grid_h=getattr(cfg, "grid_h", 4),
        grid_w=getattr(cfg, "grid_w", 6),
        out_dim=getattr(cfg, "out_dim", 76),
        d_model=getattr(cfg, "d_model", 128),
        depth=getattr(cfg, "depth", 4),
        decoder_depth=getattr(cfg, "decoder_depth", 2),
        num_heads=getattr(cfg, "num_heads", 4),
        mlp_ratio=getattr(cfg, "mlp_ratio", 4.0),
        drop=getattr(cfg, "drop", 0.1),
        num_basis=getattr(cfg, "num_basis", 12),
        out_activation=getattr(cfg, "out_activation", "linear"),
        nonnegative_coef=getattr(cfg, "nonnegative_coef", False),
        use_patch_confidence=getattr(cfg, "use_patch_confidence", True),
    )


if __name__ == "__main__":
    x = torch.randn(2, 3, 4, 6)
    model = TokenSpectralNet(out_dim=76, d_model=128, depth=3, decoder_depth=2)
    y = model(x)
    print(y.shape)
