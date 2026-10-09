# -*- coding: utf-8 -*-
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_dct_basis(n_wave: int, k: int) -> torch.Tensor:
    B = torch.zeros(n_wave, k)
    for i in range(n_wave):
        for j in range(k):
            alpha = (1.0 / n_wave) ** 0.5 if j == 0 else (2.0 / n_wave) ** 0.5
            B[i, j] = alpha * math.cos(math.pi * (i + 0.5) * j / n_wave)
    return B

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1)
        self.conv2 = ConvBNAct(ch, ch, 3, 1, act=False)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return self.act(x + y)

class DualHeadSpectralNet(nn.Module):
    """
    输入:  x ∈ R^{B×3×4×6}（来自 dataset.py 的稳健聚合输出）
    输出:  spec ∈ R^{B×76}，coef ∈ R^{B×K}
    结构:  spec = (B_dct @ coef) + residual(λ)
           - 主干: Stem + 多层 ResBlock（保留空间结构）
           - 逐波长软注意: 为每个 λ 从 4×6 token 自适应汇聚
           - Head1: 低维基系数（Softplus ≥ 0）
           - Head2: 逐波长残差（对角抽取）
    """
    def __init__(
        self,
        n_wave: int = 76,
        token_grid: Tuple[int, int] = (4, 6),
        base_ch: int = 64,
        n_blocks: int = 4,
        k_basis: int = 12,
        out_activation: str = "softplus",
    ):
        super().__init__()
        self.n_wave = n_wave
        self.k_basis = k_basis

        self.stem   = nn.Sequential(ConvBNAct(3, base_ch, 3, 1),
                                    ConvBNAct(base_ch, base_ch, 3, 1))
        self.blocks = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])
        self.proj   = nn.Conv2d(base_ch, base_ch, 1, bias=False)

        Ht, Wt = token_grid
        self.keep_grid = nn.AdaptiveAvgPool2d((Ht, Wt))
        Nt = Ht * Wt

        # per-λ token attention
        self.attn_weight = nn.Parameter(torch.zeros(n_wave, Nt))
        nn.init.xavier_uniform_(self.attn_weight)

        # heads
        self.head_coef = nn.Linear(base_ch, k_basis)   # ≥0
        self.head_res  = nn.Linear(base_ch, n_wave)    # per-λ

        # fixed DCT basis
        B = make_dct_basis(n_wave, k_basis)  # [76, K]
        self.register_buffer("basis", B, persistent=True)

        oa = out_activation.lower()
        self.out_act = nn.Softplus() if oa == "softplus" else (nn.ReLU() if oa == "relu" else nn.Identity())

    def _attn_pool(self, tokens: torch.Tensor):
        # tokens: [B,C,Ht,Wt] -> [B,C,Nt]
        Bsz, Cc, Ht, Wt = tokens.shape
        T = tokens.view(Bsz, Cc, Ht*Wt)               # [B,C,Nt]
        A = self.attn_weight.softmax(dim=-1)          # [L,Nt]
        f_lambda = torch.matmul(T, A.t())             # [B,C,L]
        f_global = T.mean(dim=-1)                     # [B,C]
        return f_global, f_lambda

    def forward(self, x: torch.Tensor):
        x = self.proj(self.blocks(self.stem(x)))
        tokens = self.keep_grid(x)
        f_global, f_lambda = self._attn_pool(tokens)     # [B,C], [B,C,L]

        # head1: nonnegative low-dim coefficients
        coef = F.softplus(self.head_coef(f_global))      # [B,K]
        spec_base = torch.matmul(coef, self.basis.t())   # [B,L]

        # head2: per-λ residual (diagonal take)
        Bsz, Cc, L = f_lambda.shape
        res_in  = f_lambda.permute(0,2,1).contiguous().view(Bsz*L, Cc)  # [B*L,C]
        res_all = self.head_res(res_in).view(Bsz, L, L)                 # [B,L,L]
        idx = torch.arange(L, device=x.device)
        res = res_all[:, idx, idx]                                      # [B,L]

        spec = self.out_act(spec_base + res)                            # [B,L]
        return spec, coef
