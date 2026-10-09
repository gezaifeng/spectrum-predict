# model_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math   # ← 新增：修复 NameError
# -------------------------
# 基础组件
# -------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act="silu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch, act="silu"):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, k=3, s=1, p=1, act=act)
        self.conv2 = ConvBNAct(ch, ch, k=3, s=1, p=1, act="linear")
        self.act   = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = out + x
        return self.act(out)

# -------------------------
# 模型主体
# -------------------------
class CNNResidualBaseline(nn.Module):
    """
    输入:  Bx3x4x6  （色卡网格的通道排列为 [C,H,W]）
    输出:  Bx76     （380~755nm, Δλ=5nm，共76点，可按你数据集定义）
    """
    def __init__(self, out_dim=76, base_ch=56, n_blocks=4,
                 mlp_hidden=128, out_activation="linear"):
        super().__init__()
        self.stem = ConvBNAct(3, base_ch, k=3, s=1, p=1, act="silu")

        self.body = nn.Sequential(
            *[ResidualBlock(base_ch, act="silu") for _ in range(n_blocks)]
        )

        # 全局平均池化到 1x1
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # MLP 映射到谱域
        self.head = nn.Sequential(
            nn.Linear(base_ch, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_hidden, out_dim)
        )

        if out_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0)
        else:
            self.out_act = nn.Identity()

        # 参数初始化（Kaiming）
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B,3,4,6], 归一化到 [0,1]
        """
        feat = self.stem(x)
        feat = self.body(feat)
        feat = self.pool(feat).flatten(1)         # [B, base_ch]
        out  = self.head(feat)                    # [B, 76]
        return self.out_act(out)

# 便捷统计
if __name__ == "__main__":
    import math
    model = CNNResidualBaseline(out_dim=76, base_ch=56, n_blocks=4, mlp_hidden=128, out_activation="linear")
    x = torch.randn(2,3,4,6)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("CNNResidualBaseline:", y.shape, "params =", n_params)
