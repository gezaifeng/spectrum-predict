# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1)
        self.c2 = ConvBNAct(ch, ch, 3, 1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.c2(self.c1(x)))


class ResCNN(nn.Module):
    """
    输入:  (B,3,H,W)
    输出:  (B,76)
    """
    def __init__(
        self,
        out_dim=76,
        base_ch=64,
        n_blocks=6,
        mlp_hidden=256,
        dropout=0.0,
        out_activation="linear",
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, base_ch, 3, 1),
            ConvBNAct(base_ch, base_ch, 3, 1),
        )
        self.body = nn.Sequential(*[ResBlock(base_ch) for _ in range(n_blocks)])
        self.head_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(base_ch, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim),
        )

        oa = str(out_activation).lower()
        if oa == "linear":
            self.out_act = nn.Identity()
        elif oa == "relu":
            self.out_act = nn.ReLU(inplace=True)
        elif oa == "softplus":
            self.out_act = nn.Softplus()
        else:
            raise ValueError("out_activation must be 'linear' | 'relu' | 'softplus'")

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head_conv(x)
        x = self.pool(x).flatten(1)
        y = self.mlp(x)
        y = self.out_act(y)
        return y