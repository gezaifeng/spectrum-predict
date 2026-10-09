# model_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 基础组件
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act="silu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, act="silu", mode="bilinear"):
        super().__init__()
        self.mode = mode
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, act=act)
    def forward(self, x, skip):
        # 上采样到 skip 的空间尺寸
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# -------------------------
# 微型 U-Net 主体
# -------------------------
class MicroUNetSpectral(nn.Module):
    """
    适配超小空间输入的 U-Net（4x6）：
      4x6 --(pool)--> 2x3 --(pool)--> 1x1 --(up)--> 2x3 --(up)--> 4x6
    通过瓶颈通道数控制参数量，整体与 CNN 基线处于同一数量级。
    """
    def __init__(self, out_dim=76,
                 c1=24, c2=48, c3=96,   # stage 通道：与参数量强相关
                 mlp_hidden=128, out_activation="linear"):
        super().__init__()
        # 编码器
        self.enc1 = DoubleConv(3,  c1, act="silu")  # 4x6
        self.down1= nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # -> 2x3
        self.enc2 = DoubleConv(c1, c2, act="silu")  # 2x3
        self.down2= nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # -> 1x1
        self.bott = DoubleConv(c2, c3, act="silu")  # 1x1

        # 解码器
        self.up1  = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2, act="silu")
        self.up2  = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1, act="silu")

        # 谱域回归头（全局池化 + MLP）
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(
            nn.Linear(c1, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_hidden, out_dim)
        )

        if out_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0)
        else:
            self.out_act = nn.Identity()

        self._init_kaiming()

    def _init_kaiming(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: [B,3,4,6], 归一化到 [0,1]
        """
        e1 = self.enc1(x)           # 4x6, c1
        x  = self.down1(e1)         # 2x3
        e2 = self.enc2(x)           # 2x3, c2
        x  = self.down2(e2)         # 1x1
        x  = self.bott(x)           # 1x1, c3

        x  = self.up1(x, e2)        # 2x3, c2
        x  = self.up2(x, e1)        # 4x6, c1

        feat = self.pool(x).flatten(1)  # [B, c1]
        out  = self.head(feat)          # [B, 76]
        return self.out_act(out)

# 便捷统计
if __name__ == "__main__":
    model = MicroUNetSpectral(out_dim=76, c1=24, c2=48, c3=96, mlp_hidden=128, out_activation="linear")
    x = torch.randn(2,3,4,6)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("MicroUNetSpectral:", y.shape, "params =", n_params)
