
import torch.nn as nn
import torch.nn.functional as F


# ========================= 模型定义 =========================
class SpectralCNNTransformer(nn.Module):
    def __init__(self):
        super(SpectralCNNTransformer, self).__init__()

        # CNN 处理 (3, 100, 4, 6) → (64, 100, 2, 3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化 (4x6 → 2x3)

        # Transformer 处理 100 维度信息
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64 * 2 * 3, nhead=4, dim_feedforward=256, batch_first=True),
            num_layers=2
        )

        # 全连接层

        self.fc1 = nn.Linear(64 * 2 * 3, 128)
        self.fc2 = nn.Linear(128, 76)


    def forward(self, x):
        # CNN 处理空间特征
        b, c, seq, h, w = x.shape  # (batch, 3, 100, 4, 6)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * seq, c, h, w)  # 合并批次和序列 (batch*100, 3, 4, 6)

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # 变成 (batch*100, 64, 2, 3)

        # 变换形状以适应 Transformer
        x = x.view(b, seq, -1)  # (batch, 100, 64*2*3)
        x = self.transformer_encoder(x)  # Transformer 处理

        # 取 Transformer 最后一个时间步的信息
        x = x[:, -1, :]

        # 全连接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x


