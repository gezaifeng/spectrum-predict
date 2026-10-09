# 放在文件顶部其他import下方
import torch.nn as nn

class MSELossCompat(nn.MSELoss):
    def forward(self, input, target, *args, **kwargs):
        # 忽略第三个及之后的参数（例如 coef）
        return super().forward(input, target)


