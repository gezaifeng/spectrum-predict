# 分支指南

| 分支 | 活跃模型／用途 |
| --- | --- |
| master | 当前工程整合：72色块、190点残差双头，MSE+DC+TV+系数L2 |
| main | 发布后与master相同，兼顾旧clone的默认分支 |
| feature/cnn | 找回的CNN+Transformer，24色块100采样点、76点输出 |
| feature/unet | 找回的MicroUNet比较模型，24色块、76点输出 |
| feature/res-baseline | 早期采样序列Res回归，24色块、76点输出 |
| feature/single-head-72 | 单头残差与差分对齐对照，72色块、190点输出 |
| feature/token-attention | 色块全局关系、波长查询，24色块、76点输出 |
| feature/token-72 | 九点编码与色块token，72色块、190点输出 |
| feature/multilight-72 | 八光源token，8×72色块、190点输出 |
| archive/original-main-2026-10-09 | 替换前GitHub仓库及原始提交 |

每个feature分支通过替换根目录model.py与configs/current.json选择自身模型；共享训练和预测入口。
仅安装torch/numpy/scipy即可跑模型检查；原始GUI与绘图需额外依赖。
根目录配置是整理后的可运行复现配置，不代表参数量已严格匹配，也不代表论文最佳超参。
