# 复现修复与历史源文件的区别

historical/和provenance/manifest.json保存找回的原始源码及SHA-256，原始文件中的旧路径和已知问题未在快照内修改。
根目录runtime.py、train.py、predict.py和tests是本次新增的复现入口，不是当年的原文件。

1. CNN样本序列Transformer补上batch_first=True，避免把不同batch样本当成序列；reshape前先permute采样轴，避免RGB与采样点错位。
2. 早期残差序列模型同样修正reshape前轴排列。网络层和返回谱维度保留。
3. 统一数据读入与推理预处理；禁止静默猜测RGB/ratio/log_ratio缩放，不对标签按不同split各自标准化。
4. 输入维度、有限值、标签配对与跨split重复ID检查；这些检查不替代按溶液分组的实验划分。
5. 使用命令行传入数据目录，checkpoint包含配置与模型权重；推理严格加载同一模型。
6. 双头统一(pred,coef)和损失调用，消除MSELoss参数数量不一致。新默认loss为MSE+DC+TV+系数L2；单头与双头使用同一接口。
7. 新72色块双头使用token_grid=(6,12), n_wave=190；九点采样在共享预处理里取均值，特征为RGB时显式除255。
8. 峰值权重来自训练标签，原始逐波长动态加权保留为实验配置。使用验证集调权意味着该验证集参与调参，须另保留独立测试集。
