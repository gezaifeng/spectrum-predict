# 色卡编码溶液光谱重建 · Spectrum Predict

用相机采集经溶液和多色块编码的RGB信号，重建溶液吸光度光谱。本项目重建对象是溶液光谱，不是整幅照片的高光谱。

这是从实际代码与项目记录整理的开发历史。请先读[历史说明](docs/HISTORY.md)及[修复记录](docs/REPAIRS.md)。
每个历史阶段提供原始源码快照、来源哈希、可运行入口与阶段标签；没有伪造当年提交时间。

## 安装与快速复现

Python 3.11；整理时在Windows、PyTorch 2.7.0 CPU上验证。

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --train-dir /path/to/train --val-dir /path/to/val --output runs/example
python predict.py --checkpoint runs/example/best.pt --input /path/to/rgb_sample.npy --output runs/example/prediction.npy
```

在Windows PowerShell中激活环境：`.\.venv\Scripts\Activate.ps1`。
数据目录包含配对的`rgb_<ID>.npy`与`spectral_<ID>.npy`；输入形状、缩放和标签维度见configs/current.json。
ratio或log_ratio必须使用feature_scale=1的显式配置，不能套用RGB的255缩放；可用--feature-prefix指定文件名前缀。
每个checkpoint写入配置；必须在对应阶段／模型分支推理。训练数据、真实标签、权重均不上传。

## 分支和阶段

主线master逐阶段推进至残差双头；阶段标签见docs/HISTORY.md。CNN和U-Net比较路线保留feature分支。
`archive/original-main-2026-10-09`保留替换前的GitHub main和全部历史。
其他分支及最终配置将在整理完成后列入docs/BRANCHES.md。

```bash
git log --graph --oneline --decorate --all
git checkout stage-01-cnn
python -m unittest discover -s tests -v
git checkout master
```

## 复现边界

合成数据测试不证明论文指标或真实泛化。重复采样、噪声扩增和不同光源照片应先按物理溶液／拍摄组划分再扩增；同一溶液不得同时进入训练与独立测试。
本包没有数据集及权重，不能据代码声明93%、96%或99.5%准确率复现成功。
原始对话只在本地local_history中保存，默认不加入Git，也不发布到公开仓库。
