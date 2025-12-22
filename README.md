# spectrum-predict

# Spectrum Predict with Color-Card Optical Encoding (Low-cost Camera Spectrometer)

> 使用低成本相机拍摄“溶液 + 多色块色卡编码”的照片，仅提取色卡区域 RGB 信号，重建溶液的可见光谱（spectrum），实现“相机 + 色卡 + 算法”式微型光谱仪。

---

## 1. 项目简介

传统光谱仪依赖光栅/棱镜等精密色散器件，成本与体积较高。本项目提出一种**低成本相机光谱重建**方案：在比色皿（cuvette）后放置**多色块色卡（例如 4×6=24 patches）**作为光学编码器。相机采集到的 RGB 不是“溶液本身的颜色”，而是**光源经溶液吸收 + 色块滤光/反射编码后**的混合响应。我们仅对图像中色卡色块区域提取 RGB（而非整幅图像高光谱重建），并训练神经网络学习从有限 3 通道信号到高维光谱的映射。

**核心目标**  
- Input: 色卡多色块 RGB 编码向量（来自图像 ROI 提取）  
- Output: 溶液真实光谱（由科学级光谱仪测得的 ground truth）

---

## 2. 方法概述（What we do）

整体流程可分为四步：

1) **数据采集（Acquisition）**  
- 配制大量样本溶液  
- 用科学级光谱仪采集真实光谱（GT spectrum）  
- 用相机在统一光源下拍摄：`光源 → 溶液 → 色卡 → 相机`

2) **数据处理（Pre-processing）**  
- 色卡检测/对齐，分割出每个色块（patch）ROI  
- 每个 patch 随机采样多个像素点，形成更稳定的 RGB 表征  
- 可选：灰度/白平衡/曝光校正，剔除异常样本

3) **模型训练（Training）**  
- 以“色卡 RGB 编码向量”为输入  
- 以“光谱仪真实光谱”为监督信号  
- 训练神经网络回归全光谱（可选加入物理先验约束，例如平滑/基线约束等）

4) **推理与评估（Inference & Evaluation）**  
- 输入新拍摄样本的色卡 RGB  
- 输出预测光谱，并用 MAE/RMSE/相关系数等指标评估

---

## 3. 代码结构（Repository Structure）

> 典型最小闭环：`dataset.py / model.py / train.py / predict.py`

建议组织如下（你可按现状调整）：

```text
.
├── dataset.py              # 数据集读取、归一化、划分
├── model.py                # 网络结构（如残差 CNN/MLP 等）
├── train.py                # 训练入口（保存模型、日志）
├── predict.py              # 推理入口（加载权重、输出光谱）
├── requirements.txt        # 环境依赖
├── README.md
└── data/                   # 数据（建议 gitignore，不直接上传）
    ├── rgb/                # RGB npy
    └── spectrum/           # 光谱 GT npy

<img width="2816" height="1536" alt="image" src="https://github.com/user-attachments/assets/0fb6160f-da1e-4b97-a47f-3676ef833d5f" />

![image](https://github.com/user-attachments/assets/7726a5ec-9300-4484-8eff-17cbdbf36c85)
