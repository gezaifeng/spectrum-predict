# 色块提取前端

从card_detection_26.5.29恢复的实际源码。安装requirements-extra.txt后从此目录运行`python ui_main.py`。
默认网格6×12；normal为每块多点均值，aberration为每块9点，extract_repeats控制重复提取。
输出RGB、ratio、log_ratio的语义和缩放不同，请与训练配置匹配。
原始GUI未在本次完成交互视觉验证；测试只覆盖模型数据入口。
