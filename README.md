# 声音事件分类小项目（第 10 课配套）

一个完整的「声音识别」最小项目：用 MLP 把音频（mel 频谱特征）分类成环境声类别（如鸟鸣、狗吠、雨声…）。对应第 10 课《深度学习基础》的 MLP 实战。

## 项目结构

```
code/
├── REPORT.md         # 📄 项目报告 + 逐步骤讲解解析（先读这个）
├── model.py          # MLP 模型定义（128→64→32→类别数）
├── dataset.py        # 数据加载 + mel 特征提取（含合成数据兜底）
├── train.py          # 完整训练脚本（训练/验证/测试/保存）
├── predict.py        # 单条音频推理
├── download_data.py  # 下载 ESC-50 真实数据集
└── requirements.txt  # 依赖
```

## 快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt
```

### 2. 跑通流程（用合成数据，无需下载）

直接训练，会自动用 4 类合成"环境声"（鸟鸣/风噪/警报/雨声）跑通全流程：

```bash
python train.py
```

### 3. 用真实数据（ESC-50，50 类环境声）

数据下载（三选一）：官网 https://github.com/karolpiczak/ESC-50 ，或直接下载 zip https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip ，或一键：

```bash
python download_data.py          # 自动下载 ESC-50（约 600MB）
python train.py --data_dir data/ESC-50 --epochs 20
```

### 4. 推理单条音频

```bash
python predict.py --model sound_mlp.pth --audio data/ESC-50/audio/1-100032-A-0.wav --labels data/ESC-50/meta/esc50.csv
```

## 技术要点（对应课件）

| 步骤 | 实现 | 课件位置 |
|------|------|---------|
| 特征 | 音频 → MelSpectrogram → log → 时间维平均 → 128 维 | 3.2 节 |
| 模型 | MLP：128→64→32→50（ReLU + Dropout） | 1.2 多层网络 |
| 前向 | `logits = model(x)` | 2.1 单层前向 |
| 损失 | CrossEntropyLoss | 第 11 课 2.1 |
| 优化 | Adam | 第 11 课 2.3 |
| 训练循环 | zero_grad → forward → loss → backward → step | 第 11 课 1.1 |

## 数据说明

- **ESC-50**：2000 个 5 秒环境声，50 类，官方按 fold 1-4 训练 / fold 5 测试。
- **合成数据兜底**：没网络/没下载时，自动生成 4 类合成声，让你先跑通「加载→训练→评估→推理」的完整闭环，理解流程后再换真实数据。

## 预期结果（实测）

| 数据 | 结果 | 说明 |
|------|------|------|
| 合成数据（4 类） | val_acc ≈ 100% | 合成数据太简单，验证流程用 |
| ESC-50 真实数据（50 类） | 测试 acc ≈ **20%**（随机是 2%） | MLP 学到东西了，但 50 类复杂环境声对 MLP 太难 |

> 💡 **为什么 MLP 只有 20%**：这正是课件 3.3 节陷阱说的——MLP 把 mel 谱平均成 128 维向量，**丢失了"时间先后"和"频率邻近"结构**。音频分类真正的主力是 CNN（第 11 课前沿会讲 PANNs 等，能在 ESC-50 上到 90%+）。这个项目让你亲手验证"MLP 是起点、不是终点"。
# Demo PR Test
