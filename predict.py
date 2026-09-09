"""单条音频推理（第 10 课配套项目）。

运行：
    python predict.py --model sound_mlp.pth --audio 某段.wav
（可选 --labels data/ESC-50/meta/esc50.csv 读取真实类别名）
"""
import argparse
import csv
import os

import torch

from model import SoundMLP
from dataset import extract_mel, SyntheticDataset, load_audio


def load_labels(csv_path, num_classes):
    """从 ESC-50 的 meta/esc50.csv 读类别名（按 target 排序去重）。"""
    if not csv_path or not os.path.exists(csv_path):
        return None
    id2name = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id2name[int(row["target"])] = row["category"]
    return [id2name[i] for i in range(num_classes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sound_mlp.pth")
    parser.add_argument("--audio", required=True, help="要识别的音频文件路径")
    parser.add_argument("--labels", default=None, help="ESC-50 的 meta/esc50.csv（读类别名）")
    parser.add_argument("--num_classes", type=int, default=50)
    args = parser.parse_args()

    # 1. 加载模型
    model = SoundMLP(input_dim=128, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # 2. 加载音频 + 提特征
    waveform, sr = load_audio(args.audio)
    feat = extract_mel(waveform, sr).unsqueeze(0)   # (1, 128)

    # 3. 推理
    with torch.no_grad():
        logits = model(feat)
        probs = torch.softmax(logits, dim=1)
        pred = int(logits.argmax(dim=1))

    # 4. 输出
    names = load_labels(args.labels, args.num_classes)
    if names:
        name = names[pred]
    elif args.num_classes == len(SyntheticDataset.CLASSES):
        name = SyntheticDataset.CLASSES[pred]
    else:
        name = str(pred)
    print(f"识别结果：{name}（类别 {pred}，置信度 {probs[0, pred].item():.3f}）")


if __name__ == "__main__":
    main()
