"""训练声音事件分类器（第 10 课配套项目）。

完整训练流程：加载数据 → 定义模型 → 训练 → 验证 → 测试评估 → 保存模型。

运行：
    python train.py --data_dir data/ESC-50 --epochs 20
（未下载 ESC-50 时会自动用合成数据跑通流程）
"""
import argparse
import torch
import torch.nn as nn

from model import SoundMLP
from dataset import get_dataloaders


def evaluate(model, loader, criterion):
    """在给定数据上算平均 loss 和准确率。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for feat, label in loader:
            logits = model(feat)
            total_loss += criterion(logits, label).item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            total += len(label)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ESC-50", help="ESC-50 数据目录（含 meta/ 和 audio/）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", default="sound_mlp.pth")
    args = parser.parse_args()

    torch.manual_seed(0)

    # 1. 数据
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.data_dir, args.batch_size)

    # 2. 模型 + 损失 + 优化器
    model = SoundMLP(input_dim=128, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. 训练循环
    print("\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for feat, label in train_loader:
            optimizer.zero_grad()               # 清空上一轮梯度
            logits = model(feat)                # 前向传播
            loss = criterion(logits, label)     # 计算损失
            loss.backward()                     # 反向传播（算梯度）
            optimizer.step()                    # 更新参数
            total_loss += loss.item() * len(label)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

    # 4. 测试集评估
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n测试集：loss {test_loss:.4f}，准确率 {test_acc:.3f}")

    # 5. 保存模型
    torch.save(model.state_dict(), args.save)
    print(f"模型已保存到 {args.save}")


if __name__ == "__main__":
    main()
