"""声音事件分类 MLP 模型（第 10 课配套项目）。

输入：mel 频谱的平均特征向量（128 维）
输出：环境声类别概率（ESC-50 为 50 类）

用法：
    from model import SoundMLP
    model = SoundMLP(input_dim=128, num_classes=50)
"""
import torch
import torch.nn as nn


class SoundMLP(nn.Module):
    """多层感知机：mel 特征 → 隐藏层 → 类别。

    结构：128 → 64 → 32 → num_classes（与第 10 课课件一致）
    """

    def __init__(self, input_dim: int = 128, num_classes: int = 50,
                 hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),   # 第 1 层：128 → 64
            nn.ReLU(),                        # 非线性激活
            nn.Dropout(dropout),              # 防过拟合：随机关神经元
            nn.Linear(hidden1, hidden2),      # 第 2 层：64 → 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),  # 输出层：32 → 类别数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：输入 (batch, 128)，输出 (batch, num_classes) 的 logits。"""
        return self.net(x)


if __name__ == "__main__":
    # 快速自测：随机输入，看输出形状对不对
    model = SoundMLP(input_dim=128, num_classes=50)
    x = torch.randn(4, 128)          # 模拟 batch=4 的 mel 特征
    logits = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", logits.shape)   # 期望 (4, 50)
    print("预测类别:", logits.argmax(dim=1).tolist())
