import torch
from model import SoundMLP


def test_output_shape():
    """输入 (4,128)，输出应该是 (4,50)。"""
    model = SoundMLP(input_dim=128, num_classes=50)
    x = torch.randn(4, 128)
    assert model(x).shape == (4, 50)


def test_forward_pass():
    """前向传播不应该报错。"""
    model = SoundMLP(input_dim=128, num_classes=50)
    x = torch.randn(2, 128)
    output = model(x)
    assert output is not None
