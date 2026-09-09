"""数据加载 + mel 特征提取（第 10 课配套项目）。

支持两种数据源：
1. ESC-50 真实环境声数据集（50 类，需先运行 download_data.py 下载）
2. 合成数据兜底（没下载 ESC-50 时，用几类合成"环境声"跑通流程）

特征：音频 → 重采样 16kHz → MelSpectrogram → 时间维平均 → 128 维向量
"""
import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram
    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


def load_audio(path):
    """读音频文件 → (waveform: (channels, samples) tensor, sr)。用 soundfile 避免 torchcodec 依赖。"""
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:               # 单声道 (samples,)
        data = data[np.newaxis, :]   # → (1, samples)
    else:                             # 多声道 (samples, channels)
        data = data.T                 # → (channels, samples)
    return torch.from_numpy(data).contiguous(), sr


SR = 16000            # 统一采样率
N_MELS = 128          # mel 频段数（= 模型输入维度）
DURATION = 5.0        # 每条音频取 5 秒


def extract_mel(waveform: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """把波形转成 128 维特征向量（mel 频谱 + 时间维平均）。"""
    if sr != SR:
        waveform = torchaudio.functional.resample(waveform, sr, SR)
    # 裁剪/填充到 5 秒
    target_len = int(SR * DURATION)
    if waveform.shape[-1] > target_len:
        waveform = waveform[..., :target_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
    # 单声道
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    mel_spec = MelSpectrogram(sample_rate=SR, n_mels=N_MELS)(waveform)  # (n_mels, time)
    mel_spec = torch.log(mel_spec + 1e-9)                                # log 压缩动态范围
    return mel_spec.mean(dim=-1)                                          # 时间维平均 → (128,)


class ESC50Dataset(Dataset):
    """ESC-50 数据集封装。root 需包含 meta/esc50.csv 和 audio/*.wav。"""

    def __init__(self, root: str, split: str = "train"):
        self.root = root
        self.meta = []
        with open(os.path.join(root, "meta", "esc50.csv"), encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fold = int(row["fold"])          # 官方 5 折划分（1-5）
                # fold 1-4 训练，fold 5 测试（官方推荐）
                in_train = fold in (1, 2, 3, 4)
                if split == "train" and in_train:
                    self.meta.append(row)
                elif split == "test" and not in_train:
                    self.meta.append(row)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta[idx]
        path = os.path.join(self.root, "audio", row["filename"])
        waveform, sr = load_audio(path)
        feat = extract_mel(waveform, sr)
        label = int(row["target"])
        return feat, label


class SyntheticDataset(Dataset):
    """合成"环境声"兜底数据（4 类，无需下载）。用于在没 ESC-50 时跑通流程。"""

    CLASSES = ["鸟鸣", "风噪", "警报", "雨声"]

    def __init__(self, samples_per_class: int = 80, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.data, self.labels = [], []
        for cls in range(len(self.CLASSES)):
            for _ in range(samples_per_class):
                self.data.append(self._synth(cls, rng))
                self.labels.append(cls)
        self.data = torch.stack(self.data)

    def _synth(self, cls: int, rng) -> torch.Tensor:
        t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
        if cls == 0:      # 鸟鸣：多个正弦波 + 频率调制
            w = (np.sin(2*np.pi*2000*t) + 0.5*np.sin(2*np.pi*3500*t)) * (0.5 + 0.5*np.sin(2*np.pi*5*t))
        elif cls == 1:    # 风噪：低频噪声
            w = np.cumsum(rng.standard_normal(len(t))) * 0.02
            w = w / (np.abs(w).max() + 1e-9)
        elif cls == 2:    # 警报：扫频正弦波
            f = 800 + 400 * np.sin(2*np.pi*1*t)
            w = np.sin(2*np.pi*np.cumsum(f)/SR)
        else:             # 雨声：带通白噪声
            w = rng.standard_normal(len(t))
            w = np.convolve(w, np.ones(16)/16, mode="same")
        w = w.astype(np.float32)
        w = w / (np.abs(w).max() + 1e-9)
        return extract_mel(torch.from_numpy(w).unsqueeze(0), SR)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloaders(data_root: str, batch_size: int = 32):
    """返回 train/val/test 三个 DataLoader，以及类别数。

    优先用 ESC-50（若 data_root 有 meta/esc50.csv），否则用合成数据。
    """
    if _HAS_TORCHAUDIO and os.path.exists(os.path.join(data_root, "meta", "esc50.csv")):
        full = ESC50Dataset(data_root, split="train")
        test = ESC50Dataset(data_root, split="test")
        n = len(full)
        n_val = int(n * 0.1)
        train, val = random_split(full, [n - n_val, n_val])
        num_classes = 50
        print(f"[ESC-50] train={len(train)} val={len(val)} test={len(test)} classes={num_classes}")
    else:
        synth = SyntheticDataset()
        n = len(synth)
        n_val = int(n * 0.15)
        train, val, test = random_split(
            synth, [n - 2*n_val, n_val, n_val],
            generator=torch.Generator().manual_seed(0))
        num_classes = len(SyntheticDataset.CLASSES)
        print(f"[合成数据兜底] train={len(train)} val={len(val)} test={len(test)} classes={num_classes}")
        print("  （未检测到 ESC-50，请先运行 python download_data.py 下载真实数据）")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader, num_classes
