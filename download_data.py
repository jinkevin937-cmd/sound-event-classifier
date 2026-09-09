"""下载 ESC-50 环境声数据集（第 10 课配套项目）。

ESC-50：2000 个 5 秒环境声，50 类（鸟叫、狗吠、雨声、警报...），约 600MB。

运行：
    python download_data.py            # 下载到 ./data/ESC-50
"""
import os
import sys
import zipfile
import urllib.request

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
TARGET = os.path.join("data", "ESC-50")


def download(url, dest):
    print(f"正在下载 {url} ...（约 600MB，请耐心等待）")
    def _hook(blocks, block_size, total_size):
        done = blocks * block_size
        if total_size > 0:
            pct = done * 100 / total_size
            print(f"\r  进度 {pct:.1f}%", end="")
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()


def main():
    os.makedirs("data", exist_ok=True)
    zip_path = os.path.join("data", "esc50.zip")

    if os.path.exists(os.path.join(TARGET, "meta", "esc50.csv")):
        print(f"ESC-50 已存在：{TARGET}")
        return

    if not os.path.exists(zip_path):
        try:
            download(URL, zip_path)
        except Exception as e:
            print(f"\n自动下载失败：{e}")
            print("请手动下载 ESC-50 并解压到 data/ESC-50/：")
            print("  https://github.com/karolpiczak/ESC-50")
            sys.exit(1)

    print("正在解压...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall("data")
    # GitHub zip 解压后是 ESC-50-master/，重命名为 ESC-50/
    extracted = os.path.join("data", "ESC-50-master")
    if os.path.exists(extracted) and not os.path.exists(TARGET):
        os.rename(extracted, TARGET)
    print(f"完成！数据在 {TARGET}/ （meta/esc50.csv + audio/）")


if __name__ == "__main__":
    main()
