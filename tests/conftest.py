import sys
import os
# 把项目根目录加到 Python path，这样 tests/ 里的 import 能找到 model.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
