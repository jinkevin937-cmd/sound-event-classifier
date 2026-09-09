# Contributing to Sound Event Classifier

感谢你对这个项目的关注！这是一个教学项目，欢迎提出问题和改进建议。

## 如何贡献

### 报告问题
- 使用 GitHub Issues 报告问题
- 描述问题时请包含：复现步骤、期望行为、实际行为、环境信息

### 提交代码
1. Fork 这个仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交改动：`git commit -m "feat: add your feature"`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request

### 代码规范
- 遵循 PEP 8 风格指南
- 添加适当的注释和文档字符串
- 确保所有测试通过：`pytest tests/`

## 开发环境

```bash
# 克隆仓库
git clone https://github.com/jinkevin937-cmd/sound-event-classifier.git
cd sound-event-classifier

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/
```

## 问题反馈

如果你发现任何问题或有改进建议，请在 GitHub Issues 中提出。

谢谢你的贡献！
