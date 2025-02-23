```markdown

# MNIST MLP Project

## 项目简介
本项目实现了一个多层感知器（MLP）模型，用于对 MNIST 数据集中的手写数字进行分类。使用 PyTorch 框架构建和训练模型，并评估其性能。

## 依赖
本项目依赖以下 Python 库：
- `torch`
- `torchvision`
- `torchaudio`
- `ipykernel`

可以通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集
本项目使用 MNIST 数据集，该数据集包含 70,000 张手写数字图像（0-9），每张图像的大小为 28x28 像素。数据集将自动下载并存储在 `./data` 目录中。

## 超参数
在 `mnist_mlp.py` 文件中设置了以下超参数：
- `batch_size`: 每个批次的样本数量（默认为 64）
- `num_epochs`: 训练周期数（默认为 10）
- `learning_rate`: 优化的步长（默认为 0.001）

## 模型结构
模型由以下层组成：
- 输入层：784（28x28 展平）
- 隐藏层 1：128 个神经元
- 隐藏层 2：64 个神经元
- 输出层：10 个神经元（对应 10 个数字）

## 训练与评估
训练过程在 `mnist_mlp.py` 文件中实现。模型在每个训练周期结束时打印训练损失，并在所有训练周期结束后评估模型的性能，输出平均测试损失和准确率。

## 运行项目
要运行项目，请执行以下命令：
```bash
python mnist_mlp.py
```

## 代码结构
- `mnist_mlp.py`: 主训练和评估脚本
- `requirements.txt`: 项目依赖列表
- `test.py`: 测试 PyTorch 安装的脚本

## 许可证
本项目遵循 MIT 许可证。请查看 LICENSE 文件以获取更多信息。
```

