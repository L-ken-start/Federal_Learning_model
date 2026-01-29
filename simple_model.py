"""
简单的神经网络模型定义
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络，适用于MNIST/CIFAR-10
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1 if num_classes == 10 else 3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层1
        x = self.pool(F.relu(self.conv1(x)))

        # 卷积层2
        x = self.pool(F.relu(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 输出层
        x = self.fc2(x)

        return x


class SimpleMLP(nn.Module):
    """
    简单的多层感知机，适用于简单分类任务
    """

    def __init__(self, input_size=784, num_classes=10):
        super(SimpleMLP, self).__init__()
        # 隐藏层
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # 输出层
        self.fc3 = nn.Linear(64, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 展平输入
        x = x.view(x.size(0), -1)

        # 隐藏层1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 隐藏层2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # 输出层
        x = self.fc3(x)

        return x


def create_model(model_type='mlp', num_classes=10, dataset='mnist'):
    """
    创建模型实例

    Args:
        model_type: 模型类型，'mlp' 或 'cnn'
        num_classes: 类别数量
        dataset: 数据集名称，用于确定输入维度

    Returns:
        nn.Module: 模型实例
    """
    if model_type == 'cnn':
        return SimpleCNN(num_classes)
    else:  # 默认使用MLP
        if dataset == 'mnist':
            input_size = 784  # 28*28
        elif dataset == 'cifar10':
            input_size = 3072  # 32*32*3
        else:
            input_size = 784  # 默认

        return SimpleMLP(input_size, num_classes)


if __name__ == "__main__":
    # 测试模型
    model = create_model('cnn', num_classes=10)
    print(f"模型结构:\n{model}")

    # 测试前向传播
    if model.__class__.__name__ == 'SimpleCNN':
        test_input = torch.randn(4, 1, 28, 28)  # MNIST-like
    else:
        test_input = torch.randn(4, 1, 28, 28)

    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")