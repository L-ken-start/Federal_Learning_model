"""
简化的联邦学习客户端
"""
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import time
from typing import Dict, Tuple, Optional, Any


class SimpleFLClient:
    """
    简化的联邦学习客户端
    """

    def __init__(
            self,
            client_id: int,
            model: nn.Module,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化客户端

        Args:
            client_id: 客户端ID
            model: 本地模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            config: 配置字典
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 配置
        self.config = config or {}
        self.device = self.config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.local_epochs = self.config.get('local_epochs', 1)
        self.local_lr = self.config.get('learning_rate', 0.01)
        self.batch_size = self.config.get('batch_size', 32)

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.local_lr,
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'round_times': []
        }

        # 数据统计
        self.num_samples = len(train_loader.dataset) if train_loader else 0

    def local_train(self, global_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        本地训练

        Args:
            global_state: 全局模型状态

        Returns:
            Dict: 包含更新信息
        """
        start_time = time.time()

        # 加载全局模型参数
        self.model.load_state_dict(global_state)

        # 训练模式
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # 本地训练多个epoch
        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)

                # 打印进度
                if batch_idx % 50 == 0:
                    print(f"客户端 {self.client_id} - Epoch {epoch + 1}/{self.local_epochs} "
                          f"- Batch {batch_idx}/{len(self.train_loader)} "
                          f"- Loss: {loss.item():.4f}")

        # 计算平均损失和准确率
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        # 记录历史
        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)

        # 计算训练时间
        train_time = time.time() - start_time
        self.history['round_times'].append(train_time)

        print(f"客户端 {self.client_id} 训练完成 - "
              f"损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%, "
              f"时间: {train_time:.2f}秒")

        # 返回更新
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'num_samples': self.num_samples,
            'client_id': self.client_id,
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'train_time': train_time
        }

    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        评估模型

        Args:
            data_loader: 数据加载器

        Returns:
            Tuple[float, float]: (loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # 前向传播
                output = self.model(data)
                loss = self.criterion(output, target)

                # 统计
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def get_client_info(self) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples,
            'device': str(self.device),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'train_history': self.history
        }


def create_client(
        client_id: int,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: Optional[Dict[str, Any]] = None
) -> SimpleFLClient:
    """创建客户端实例"""
    return SimpleFLClient(client_id, model, train_loader, config=config)


if __name__ == "__main__":
    # 测试客户端
    import torchvision
    import torchvision.transforms as transforms

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 创建测试数据集
    from torch.utils.data import DataLoader, random_split

    # 使用MNIST的示例
    try:
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )

        # 分割为训练集和验证集
        train_size = int(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        train_dataset, val_dataset = random_split(trainset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 创建模型
        from simple_model import create_model

        model = create_model('mlp', num_classes=10, dataset='mnist')

        # 创建客户端
        client = SimpleFLClient(
            client_id=0,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={'local_epochs': 1, 'learning_rate': 0.01}
        )

        print(f"客户端 {client.client_id} 创建成功")
        print(f"数据量: {client.num_samples}")
        print(f"设备: {client.device}")

        # 测试本地训练
        global_state = model.state_dict()
        update = client.local_train(global_state)
        print(f"训练完成，更新大小: {len(update['model_state'])} 个参数")

    except Exception as e:
        print(f"测试客户端时出错: {e}")
        print("请确保已安装torchvision: pip install torchvision")