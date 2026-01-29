"""
联邦学习训练主脚本
直接运行此文件开始训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader, random_split, Subset

# 添加当前目录到Python路径
sys.path.append('.')
from simple_model import create_model
from simple_client import SimpleFLClient, create_client
from simple_server import SimpleFLServer


def load_mnist_data(
        num_clients: int = 10,
        batch_size: int = 32,
        iid: bool = True,
        data_dir: str = './data'
) -> List[DataLoader]:
    """
    加载MNIST数据并分割给多个客户端

    Args:
        num_clients: 客户端数量
        batch_size: 批次大小
        iid: 是否使用独立同分布数据
        data_dir: 数据目录

    Returns:
        List[DataLoader]: 客户端数据加载器列表
    """
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载MNIST训练集
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )

    # 分割数据给客户端
    if iid:
        # IID：均匀随机分割
        client_loaders = []
        data_size = len(trainset)
        client_size = data_size // num_clients

        for i in range(num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < num_clients - 1 else data_size

            client_data = Subset(trainset, list(range(start_idx, end_idx)))
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            client_loaders.append(client_loader)

            print(f"客户端 {i}: {len(client_data)} 个样本")
    else:
        # Non-IID：按标签分割，模拟真实世界数据分布
        # 每个客户端只有少数几个类别的数据
        client_loaders = []

        # 按标签分组数据
        label_to_indices = {}
        for idx, (_, label) in enumerate(trainset):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # 分配数据给客户端
        labels_per_client = 2  # 每个客户端只有2个类别
        client_indices = [[] for _ in range(num_clients)]

        # 随机分配类别给客户端
        all_labels = list(range(10))
        np.random.shuffle(all_labels)

        for client_id in range(num_clients):
            client_labels = all_labels[client_id * labels_per_client:
                                       (client_id + 1) * labels_per_client]

            for label in client_labels:
                indices = label_to_indices[label]
                num_samples = len(indices) // num_clients
                start_idx = client_id * num_samples
                end_idx = (client_id + 1) * num_samples if client_id < num_clients - 1 else len(indices)

                client_indices[client_id].extend(indices[start_idx:end_idx])

            print(f"客户端 {client_id}: {len(client_indices[client_id])} 个样本, "
                  f"类别: {client_labels}")

        # 创建数据加载器
        for indices in client_indices:
            client_data = Subset(trainset, indices)
            client_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
            client_loaders.append(client_loader)

    return client_loaders


def load_test_data(batch_size: int = 100, data_dir: str = './data') -> DataLoader:
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    print(f"测试集: {len(testset)} 个样本")

    return test_loader


def create_clients(
        num_clients: int,
        model_template: nn.Module,
        train_loaders: List[DataLoader],
        config: Dict[str, Any]
) -> List[SimpleFLClient]:
    """创建客户端列表"""
    clients = []

    for i in range(num_clients):
        # 为每个客户端创建独立的模型副本
        client_model = create_model(
            model_type=config.get('model_type', 'mlp'),
            num_classes=10,
            dataset='mnist'
        )

        client = create_client(
            client_id=i,
            model=client_model,
            train_loader=train_loaders[i],
            config={
                'local_epochs': config.get('local_epochs', 1),
                'learning_rate': config.get('learning_rate', 0.01),
                'batch_size': config.get('batch_size', 32),
                'device': config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            }
        )

        clients.append(client)
        print(f"客户端 {i} 创建完成，数据量: {client.num_samples}")

    return clients


def main():
    """主函数"""
    print("=" * 60)
    print("联邦学习训练开始")
    print("=" * 60)

    # 配置参数
    config = {
        # 联邦学习参数
        'num_clients': 10,# 增加客户端数量
        'num_rounds': 20,# 增加训练轮次
        'fraction': 0.3,  # 每轮选择的客户端比例
        'aggregation_method': 'fedavg',

        # 客户端训练参数
        'local_epochs': 1,
        'learning_rate': 0.01,
        'batch_size': 32,

        # 模型参数
        'model_type': 'cnn',  # 'mlp' 或 'cnn'

        # 数据参数
        'iid': True,  # 是否使用IID数据分布

        # 设备
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

        # 保存路径
        'save_dir': './federated_learning_results',
    }

    print(f"设备: {config['device']}")
    print(f"模型类型: {config['model_type']}")
    print(f"数据分布: {'IID' if config['iid'] else 'Non-IID'}")

    # 1. 准备数据
    print("\n1. 准备数据...")
    train_loaders = load_mnist_data(
        num_clients=config['num_clients'],
        batch_size=config['batch_size'],
        iid=config['iid']
    )

    test_loader = load_test_data(batch_size=100)

    # 2. 创建全局模型
    print("\n2. 创建全局模型...")
    global_model = create_model(
        model_type=config['model_type'],
        num_classes=10,
        dataset='mnist'
    )

    # 3. 创建服务器
    print("\n3. 创建服务器...")
    server = SimpleFLServer(global_model, config)

    # 4. 创建客户端
    print("\n4. 创建客户端...")
    clients = create_clients(
        num_clients=config['num_clients'],
        model_template=global_model,
        train_loaders=train_loaders,
        config=config
    )

    # 5. 初始评估
    print("\n5. 初始评估...")
    initial_loss, initial_acc = server.evaluate(test_loader)
    print(f"初始模型 - 测试集准确率: {initial_acc:.2f}%")

    # 6. 联邦学习训练
    print("\n6. 开始联邦学习训练...")
    print("=" * 60)

    for round_idx in range(config['num_rounds']):
        # 运行一轮训练
        round_results = server.run_training_round(clients, test_loader)

        # 打印本轮结果
        print(f"\n第 {round_results['round']} 轮结果:")
        print(f"  选择的客户端: {round_results['selected_clients']}")
        print(f"  有效更新数: {round_results['num_valid_updates']}")
        print(f"  测试准确率: {round_results.get('test_accuracy', 0):.2f}%")
        print(f"  本轮耗时: {round_results['round_time']:.2f}秒")
        print("-" * 40)

    # 7. 最终评估和保存
    print("\n7. 最终评估...")
    final_loss, final_acc = server.evaluate(test_loader)
    print(f"最终模型 - 测试集准确率: {final_acc:.2f}%")
    print(f"提升: {final_acc - initial_acc:.2f}%")

    # 保存结果
    server.save_results()
    server.print_summary()

    # 8. 保存最终模型
    final_model_path = Path(config['save_dir']) / 'final_model.pth'
    torch.save({
        'model_state_dict': server.global_model.state_dict(),
        'config': config,
        'best_accuracy': server.best_accuracy,
        'test_accuracy': final_acc
    }, final_model_path)

    print(f"\n最终模型已保存: {final_model_path}")
    print("=" * 60)
    print("联邦学习训练完成！")
    print("=" * 60)



if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 运行主函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback

        traceback.print_exc()