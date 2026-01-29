"""
简化的联邦学习服务器
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class SimpleFLServer:
    """
    简化的联邦学习服务器
    """

    def __init__(
            self,
            global_model: nn.Module,
            config: Dict[str, Any],
            logger: Optional[Any] = None
    ):
        """
        初始化服务器

        Args:
            global_model: 全局模型
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config

        # 基础配置
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.num_clients = config.get('num_clients', 10)
        self.num_rounds = config.get('num_rounds', 50)
        self.fraction = config.get('fraction', 0.3)
        self.aggregation_method = config.get('aggregation_method', 'fedavg')

        # 模型
        self.global_model = global_model.to(self.device)

        # 训练历史
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'client_selection': [],
            'round_time': [],
            'timestamp': []
        }

        # 状态
        self.current_round = 0
        self.best_accuracy = 0.0
        self.best_model_state = None

        # 保存目录
        self.save_dir = Path(config.get('save_dir', './fl_results'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.logger = logger

        print(f"服务器初始化完成")
        print(f"设备: {self.device}")
        print(f"总轮次: {self.num_rounds}")
        print(f"客户端数量: {self.num_clients}")
        print(f"每轮选择比例: {self.fraction}")

    def select_clients(self) -> List[int]:
        """选择参与本轮训练的客户端"""
        num_selected = max(1, int(self.num_clients * self.fraction))
        selected = np.random.choice(self.num_clients, num_selected, replace=False).tolist()

        self.history['client_selection'].append(selected)

        print(f"第 {self.current_round} 轮选择客户端: {selected}")
        return selected

    def aggregate_updates(
            self,
            client_updates: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """聚合客户端更新"""

        # 提取模型状态和数据量
        model_states = [update['model_state'] for update in client_updates]
        num_samples = [update['num_samples'] for update in client_updates]

        total_samples = sum(num_samples)

        # 初始化聚合状态
        aggregated_state = {}

        # 获取所有键
        keys = model_states[0].keys()
        for key in keys:
            # 初始化加权和
            weighted_sum = torch.zeros_like(model_states[0][key])

            # 加权求和
            for state, weight in zip(model_states, num_samples):
                weighted_sum += state[key] * (weight / total_samples)

            aggregated_state[key] = weighted_sum

        print(f"聚合完成，客户端数量: {len(client_updates)}")
        return aggregated_state

    def run_training_round(
            self,
            clients: List[Any],
            test_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """运行一轮训练"""
        start_time = time.time()

        print(f"\n{'=' * 50}")
        print(f"开始第 {self.current_round}/{self.num_rounds} 轮训练")
        print(f"{'=' * 50}")

        # 1. 选择客户端
        selected_indices = self.select_clients()
        selected_clients = [clients[i] for i in selected_indices]

        # 2. 准备全局模型状态
        global_state = copy.deepcopy(self.global_model.state_dict())

        # 3. 客户端本地训练
        client_updates = []
        for idx, client in zip(selected_indices, selected_clients):
            try:
                update = client.local_train(global_state)
                client_updates.append(update)
                print(f"客户端 {idx} 训练完成")
            except Exception as e:
                print(f"客户端 {idx} 训练失败: {e}")
                continue

        if not client_updates:
            raise ValueError("没有客户端完成训练")

        # 4. 聚合更新
        aggregated_state = self.aggregate_updates(client_updates)

        # 5. 更新全局模型
        self.global_model.load_state_dict(aggregated_state)

        # 6. 评估全局模型
        round_results = {
            'round': self.current_round,
            'selected_clients': selected_indices,
            'num_valid_updates': len(client_updates)
        }

        if test_loader:
            test_loss, test_acc = self.evaluate(test_loader)
            round_results.update({
                'test_loss': test_loss,
                'test_accuracy': test_acc
            })

            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_acc)

            print(f"测试集 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")

            # 保存最佳模型
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_model_state = copy.deepcopy(self.global_model.state_dict())
                print(f"🎉 新的最佳准确率: {test_acc:.2f}%")

        # 记录本轮结果
        round_time = time.time() - start_time
        self.history['round_time'].append(round_time)
        self.history['rounds'].append(self.current_round)
        self.history['timestamp'].append(datetime.now())

        round_results['round_time'] = round_time

        print(f"第 {self.current_round} 轮完成 - 耗时: {round_time:.2f}秒")

        # 保存检查点
        if self.current_round % 10 == 0:
            self.save_checkpoint()

        # 更新轮次
        self.current_round += 1

        return round_results

    def evaluate(
            self,
            data_loader: torch.utils.data.DataLoader,
            criterion: Optional[nn.Module] = None
    ) -> Tuple[float, float]:
        """评估模型"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.global_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += target.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        return avg_loss, accuracy

    def save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'round': self.current_round,
            'global_model_state': self.global_model.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_accuracy': self.best_accuracy,
            'best_model_state': self.best_model_state
        }

        checkpoint_path = self.save_dir / f'checkpoint_round_{self.current_round:03d}.pt'
        torch.save(checkpoint, checkpoint_path)

        # 同时保存为JSON格式
        json_checkpoint = {
            'round': self.current_round,
            'best_accuracy': self.best_accuracy,
            'test_accuracy': self.history['test_accuracy'][-1] if self.history['test_accuracy'] else None,
            'round_time': self.history['round_time'][-1],
            'timestamp': datetime.now().isoformat()
        }

        json_path = self.save_dir / f'checkpoint_round_{self.current_round:03d}.json'
        with open(json_path, 'w') as f:
            json.dump(json_checkpoint, f, indent=2)

        print(f"检查点已保存: {checkpoint_path}")

    def save_results(self):
        """保存训练结果"""
        results = {
            'history': self.history,
            'config': self.config,
            'best_accuracy': self.best_accuracy,
            'final_round': self.current_round
        }

        results_path = self.save_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"训练结果已保存: {results_path}")

    def print_summary(self):
        """打印训练摘要"""
        print(f"\n{'=' * 60}")
        print("联邦学习训练摘要")
        print(f"{'=' * 60}")

        if self.history['test_accuracy']:
            best_idx = np.argmax(self.history['test_accuracy'])
            best_acc = self.history['test_accuracy'][best_idx]
            best_round = self.history['rounds'][best_idx]

            print(f"最佳准确率: {best_acc:.2f}% (第 {best_round} 轮)")
            print(f"最终准确率: {self.history['test_accuracy'][-1]:.2f}%")
        else:
            print("未进行测试集评估")

        print(f"总训练轮次: {self.current_round}")
        print(f"平均每轮时间: {np.mean(self.history['round_time']):.2f}秒")
        print(f"结果保存目录: {self.save_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    # 测试服务器
    from simple_model import create_model

    # 创建模型
    model = create_model('mlp', num_classes=10, dataset='mnist')

    # 配置
    config = {
        'num_clients': 10,
        'num_rounds': 5,
        'fraction': 0.3,
        'aggregation_method': 'fedavg',
        'save_dir': './test_results'
    }

    # 创建服务器
    server = SimpleFLServer(model, config)

    print(f"服务器创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")