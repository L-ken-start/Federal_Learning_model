联邦学习系统 (Federated Learning System)
🚀 快速开始
1. 安装依赖
bash
pip install torch torchvision numpy matplotlib
2. 运行训练
bash
python run_federated_learning.py
3. 查看结果
bash
python show_results.py
📁 项目结构
text
├── simple_model.py          # 神经网络模型定义
├── simple_client.py         # 联邦学习客户端
├── simple_server.py         # 联邦学习服务器
├── run_federated_learning.py # 主训练脚本（直接运行这个）
├── show_results.py          # 可视化训练结果
├── check_results.py         # 文本结果检查
└── requirements.txt         # 依赖包列表
⚡ 一键训练
bash
# 运行以下命令即可开始联邦学习训练：
python run_federated_learning.py
训练完成后会自动：

✅ 保存训练结果到 ./federated_learning_results/

✅ 生成可视化图表 simple_training_plot.png

✅ 保存最佳模型 final_model.pth

📊 查看训练结果
方法1：图表查看
bash
python show_results.py
显示准确率和损失变化曲线，并打印训练统计信息。

方法2：文本查看
bash
python check_results.py
纯文本显示训练结果，无需matplotlib。

方法3：测试模型
bash
python test_trained_model.py --find_latest
自动查找最新模型并在测试集上评估。

⚙️ 配置说明
在 run_federated_learning.py 中可修改配置：

python
config = {
    'num_clients': 10,      # 客户端数量
    'num_rounds': 20,       # 训练轮次
    'fraction': 0.3,        # 每轮选择30%客户端
    'local_epochs': 1,      # 客户端本地训练轮次
    'learning_rate': 0.01,  # 学习率
    'model_type': 'mlp',    # 'mlp' 或 'cnn'
    'iid': True,            # 数据分布：True=IID, False=Non-IID
}
🎯 模型性能
预期结果：
MLP模型：96-98% 准确率

CNN模型：97-99% 准确率

训练时间：1-3分钟（CPU）

实际测试结果示例：
text
训练轮次: 20
初始准确率: 90.90%
最终准确率: 97.63%
最佳准确率: 97.70%
提升幅度: 6.73%
总训练时间: 87.9秒
🔧 常见问题
Q1: 内存不足？
减少 batch_size 到16

减少 num_clients 到5

使用MLP而不是CNN

Q2: 训练太慢？
确保安装了PyTorch GPU版本

减少 num_rounds 到10

保持 local_epochs 为1

Q3: 准确率不高？
增加 num_rounds 到50

增加 local_epochs 到3

尝试CNN模型

Q4: 报错缺少依赖？
bash
pip install torch torchvision numpy matplotlib
📈 结果解读
训练成功标志：
✅ 准确率持续上升（左边图向上）

✅ 损失持续下降（右边图向下）

✅ 曲线趋于平稳（后期变化小）

图表说明：
左图：准确率变化，越高越好

右图：损失变化，越低越好

红色星星：标记最佳准确率点

🎮 实验建议
尝试不同配置：
python
# 1. 使用CNN模型（准确率更高）
config['model_type'] = 'cnn'

# 2. 模拟真实场景（Non-IID数据）
config['iid'] = False

# 3. 增加训练强度
config['num_rounds'] = 50
config['local_epochs'] = 3

# 4. 使用不同聚合算法
config['aggregation_method'] = 'fedprox'  # 或 'scaffold'
📚 学习资源
联邦学习概念：
隐私保护：数据不出本地，只传模型

分布式训练：多个客户端协作学习

模型聚合：服务器整合客户端更新

本项目特点：
完整的联邦学习流程

支持IID/Non-IID数据分布

自动可视化结果

简单易用，一键运行

🆘 故障排除
如果遇到问题：
检查依赖：确保安装了所有包

查看错误信息：根据错误提示排查

降低配置：减少客户端数量或批次大小

重新运行：有时候重新运行即可

常见错误：
bash
# 如果显示缺少matplotlib：
pip install matplotlib

# 如果显示torch错误：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
🎉 开始你的联邦学习之旅！
只需三步：

安装依赖

运行训练

查看结果

bash
# 完整流程
pip install torch torchvision numpy matplotlib
python run_federated_learning.py
python show_results.py
祝您训练顺利！