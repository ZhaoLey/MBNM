import os

from avalanche.models import SimpleCNN
from avalanche.models.resnet32 import resnet32

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决 OpenMP 错误

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitCIFAR100

from avalanche.training import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.plugins import EvaluationPlugin
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 创建 SplitCIFAR100 数据集
scenario = SplitCIFAR100(n_experiences=10, return_task_id=False)

# 定义模型，并将其移动到 GPU
# model = SimpleCNN(num_classes=scenario.n_classes).to(device)
model = resnet32(num_classes=scenario.n_classes).to(device)

# 定义优化器和损失函数
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)  # 调整学习率
criterion = CrossEntropyLoss()

# 定义日志记录器
interactive_logger = InteractiveLogger()
tensorboard_logger = TensorboardLogger()
text_logger = TextLogger(open("log.txt", "a"))
loggers = [interactive_logger, tensorboard_logger,text_logger]

# 定义评估插件
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),  # 记录任务准确率和数据流准确率
    forgetting_metrics(experience=True, stream=True),           # 记录遗忘度量
    loggers=loggers
)

# 定义 ReplayPlugin
replay_plugin = ReplayPlugin(mem_size=2000)

# 定义持续学习策略
strategy = Naive(
    model, optimizer, criterion,
    train_mb_size=128, train_epochs=5, eval_mb_size=128,  # 调整批量大小和训练周期
    plugins=[replay_plugin],
    evaluator=eval_plugin,
    device=device  # 指定使用 GPU
)

# 用于存储平均任务准确率
avg_accuracies = []

# 训练和评估循环
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # 训练模型
    strategy.train(experience)
    print('Training completed')

    # 评估模型
    print('Evaluating on current test set...')
    results = strategy.eval(scenario.test_stream)

    # 计算平均任务准确率
    avg_acc = 0.0
    for exp_id in range(experience.current_experience + 1):
        exp_key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{exp_id:03d}"
        if exp_key in results:
            avg_acc += results[exp_key]
    avg_acc /= (experience.current_experience + 1)
    avg_accuracies.append(avg_acc)
    print(f"Average Accuracy up to experience {experience.current_experience}: {avg_acc:.4f}")

# 绘制平均任务准确率折线图
plt.figure(figsize=(8, 5))
plt.plot(range(len(avg_accuracies)), avg_accuracies, marker='o', linestyle='-', color='b', label='Average Accuracy')
plt.title("Average Task Accuracy Over Experiences")
plt.xlabel("Experience")
plt.ylabel("Average Accuracy")
plt.xticks(range(len(avg_accuracies)), labels=[f"Exp {i}" for i in range(len(avg_accuracies))])
plt.ylim(0, 1)  # 设置纵轴范围为 0 到 1
plt.grid(True)
plt.legend()
plt.show()
