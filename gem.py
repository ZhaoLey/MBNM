import avalanche as avl
import torch
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from experiments.utils import set_seed, create_default_args
import matplotlib.pyplot as plt
from models.reduced_resnet18 import MultiHeadReducedResNet18
import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

# 现在可以正常导入 models 模块
from models.reduced_resnet18 import MultiHeadReducedResNet18


def gem_scifar100(override_args=None):
    """
    "Gradient Episodic Memory for Continual Learning" by Lopez-paz et. al. (2017).
    https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html
    """
    args = create_default_args({'cuda': 0, 'patterns_per_exp': 256, 'epochs': 5,
                                'mem_strength': 0.5, 'learning_rate': 0.1, 'train_mb_size': 10,
                                'seed': None}, override_args)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    benchmark = avl.benchmarks.SplitCIFAR100(10, return_task_id=True)

    model = MultiHeadReducedResNet18()
    criterion = CrossEntropyLoss()

    # interactive_logger = avl.logging.InteractiveLogger()
    # 定义日志记录器
    interactive_logger = InteractiveLogger()
    tensorboard_logger = TensorboardLogger()
    text_logger = TextLogger(open("log22.txt", "a"))
    loggers = [interactive_logger, tensorboard_logger, text_logger]

    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=loggers)

    cl_strategy = avl.training.GEM(
        model, SGD(model.parameters(), lr=args.learning_rate, momentum=0.9), criterion,
        patterns_per_exp=args.patterns_per_exp, memory_strength=args.mem_strength,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    # 用于存储平均任务准确率
    avg_accuracies = []
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        results = cl_strategy.eval(benchmark.test_stream)
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

    return results


if __name__ == '__main__':
    res = gem_scifar100()
    print(res)
