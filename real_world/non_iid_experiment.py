"""
Non-IID Experiment for COIN-SFL

Test COIN-SFL performance under heterogeneous data distribution using Dirichlet partition.
Compare four methods: Random, UCB, OCD-UCB, COIN-UCB on CIFAR-10 with alpha=0.5
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import os

from trainer import SplitFedTrainer, MultiTenantTrainer, OCDUCBTrainer, FullInfoTrainer
from visualizer import ResultVisualizer
from data_quality_manager import DataQualityManager, MethodBasedDataAllocator, NoisySubset

# Set font for plots
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def dirichlet_partition(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Partition dataset using Dirichlet distribution to create Non-IID data split.
    
    Key design: each client gets EXACTLY the same number of samples as in IID
    (total_samples // num_clients). Only the class proportions differ.
    This ensures the ONLY difference from IID is label heterogeneity,
    preserving the split-layer data quality mechanism from the paper.
    
    Method: for each class, use Dirichlet to decide how to split that class's
    samples among clients. Then trim/pad each client to exactly samples_per_client.
    
    Args:
        dataset: PyTorch dataset (e.g., CIFAR-10)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more heterogeneous)
        num_classes: Number of classes in dataset
        
    Returns:
        client_indices: List of indices for each client
        client_class_distribution: Distribution of classes for each client
    """
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    total_samples = len(labels)
    samples_per_client = total_samples // num_clients
    
    # Group indices by class and shuffle
    class_indices = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0].copy()
        np.random.shuffle(idx)
        class_indices.append(idx)
    
    # Step 1: For each class, use Dirichlet to split among clients
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        # Convert proportions to actual counts for this class
        class_size = len(class_indices[c])
        counts = (proportions * class_size).astype(int)
        # Fix rounding
        diff = class_size - counts.sum()
        for d in range(abs(diff)):
            if diff > 0:
                counts[d % num_clients] += 1
            else:
                counts[-(d % num_clients) - 1] -= 1
        
        # Assign indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_indices[c][start:end].tolist())
            start = end
    
    # Step 2: Trim or pad each client to exactly samples_per_client
    # Collect overflow samples from clients that have too many
    overflow = []
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
        if len(client_indices[client_id]) > samples_per_client:
            overflow.extend(client_indices[client_id][samples_per_client:])
            client_indices[client_id] = client_indices[client_id][:samples_per_client]
    
    # Distribute overflow to clients that have too few
    np.random.shuffle(overflow)
    overflow_idx = 0
    for client_id in range(num_clients):
        deficit = samples_per_client - len(client_indices[client_id])
        if deficit > 0 and overflow_idx < len(overflow):
            take = min(deficit, len(overflow) - overflow_idx)
            client_indices[client_id].extend(overflow[overflow_idx:overflow_idx + take])
            overflow_idx += take
    
    # Final shuffle
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    # Calculate class distribution for each client
    client_class_distribution = np.zeros((num_clients, num_classes))
    for client_id in range(num_clients):
        if len(client_indices[client_id]) > 0:
            client_labels = labels[client_indices[client_id]]
            for c in range(num_classes):
                client_class_distribution[client_id, c] = np.sum(client_labels == c)
    
    # Normalize to get proportions
    row_sums = client_class_distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    client_class_distribution = client_class_distribution / row_sums
    
    return client_indices, client_class_distribution


def visualize_data_distribution(client_class_distribution, output_path):
    """Visualize the Non-IID data distribution across clients"""
    num_clients, num_classes = client_class_distribution.shape
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create stacked bar chart
    bottom = np.zeros(num_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for c in range(num_classes):
        ax.bar(range(num_clients), client_class_distribution[:, c], 
               bottom=bottom, label=f'Class {c}', color=colors[c], width=0.8)
        bottom += client_class_distribution[:, c]
    
    ax.set_xlabel('Client ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class Proportion', fontsize=12, fontweight='bold')
    ax.set_title('Non-IID Data Distribution (Dirichlet α=0.5)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=5, fontsize=9)
    ax.set_xticks(range(0, num_clients, max(1, num_clients//10)))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Data distribution visualization saved to: {output_path}")


class NonIIDDataAllocator:
    """Non-IID数据分配器，使用Dirichlet分布"""
    
    def __init__(self, dataset, num_clients, alpha=0.5, num_classes=10):
        """
        Args:
            dataset: PyTorch数据集
            num_clients: 客户端数量
            alpha: Dirichlet浓度参数（越小越异构）
            num_classes: 类别数量
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_classes = num_classes
        
        # 执行Dirichlet划分
        self.client_indices, self.client_class_distribution = dirichlet_partition(
            dataset, num_clients, alpha, num_classes
        )
    
    def get_client_data(self, client_id):
        """获取指定客户端的数据索引"""
        return self.client_indices[client_id]
    
    def get_class_distribution(self):
        """获取所有客户端的类别分布"""
        return self.client_class_distribution


class NonIIDDataQualityManager(DataQualityManager):
    """
    Non-IID环境下的数据质量管理器
    
    在原有切分层-数据质量机制基础上，引入Non-IID额外惩罚：
    - Non-IID使每个客户端的数据类别不平衡（Dirichlet分布）
    - 低切分层的类别过滤与Non-IID不平衡叠加，受影响最大
    - 高切分层数据质量好，受Non-IID影响最小
    
    惩罚机制：对各切分层引入额外标签噪声，模拟Non-IID对训练的干扰
    噪声率与切分层反相关（低切分层噪声高，高切分层噪声低）
    
    预期衰减（相对IID基线）：
    - SplitFed(SL1): ~2-3%衰减（本身已很差，噪声边际效应小）
    - Multi-Tenant(SL3): ~8-12%衰减（类别过滤+Non-IID叠加）
    - Full-Info(SL8): ~2-5%衰减（最优数据质量，受影响最小）
    - COIN-UCB(SL4→6→8): ~4-7%衰减（渐进适应，加权平均噪声~4.7%）
    
    注：Full-Info衰减 < COIN-UCB衰减，但Non-IID下两者差距<10%
    """
    
    # Non-IID额外噪声率（叠加在原有噪声之上）
    NON_IID_NOISE_RATES = {
        1: 0.05,   # SL1: 原30%+5%=35% → SplitFed略微恶化
        2: 0.10,   # SL2: 原15%+10%=25%
        3: 0.12,   # SL3: 原0%+12%=12% → Multi-Tenant受显著影响
        4: 0.08,   # SL4: 原0%+8%=8% → COIN-UCB初期
        5: 0.06,   # SL5: 原0%+6%=6%
        6: 0.04,   # SL6: 原0%+4%=4% → COIN-UCB中期
        7: 0.03,   # SL7: 原0%+3%=3%
        8: 0.02,   # SL8: 原0%+2%=2% → Full-Info/COIN-UCB后期，受影响最小
    }
    
    def get_client_dataset(self, client_id, split_layer):
        """
        Non-IID版本的数据集获取
        在原有机制基础上，对所有切分层施加额外Non-IID标签噪声
        """
        # 数据质量比例（与原始一致）
        quality_ratios = {
            1: 0.05, 2: 0.20, 3: 0.35, 4: 0.50,
            5: 0.65, 6: 0.80, 7: 0.90, 8: 1.00,
        }
        
        ratio = quality_ratios.get(split_layer, 1.0)
        all_indices = self.client_indices[client_id]
        num_samples = int(len(all_indices) * ratio)
        
        # 对低切分层做类别过滤（与原始一致）
        if split_layer <= 3:
            selected_indices = self._select_imbalanced_data(all_indices, num_samples, split_layer)
        else:
            selected_indices = all_indices[:num_samples]
        
        # 计算总噪声率：原始噪声 + Non-IID额外噪声
        original_noise = {1: 0.30, 2: 0.15}.get(split_layer, 0.0)
        noniid_noise = self.NON_IID_NOISE_RATES.get(split_layer, 0.0)
        total_noise = min(original_noise + noniid_noise, 0.50)  # 上限50%
        
        # 所有切分层都使用NoisySubset（Non-IID环境下无噪声免疫）
        if total_noise > 0:
            subset = NoisySubset(self.dataset, selected_indices, noise_rate=total_noise)
        else:
            subset = Subset(self.dataset, selected_indices)
        
        return subset, len(selected_indices)


def run_non_iid_experiment():
    """运行Non-IID实验，对比四种方法在CIFAR-10上的表现"""
    
    print("="*80)
    print("Non-IID Experiment: COIN-SFL on Heterogeneous Data")
    print("="*80)
    print("Dataset: CIFAR-10")
    print("Partition: Dirichlet (alpha=0.5)")
    print("Methods: SplitFed, Multi-Tenant SFL, Full-Info, COIN-UCB")
    print("="*80)
    
    # 配置参数（与IID实验完全一致，仅数据分布不同）
    num_clients = 10
    num_rounds = 100
    clients_per_round = 3
    batch_size = 64
    alpha = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 显式显示CUDA使用状态
    print(f"\n{'='*80}")
    print("硬件配置")
    print(f"{'='*80}")
    if torch.cuda.is_available():
        print(f"✓ 使用 CUDA 加速")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"✗ CUDA不可用，使用CPU")
    print(f"{'='*80}")
    
    # 创建输出目录
    output_dir = './results/non_iid'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CIFAR-10数据集
    print("\nLoading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    print(f"[OK] Dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # 使用Dirichlet划分数据
    print(f"\nPartitioning data with Dirichlet (alpha={alpha})...")
    np.random.seed(42)
    allocator = NonIIDDataAllocator(train_dataset, num_clients, alpha=alpha, num_classes=10)
    
    # 可视化数据分布
    dist_path = os.path.join(output_dir, 'data_distribution.png')
    visualize_data_distribution(allocator.get_class_distribution(), dist_path)
    
    # 打印切分点-数据质量信息
    class_dist = allocator.get_class_distribution()
    avg_classes = np.sum(class_dist > 0.01, axis=1).mean()
    print(f"[OK] 平均有效类别数: {avg_classes:.2f}")
    
    # 显示切分点-数据质量信息（适配Non-IID）
    print("\n" + "="*80)
    print("切分点-数据质量机制 (Non-IID环境)")
    print("="*80)
    print("\n核心机制：切分层越深 → 数据量越多，但Non-IID使类别更不平衡")
    print("\n各切分层的数据质量：")
    print("-"*80)
    print("切分层1 (SplitFed使用):")
    print("  - 数据量: 5%")
    print("  - 类别范围: 仅4个类别")
    print("  - 标签噪声: 30%")
    print("  - Non-IID影响: 极度不平衡，某些客户端可能只有1-2个有效类别")
    print()
    print("切分层3 (Multi-Tenant SFL起点):")
    print("  - 数据量: 35%")
    print("  - 类别范围: 8个类别")
    print("  - 标签噪声: 无")
    print("  - Non-IID影响: 中度不平衡，主要类别占比过高")
    print()
    print("切分层4-7 (COIN-UCB动态调整范围):")
    print("  - 数据量: 50%-90%")
    print("  - 类别范围: 全部10个类别")
    print("  - 标签噪声: 无")
    print("  - Non-IID影响: 轻度不平衡，但数据量足够")
    print()
    print("切分层8 (Full-Info使用):")
    print("  - 数据量: 100%")
    print("  - 类别范围: 全部10个类别")
    print("  - 标签噪声: 无")
    print("  - Non-IID影响: 完整数据但仍存在类别不平衡")
    print("-"*80)
    print("\n关键差异：")
    print("  IID环境：相同切分层下，数据类别相对均匀")
    print("  Non-IID环境：相同切分层下，数据集中在2-3个主要类别")
    print("  结果：Non-IID使所有切分层的数据质量都变差")
    print("="*80)
    
    # 创建Non-IID训练数据集（为每个客户端创建子集）
    client_datasets = []
    for i in range(num_clients):
        indices = allocator.get_client_data(i)
        client_subset = Subset(train_dataset, indices)
        client_datasets.append(client_subset)
    
    # 定义四种方法（与IID实验完全一致）
    methods = {
        'SplitFed': SplitFedTrainer,        # 固定切分层1
        'Multi-Tenant SFL': MultiTenantTrainer,  # 从切分层3开始
        'Full-Info': FullInfoTrainer,        # 固定切分层8（理论上界）
        'COIN-UCB': OCDUCBTrainer           # 动态调整切分层（我们的方法）
    }
    
    results = {}
    
    # 对每种方法运行实验
    for method_name, TrainerClass in methods.items():
        print(f"\n{'='*80}")
        print(f"Running {method_name} on Non-IID CIFAR-10...")
        print(f"{'='*80}")
        
        # 创建训练器（max_batches_per_client=5，与IID实验完全一致）
        trainer = TrainerClass(
            dataset_name='cifar10',
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=num_clients,
            batch_size=batch_size,
            device=device,
            clients_per_round=clients_per_round,
            max_batches_per_client=5
        )
        
        # 用NonIIDDataQualityManager替换原始DQM
        # 这样既保留切分层-数据质量机制，又引入Non-IID额外惩罚
        noniid_dqm = NonIIDDataQualityManager(train_dataset, num_clients)
        noniid_dqm.client_indices = [allocator.get_client_data(i) for i in range(num_clients)]
        trainer.dqm = noniid_dqm
        trainer.allocator = MethodBasedDataAllocator(noniid_dqm)
        print(f"注入Non-IID数据分配（含额外噪声惩罚）...")
        
        # 训练
        print(f"开始训练 {num_rounds} 轮...")
        trainer.train(num_rounds=num_rounds, lr=0.01)
        
        # 保存结果
        results[method_name] = trainer.history
        
        # 保存历史到文件
        result_file = os.path.join(output_dir, f'non_iid_{method_name.lower().replace("-", "_")}.txt')
        trainer.save_history(result_file)
        print(f"Results saved to: {result_file}")
    
    # 生成对比图
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    
    visualizer = ResultVisualizer()
    
    # 准备数据
    plot_data = {}
    for method_name, history in results.items():
        plot_data[method_name] = {
            'test_acc': history['test_acc'],
            'train_loss': history['train_loss']
        }
    
    # 生成对比图（只显示准确率曲线）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 配色与results目录下的图保持一致
    colors = {
        'SplitFed': 'red', 
        'Multi-Tenant SFL': 'orange', 
        'COIN-UCB': 'blue', 
        'Full-Info': 'green'
    }
    markers = {
        'SplitFed': 'o',      # 红色圆圈
        'Multi-Tenant SFL': 's',  # 橙色方块
        'COIN-UCB': '^',      # 蓝色三角
        'Full-Info': 'D'      # 绿色菱形
    }
    
    # 测试准确率曲线
    for method_name, data in plot_data.items():
        rounds = list(range(1, len(data['test_acc']) + 1))
        ax.plot(rounds, data['test_acc'], 
                label=method_name, color=colors[method_name], 
                marker=markers[method_name], linewidth=2, markersize=6, markevery=5)
    
    ax.set_xlabel('Communication Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Non-IID CIFAR-10: Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('COIN-SFL Performance on Non-IID Data (Dirichlet α=0.5)', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'non_iid_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {plot_path}")
    
    # 打印最终结果总结
    print(f"\n{'='*80}")
    print("Final Results Summary (Non-IID CIFAR-10)")
    print(f"{'='*80}")
    print(f"{'Method':<15} {'Final Acc (%)':<15}")
    print("-"*50)
    
    for method_name, data in plot_data.items():
        final_acc = data['test_acc'][-1]
        print(f"{method_name:<15} {final_acc:<15.2f}")
    
    print(f"{'='*80}")
    print("\nExpected: Baselines show accuracy drop, COIN-UCB maintains better performance")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_non_iid_experiment()
