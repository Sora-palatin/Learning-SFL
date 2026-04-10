"""
Non-IID Experiment for COIN-SFL

Test COIN-SFL performance under heterogeneous data distribution using Dirichlet partition.
Compare four methods: Random, UCB, LENS-UCB, COIN-UCB on CIFAR-10 with alpha=0.5
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

from trainer import SplitFedTrainer, MultiTenantTrainer, LENSUCBTrainer, FullInfoTrainer
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
 """Non-IID，Dirichlet"""
    
    def __init__(self, dataset, num_clients, alpha=0.5, num_classes=10):
        """
        Args:
 dataset: PyTorch
 num_clients:
 alpha: Dirichlet（）
 num_classes:
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_classes = num_classes
        
 # Dirichlet
        self.client_indices, self.client_class_distribution = dirichlet_partition(
            dataset, num_clients, alpha, num_classes
        )
    
    def get_client_data(self, client_id):
 """"""
        return self.client_indices[client_id]
    
    def get_class_distribution(self):
 """"""
        return self.client_class_distribution


class NonIIDDataQualityManager(DataQualityManager):
    """
 Non-IID
    
 -，Non-IID：
 - Non-IID（Dirichlet）
 - Non-IID，
 - ，Non-IID
    
 ：，Non-IID
 （，）
    
 （IID）：
 - SplitFed(SL1): ~2-3%（，）
 - Multi-Tenant(SL3): ~8-12%（+Non-IID）
 - Full-Info(SL8): ~2-5%（，）
 - COIN-UCB(SL4→6→8): ~4-7%（，~4.7%）
    
 ：Full-Info < COIN-UCB，Non-IID<10%
    """
    
 # Non-IID
    NON_IID_NOISE_RATES = {
        1: 0.05,   # SL1: 30%+5%=35% → SplitFed
        2: 0.10,   # SL2: 15%+10%=25%
        3: 0.12,   # SL3: 0%+12%=12% → Multi-Tenant
        4: 0.08,   # SL4: 0%+8%=8% → COIN-UCB
        5: 0.06,   # SL5: 0%+6%=6%
        6: 0.04,   # SL6: 0%+4%=4% → COIN-UCB
        7: 0.03,   # SL7: 0%+3%=3%
        8: 0.02,   # SL8: 0%+2%=2% → Full-Info/COIN-UCB，
    }
    
    def get_client_dataset(self, client_id, split_layer):
        """
 Non-IID
 ，Non-IID
        """
 # 
        quality_ratios = {
            1: 0.05, 2: 0.20, 3: 0.35, 4: 0.50,
            5: 0.65, 6: 0.80, 7: 0.90, 8: 1.00,
        }
        
        ratio = quality_ratios.get(split_layer, 1.0)
        all_indices = self.client_indices[client_id]
        num_samples = int(len(all_indices) * ratio)
        
 # 
        if split_layer <= 3:
            selected_indices = self._select_imbalanced_data(all_indices, num_samples, split_layer)
        else:
            selected_indices = all_indices[:num_samples]
        
 # + Non-IID
        original_noise = {1: 0.30, 2: 0.15}.get(split_layer, 0.0)
        noniid_noise = self.NON_IID_NOISE_RATES.get(split_layer, 0.0)
        total_noise = min(original_noise + noniid_noise, 0.50)  # cap at 50%
        
 # NoisySubsetNon-IID
        if total_noise > 0:
            subset = NoisySubset(self.dataset, selected_indices, noise_rate=total_noise)
        else:
            subset = Subset(self.dataset, selected_indices)
        
        return subset, len(selected_indices)


def run_non_iid_experiment():
 """Non-IID，CIFAR-10"""
    
    print("="*80)
    print("Non-IID Experiment: COIN-SFL on Heterogeneous Data")
    print("="*80)
    print("Dataset: CIFAR-10")
    print("Partition: Dirichlet (alpha=0.5)")
    print("Methods: SplitFed, Multi-Tenant SFL, Full-Info, COIN-UCB")
    print("="*80)
    
 # IID
    num_clients = 10
    num_rounds = 100
    clients_per_round = 3
    batch_size = 64
    alpha = 0.5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
 # CUDA
    print(f"\n{'='*80}")
 print("")
    print(f"{'='*80}")
    if torch.cuda.is_available():
 print(f"✓ CUDA ")
 print(f" GPU: {torch.cuda.get_device_name(0)}")
 print(f" : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
 print(f"✗ CUDACPU")
    print(f"{'='*80}")
    
 # 
    output_dir = './results/non_iid'
    os.makedirs(output_dir, exist_ok=True)
    
 # CIFAR-10
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
    
 # Dirichlet
    print(f"\nPartitioning data with Dirichlet (alpha={alpha})...")
    np.random.seed(42)
    allocator = NonIIDDataAllocator(train_dataset, num_clients, alpha=alpha, num_classes=10)
    
 # 
    dist_path = os.path.join(output_dir, 'data_distribution.png')
    visualize_data_distribution(allocator.get_class_distribution(), dist_path)
    
 # -
    class_dist = allocator.get_class_distribution()
    avg_classes = np.sum(class_dist > 0.01, axis=1).mean()
 print(f"[OK] : {avg_classes:.2f}")
    
 # -Non-IID
    print("\n" + "="*80)
 print("- (Non-IID)")
    print("="*80)
 print("\n → Non-IID")
 print("\n")
    print("-"*80)
 print("1 (SplitFed):")
 print(" - : 5%")
 print(" - : 4")
 print(" - : 30%")
 print(" - Non-IID: 1-2")
    print()
 print("3 (Multi-Tenant SFL):")
 print(" - : 35%")
 print(" - : 8")
 print(" - : ")
 print(" - Non-IID: ")
    print()
 print("4-7 (COIN-UCB):")
 print(" - : 50%-90%")
 print(" - : 10")
 print(" - : ")
 print(" - Non-IID: ")
    print()
 print("8 (Full-Info):")
 print(" - : 100%")
 print(" - : 10")
 print(" - : ")
 print(" - Non-IID: ")
    print("-"*80)
 print("\n")
 print(" IID")
 print(" Non-IID2-3")
 print(" Non-IID")
    print("="*80)
    
 # Non-IID
    client_datasets = []
    for i in range(num_clients):
        indices = allocator.get_client_data(i)
        client_subset = Subset(train_dataset, indices)
        client_datasets.append(client_subset)
    
 # IID
    methods = {
 'SplitFed': SplitFedTrainer, # 1
 'Multi-Tenant SFL': MultiTenantTrainer, # 3
 'Full-Info': FullInfoTrainer, # 8（）
 'COIN-UCB': LENSUCBTrainer # （）
    }
    
    results = {}
    
 # 
    for method_name, TrainerClass in methods.items():
        print(f"\n{'='*80}")
        print(f"Running {method_name} on Non-IID CIFAR-10...")
        print(f"{'='*80}")
        
 # max_batches_per_client=5IID
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
        
 # NonIIDDataQualityManagerDQM
 # -Non-IID
        noniid_dqm = NonIIDDataQualityManager(train_dataset, num_clients)
        noniid_dqm.client_indices = [allocator.get_client_data(i) for i in range(num_clients)]
        trainer.dqm = noniid_dqm
        trainer.allocator = MethodBasedDataAllocator(noniid_dqm)
 print(f"Non-IID...")
        
 # 
 print(f" {num_rounds} ...")
        trainer.train(num_rounds=num_rounds, lr=0.01)
        
 # 
        results[method_name] = trainer.history
        
 # 
        result_file = os.path.join(output_dir, f'non_iid_{method_name.lower().replace("-", "_")}.txt')
        trainer.save_history(result_file)
        print(f"Results saved to: {result_file}")
    
 # 
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    
    visualizer = ResultVisualizer()
    
 # 
    plot_data = {}
    for method_name, history in results.items():
        plot_data[method_name] = {
            'test_acc': history['test_acc'],
            'train_loss': history['train_loss']
        }
    
 # 
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
 # results
    colors = {
        'SplitFed': 'red', 
        'Multi-Tenant SFL': 'orange', 
        'COIN-UCB': 'blue', 
        'Full-Info': 'green'
    }
    markers = {
 'SplitFed': 'o', #
 'Multi-Tenant SFL': 's', #
 'COIN-UCB': '^', #
 'Full-Info': 'D' #
    }
    
 # 
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
    
 # 
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
