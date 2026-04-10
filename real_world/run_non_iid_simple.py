"""
Non-IID
Non-IID CIFAR-10
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
import sys

from trainer import SplitFedTrainer, MultiTenantTrainer, LENSUCBTrainer, FullInfoTrainer
from data_quality_manager import DataQualityManager, MethodBasedDataAllocator, NoisySubset
from non_iid_config import (
    IID_RESULTS, NON_IID_EXPECTED, PLOT_CONFIG, 
    EXPERIMENT_CONFIG, print_expected_results
)

# Set font for plots
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def dirichlet_partition(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Balanced Dirichlet partition: each client gets EXACTLY the same number
    of samples as in IID (total // num_clients). Only class proportions differ.
    This preserves the split-layer data quality mechanism from the paper.
    """
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
        class_size = len(class_indices[c])
        counts = (proportions * class_size).astype(int)
        diff = class_size - counts.sum()
        for d in range(abs(diff)):
            if diff > 0:
                counts[d % num_clients] += 1
            else:
                counts[-(d % num_clients) - 1] -= 1
        
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_indices[c][start:end].tolist())
            start = end
    
    # Step 2: Trim or pad each client to exactly samples_per_client
    overflow = []
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
        if len(client_indices[client_id]) > samples_per_client:
            overflow.extend(client_indices[client_id][samples_per_client:])
            client_indices[client_id] = client_indices[client_id][:samples_per_client]
    
    np.random.shuffle(overflow)
    overflow_idx = 0
    for client_id in range(num_clients):
        deficit = samples_per_client - len(client_indices[client_id])
        if deficit > 0 and overflow_idx < len(overflow):
            take = min(deficit, len(overflow) - overflow_idx)
            client_indices[client_id].extend(overflow[overflow_idx:overflow_idx + take])
            overflow_idx += take
    
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    # Calculate class distribution
    client_class_distribution = np.zeros((num_clients, num_classes))
    for client_id in range(num_clients):
        if len(client_indices[client_id]) > 0:
            client_labels = labels[client_indices[client_id]]
            for c in range(num_classes):
                client_class_distribution[client_id, c] = np.sum(client_labels == c)
    
    row_sums = client_class_distribution.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    client_class_distribution = client_class_distribution / row_sums
    
    return client_indices, client_class_distribution


def main():
    print("="*80)
    print("Non-IID Experiment: COIN-SFL on Heterogeneous CIFAR-10")
    print("="*80)
    print("Methods: SplitFed, Multi-Tenant SFL, Full-Info, COIN-UCB")
    print(f"Dirichlet α={EXPERIMENT_CONFIG['alpha']} | {EXPERIMENT_CONFIG['num_clients']} clients | {EXPERIMENT_CONFIG['num_rounds']} rounds")
    print("="*80)
    
 # 
    num_clients = EXPERIMENT_CONFIG['num_clients']
    num_rounds = EXPERIMENT_CONFIG['num_rounds']
    clients_per_round = EXPERIMENT_CONFIG['clients_per_round']
    batch_size = EXPERIMENT_CONFIG['batch_size']
    alpha = EXPERIMENT_CONFIG['alpha']
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
 print("\n (IID CIFAR-10):")
    print_expected_results()
    
 # 
    output_dir = './results/non_iid'
    os.makedirs(output_dir, exist_ok=True)
    
 # 
 print("\nCIFAR-10...")
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
    
 print(f"[OK] : {len(train_dataset)}, : {len(test_dataset)}")
    
 # Dirichlet
 print(f"\nDirichlet (α={alpha})...")
    np.random.seed(42)
    client_indices, client_class_dist = dirichlet_partition(
        train_dataset, num_clients, alpha=alpha, num_classes=10
    )
    
    avg_classes = np.sum(client_class_dist > 0.01, axis=1).mean()
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
 print(f" {method_name}...")
        print(f"{'='*80}")
        
        try:
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
            from non_iid_experiment import NonIIDDataQualityManager
            noniid_dqm = NonIIDDataQualityManager(train_dataset, num_clients)
            noniid_dqm.client_indices = client_indices
            trainer.dqm = noniid_dqm
            trainer.allocator = MethodBasedDataAllocator(noniid_dqm)
 print(f"Non-IID...")
            
 # 
 print(f" {num_rounds} ...")
            trainer.train(num_rounds=num_rounds, lr=0.01)
            
 # 
            results[method_name] = trainer.history
            
            result_file = os.path.join(output_dir, f'non_iid_{method_name.lower().replace("-", "_")}.txt')
            trainer.save_history(result_file)
            
            final_acc = trainer.history['test_acc'][-1] if trainer.history['test_acc'] else 0
 print(f"[OK] {method_name} - : {final_acc:.2f}%")
            
        except Exception as e:
 print(f"[ERROR] {method_name} : {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
 # 
    if len(results) > 0:
        print(f"\n{'='*80}")
 print("...")
        print(f"{'='*80}")
        
 print(f"\n...")
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
        for method_name, history in results.items():
            if 'test_acc' in history and len(history['test_acc']) > 0:
                rounds = list(range(1, len(history['test_acc']) + 1))
                ax.plot(rounds, history['test_acc'], 
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
        
 print(f"[OK] : {plot_path}")
        
 # 
        print(f"\n{'='*80}")
 print(" (Non-IID vs IID)")
        print(f"{'='*80}")
 print(f"{'':<20} {'IID':<12} {'Non-IID':<15} {'':<15} {'':<10}")
        print("-"*80)
        
        for method_name, history in results.items():
            if 'test_acc' in history and len(history['test_acc']) > 0:
                final_acc = history['test_acc'][-1]
                iid_acc = IID_RESULTS.get(method_name, 0)
                expected = NON_IID_EXPECTED.get(method_name, {})
                exp_min = expected.get('min', 0)
                exp_max = expected.get('max', 0)
                actual_drop = iid_acc - final_acc
                
                status = "✓" if exp_min <= final_acc <= exp_max else "✗"
                print(f"{method_name:<20} {iid_acc:>6.2f}%      {final_acc:>6.2f}%         "
                      f"{exp_min:>5.1f}-{exp_max:<5.1f}%    {actual_drop:>5.2f}% {status}")
        
        print(f"{'='*80}")
 print("\n:")
 print("1. SplitFed: 1Non-IID")
 print("2. Multi-Tenant SFL: 3")
 print("3. COIN-UCB: [OK]")
 print("4. Full-Info: 100%")
        print(f"{'='*80}")
    else:
 print("\n[WARNING] ")
    
 print("\n")


if __name__ == '__main__':
    main()
