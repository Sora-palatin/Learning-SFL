"""


"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
from datetime import datetime

from trainer import SplitFedTrainer, MultiTenantTrainer, LENSUCBTrainer, FullInfoTrainer
from visualizer import ResultVisualizer

def load_dataset(dataset_name, data_dir='./data'):
    """

    
    Args:
 dataset_name: ('mnist', 'fmnist', 'cifar10')
 data_dir:
        
    Returns:
        train_dataset, test_dataset
    """
    print(f"\n{'='*80}")
 print(f" {dataset_name.upper()} ...")
    print(f"{'='*80}\n")
    
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
 print(f"[OK] ")
 print(f" : {len(train_dataset)} ")
 print(f" : {len(test_dataset)} \n")
    
    return train_dataset, test_dataset

def run_single_experiment(dataset_name, method_name, train_dataset, test_dataset, 
                         num_rounds=100, num_clients=10, batch_size=64, lr=0.01, device='cuda',
                         clients_per_round=3, max_batches_per_client=5):
    """

    
    Args:
 dataset_name:
 method_name:
 train_dataset:
 test_dataset:
 num_rounds:
 num_clients:
 batch_size:
 lr:
 device:
        
    Returns:

    """
    print(f"\n{'='*80}")
 print(f": {dataset_name.upper()} - {method_name}")
    print(f"{'='*80}")
 print(f":")
 print(f" : {num_rounds}")
 print(f" : {num_clients}")
 print(f" : {batch_size}")
 print(f" : {lr}")
 print(f" : {device}")
    print(f"{'='*80}\n")
    
 # 
    if method_name == 'SplitFed':
        trainer = SplitFedTrainer(dataset_name, train_dataset, test_dataset, 
                                 num_clients, batch_size, device,
                                 clients_per_round, max_batches_per_client)
    elif method_name == 'Multi-Tenant SFL':
        trainer = MultiTenantTrainer(dataset_name, train_dataset, test_dataset,
                                    num_clients, batch_size, device,
                                    clients_per_round, max_batches_per_client)
    elif method_name == 'COIN-UCB':
        trainer = LENSUCBTrainer(dataset_name, train_dataset, test_dataset,
                               num_clients, batch_size, device,
                               clients_per_round, max_batches_per_client)
    elif method_name == 'Full-Info':
        trainer = FullInfoTrainer(dataset_name, train_dataset, test_dataset,
                                 num_clients, batch_size, device,
                                 clients_per_round, max_batches_per_client)
    else:
        raise ValueError(f"Unsupported method: {method_name}")
    
 # 
    history = trainer.train(num_rounds=num_rounds, lr=lr)
    
    return history

def save_training_log(dataset_name, method_name, history, output_dir='./results'):
    """

    
    Args:
 dataset_name:
 method_name:
 history:
 output_dir:
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, f'test_{dataset_name}_{method_name.replace(" ", "_")}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} - {method_name} Training Log\n")
        f.write("="*80 + "\n\n")
        
        f.write("Training process:\n")
        f.write("-"*80 + "\n")
        
        for i in range(len(history['time'])):
            f.write(f"Round {i+1:3d}: ")
            f.write(f"Time={history['time'][i]:.1f}s, ")
            f.write(f"Loss={history['train_loss'][i]:.4f}, ")
            f.write(f"Test Acc={history['test_acc'][i]:.2f}%")
            if 'split_layers' in history:
                f.write(f", Split Layer={history['split_layers'][i]}")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Final results:\n")
        f.write("="*80 + "\n")
        f.write(f"  Final test accuracy : {history['test_acc'][-1]:.2f}%\n")
        f.write(f"  Best test accuracy  : {max(history['test_acc']):.2f}%\n")
        f.write(f"  Total training time : {history['time'][-1]:.1f}s\n")
        
        if 'split_layers' in history:
            f.write(f"  Final split layer   : {history['split_layers'][-1]}\n")
    
 print(f"[OK] : {log_path}")

def run_all_experiments(dataset_name, num_rounds=100, num_clients=10, 
                       batch_size=64, lr=0.01, output_dir='./results',
                       clients_per_round=3, max_batches_per_client=5):
    """

    
    Args:
 dataset_name:
 num_rounds:
 num_clients:
 batch_size:
 lr:
 output_dir:
    """
 # GPU
    if torch.cuda.is_available():
        device = 'cuda'
 print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
 print(": GPUCPU")
    
    print(f"\n{'='*80}")
 print(f" {dataset_name.upper()} ")
    print(f"{'='*80}")
 print(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 print(f": {device}")
    print(f"{'='*80}\n")
    
 # 
    train_dataset, test_dataset = load_dataset(dataset_name)
    
 # 
    methods = ['SplitFed', 'Multi-Tenant SFL', 'COIN-UCB', 'Full-Info']
    
    histories = {}
    
 # 
    for method_name in methods:
        try:
            history = run_single_experiment(
                dataset_name, method_name, train_dataset, test_dataset,
                num_rounds, num_clients, batch_size, lr, device,
                clients_per_round, max_batches_per_client
            )
            
            histories[method_name] = history
            
 # 
            save_training_log(dataset_name, method_name, history, output_dir)
            
        except Exception as e:
 print(f"\n❌ {method_name} : {str(e)}")
            import traceback
            traceback.print_exc()
            histories[method_name] = None
    
 # 
    print(f"\n{'='*80}")
 print("...")
    print(f"{'='*80}\n")
    
    visualizer = ResultVisualizer(output_dir)
    visualizer.generate_report(histories, dataset_name)
    
    print(f"\n{'='*80}")
 print(f"[OK] {dataset_name.upper()} ")
    print(f"{'='*80}\n")

def main():
 """"""
    parser = argparse.ArgumentParser(description='Run real-dataset comparison experiments for LENS-SFL')
    
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10', 'all'],
                       help='Dataset name (default: mnist)')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of communication rounds (default: 100)')
    parser.add_argument('--clients', type=int, default=10,
                       help='Number of clients (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Mini-batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='SGD learning rate (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for logs and figures (default: ./results)')
    parser.add_argument('--clients_per_round', type=int, default=3,
                       help='Clients sampled per round (default: 3)')
    parser.add_argument('--max_batches', type=int, default=5,
                       help='Max batches per client per round (default: 5)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast-test mode: fewer clients and batches for quick smoke-test')
    
    args = parser.parse_args()
    
 # 
    if args.fast:
        args.clients_per_round = 2
        args.max_batches = 3
        print("\n" + "="*80)
 print(" - ")
        print("="*80)
 print(": 23batch")
    else:
        print("\n" + "="*80)
 print("")
        print("="*80)
 print(":")
 print(" 1. SplitFed: ")
 print(" 2. Multi-Tenant SFL: ")
    print("  3. COIN-UCB: Contract-based Online Incentive with UCB (Our Method)")
 print(" 4. Full-Info: ")
    print("="*80 + "\n")
    
 # 
    if args.dataset == 'all':
        datasets = ['mnist', 'fmnist', 'cifar10']
        for dataset_name in datasets:
            run_all_experiments(
                dataset_name, args.rounds, args.clients,
                args.batch_size, args.lr, args.output_dir,
                args.clients_per_round, args.max_batches
            )
    else:
        run_all_experiments(
            args.dataset, args.rounds, args.clients,
            args.batch_size, args.lr, args.output_dir,
            args.clients_per_round, args.max_batches
        )
    
    print("\n" + "="*80)
 print("[OK] ")
    print("="*80)
 print(f": {os.path.abspath(args.output_dir)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
