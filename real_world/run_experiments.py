"""
主运行脚本
运行四种方法在三个数据集上的对比实验
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
from datetime import datetime

from trainer import SplitFedTrainer, MultiTenantTrainer, OCDUCBTrainer, FullInfoTrainer
from visualizer import ResultVisualizer

def load_dataset(dataset_name, data_dir='./data'):
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 ('mnist', 'fmnist', 'cifar10')
        data_dir: 数据目录
        
    Returns:
        train_dataset, test_dataset
    """
    print(f"\n{'='*80}")
    print(f"加载 {dataset_name.upper()} 数据集...")
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
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    print(f"[OK] 数据集加载完成！")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   测试集: {len(test_dataset)} 样本\n")
    
    return train_dataset, test_dataset

def run_single_experiment(dataset_name, method_name, train_dataset, test_dataset, 
                         num_rounds=100, num_clients=10, batch_size=64, lr=0.01, device='cuda',
                         clients_per_round=3, max_batches_per_client=5):
    """
    运行单个实验
    
    Args:
        dataset_name: 数据集名称
        method_name: 方法名称
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        num_rounds: 训练轮数
        num_clients: 客户端数量
        batch_size: 批次大小
        lr: 学习率
        device: 设备
        
    Returns:
        训练历史
    """
    print(f"\n{'='*80}")
    print(f"实验: {dataset_name.upper()} - {method_name}")
    print(f"{'='*80}")
    print(f"参数设置:")
    print(f"  训练轮数: {num_rounds}")
    print(f"  客户端数: {num_clients}")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {lr}")
    print(f"  设备: {device}")
    print(f"{'='*80}\n")
    
    # 创建训练器（传递加速参数）
    if method_name == 'SplitFed':
        trainer = SplitFedTrainer(dataset_name, train_dataset, test_dataset, 
                                 num_clients, batch_size, device,
                                 clients_per_round, max_batches_per_client)
    elif method_name == 'Multi-Tenant SFL':
        trainer = MultiTenantTrainer(dataset_name, train_dataset, test_dataset,
                                    num_clients, batch_size, device,
                                    clients_per_round, max_batches_per_client)
    elif method_name == 'COIN-UCB':
        trainer = OCDUCBTrainer(dataset_name, train_dataset, test_dataset,
                               num_clients, batch_size, device,
                               clients_per_round, max_batches_per_client)
    elif method_name == 'Full-Info':
        trainer = FullInfoTrainer(dataset_name, train_dataset, test_dataset,
                                 num_clients, batch_size, device,
                                 clients_per_round, max_batches_per_client)
    else:
        raise ValueError(f"不支持的方法: {method_name}")
    
    # 训练
    history = trainer.train(num_rounds=num_rounds, lr=lr)
    
    return history

def save_training_log(dataset_name, method_name, history, output_dir='./results'):
    """
    保存训练日志
    
    Args:
        dataset_name: 数据集名称
        method_name: 方法名称
        history: 训练历史
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, f'test_{dataset_name}_{method_name.replace(" ", "_")}.txt')
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"{dataset_name.upper()} - {method_name} 训练日志\n")
        f.write("="*80 + "\n\n")
        
        f.write("训练过程:\n")
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
        f.write("最终结果:\n")
        f.write("="*80 + "\n")
        f.write(f"  最终测试准确率: {history['test_acc'][-1]:.2f}%\n")
        f.write(f"  最佳测试准确率: {max(history['test_acc']):.2f}%\n")
        f.write(f"  总训练时间: {history['time'][-1]:.1f}秒\n")
        
        if 'split_layers' in history:
            f.write(f"  最终切分层: {history['split_layers'][-1]}\n")
    
    print(f"[OK] 训练日志已保存: {log_path}")

def run_all_experiments(dataset_name, num_rounds=100, num_clients=10, 
                       batch_size=64, lr=0.01, output_dir='./results',
                       clients_per_round=3, max_batches_per_client=5):
    """
    运行所有方法的对比实验
    
    Args:
        dataset_name: 数据集名称
        num_rounds: 训练轮数
        num_clients: 客户端数量
        batch_size: 批次大小
        lr: 学习率
        output_dir: 输出目录
    """
    # 强制使用GPU（如果可用）
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"使用GPU加速: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("警告: GPU不可用，使用CPU（速度会很慢）")
    
    print(f"\n{'='*80}")
    print(f"开始 {dataset_name.upper()} 数据集的完整实验")
    print(f"{'='*80}")
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用设备: {device}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    # 四种方法
    methods = ['SplitFed', 'Multi-Tenant SFL', 'COIN-UCB', 'Full-Info']
    
    histories = {}
    
    # 运行每种方法
    for method_name in methods:
        try:
            history = run_single_experiment(
                dataset_name, method_name, train_dataset, test_dataset,
                num_rounds, num_clients, batch_size, lr, device,
                clients_per_round, max_batches_per_client
            )
            
            histories[method_name] = history
            
            # 保存训练日志
            save_training_log(dataset_name, method_name, history, output_dir)
            
        except Exception as e:
            print(f"\n❌ {method_name} 训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            histories[method_name] = None
    
    # 生成可视化报告
    print(f"\n{'='*80}")
    print("生成可视化报告...")
    print(f"{'='*80}\n")
    
    visualizer = ResultVisualizer(output_dir)
    visualizer.generate_report(histories, dataset_name)
    
    print(f"\n{'='*80}")
    print(f"[OK] {dataset_name.upper()} 数据集实验全部完成！")
    print(f"{'='*80}\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行真实数据集对比实验')
    
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fmnist', 'cifar10', 'all'],
                       help='数据集名称 (default: mnist)')
    parser.add_argument('--rounds', type=int, default=100,
                       help='训练轮数 (default: 100)')
    parser.add_argument('--clients', type=int, default=10,
                       help='客户端数量 (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='学习率 (default: 0.01)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录 (default: ./results)')
    parser.add_argument('--clients_per_round', type=int, default=3,
                       help='每轮训练的客户端数量 (default: 3, 加速训练)')
    parser.add_argument('--max_batches', type=int, default=5,
                       help='每个客户端最多训练的batch数 (default: 5, 加速训练)')
    parser.add_argument('--fast', action='store_true',
                       help='快速测试模式（更少的客户端和batch）')
    
    args = parser.parse_args()
    
    # 快速测试模式
    if args.fast:
        args.clients_per_round = 2
        args.max_batches = 3
        print("\n" + "="*80)
        print("真实数据集对比实验 - 快速测试模式")
        print("="*80)
        print("注意: 使用快速模式（每轮2个客户端，每客户端3个batch）")
    else:
        print("\n" + "="*80)
        print("真实数据集对比实验")
        print("="*80)
    print("对比方法:")
    print("  1. SplitFed: 无激励，客户端说谎，最差数据分配")
    print("  2. Multi-Tenant SFL: 常规激励，大锅饭，中等数据分配")
    print("  3. COIN-UCB: Contract-based Online Incentive with UCB (Our Method)")
    print("  4. Full-Info: 完全信息，理论上界")
    print("="*80 + "\n")
    
    # 运行实验
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
    print("[OK] 所有实验完成！")
    print("="*80)
    print(f"结果保存在: {os.path.abspath(args.output_dir)}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
