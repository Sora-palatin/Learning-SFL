"""
数据集下载脚本
支持MNIST、Fashion-MNIST、CIFAR-10三个数据集的自动下载
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_mnist(data_dir='./data'):
    """
    下载MNIST数据集
    
    Args:
        data_dir: 数据保存目录
    """
    print("="*80)
    print("下载MNIST数据集...")
    print("="*80)
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载训练集
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载测试集
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"[OK] MNIST数据集下载完成！")
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   测试集大小: {len(test_dataset)}")
    print(f"   保存位置: {data_dir}/MNIST")
    print()
    
    return train_dataset, test_dataset

def download_fashion_mnist(data_dir='./data'):
    """
    下载Fashion-MNIST数据集
    
    Args:
        data_dir: 数据保存目录
    """
    print("="*80)
    print("下载Fashion-MNIST数据集...")
    print("="*80)
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    # 下载训练集
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载测试集
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"[OK] Fashion-MNIST数据集下载完成！")
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   测试集大小: {len(test_dataset)}")
    print(f"   保存位置: {data_dir}/FashionMNIST")
    print()
    
    return train_dataset, test_dataset

def download_cifar10(data_dir='./data'):
    """
    下载CIFAR-10数据集
    
    Args:
        data_dir: 数据保存目录
    """
    print("="*80)
    print("下载CIFAR-10数据集...")
    print("="*80)
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 下载训练集
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # 下载测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"[OK] CIFAR-10数据集下载完成！")
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   测试集大小: {len(test_dataset)}")
    print(f"   保存位置: {data_dir}/cifar-10-batches-py")
    print()
    
    return train_dataset, test_dataset

def download_all_datasets(data_dir='./data'):
    """
    下载所有数据集
    
    Args:
        data_dir: 数据保存目录
    """
    print("\n" + "="*80)
    print("开始下载所有数据集...")
    print("="*80 + "\n")
    
    # 下载MNIST
    mnist_train, mnist_test = download_mnist(data_dir)
    
    # 下载Fashion-MNIST
    fmnist_train, fmnist_test = download_fashion_mnist(data_dir)
    
    # 下载CIFAR-10
    cifar10_train, cifar10_test = download_cifar10(data_dir)
    
    print("="*80)
    print("所有数据集下载完成！")
    print("="*80)
    print(f"\n数据集保存在: {os.path.abspath(data_dir)}")
    print("\n数据集信息:")
    print(f"  1. MNIST:         训练集 {len(mnist_train):>5}, 测试集 {len(mnist_test):>5}")
    print(f"  2. Fashion-MNIST: 训练集 {len(fmnist_train):>5}, 测试集 {len(fmnist_test):>5}")
    print(f"  3. CIFAR-10:      训练集 {len(cifar10_train):>5}, 测试集 {len(cifar10_test):>5}")
    print()

if __name__ == '__main__':
    # 设置数据保存目录
    data_dir = './data'
    
    # 下载所有数据集
    download_all_datasets(data_dir)
    
    print("[OK] 数据集准备完成！可以开始训练了。")
