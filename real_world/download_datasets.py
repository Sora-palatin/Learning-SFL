"""

MNIST、Fashion-MNIST、CIFAR-10
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os

def download_mnist(data_dir='./data'):
    """
 MNIST
    
    Args:
 data_dir:
    """
    print("="*80)
 print("MNIST...")
    print("="*80)
    
 # 
    os.makedirs(data_dir, exist_ok=True)
    
 # 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
 # 
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
 # 
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
 print(f"[OK] MNIST")
 print(f" : {len(train_dataset)}")
 print(f" : {len(test_dataset)}")
 print(f" : {data_dir}/MNIST")
    print()
    
    return train_dataset, test_dataset

def download_fashion_mnist(data_dir='./data'):
    """
 Fashion-MNIST
    
    Args:
 data_dir:
    """
    print("="*80)
 print("Fashion-MNIST...")
    print("="*80)
    
 # 
    os.makedirs(data_dir, exist_ok=True)
    
 # 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
 # 
    train_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
 # 
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
 print(f"[OK] Fashion-MNIST")
 print(f" : {len(train_dataset)}")
 print(f" : {len(test_dataset)}")
 print(f" : {data_dir}/FashionMNIST")
    print()
    
    return train_dataset, test_dataset

def download_cifar10(data_dir='./data'):
    """
 CIFAR-10
    
    Args:
 data_dir:
    """
    print("="*80)
 print("CIFAR-10...")
    print("="*80)
    
 # 
    os.makedirs(data_dir, exist_ok=True)
    
 # 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
 # 
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
 # 
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
 print(f"[OK] CIFAR-10")
 print(f" : {len(train_dataset)}")
 print(f" : {len(test_dataset)}")
 print(f" : {data_dir}/cifar-10-batches-py")
    print()
    
    return train_dataset, test_dataset

def download_all_datasets(data_dir='./data'):
    """

    
    Args:
 data_dir:
    """
    print("\n" + "="*80)
 print("...")
    print("="*80 + "\n")
    
 # MNIST
    mnist_train, mnist_test = download_mnist(data_dir)
    
 # Fashion-MNIST
    fmnist_train, fmnist_test = download_fashion_mnist(data_dir)
    
 # CIFAR-10
    cifar10_train, cifar10_test = download_cifar10(data_dir)
    
    print("="*80)
 print("")
    print("="*80)
 print(f"\n: {os.path.abspath(data_dir)}")
 print("\n:")
 print(f" 1. MNIST: {len(mnist_train):>5}, {len(mnist_test):>5}")
 print(f" 2. Fashion-MNIST: {len(fmnist_train):>5}, {len(fmnist_test):>5}")
 print(f" 3. CIFAR-10: {len(cifar10_train):>5}, {len(cifar10_test):>5}")
    print()

if __name__ == '__main__':
 # 
    data_dir = './data'
    
 # 
    download_all_datasets(data_dir)
    
 print("[OK] 。")
