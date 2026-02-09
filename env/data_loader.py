"""
CIFAR-10 Data Loading and Preprocessing for Contract-based SFL
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import os


def get_cifar10_transforms():
    """
    Get CIFAR-10 data transformations
    
    Returns:
        train_transform, test_transform
    """
    # CIFAR-10 normalization (standard values)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, test_transform


def load_cifar10(data_dir='./data'):
    """
    Load CIFAR-10 dataset
    
    Args:
        data_dir: directory to store/load data
    
    Returns:
        train_dataset, test_dataset
    """
    train_transform, test_transform = get_cifar10_transforms()
    
    # Create data directory if not exists
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    return train_dataset, test_dataset


def create_iid_split(dataset, num_clients):
    """
    Create IID data split for clients
    Each client gets equal amount of randomly sampled data
    
    Args:
        dataset: PyTorch dataset
        num_clients: number of clients
    
    Returns:
        dict: {client_id: list of indices}
    """
    num_items = len(dataset) // num_clients
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]
    
    np.random.shuffle(all_idxs)
    
    for i in range(num_clients):
        dict_users[i] = set(all_idxs[i * num_items:(i + 1) * num_items])
    
    return dict_users


def create_non_iid_split(dataset, num_clients, num_shards=200):
    """
    Create Non-IID data split for clients
    Each client gets data from limited number of classes
    
    Args:
        dataset: PyTorch dataset
        num_clients: number of clients
        num_shards: number of shards to split data into
    
    Returns:
        dict: {client_id: list of indices}
    """
    num_imgs = len(dataset) // num_shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    
    # Get labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
    
    # Sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Divide and assign shards
    shards_per_client = num_shards // num_clients
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shards_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users


class DatasetSplit(Dataset):
    """Custom dataset for client data split"""
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_client_dataloaders(dataset_train, dataset_test, dict_users_train, dict_users_test, 
                           batch_size=64, num_workers=0):
    """
    Create data loaders for all clients
    
    Args:
        dataset_train: training dataset
        dataset_test: test dataset
        dict_users_train: train data split dict
        dict_users_test: test data split dict
        batch_size: batch size
        num_workers: number of workers for data loading (0 for Windows compatibility)
    
    Returns:
        train_loaders, test_loaders (lists)
    """
    num_clients = len(dict_users_train)
    train_loaders = []
    test_loaders = []
    
    for idx in range(num_clients):
        train_loader = DataLoader(
            DatasetSplit(dataset_train, dict_users_train[idx]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False  # Disable for Windows compatibility
        )
        test_loader = DataLoader(
            DatasetSplit(dataset_test, dict_users_test[idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loaders, test_loaders


if __name__ == '__main__':
    # Test data loading
    print("Testing CIFAR-10 data loading...")
    
    train_dataset, test_dataset = load_cifar10()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test IID split
    num_clients = 20
    dict_users_train = create_iid_split(train_dataset, num_clients)
    dict_users_test = create_iid_split(test_dataset, num_clients)
    
    print(f"\nIID split for {num_clients} clients:")
    for i in range(min(5, num_clients)):
        print(f"  Client {i}: {len(dict_users_train[i])} train samples, {len(dict_users_test[i])} test samples")
    
    # Test data loaders
    train_loaders, test_loaders = get_client_dataloaders(
        train_dataset, test_dataset, dict_users_train, dict_users_test, batch_size=64
    )
    
    print(f"\nData loaders created: {len(train_loaders)} train, {len(test_loaders)} test")
    print(f"First client train batches: {len(train_loaders[0])}")
    
    # Test one batch
    images, labels = next(iter(train_loaders[0]))
    print(f"\nBatch shape: images={images.shape}, labels={labels.shape}")
    
    print("\nData loading test completed!")
