"""
数据质量管理器
实现数据量与切分点的绑定机制
切分越深 -> 数据质量越好 -> 训练效果越好
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

class NoisySubset(Dataset):
    """
    带标签噪声的数据子集
    """
    def __init__(self, dataset, indices, noise_rate=0.3):
        """
        Args:
            dataset: 原始数据集
            indices: 数据索引
            noise_rate: 标签噪声比例
        """
        self.dataset = dataset
        self.indices = indices
        self.noise_rate = noise_rate
        
        # 为每个样本决定是否添加噪声
        np.random.seed(sum(indices) % 10000)  # 确保可重复
        self.noisy_mask = np.random.random(len(indices)) < noise_rate
        
        # 获取所有可能的标签
        if hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        else:
            # 尝试推断类别数
            sample_labels = [dataset[i][1] for i in indices[:min(100, len(indices))]]
            self.num_classes = max(sample_labels) + 1
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取原始数据和标签
        data, label = self.dataset[self.indices[idx]]
        
        # 如果需要添加噪声，随机改变标签
        if self.noisy_mask[idx]:
            # 随机选择一个不同的标签
            noisy_label = np.random.randint(0, self.num_classes)
            while noisy_label == label and self.num_classes > 1:
                noisy_label = np.random.randint(0, self.num_classes)
            label = noisy_label
        
        return data, label

class DataQualityManager:
    """
    数据质量管理器
    根据切分点分配不同质量的数据
    """
    
    def __init__(self, dataset, num_clients=10):
        """
        Args:
            dataset: 完整数据集
            num_clients: 客户端数量
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.total_samples = len(dataset)
        
        # 为每个客户端分配数据索引
        self.client_indices = self._split_data()
        
    def _split_data(self):
        """将数据集分割给不同客户端"""
        indices = np.arange(self.total_samples)
        np.random.shuffle(indices)
        
        # 均匀分割
        client_indices = np.array_split(indices, self.num_clients)
        return [idx.tolist() for idx in client_indices]
    
    def get_client_dataset(self, client_id, split_layer):
        """
        根据切分点获取客户端的数据集
        
        Args:
            client_id: 客户端ID (0-9)
            split_layer: 切分层 (1-8)
                1: 最差数据质量（少量数据+类别不平衡+标签噪声）
                8: 最好数据质量（全部数据+平衡+无噪声）
        
        Returns:
            客户端的数据子集
        """
        # 数据质量比例：切分层越深，数据质量越好
        quality_ratios = {
            1: 0.05,  # 5%数据（SplitFed专用）
            2: 0.20,  # 20%数据
            3: 0.35,  # 35%数据（Multi-Tenant新起点）
            4: 0.50,  # 50%数据（OCD-UCB初期）
            5: 0.65,  # 65%数据
            6: 0.80,  # 80%数据
            7: 0.90,  # 90%数据
            8: 1.00,  # 100%数据（Full-Info和OCD-UCB后期）
        }
        
        ratio = quality_ratios.get(split_layer, 1.0)
        
        # 获取该客户端的所有数据索引
        all_indices = self.client_indices[client_id]
        
        # 根据质量比例选择数据
        num_samples = int(len(all_indices) * ratio)
        
        # 对于低质量数据，引入类别不平衡
        if split_layer <= 3:
            selected_indices = self._select_imbalanced_data(all_indices, num_samples, split_layer)
        else:
            selected_indices = all_indices[:num_samples]
        
        # 创建子集（可能带噪声标签）
        if split_layer <= 2:
            # 为最差的数据添加标签噪声
            subset = NoisySubset(self.dataset, selected_indices, noise_rate=0.3 if split_layer == 1 else 0.15)
        else:
            subset = Subset(self.dataset, selected_indices)
        
        return subset, len(selected_indices)
    
    def _select_imbalanced_data(self, all_indices, num_samples, split_layer):
        """
        选择类别不平衡的数据
        
        Args:
            all_indices: 所有可用索引
            num_samples: 需要选择的样本数
            split_layer: 切分层
        
        Returns:
            不平衡的数据索引
        """
        # 获取每个索引对应的标签
        labels = [self.dataset[idx][1] for idx in all_indices]
        labels = np.array(labels)
        
        # 根据切分层决定类别不平衡程度
        if split_layer == 1:
            # 最差：只保留3-4个类别
            num_classes = 4
        elif split_layer == 2:
            # 较差：保留5-6个类别
            num_classes = 6
        else:
            # 中下：保留7-8个类别
            num_classes = 8
        
        # 随机选择要保留的类别
        unique_labels = np.unique(labels)
        if len(unique_labels) > num_classes:
            np.random.seed(split_layer + len(all_indices))  # 确保可重复
            selected_classes = np.random.choice(unique_labels, num_classes, replace=False)
        else:
            selected_classes = unique_labels
        
        # 只保留选定类别的样本
        mask = np.isin(labels, selected_classes)
        filtered_indices = [all_indices[i] for i in range(len(all_indices)) if mask[i]]
        
        # 如果过滤后样本不够，就用全部
        if len(filtered_indices) < num_samples:
            return filtered_indices
        else:
            return filtered_indices[:num_samples]
    
    def get_quality_info(self, split_layer):
        """
        获取数据质量信息
        
        Args:
            split_layer: 切分层
            
        Returns:
            质量信息字典
        """
        quality_ratios = {
            1: 0.10, 2: 0.25, 3: 0.40, 4: 0.55,
            5: 0.70, 6: 0.80, 7: 0.90, 8: 1.00
        }
        
        ratio = quality_ratios.get(split_layer, 1.0)
        
        return {
            'split_layer': split_layer,
            'quality_ratio': ratio,
            'quality_level': self._get_quality_level(ratio),
            'description': self._get_quality_description(split_layer)
        }
    
    def _get_quality_level(self, ratio):
        """获取质量等级"""
        if ratio <= 0.25:
            return '差'
        elif ratio <= 0.55:
            return '中等'
        elif ratio <= 0.80:
            return '良好'
        else:
            return '优秀'
    
    def _get_quality_description(self, split_layer):
        """获取质量描述"""
        descriptions = {
            1: '最差数据质量（5%数据+4类别+30%噪声）',
            2: '较差数据质量（20%数据+6类别+15%噪声）',
            3: '中下数据质量（35%数据+8类别）',
            4: '中等数据质量（50%数据+全类别）',
            5: '中上数据质量（65%数据+全类别）',
            6: '良好数据质量（80%数据+全类别）',
            7: '优秀数据质量（90%数据+全类别）',
            8: '最优数据质量（100%数据+全类别）',
        }
        return descriptions.get(split_layer, '未知')

class MethodBasedDataAllocator:
    """
    基于不同方法的数据分配策略
    """
    
    def __init__(self, data_quality_manager):
        """
        Args:
            data_quality_manager: 数据质量管理器
        """
        self.dqm = data_quality_manager
    
    def allocate_splitfed(self, client_id):
        """
        SplitFed方法：无激励，客户端说谎
        分配最差的数据（切分层1）
        
        Args:
            client_id: 客户端ID
            
        Returns:
            数据集, 样本数, 切分层
        """
        split_layer = 1  # 最浅切分，最差数据
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_multi_tenant(self, client_id):
        """
        Multi-Tenant SFL方法：常规激励，大锅饭
        分配中下等数据（切分层3，35%数据）
        
        Args:
            client_id: 客户端ID
            
        Returns:
            数据集, 样本数, 切分层
        """
        split_layer = 3  # 降低到35%数据，拉开与Full-Info差距
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_ocd_ucb(self, client_id, round_num, total_rounds):
        """
        OCD-UCB方法（COIN-UCB）：智能分配，逐步优化
        初期：50%数据（切分层4）
        中期：80%数据（切分层6）
        后期：100%数据（切分层8）
        
        注意：这个函数只负责数据分配，实际的切分层由OCDUCBTrainer控制
        
        Args:
            client_id: 客户端ID
            round_num: 当前轮次
            total_rounds: 总轮次
            
        Returns:
            数据集, 样本数, 切分层
        """
        # 根据训练进度动态调整切分层（与OCDUCBTrainer保持一致）
        progress = round_num / total_rounds
        
        if progress < 0.33:  # 前1/3
            split_layer = 4  # 50%数据
        elif progress < 0.67:  # 中1/3
            split_layer = 6  # 80%数据
        else:  # 后1/3
            split_layer = 8  # 100%数据
        
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_full_info(self, client_id):
        """
        完全信息方法：最优分配
        分配最好的数据（切分层8）
        
        Args:
            client_id: 客户端ID
            
        Returns:
            数据集, 样本数, 切分层
        """
        split_layer = 8  # 最深切分，最优数据
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer

def test_data_quality_manager():
    """测试数据质量管理器"""
    print("="*80)
    print("测试数据质量管理器")
    print("="*80)
    
    # 创建模拟数据集
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 创建数据质量管理器
    dqm = DataQualityManager(dataset, num_clients=10)
    
    print(f"\n总样本数: {dqm.total_samples}")
    print(f"客户端数: {dqm.num_clients}")
    
    # 测试不同切分层的数据质量
    print("\n" + "="*80)
    print("不同切分层的数据质量")
    print("="*80)
    
    for split_layer in range(1, 9):
        info = dqm.get_quality_info(split_layer)
        print(f"\n切分层 {split_layer}:")
        print(f"  质量比例: {info['quality_ratio']*100:.0f}%")
        print(f"  质量等级: {info['quality_level']}")
        print(f"  描述: {info['description']}")
    
    # 测试不同方法的数据分配
    print("\n" + "="*80)
    print("不同方法的数据分配策略")
    print("="*80)
    
    allocator = MethodBasedDataAllocator(dqm)
    
    client_id = 0
    
    # SplitFed
    dataset_sf, num_sf, layer_sf = allocator.allocate_splitfed(client_id)
    print(f"\n1. SplitFed: 切分层 {layer_sf}, 样本数 {num_sf}")
    
    # Multi-Tenant SFL
    dataset_mt, num_mt, layer_mt = allocator.allocate_multi_tenant(client_id)
    print(f"2. Multi-Tenant SFL: 切分层 {layer_mt}, 样本数 {num_mt}")
    
    # OCD-UCB (不同阶段)
    for stage, (round_num, total_rounds) in enumerate([
        (10, 100),   # 初期
        (50, 100),   # 中期
        (90, 100),   # 后期
    ]):
        dataset_ocd, num_ocd, layer_ocd = allocator.allocate_ocd_ucb(client_id, round_num, total_rounds)
        stage_name = ['初期', '中期', '后期'][stage]
        print(f"3. OCD-UCB ({stage_name}): 切分层 {layer_ocd}, 样本数 {num_ocd}")
    
    # 完全信息
    dataset_fi, num_fi, layer_fi = allocator.allocate_full_info(client_id)
    print(f"4. 完全信息: 切分层 {layer_fi}, 样本数 {num_fi}")
    
    print("\n" + "="*80)
    print("[OK] 测试完成！")
    print("="*80)

if __name__ == '__main__':
    test_data_quality_manager()
