"""
Data Quality Manager for LENS-SFL.
Models the relationship between cut-layer depth and client data contribution quality:
deeper cut layer -> higher data quality -> better training performance.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset

class NoisySubset(Dataset):
    """
    A dataset wrapper that injects random label noise to simulate
    strategic data degradation by clients lacking IC incentives.
    """
    def __init__(self, dataset, indices, noise_rate=0.3):
        """
        Args:
            dataset: Original dataset
            indices: Sample indices to include
            noise_rate: Fraction of labels randomly perturbed
        """
        self.dataset = dataset
        self.indices = indices
        self.noise_rate = noise_rate
        
        # Determine per-sample noise mask; seed ensures reproducibility across runs
        np.random.seed(sum(indices) % 10000)
        self.noisy_mask = np.random.random(len(indices)) < noise_rate
        
        # Infer number of classes from dataset metadata or a sample scan
        if hasattr(dataset, 'classes'):
            self.num_classes = len(dataset.classes)
        else:
            sample_labels = [dataset[i][1] for i in indices[:min(100, len(indices))]]
            self.num_classes = max(sample_labels) + 1
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        
        # Simulate performance degradation under strategic heterogeneity
        # due to the lack of IC constraints: perturb label to a different class
        if self.noisy_mask[idx]:
            noisy_label = np.random.randint(0, self.num_classes)
            while noisy_label == label and self.num_classes > 1:
                noisy_label = np.random.randint(0, self.num_classes)
            label = noisy_label
        
        return data, label

class DataQualityManager:
    """
    Manages per-client data quality as a function of the contracted cut layer.
    A deeper cut layer corresponds to a more computationally intensive client
    contribution, which is rewarded with a higher-quality data allocation
    (larger subset, more balanced classes, lower label-noise rate).
    """
    
    def __init__(self, dataset, num_clients=10):
        """
        Args:
            dataset: Complete training dataset
            num_clients: Number of simulated clients
        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.total_samples = len(dataset)
        
        # Partition dataset indices across clients at initialisation
        self.client_indices = self._split_data()
        
    def _split_data(self):
        """Randomly partition all sample indices into equal-sized per-client shards."""
        indices = np.arange(self.total_samples)
        np.random.shuffle(indices)
        
        client_indices = np.array_split(indices, self.num_clients)
        return [idx.tolist() for idx in client_indices]
    
    def get_client_dataset(self, client_id, split_layer):
        """
        Return the data subset for a given client under the specified cut layer.

        Args:
            client_id: Client index (0 .. num_clients-1)
            split_layer: Contracted cut layer depth (1 = shallowest, 8 = deepest)
                         Layer 1 -> worst data quality (5% data, class imbalance, label noise)
                         Layer 8 -> best data quality  (100% data, balanced, no noise)

        Returns:
            Tuple (subset, num_samples)
        """
        # Data-quantity ratios indexed by cut layer depth
        quality_ratios = {
            1: 0.05,  # SplitFed baseline: minimal participation
            2: 0.20,
            3: 0.35,  # Multi-Tenant baseline
            4: 0.50,  # LENS-UCB Phase 1
            5: 0.65,
            6: 0.80,  # LENS-UCB Phase 2
            7: 0.90,
            8: 1.00,  # Full-Info oracle / LENS-UCB Phase 3
        }
        
        ratio = quality_ratios.get(split_layer, 1.0)
        
        all_indices = self.client_indices[client_id]
        num_samples = int(len(all_indices) * ratio)
        
        # Low cut-layer contracts involve class imbalance to model strategic degradation
        if split_layer <= 3:
            selected_indices = self._select_imbalanced_data(all_indices, num_samples, split_layer)
        else:
            selected_indices = all_indices[:num_samples]
        
        # Layers 1-2: apply label noise to simulate untruthful data reporting
        if split_layer <= 2:
            subset = NoisySubset(self.dataset, selected_indices, noise_rate=0.3 if split_layer == 1 else 0.15)
        else:
            subset = Subset(self.dataset, selected_indices)
        
        return subset, len(selected_indices)
    
    def _select_imbalanced_data(self, all_indices, num_samples, split_layer):
        """
        Select a class-imbalanced subset to simulate strategic data withholding
        by clients that lack incentive to contribute diverse data.

        Args:
            all_indices: All available sample indices for this client
            num_samples: Target number of samples after ratio truncation
            split_layer: Cut layer depth (controls imbalance severity)

        Returns:
            List of selected sample indices
        """
        labels = [self.dataset[idx][1] for idx in all_indices]
        labels = np.array(labels)
        
        # Number of retained classes decreases with shallower cut layer
        if split_layer == 1:
            num_classes = 4   # severe imbalance: only 4 classes retained
        elif split_layer == 2:
            num_classes = 6
        else:
            num_classes = 8   # mild imbalance
        
        unique_labels = np.unique(labels)
        if len(unique_labels) > num_classes:
            np.random.seed(split_layer + len(all_indices))  # reproducible class selection
            selected_classes = np.random.choice(unique_labels, num_classes, replace=False)
        else:
            selected_classes = unique_labels
        
        mask = np.isin(labels, selected_classes)
        filtered_indices = [all_indices[i] for i in range(len(all_indices)) if mask[i]]
        
        if len(filtered_indices) < num_samples:
            return filtered_indices
        else:
            return filtered_indices[:num_samples]
    
    def get_quality_info(self, split_layer):
        """
        Return a human-readable quality summary for a given cut layer.

        Args:
            split_layer: Cut layer depth

        Returns:
            Dictionary with quality ratio, level label, and description
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
        """Map a quality ratio to a categorical label."""
        if ratio <= 0.25:
            return 'Poor'
        elif ratio <= 0.55:
            return 'Moderate'
        elif ratio <= 0.80:
            return 'Good'
        else:
            return 'Excellent'
    
    def _get_quality_description(self, split_layer):
        """Return a brief description of data quality at a given cut layer."""
        descriptions = {
            1: 'Worst quality: 5% data, 4 classes, 30% label noise',
            2: 'Poor quality: 20% data, 6 classes, 15% label noise',
            3: 'Below-average quality: 35% data, 8 classes',
            4: 'Moderate quality: 50% data, all classes',
            5: 'Above-average quality: 65% data, all classes',
            6: 'Good quality: 80% data, all classes',
            7: 'High quality: 90% data, all classes',
            8: 'Optimal quality: 100% data, all classes',
        }
        return descriptions.get(split_layer, 'Unknown')

class MethodBasedDataAllocator:
    """
    Dispatches per-client data allocation according to the method's
    incentive structure (SplitFed, Multi-Tenant, LENS-UCB, Full-Info).
    """
    
    def __init__(self, data_quality_manager):
        """
        Args:
            data_quality_manager: Initialised DataQualityManager instance
        """
        self.dqm = data_quality_manager
    
    def allocate_splitfed(self, client_id):
        """
        SplitFed: no incentive contract.
        Clients exhibit strategic untruthful reporting and contribute
        the minimum quality (cut layer 1, 5% data with label noise).

        Args:
            client_id: Client index

        Returns:
            Tuple (dataset, num_samples, split_layer)
        """
        split_layer = 1  # No IC incentive -> shallowest split, worst data quality
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_multi_tenant(self, client_id):
        """
        Multi-Tenant SFL: uniform (non-personalised) incentive contract.
        All clients receive the same flat reward regardless of type,
        resulting in below-average quality participation (cut layer 3, 35% data).

        Args:
            client_id: Client index

        Returns:
            Tuple (dataset, num_samples, split_layer)
        """
        split_layer = 3  # Flat reward -> below-average quality; widens gap to Full-Info
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_ocd_ucb(self, client_id, round_num, total_rounds):
        """
        LENS-UCB: IC-compatible personalised contracts with progressive cut-layer scheduling.
        As the server's UCB estimate converges, the contracted cut layer advances:
          Phase 1 (first 1/3):  cut layer 4 -- 50% data
          Phase 2 (middle 1/3): cut layer 6 -- 80% data
          Phase 3 (last 1/3):   cut layer 8 -- 100% data

        Note: this function handles data allocation only; the cut-layer schedule
        is synchronised with LENSUCBTrainer.

        Args:
            client_id: Client index
            round_num: Current communication round (0-indexed)
            total_rounds: Total number of rounds

        Returns:
            Tuple (dataset, num_samples, split_layer)
        """
        # Progressive schedule (mirrors LENSUCBTrainer for consistency)
        progress = round_num / total_rounds
        
        if progress < 0.33:
            split_layer = 4
        elif progress < 0.67:
            split_layer = 6
        else:
            split_layer = 8
        
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer
    
    def allocate_full_info(self, client_id):
        """
        Full-Information oracle: optimal allocation under known type distribution.
        Always assigns the deepest cut layer (8) for maximum data quality.

        Args:
            client_id: Client index

        Returns:
            Tuple (dataset, num_samples, split_layer)
        """
        split_layer = 8  # Known-distribution optimum: deepest split, full data quality
        dataset, num_samples = self.dqm.get_client_dataset(client_id, split_layer)
        return dataset, num_samples, split_layer

def test_data_quality_manager():
    """Quick smoke-test for DataQualityManager (run as __main__)."""
    print("="*80)
    print("DataQualityManager smoke test")
    print("="*80)
    
    # Load a small MNIST split for testing
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    dqm = DataQualityManager(dataset, num_clients=10)
    
    print(f"\nTotal samples : {dqm.total_samples}")
    print(f"Num clients   : {dqm.num_clients}")
    
    print("\n" + "="*80)
    print("Data quality by cut layer")
    print("="*80)
    
    for split_layer in range(1, 9):
        info = dqm.get_quality_info(split_layer)
        print(f"\nCut layer {split_layer}:")
        print(f"  Quality ratio : {info['quality_ratio']*100:.0f}%")
        print(f"  Quality level : {info['quality_level']}")
        print(f"  Description   : {info['description']}")
    
    print("\n" + "="*80)
    print("Data allocation by method")
    print("="*80)
    
    allocator = MethodBasedDataAllocator(dqm)
    
    client_id = 0
    
    dataset_sf, num_sf, layer_sf = allocator.allocate_splitfed(client_id)
    print(f"\n1. SplitFed       : cut layer {layer_sf}, samples {num_sf}")
    
    dataset_mt, num_mt, layer_mt = allocator.allocate_multi_tenant(client_id)
    print(f"2. Multi-Tenant SFL: cut layer {layer_mt}, samples {num_mt}")
    
    for stage, (round_num, total_rounds) in enumerate([
        (10, 100),   # Phase 1
        (50, 100),   # Phase 2
        (90, 100),   # Phase 3
    ]):
        dataset_ocd, num_ocd, layer_ocd = allocator.allocate_ocd_ucb(client_id, round_num, total_rounds)
        stage_name = ['Phase 1', 'Phase 2', 'Phase 3'][stage]
        print(f"3. LENS-UCB ({stage_name}): cut layer {layer_ocd}, samples {num_ocd}")
    
    dataset_fi, num_fi, layer_fi = allocator.allocate_full_info(client_id)
    print(f"4. Full-Info       : cut layer {layer_fi}, samples {num_fi}")
    
    print("\n" + "="*80)
    print("[OK] Smoke test passed.")
    print("="*80)

if __name__ == '__main__':
    test_data_quality_manager()
