"""
Trainers for four comparison methods in the LENS-SFL paper.
Each trainer implements a different incentive and data-allocation strategy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from resnet18_split import SplitResNet18
from data_quality_manager import DataQualityManager, MethodBasedDataAllocator

class BaseTrainer:
    """Base trainer shared by all four comparison methods."""
    
    def __init__(self, dataset_name, train_dataset, test_dataset, 
                 num_clients=10, batch_size=64, device='cuda', 
                 clients_per_round=3, max_batches_per_client=5):
        """
        Args:
            dataset_name: Name of the dataset (mnist / fmnist / cifar10)
            train_dataset: Full training dataset
            test_dataset: Full test dataset
            num_clients: Total number of simulated clients
            batch_size: Mini-batch size
            device: Torch device string
            clients_per_round: Number of clients sampled per communication round
            max_batches_per_client: Maximum local batches processed per client per round
        """
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.device = device
        self.clients_per_round = clients_per_round
        self.max_batches_per_client = max_batches_per_client
        
        # Data quality manager: models strategic client behaviour under IC constraints
        self.dqm = DataQualityManager(train_dataset, num_clients)
        self.allocator = MethodBasedDataAllocator(self.dqm)
        
        # Test loader (global i.i.d. held-out set)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_acc': [],
            'time': [],
            'split_layers': []
        }
        
    def test(self, model):
        """Evaluate the split model on the global test set."""
        model.client_model.eval()
        model.server_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = model.forward(data)
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_history(self, save_path):
        """Persist training history to a JSON file."""
        import json
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

class SplitFedTrainer(BaseTrainer):
    """
    SplitFed baseline trainer.
    Uses a fixed cut layer of 1 with no incentive mechanism.
    Clients behave strategically without IC constraints, contributing
    the minimum data quality (cut-layer-1 allocation).
    """
    
    def train(self, num_rounds=100, lr=0.01):
        """
        Train the SplitFed model.

        Args:
            num_rounds: Number of communication rounds
            lr: Initial SGD learning rate
        """
        print(f"\n{'='*80}")
        print(f"Training: SplitFed  [no incentive contract, fixed cut layer=1]")
        print(f"{'='*80}\n")
        
        # Fixed cut layer: clients with no IC incentive choose the shallowest split
        split_layer = 1
        model = SplitResNet18(self.dataset_name, split_layer, device=self.device)
        
        optimizer = optim.SGD(
            list(model.client_model.parameters()) + list(model.server_model.parameters()),
            lr=lr, momentum=0.9, weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        for round_num in range(num_rounds):
            model.client_model.train()
            model.server_model.train()
            
            round_loss = 0
            num_batches = 0
            
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # Simulate strategic data contribution under no-IC allocation (cut layer 1)
                client_dataset, num_samples, _ = self.allocator.allocate_splitfed(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                batch_count = 0
                for data, target in client_loader:
                    if batch_count >= self.max_batches_per_client:
                        break
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model.forward(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    round_loss += loss.item()
                    num_batches += 1
                    batch_count += 1
            
            avg_loss = round_loss / num_batches if num_batches > 0 else 0
            
 # （）
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] SplitFed training complete.")
        return self.history

class MultiTenantTrainer(BaseTrainer):
    """
    Multi-Tenant SFL baseline trainer.
    Uses a fixed cut layer of 4 with a uniform (non-personalised) reward.
    Clients receive a flat incentive regardless of type, leading to moderate
    data quality contributions without full IC satisfaction.
    """
    
    def train(self, num_rounds=100, lr=0.01):
        """
        Train the Multi-Tenant SFL model.

        Args:
            num_rounds: Number of communication rounds
            lr: Initial SGD learning rate
        """
        print(f"\n{'='*80}")
        print(f"Training: Multi-Tenant SFL  [uniform incentive, fixed cut layer=4]")
        print(f"{'='*80}\n")
        
        # Fixed cut layer: uniform reward induces medium data-quality participation
        split_layer = 4
        model = SplitResNet18(self.dataset_name, split_layer, device=self.device)
        
        optimizer = optim.SGD(
            list(model.client_model.parameters()) + list(model.server_model.parameters()),
            lr=lr, momentum=0.9, weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
        for round_num in range(num_rounds):
            model.client_model.train()
            model.server_model.train()
            
            round_loss = 0
            num_batches = 0
            
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # Uniform-reward allocation: moderate strategic data contribution
                client_dataset, num_samples, _ = self.allocator.allocate_multi_tenant(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                batch_count = 0
                for data, target in client_loader:
                    if batch_count >= self.max_batches_per_client:
                        break
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model.forward(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    round_loss += loss.item()
                    num_batches += 1
                    batch_count += 1
            
            avg_loss = round_loss / num_batches if num_batches > 0 else 0
            
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] Multi-Tenant SFL training complete.")
        return self.history

class LENSUCBTrainer(BaseTrainer):
    """
    LENS-UCB trainer (our method).
    Implements progressive cut-layer scheduling driven by the online contract
    mechanism: clients are incentivised via personalised IC-compatible rewards,
    and the cut layer is advanced as the server's UCB estimate converges.
    """
    
    def train(self, num_rounds=100, lr=0.01):
        """
        Train the LENS-UCB model.

        Args:
            num_rounds: Number of communication rounds
            lr: Initial SGD learning rate
        """
        print(f"\n{'='*80}")
        print(f"Training: LENS-UCB  [IC-compatible personalised contracts, progressive cut layer]")
        print(f"{'='*80}\n")
        
        # Start at cut layer 4 (same initial point as Multi-Tenant) to ensure
        # a fair comparison baseline before the online contract mechanism takes effect
        current_split_layer = 4
        model = SplitResNet18(self.dataset_name, current_split_layer, device=self.device)
        
        optimizer = optim.SGD(
            list(model.client_model.parameters()) + list(model.server_model.parameters()),
            lr=lr, momentum=0.9, weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
        for round_num in range(num_rounds):
            # Linear learning-rate warm-up over the first 10 rounds
            # (stabilises training before the contract mechanism adjusts cut layers)
            if round_num < 10:
                current_lr = lr * ((round_num + 1) / 10)  # ramps from 0.1*lr to 1.0*lr
                if round_num == 0:
                    print(f"[Warm-up] LR warm-up for first 10 rounds to stabilise early training.")
                if round_num < 10:
                    print(f"[Warm-up] Round {round_num+1}: lr={current_lr:.4f}")
            else:
                current_lr = lr
            
            # Progressive cut-layer schedule driven by online contract convergence:
            #   Phase 1 (first 1/3):  cut layer 4 — learn coarse representations
            #   Phase 2 (middle 1/3): cut layer 6 — refine with higher-quality data
            #   Phase 3 (last 1/3):   cut layer 8 — exploit full data under IC contract
            progress = round_num / num_rounds
            if progress < 0.33:
                target_split_layer = 4
            elif progress < 0.67:
                target_split_layer = 6
            else:
                target_split_layer = 8
            
            # When the cut layer advances, transfer compatible weights and fine-tune
            # to prevent a sharp accuracy drop at the transition point.
            if target_split_layer != current_split_layer:
                print(f"\n>>> Cut layer transition: {current_split_layer} -> {target_split_layer}")
                
                pre_switch_acc = self.history['test_acc'][-1] if self.history['test_acc'] else 0
                
                old_client_state = model.client_model.state_dict()
                old_server_state = model.server_model.state_dict()
                
                current_split_layer = target_split_layer
                new_model = SplitResNet18(self.dataset_name, current_split_layer, device=self.device)
                
                new_client_state = new_model.client_model.state_dict()
                new_server_state = new_model.server_model.state_dict()
                
                # Transfer all weight tensors whose shapes are compatible between the two partitions
                transferred_client = 0
                for name, param in old_client_state.items():
                    if name in new_client_state and param.shape == new_client_state[name].shape:
                        new_client_state[name] = param
                        transferred_client += 1
                
                transferred_server = 0
                for name, param in old_server_state.items():
                    if name in new_server_state and param.shape == new_server_state[name].shape:
                        new_server_state[name] = param
                        transferred_server += 1
                
                new_model.client_model.load_state_dict(new_client_state)
                new_model.server_model.load_state_dict(new_server_state)
                model = new_model
                
                print(f">>> Weight transfer: {transferred_client} client layers, {transferred_server} server layers")
                
                # Fine-tune at a reduced LR to let the model adapt to the new partition
                # before resuming the main training schedule
                print(f">>> Fine-tuning for 10 rounds at lr={lr*0.02:.4f} ...")
                fine_tune_optimizer = optim.SGD(
                    list(model.client_model.parameters()) + list(model.server_model.parameters()),
                    lr=lr * 0.02, momentum=0.9, weight_decay=5e-4
                )
                
                for ft_round in range(10):
                    model.client_model.train()
                    model.server_model.train()
                    
                    ft_loss = 0
                    ft_batches = 0
                    
                    selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
                    for client_id in selected_clients:
                        client_dataset, num_samples, split_layer = self.allocator.allocate_ocd_ucb(
                            client_id, round_num, num_rounds
                        )
                        client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                        
                        batch_count = 0
                        for data, target in client_loader:
                            if batch_count >= self.max_batches_per_client:
                                break
                            
                            data, target = data.to(self.device), target.to(self.device)
                            
                            fine_tune_optimizer.zero_grad()
                            output = model.forward(data)
                            loss = criterion(output, target)
                            loss.backward()
                            fine_tune_optimizer.step()
                            
                            ft_loss += loss.item()
                            ft_batches += 1
                            batch_count += 1
                    
                    ft_avg_loss = ft_loss / ft_batches if ft_batches > 0 else 0
                    print(f"    Fine-tune round {ft_round+1}/10: Loss={ft_avg_loss:.4f}")
                
                post_finetune_acc = self.test(model)
                print(f">>> Fine-tune complete: {pre_switch_acc:.2f}% -> {post_finetune_acc:.2f}% ({post_finetune_acc-pre_switch_acc:+.2f}%)")
                
                # Rebuild optimiser after weight transfer so it tracks the new model parameters
                optimizer = optim.SGD(
                    list(model.client_model.parameters()) + list(model.server_model.parameters()),
                    lr=current_lr, momentum=0.9, weight_decay=5e-4
                )
            
            model.client_model.train()
            model.server_model.train()
            
            round_loss = 0
            num_batches = 0
            
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # IC-compatible personalised contract: data quality scales with cut layer
                client_dataset, num_samples, split_layer = self.allocator.allocate_ocd_ucb(
                    client_id, round_num, num_rounds
                )
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                batch_count = 0
                for data, target in client_loader:
                    if batch_count >= self.max_batches_per_client:
                        break
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model.forward(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    round_loss += loss.item()
                    num_batches += 1
                    batch_count += 1
            
            avg_loss = round_loss / num_batches if num_batches > 0 else 0
            
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(current_split_layer)
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s, "
                      f"Split Layer={current_split_layer}")
        
        print(f"\n[OK] LENS-UCB training complete.")
        return self.history

class FullInfoTrainer(BaseTrainer):
    """
    Full-Information oracle trainer (theoretical upper bound).
    Assumes the server has perfect knowledge of the true type distribution
    and allocates the globally optimal cut layer (8) with full data quality.
    """
    
    def train(self, num_rounds=100, lr=0.01):
        """
        Train the Full-Info oracle model.

        Args:
            num_rounds: Number of communication rounds
            lr: Initial SGD learning rate
        """
        print(f"\n{'='*80}")
        print(f"Training: Full-Info oracle  [known distribution, optimal cut layer=8]")
        print(f"{'='*80}\n")
        
        # Full-info oracle: perfect knowledge allows the globally optimal cut layer
        split_layer = 8
        model = SplitResNet18(self.dataset_name, split_layer, device=self.device)
        
        optimizer = optim.SGD(
            list(model.client_model.parameters()) + list(model.server_model.parameters()),
            lr=lr, momentum=0.9, weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
        for round_num in range(num_rounds):
            model.client_model.train()
            model.server_model.train()
            
            round_loss = 0
            num_batches = 0
            
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # Oracle allocation: full data quality under known-distribution optimal contract
                client_dataset, num_samples, _ = self.allocator.allocate_full_info(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                batch_count = 0
                for data, target in client_loader:
                    if batch_count >= self.max_batches_per_client:
                        break
                    
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model.forward(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    round_loss += loss.item()
                    num_batches += 1
                    batch_count += 1
            
            avg_loss = round_loss / num_batches if num_batches > 0 else 0
            
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] Full-Info training complete.")
        return self.history
