"""
Contract-based Split Federated Learning on CIFAR-10
Simplified training program - NO regret R(T) tracking (for real-world experiments)
"""
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import sys
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs.config import Config, CLIENT_TYPES, RESNET_PROFILE
from core.physics import SystemPhysics
from env.models import SplitResNet18
from env.data_loader import load_cifar10, create_iid_split, get_client_dataloaders

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Set random seeds for reproducibility
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


class ClientType:
    """Client type definition"""
    def __init__(self, client_dict):
        self.id = client_dict['id']
        self.f = client_dict['f']
        self.tau = client_dict['tau']
        self.data_size = client_dict['data_size']


class ContractSFL:
    """Contract-based Split Federated Learning System"""
    
    def __init__(self, config, num_clients=20, epochs=100, lr=0.001, 
                 batch_size=64, device='cuda', output_dir='./env/outputs'):
        self.config = config
        self.num_clients = num_clients
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*80)
        print("Contract-based Split Federated Learning - CIFAR-10 Training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Number of clients: {num_clients}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        
        # Initialize physics system
        self.physics = SystemPhysics(config)
        
        # Client types (use first num_clients from config)
        self.client_types = [ClientType(CLIENT_TYPES[i]) for i in range(num_clients)]
        
        # Distribution (uniform for simplicity)
        self.distribution_p = np.ones(num_clients) / num_clients
        
        # Get optimal contract (known distribution)
        print("\nSolving optimal contract...")
        self.optimal_menu = self.physics.solve_optimal_contract(
            self.distribution_p, self.client_types
        )
        
        # Print contract distribution
        v_distribution = {}
        for idx in range(num_clients):
            v_star, _ = self.optimal_menu[idx]
            v_distribution[v_star] = v_distribution.get(v_star, 0) + 1
        
        print("\nContract Assignment:")
        for v in sorted(v_distribution.keys()):
            count = v_distribution[v]
            percentage = 100.0 * count / num_clients
            print(f"  Split point v={v}: {count} clients ({percentage:.1f}%)")
        
        # Load CIFAR-10 data
        print("\nLoading CIFAR-10 dataset...")
        self.dataset_train, self.dataset_test = load_cifar10()
        
        # Create IID split
        print(f"Creating IID data split for {num_clients} clients...")
        self.dict_users_train = create_iid_split(self.dataset_train, num_clients)
        self.dict_users_test = create_iid_split(self.dataset_test, num_clients)
        
        # Create data loaders
        self.train_loaders, self.test_loaders = get_client_dataloaders(
            self.dataset_train, self.dataset_test,
            self.dict_users_train, self.dict_users_test,
            batch_size=batch_size, num_workers=0
        )
        
        # Initialize models based on contract
        print("\nInitializing models...")
        self.models = []
        self.optimizers = []
        self.cut_points = []
        
        for idx in range(num_clients):
            v_star, _ = self.optimal_menu[idx]
            self.cut_points.append(v_star)
            
            # Create unified model
            model = SplitResNet18(num_classes=10).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            self.models.append(model)
            self.optimizers.append(optimizer)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage (NO R(T) tracking for real-world experiments)
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        
        print("\nInitialization complete!")
        print("="*80)
        
    def train_one_epoch(self, epoch):
        """Train for one epoch using split learning"""
        epoch_loss = []
        epoch_acc = []
        
        for idx in range(self.num_clients):
            self.models[idx].train()
            
            batch_loss = []
            batch_acc = []
            
            v = self.cut_points[idx]
            
            for batch_idx, (images, labels) in enumerate(self.train_loaders[idx]):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizers[idx].zero_grad()
                
                # Client-side forward pass
                smashed_data = self.models[idx].forward_client(images, v)
                smashed_data_detached = smashed_data.clone().detach().requires_grad_(True)
                
                # Server-side forward pass
                outputs = self.models[idx].forward_server(smashed_data_detached, v)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                acc = 100.0 * correct / labels.size(0)
                
                # Backward pass: server side
                loss.backward()
                grad_smashed = smashed_data_detached.grad.clone().detach()
                
                # Backward pass: client side
                smashed_data.backward(grad_smashed)
                
                # Update parameters
                self.optimizers[idx].step()
                
                batch_loss.append(loss.item())
                batch_acc.append(acc)
            
            # Average over batches
            avg_loss = np.mean(batch_loss)
            avg_acc = np.mean(batch_acc)
            epoch_loss.append(avg_loss)
            epoch_acc.append(avg_acc)
        
        # Average over clients
        train_loss = np.mean(epoch_loss)
        train_acc = np.mean(epoch_acc)
        
        return train_loss, train_acc
    
    def test(self, epoch):
        """Test all clients"""
        epoch_loss = []
        epoch_acc = []
        
        for idx in range(self.num_clients):
            self.models[idx].eval()
            
            batch_loss = []
            batch_acc = []
            
            v = self.cut_points[idx]
            
            with torch.no_grad():
                for images, labels in self.test_loaders[idx]:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Split forward pass
                    smashed_data = self.models[idx].forward_client(images, v)
                    outputs = self.models[idx].forward_server(smashed_data, v)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    correct = predicted.eq(labels).sum().item()
                    acc = 100.0 * correct / labels.size(0)
                    
                    batch_loss.append(loss.item())
                    batch_acc.append(acc)
            
            # Average over batches
            avg_loss = np.mean(batch_loss)
            avg_acc = np.mean(batch_acc)
            epoch_loss.append(avg_loss)
            epoch_acc.append(avg_acc)
        
        # Average over clients
        test_loss = np.mean(epoch_loss)
        test_acc = np.mean(epoch_acc)
        
        return test_loss, test_acc
    
    def federated_averaging(self):
        """Perform federated averaging on all models"""
        # Average model parameters across all clients
        w_avg = copy.deepcopy(self.models[0].state_dict())
        for k in w_avg.keys():
            for i in range(1, self.num_clients):
                w_avg[k] += self.models[i].state_dict()[k]
            w_avg[k] = torch.div(w_avg[k], self.num_clients)
        
        # Update all models
        for i in range(self.num_clients):
            self.models[i].load_state_dict(w_avg)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}/{self.epochs}]")
            
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Test
            test_loss, test_acc = self.test(epoch)
            
            # Federated averaging every 5 epochs
            if (epoch + 1) % 5 == 0:
                print("  Performing federated averaging...")
                self.federated_averaging()
            
            # Store metrics
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)
            
            # Print summary
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("Training completed!")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print("="*80)
        
        # Save results
        self.save_results(elapsed_time)
    
    def save_results(self, elapsed_time):
        """Save training results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"训练输出数据_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Contract-based Split Federated Learning - CIFAR-10 Training Results\n")
            f.write("="*80 + "\n\n")
            
            f.write("Configuration:\n")
            f.write(f"  Number of clients: {self.num_clients}\n")
            f.write(f"  Epochs: {self.epochs}\n")
            f.write(f"  Batch size: {self.batch_size}\n")
            f.write(f"  Learning rate: {self.lr}\n")
            f.write(f"  Device: {self.device}\n")
            f.write(f"  ALPHA: {self.config.ALPHA}\n")
            f.write(f"  BETA: {self.config.BETA}\n")
            f.write(f"  MU: {self.config.MU}\n")
            f.write(f"  Training time: {elapsed_time/60:.2f} minutes\n\n")
            
            f.write("ResNet-18 Load Profile (Real Measured):\n")
            for v in range(1, 6):
                profile = RESNET_PROFILE[v]
                f.write(f"  v={v}: W={profile['W']:.2f}, D={profile['D']:.2f} MB\n")
            f.write("\n")
            
            f.write("Contract Assignment:\n")
            v_distribution = {}
            for idx in range(self.num_clients):
                v_star, r_star = self.optimal_menu[idx]
                v_distribution[v_star] = v_distribution.get(v_star, 0) + 1
                f.write(f"  Client {idx:2d}: v*={v_star}, R*={r_star:.4f}\n")
            
            f.write("\nContract Distribution:\n")
            for v in sorted(v_distribution.keys()):
                count = v_distribution[v]
                percentage = 100.0 * count / self.num_clients
                f.write(f"  v={v}: {count} clients ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Final Results:\n")
            f.write(f"  Final Train Loss: {self.train_loss_history[-1]:.4f}\n")
            f.write(f"  Final Train Acc:  {self.train_acc_history[-1]:.2f}%\n")
            f.write(f"  Final Test Loss:  {self.test_loss_history[-1]:.4f}\n")
            f.write(f"  Final Test Acc:   {self.test_acc_history[-1]:.2f}%\n\n")
            
            f.write("Training History:\n")
            f.write("Epoch | Train Loss | Train Acc | Test Loss | Test Acc\n")
            f.write("-" * 60 + "\n")
            for i in range(len(self.train_loss_history)):
                f.write(f"{i+1:5d} | {self.train_loss_history[i]:10.4f} | "
                       f"{self.train_acc_history[i]:9.2f} | "
                       f"{self.test_loss_history[i]:9.4f} | "
                       f"{self.test_acc_history[i]:8.2f}\n")
        
        print(f"\nResults saved to: {output_file}")
        
        # Plot and save figures
        self.plot_results(timestamp)
    
    def plot_results(self, timestamp):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.train_loss_history) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_loss_history, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.test_loss_history, 'r-', label='Test Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_acc_history, 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.test_acc_history, 'r-', label='Test Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Contract distribution
        v_stars = [self.optimal_menu[i][0] for i in range(self.num_clients)]
        unique, counts = np.unique(v_stars, return_counts=True)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        axes[1, 0].bar(unique, counts, color=[colors[v-1] for v in unique], edgecolor='black', linewidth=1.5)
        axes[1, 0].set_xlabel('Split Point (v*)', fontsize=12)
        axes[1, 0].set_ylabel('Number of Clients', fontsize=12)
        axes[1, 0].set_title('Contract Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(range(1, 6))
        axes[1, 0].grid(True, axis='y', alpha=0.3)
        
        # Load profile visualization
        v_list = list(range(1, 6))
        W_list = [RESNET_PROFILE[v]['W'] for v in v_list]
        D_list = [RESNET_PROFILE[v]['D'] for v in v_list]
        
        ax2 = axes[1, 1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(v_list, W_list, 'b-o', label='W (Computation)', linewidth=2, markersize=8)
        line2 = ax2_twin.plot(v_list, D_list, 'r-s', label='D (Communication)', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Split Point (v)', fontsize=12)
        ax2.set_ylabel('W (Normalized)', fontsize=12, color='b')
        ax2_twin.set_ylabel('D (MB)', fontsize=12, color='r')
        ax2.set_title('ResNet-18 Load Profile', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.set_xticks(v_list)
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, f"training_curves_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to: {plot_file}")


def main():
    """Main function"""
    # Configuration
    config = Config()
    
    # Training parameters
    num_clients = 20
    epochs = 100
    lr = 0.001
    batch_size = 64
    
    # Create and run training
    sfl = ContractSFL(
        config=config,
        num_clients=num_clients,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device='cuda',
        output_dir='./env/outputs'
    )
    
    sfl.train()


if __name__ == '__main__':
    main()
