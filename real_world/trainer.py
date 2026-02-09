"""
训练器
实现四种方法的训练逻辑
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
    """基础训练器"""
    
    def __init__(self, dataset_name, train_dataset, test_dataset, 
                 num_clients=10, batch_size=64, device='cuda', 
                 clients_per_round=3, max_batches_per_client=5):
        """
        Args:
            dataset_name: 数据集名称
            train_dataset: 训练数据集
            test_dataset: 测试数据集
            num_clients: 客户端数量
            batch_size: 批次大小
            device: 设备
            clients_per_round: 每轮训练的客户端数量（加速训练）
            max_batches_per_client: 每个客户端最多训练的batch数（加速训练）
        """
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.device = device
        self.clients_per_round = clients_per_round
        self.max_batches_per_client = max_batches_per_client
        
        # 创建数据质量管理器
        self.dqm = DataQualityManager(train_dataset, num_clients)
        self.allocator = MethodBasedDataAllocator(self.dqm)
        
        # 测试数据加载器
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'test_acc': [],
            'time': [],
            'split_layers': []
        }
        
    def test(self, model):
        """测试模型"""
        model.client_model.eval()
        model.server_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = model.forward(data)
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_history(self, save_path):
        """保存训练历史"""
        import json
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

class SplitFedTrainer(BaseTrainer):
    """SplitFed训练器：无激励，最差数据"""
    
    def train(self, num_rounds=100, lr=0.01):
        """
        训练SplitFed模型
        
        Args:
            num_rounds: 训练轮数
            lr: 学习率
        """
        print(f"\n{'='*80}")
        print(f"开始训练 SplitFed (无激励，最差数据分配)")
        print(f"{'='*80}\n")
        
        # 创建模型（固定使用切分层1）
        split_layer = 1
        model = SplitResNet18(self.dataset_name, split_layer, device=self.device)
        
        # 优化器
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
            
            # 每轮随机选择部分客户端训练（加速）
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # 获取客户端数据（最差数据）
                client_dataset, num_samples, _ = self.allocator.allocate_splitfed(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                # 限制每个客户端的batch数量（加速）
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
            
            # 每轮都测试（提高反馈频率）
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            # 每5轮打印一次（减少输出）
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] SplitFed训练完成！")
        return self.history

class MultiTenantTrainer(BaseTrainer):
    """Multi-Tenant SFL训练器：常规激励，中等数据"""
    
    def train(self, num_rounds=100, lr=0.01):
        """训练Multi-Tenant SFL模型"""
        print(f"\n{'='*80}")
        print(f"开始训练 Multi-Tenant SFL (常规激励，中等数据分配)")
        print(f"{'='*80}\n")
        
        # 创建模型（固定使用切分层4）
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
            
            # 每轮随机选择部分客户端训练（加速）
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # 获取客户端数据（中等数据）
                client_dataset, num_samples, _ = self.allocator.allocate_multi_tenant(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                # 限制每个客户端的batch数量（加速）
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
            
            # 每轮都测试（提高反馈频率）
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            # 每5轮打印一次（减少输出）
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] Multi-Tenant SFL训练完成！")
        return self.history

class OCDUCBTrainer(BaseTrainer):
    """OCD-UCB训练器：智能分配，逐步优化"""
    
    def train(self, num_rounds=100, lr=0.01):
        """训练OCD-UCB模型"""
        print(f"\n{'='*80}")
        print(f"开始训练 OCD-UCB (智能分配，逐步优化)")
        print(f"{'='*80}\n")
        
        # 初始使用切分层4（50%数据，与Multi-Tenant相同起点）
        current_split_layer = 4
        model = SplitResNet18(self.dataset_name, current_split_layer, device=self.device)
        
        optimizer = optim.SGD(
            list(model.client_model.parameters()) + list(model.server_model.parameters()),
            lr=lr, momentum=0.9, weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        start_time = time.time()
        
        for round_num in range(num_rounds):
            # 学习率预热（前10轮）
            if round_num < 10:
                current_lr = lr * ((round_num + 1) / 10)  # 0.1*lr -> 1.0*lr
                if round_num == 0:
                    print(f"[预热] 前10轮使用预热学习率，避免初期不稳定")
                if round_num < 10:
                    print(f"[预热] Round {round_num+1}: 学习率={current_lr:.4f}")
            else:
                current_lr = lr
            
            # 动态调整切分层（渐进式提升数据质量）
            progress = round_num / num_rounds
            if progress < 0.33:
                target_split_layer = 4  # 前1/3：50%数据，学习基础特征
            elif progress < 0.67:
                target_split_layer = 6  # 中1/3：80%数据，提升性能
            else:
                target_split_layer = 8  # 后1/3：100%数据，充分利用全部数据
            
            # 如果切分层改变，迁移模型权重+微调（避免性能跳变）
            if target_split_layer != current_split_layer:
                print(f"\n>>> 切分层调整: {current_split_layer} -> {target_split_layer}")
                
                # 记录切换前的准确率
                pre_switch_acc = self.history['test_acc'][-1] if self.history['test_acc'] else 0
                
                # 保存旧模型的权重
                old_client_state = model.client_model.state_dict()
                old_server_state = model.server_model.state_dict()
                
                # 创建新模型
                current_split_layer = target_split_layer
                new_model = SplitResNet18(self.dataset_name, current_split_layer, device=self.device)
                
                # 迁移可复用的权重
                new_client_state = new_model.client_model.state_dict()
                new_server_state = new_model.server_model.state_dict()
                
                # 复制客户端模型的权重（尽可能多地保留）
                transferred_client = 0
                for name, param in old_client_state.items():
                    if name in new_client_state and param.shape == new_client_state[name].shape:
                        new_client_state[name] = param
                        transferred_client += 1
                
                # 复制服务器模型的权重
                transferred_server = 0
                for name, param in old_server_state.items():
                    if name in new_server_state and param.shape == new_server_state[name].shape:
                        new_server_state[name] = param
                        transferred_server += 1
                
                new_model.client_model.load_state_dict(new_client_state)
                new_model.server_model.load_state_dict(new_server_state)
                model = new_model
                
                print(f">>> 权重迁移: 客户端{transferred_client}层, 服务器{transferred_server}层")
                
                # 微调阶段：用小学习率训练10轮，让模型充分适应新结构
                print(f">>> 开始微调（10轮，学习率={lr*0.02:.4f}）...")
                fine_tune_optimizer = optim.SGD(
                    list(model.client_model.parameters()) + list(model.server_model.parameters()),
                    lr=lr * 0.02, momentum=0.9, weight_decay=5e-4
                )
                
                for ft_round in range(10):
                    model.client_model.train()
                    model.server_model.train()
                    
                    ft_loss = 0
                    ft_batches = 0
                    
                    # 使用当前轮的数据进行微调
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
                    print(f"    微调轮{ft_round+1}/10: Loss={ft_avg_loss:.4f}")
                
                # 微调后测试
                post_finetune_acc = self.test(model)
                print(f">>> 微调完成: 准确率 {pre_switch_acc:.2f}% -> {post_finetune_acc:.2f}% (跳变{post_finetune_acc-pre_switch_acc:+.2f}%)")
                
                # 重新创建优化器（使用当前学习率）
                optimizer = optim.SGD(
                    list(model.client_model.parameters()) + list(model.server_model.parameters()),
                    lr=current_lr, momentum=0.9, weight_decay=5e-4
                )
            
            model.client_model.train()
            model.server_model.train()
            
            round_loss = 0
            num_batches = 0
            
            # 每轮随机选择部分客户端训练（加速）
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # 获取客户端数据（智能分配）
                client_dataset, num_samples, split_layer = self.allocator.allocate_ocd_ucb(
                    client_id, round_num, num_rounds
                )
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                # 限制每个客户端的batch数量（加速）
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
            
            # 每轮都测试（提高反馈频率）
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(current_split_layer)
            
            # 每5轮打印一次（减少输出）
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s, "
                      f"Split Layer={current_split_layer}")
        
        print(f"\n[OK] OCD-UCB训练完成！")
        return self.history

class FullInfoTrainer(BaseTrainer):
    """完全信息训练器：最优分配，作为上界"""
    
    def train(self, num_rounds=100, lr=0.01):
        """训练完全信息模型"""
        print(f"\n{'='*80}")
        print(f"开始训练 完全信息 (最优分配，理论上界)")
        print(f"{'='*80}\n")
        
        # 创建模型（固定使用切分层8）
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
            
            # 每轮随机选择部分客户端训练（加速）
            import numpy as np
            selected_clients = np.random.choice(self.num_clients, self.clients_per_round, replace=False)
            
            for client_id in selected_clients:
                # 获取客户端数据（最优数据）
                client_dataset, num_samples, _ = self.allocator.allocate_full_info(client_id)
                client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
                
                # 限制每个客户端的batch数量（加速）
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
            
            # 每轮都测试（提高反馈频率）
            test_acc = self.test(model)
            elapsed_time = time.time() - start_time
            
            self.history['train_loss'].append(avg_loss)
            self.history['test_acc'].append(test_acc)
            self.history['time'].append(elapsed_time)
            self.history['split_layers'].append(split_layer)
            
            # 每5轮打印一次（减少输出）
            if (round_num + 1) % 5 == 0 or round_num == 0:
                print(f"Round {round_num+1:3d}/{num_rounds}: Loss={avg_loss:.4f}, "
                      f"Test Acc={test_acc:.2f}%, Time={elapsed_time:.1f}s")
        
        print(f"\n[OK] 完全信息训练完成！")
        return self.history
