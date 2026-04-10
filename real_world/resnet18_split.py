"""
Split ResNet-18
，MNIST、Fashion-MNIST、CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
 """ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ClientModel(nn.Module):
 """（）"""
    
    def __init__(self, dataset='cifar10', split_layer=4):
        """
        Args:
 dataset: ('mnist', 'fmnist', 'cifar10')
 split_layer: (1-8)
 1: （，）
 8: （，）
        """
        super(ClientModel, self).__init__()
        self.dataset = dataset
        self.split_layer = split_layer
        
 # 
        if dataset in ['mnist', 'fmnist']:
            in_channels = 1
 # MNIST/FMNIST
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # cifar10
            in_channels = 3
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        
 # ResNet-184layer
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 1-2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 3-4
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 5-6
        self.layer4 = self._make_layer(256, 512, 2, stride=2) # 7-8
        
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
 """，split_layer"""
 # 
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.split_layer >= 1:
            out = self.layer1[0](out)  # 1
        if self.split_layer >= 2:
            out = self.layer1[1](out)  # 2
        if self.split_layer >= 3:
            out = self.layer2[0](out)  # 3
        if self.split_layer >= 4:
            out = self.layer2[1](out)  # 4
        if self.split_layer >= 5:
            out = self.layer3[0](out)  # 5
        if self.split_layer >= 6:
            out = self.layer3[1](out)  # 6
        if self.split_layer >= 7:
            out = self.layer4[0](out)  # 7
        if self.split_layer >= 8:
            out = self.layer4[1](out)  # 8
            
        return out

class ServerModel(nn.Module):
 """（）"""
    
    def __init__(self, dataset='cifar10', split_layer=4, num_classes=10):
        """
        Args:
 dataset:
 split_layer: (1-8)
 num_classes:
        """
        super(ServerModel, self).__init__()
        self.dataset = dataset
        self.split_layer = split_layer
        self.num_classes = num_classes
        
 # 
        if split_layer <= 2:
            in_channels = 64
        elif split_layer <= 4:
            in_channels = 128
        elif split_layer <= 6:
            in_channels = 256
        else:
            in_channels = 512
        
 # 
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
 # 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
 """，split_layer"""
        out = x
        
 # 
        if self.split_layer < 1:
            out = self.layer1[0](out)
        if self.split_layer < 2:
            out = self.layer1[1](out)
        if self.split_layer < 3:
            out = self.layer2[0](out)
        if self.split_layer < 4:
            out = self.layer2[1](out)
        if self.split_layer < 5:
            out = self.layer3[0](out)
        if self.split_layer < 6:
            out = self.layer3[1](out)
        if self.split_layer < 7:
            out = self.layer4[0](out)
        if self.split_layer < 8:
            out = self.layer4[1](out)
        
 # 
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

class SplitResNet18:
 """Split ResNet-18"""
    
    def __init__(self, dataset='cifar10', split_layer=4, num_classes=10, device='cuda'):
        """
        Args:
 dataset: ('mnist', 'fmnist', 'cifar10')
 split_layer: (1-8)
 num_classes:
 device:
        """
        self.dataset = dataset
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = device
        
 # 
        self.client_model = ClientModel(dataset, split_layer).to(device)
        self.server_model = ServerModel(dataset, split_layer, num_classes).to(device)
        
    def forward(self, x):
 """"""
 # 
        smashed_data = self.client_model(x)
        
 # 
        output = self.server_model(smashed_data)
        
        return output
    
    def get_client_model(self):
 """"""
        return self.client_model
    
    def get_server_model(self):
 """"""
        return self.server_model
    
    def get_split_info(self):
 """"""
        return {
            'dataset': self.dataset,
            'split_layer': self.split_layer,
            'num_classes': self.num_classes,
            'client_params': sum(p.numel() for p in self.client_model.parameters()),
            'server_params': sum(p.numel() for p in self.server_model.parameters()),
        }

def test_split_resnet18():
 """Split ResNet-18"""
    print("="*80)
 print("Split ResNet-18")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 print(f": {device}\n")
    
 # 
    datasets = ['mnist', 'fmnist', 'cifar10']
    split_layers = [1, 4, 8]
    
    for dataset in datasets:
 print(f"\n: {dataset.upper()}")
        print("-"*80)
        
 # 
        if dataset in ['mnist', 'fmnist']:
            input_shape = (2, 1, 28, 28)  # batch_size=2, channels=1, 28x28
        else:
            input_shape = (2, 3, 32, 32)  # batch_size=2, channels=3, 32x32
        
        for split_layer in split_layers:
            model = SplitResNet18(dataset=dataset, split_layer=split_layer, device=device)
            
 # 
            x = torch.randn(input_shape).to(device)
            
 # 
            output = model.forward(x)
            
 # 
            info = model.get_split_info()
            
 print(f" {split_layer}: {output.shape}, "
                  f"Client params  {info['client_params']:,}, "
                  f"Server params  {info['server_params']:,}")
    
    print("\n" + "="*80)
 print("✅ ")
    print("="*80)

if __name__ == '__main__':
    test_split_resnet18()
