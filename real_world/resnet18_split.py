"""
Split ResNet-18模型
支持在不同层进行切分，适配MNIST、Fashion-MNIST、CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet基础块"""
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
    """客户端模型（前半部分）"""
    
    def __init__(self, dataset='cifar10', split_layer=4):
        """
        Args:
            dataset: 数据集名称 ('mnist', 'fmnist', 'cifar10')
            split_layer: 切分层 (1-8)
                1: 最浅切分（客户端计算量最小，数据质量最差）
                8: 最深切分（客户端计算量最大，数据质量最好）
        """
        super(ClientModel, self).__init__()
        self.dataset = dataset
        self.split_layer = split_layer
        
        # 根据数据集确定输入通道数
        if dataset in ['mnist', 'fmnist']:
            in_channels = 1
            # MNIST/FMNIST需要调整初始卷积
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:  # cifar10
            in_channels = 3
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet-18的4个layer
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 层1-2
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 层3-4
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 层5-6
        self.layer4 = self._make_layer(256, 512, 2, stride=2) # 层7-8
        
    def _make_layer(self, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(in_planes, planes, stride))
            in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播，根据split_layer决定计算到哪一层"""
        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.split_layer >= 1:
            out = self.layer1[0](out)  # 层1
        if self.split_layer >= 2:
            out = self.layer1[1](out)  # 层2
        if self.split_layer >= 3:
            out = self.layer2[0](out)  # 层3
        if self.split_layer >= 4:
            out = self.layer2[1](out)  # 层4
        if self.split_layer >= 5:
            out = self.layer3[0](out)  # 层5
        if self.split_layer >= 6:
            out = self.layer3[1](out)  # 层6
        if self.split_layer >= 7:
            out = self.layer4[0](out)  # 层7
        if self.split_layer >= 8:
            out = self.layer4[1](out)  # 层8
            
        return out

class ServerModel(nn.Module):
    """服务器模型（后半部分）"""
    
    def __init__(self, dataset='cifar10', split_layer=4, num_classes=10):
        """
        Args:
            dataset: 数据集名称
            split_layer: 切分层 (1-8)
            num_classes: 分类数量
        """
        super(ServerModel, self).__init__()
        self.dataset = dataset
        self.split_layer = split_layer
        self.num_classes = num_classes
        
        # 根据切分层确定输入通道数
        if split_layer <= 2:
            in_channels = 64
        elif split_layer <= 4:
            in_channels = 128
        elif split_layer <= 6:
            in_channels = 256
        else:
            in_channels = 512
        
        # 构建剩余的层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # 全局平均池化和全连接层
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
        """前向传播，从split_layer之后开始计算"""
        out = x
        
        # 根据切分层继续计算
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
        
        # 全局平均池化和分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

class SplitResNet18:
    """Split ResNet-18完整模型"""
    
    def __init__(self, dataset='cifar10', split_layer=4, num_classes=10, device='cuda'):
        """
        Args:
            dataset: 数据集名称 ('mnist', 'fmnist', 'cifar10')
            split_layer: 切分层 (1-8)
            num_classes: 分类数量
            device: 设备
        """
        self.dataset = dataset
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.device = device
        
        # 创建客户端和服务器模型
        self.client_model = ClientModel(dataset, split_layer).to(device)
        self.server_model = ServerModel(dataset, split_layer, num_classes).to(device)
        
    def forward(self, x):
        """完整的前向传播"""
        # 客户端前向传播
        smashed_data = self.client_model(x)
        
        # 服务器前向传播
        output = self.server_model(smashed_data)
        
        return output
    
    def get_client_model(self):
        """获取客户端模型"""
        return self.client_model
    
    def get_server_model(self):
        """获取服务器模型"""
        return self.server_model
    
    def get_split_info(self):
        """获取切分信息"""
        return {
            'dataset': self.dataset,
            'split_layer': self.split_layer,
            'num_classes': self.num_classes,
            'client_params': sum(p.numel() for p in self.client_model.parameters()),
            'server_params': sum(p.numel() for p in self.server_model.parameters()),
        }

def test_split_resnet18():
    """测试Split ResNet-18模型"""
    print("="*80)
    print("测试Split ResNet-18模型")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}\n")
    
    # 测试不同数据集和切分层
    datasets = ['mnist', 'fmnist', 'cifar10']
    split_layers = [1, 4, 8]
    
    for dataset in datasets:
        print(f"\n数据集: {dataset.upper()}")
        print("-"*80)
        
        # 确定输入大小和类别数
        if dataset in ['mnist', 'fmnist']:
            input_shape = (2, 1, 28, 28)  # batch_size=2, channels=1, 28x28
        else:
            input_shape = (2, 3, 32, 32)  # batch_size=2, channels=3, 32x32
        
        for split_layer in split_layers:
            model = SplitResNet18(dataset=dataset, split_layer=split_layer, device=device)
            
            # 创建随机输入
            x = torch.randn(input_shape).to(device)
            
            # 前向传播
            output = model.forward(x)
            
            # 获取切分信息
            info = model.get_split_info()
            
            print(f"  切分层 {split_layer}: 输出形状 {output.shape}, "
                  f"客户端参数 {info['client_params']:,}, "
                  f"服务器参数 {info['server_params']:,}")
    
    print("\n" + "="*80)
    print("✅ 所有测试通过！")
    print("="*80)

if __name__ == '__main__':
    test_split_resnet18()
