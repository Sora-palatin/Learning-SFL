"""
Dynamic Split ResNet-18 for CIFAR-10
Supports 5 split points (v=1,2,3,4,5) with unified architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block
    Used in ResNet-18 and ResNet-34
    """
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


class SplitResNet18(nn.Module):
    """
    ResNet-18 with dynamic split support for Split Federated Learning
    
    Architecture (adapted for CIFAR-10):
    - conv1: 3x3, stride=1 (no maxpool for small images)
    - layer1: 2 BasicBlocks, 64 channels
    - layer2: 2 BasicBlocks, 128 channels, stride=2
    - layer3: 2 BasicBlocks, 256 channels, stride=2
    - layer4: 2 BasicBlocks, 512 channels, stride=2
    - avgpool + fc
    
    Split points:
    - v=1: after conv1+bn+relu
    - v=2: after layer1
    - v=3: after layer2
    - v=4: after layer3
    - v=5: after layer4
    """
    
    def __init__(self, num_classes=10):
        super(SplitResNet18, self).__init__()
        self.in_planes = 64
        
        # Initial convolution (adapted for CIFAR-10)
        # No maxpool since CIFAR-10 images are only 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """Create a ResNet layer with multiple blocks"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward_client(self, x, cut_layer_idx):
        """
        Client-side forward pass
        
        Args:
            x: input tensor (batch_size, 3, 32, 32)
            cut_layer_idx: split point (1-5)
        
        Returns:
            smashed_data: intermediate features to send to server
        """
        # v=1: conv1 + bn + relu
        out = F.relu(self.bn1(self.conv1(x)))
        if cut_layer_idx == 1:
            return out
        
        # v=2: layer1
        out = self.layer1(out)
        if cut_layer_idx == 2:
            return out
        
        # v=3: layer2
        out = self.layer2(out)
        if cut_layer_idx == 3:
            return out
        
        # v=4: layer3
        out = self.layer3(out)
        if cut_layer_idx == 4:
            return out
        
        # v=5: layer4
        out = self.layer4(out)
        if cut_layer_idx == 5:
            return out
        
        raise ValueError(f"Invalid cut_layer_idx: {cut_layer_idx}. Must be 1-5.")
    
    def forward_server(self, x, cut_layer_idx):
        """
        Server-side forward pass
        
        Args:
            x: smashed_data from client
            cut_layer_idx: split point (1-5)
        
        Returns:
            output: final classification logits (batch_size, num_classes)
        """
        out = x
        
        # Continue from the layer after the cut point
        if cut_layer_idx == 1:
            # Input is from conv1, continue with layer1
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif cut_layer_idx == 2:
            # Input is from layer1, continue with layer2
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        elif cut_layer_idx == 3:
            # Input is from layer2, continue with layer3
            out = self.layer3(out)
            out = self.layer4(out)
        elif cut_layer_idx == 4:
            # Input is from layer3, continue with layer4
            out = self.layer4(out)
        elif cut_layer_idx == 5:
            # Input is from layer4, only avgpool + fc left
            pass
        else:
            raise ValueError(f"Invalid cut_layer_idx: {cut_layer_idx}. Must be 1-5.")
        
        # Final layers (always on server)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
    def forward(self, x):
        """
        Standard forward pass (for testing/validation)
        
        Args:
            x: input tensor (batch_size, 3, 32, 32)
        
        Returns:
            output: classification logits (batch_size, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_size(model):
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        size_mb: model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_bytes = param_size + buffer_size
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def get_activation_size(tensor):
    """
    Calculate activation tensor size in MB
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        size_mb: tensor size in megabytes
    """
    size_bytes = tensor.nelement() * tensor.element_size()
    size_mb = size_bytes / (1024 ** 2)
    return size_mb


def test_split_model():
    """Test the split model with different cut points"""
    print("="*80)
    print("Testing SplitResNet18 Model")
    print("="*80)
    
    model = SplitResNet18(num_classes=10)
    model.eval()
    
    # Test input (CIFAR-10 size)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Total model size: {get_model_size(model):.2f} MB")
    print("\n" + "-"*80)
    
    # Test each split point
    for v in range(1, 6):
        print(f"\nSplit Point v={v}:")
        
        # Client forward
        smashed_data = model.forward_client(x, v)
        print(f"  Client output shape: {smashed_data.shape}")
        print(f"  Smashed data size: {get_activation_size(smashed_data):.4f} MB")
        
        # Server forward
        output = model.forward_server(smashed_data, v)
        print(f"  Server output shape: {output.shape}")
        
        # Verify correctness
        full_output = model.forward(x)
        diff = torch.abs(output - full_output).max().item()
        print(f"  Verification (max diff): {diff:.6f} {'✓' if diff < 1e-5 else '✗'}")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == '__main__':
    test_split_model()
