"""
Measure actual computational and communication loads for ResNet-18 split points
"""
import torch
import torch.nn as nn
import time
import numpy as np
from models import SplitResNet18, get_activation_size


def count_parameters(model, layers):
    """Count parameters in specific layers"""
    total_params = 0
    for name, param in model.named_parameters():
        # Check if parameter belongs to specified layers
        for layer in layers:
            if name.startswith(layer):
                total_params += param.numel()
                break
    return total_params


def measure_computation_time(model, x, cut_layer_idx, num_runs=100):
    """Measure average computation time for client-side forward pass"""
    model.eval()
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model.forward_client(x, cut_layer_idx)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model.forward_client(x, cut_layer_idx)
            end = time.time()
            times.append(end - start)
    
    return np.mean(times), np.std(times)


def get_client_layers(cut_layer_idx):
    """Get list of layers executed on client side"""
    if cut_layer_idx == 1:
        return ['conv1', 'bn1']
    elif cut_layer_idx == 2:
        return ['conv1', 'bn1', 'layer1']
    elif cut_layer_idx == 3:
        return ['conv1', 'bn1', 'layer1', 'layer2']
    elif cut_layer_idx == 4:
        return ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
    elif cut_layer_idx == 5:
        return ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    else:
        raise ValueError(f"Invalid cut_layer_idx: {cut_layer_idx}")


def measure_loads():
    """Measure W (computation) and D (communication) for each split point"""
    print("="*80)
    print("Measuring ResNet-18 Split Point Loads for CIFAR-10")
    print("="*80)
    
    model = SplitResNet18(num_classes=10)
    model.eval()
    
    # Test input (batch_size=1 for accurate measurement)
    x = torch.randn(1, 3, 32, 32)
    
    print(f"\nInput: {x.shape}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\n" + "-"*80)
    
    results = {}
    
    for v in range(1, 6):
        print(f"\nSplit Point v={v}:")
        
        # Client layers
        client_layers = get_client_layers(v)
        client_params = count_parameters(model, client_layers)
        
        # Smashed data
        smashed_data = model.forward_client(x, v)
        smashed_size_mb = get_activation_size(smashed_data)
        
        # Computation time
        comp_time_mean, comp_time_std = measure_computation_time(model, x, v, num_runs=100)
        
        # Store results
        results[v] = {
            'client_layers': client_layers,
            'client_params': client_params,
            'smashed_shape': smashed_data.shape,
            'smashed_size_mb': smashed_size_mb,
            'comp_time_ms': comp_time_mean * 1000,
            'comp_time_std_ms': comp_time_std * 1000
        }
        
        print(f"  Client layers: {', '.join(client_layers)}")
        print(f"  Client parameters: {client_params:,}")
        print(f"  Smashed data shape: {smashed_data.shape}")
        print(f"  Smashed data size: {smashed_size_mb:.4f} MB")
        print(f"  Computation time: {comp_time_mean*1000:.2f} +/- {comp_time_std*1000:.2f} ms")
    
    print("\n" + "="*80)
    print("Normalized Load Profile (for config.py)")
    print("="*80)
    
    # Normalize W (computation) - relative to max
    max_params = max(r['client_params'] for r in results.values())
    max_time = max(r['comp_time_ms'] for r in results.values())
    
    # Normalize D (communication) - use actual MB
    max_size = max(r['smashed_size_mb'] for r in results.values())
    
    print("\nRESNET_PROFILE = {")
    for v in range(1, 6):
        r = results[v]
        
        # W: normalized by parameters (0-1 scale)
        W_norm = r['client_params'] / max_params
        
        # D: actual size in MB
        D_actual = r['smashed_size_mb']
        
        print(f"    {v}: {{'W': {W_norm:.2f}, 'D': {D_actual:.2f}}},  "
              f"# params={r['client_params']:,}, size={D_actual:.2f}MB")
    
    print("}")
    
    print("\n" + "="*80)
    print("Detailed Analysis")
    print("="*80)
    
    print("\n1. Computational Load (W):")
    print("   v | Parameters | Norm W | Time (ms)")
    print("   " + "-"*45)
    for v in range(1, 6):
        r = results[v]
        W_norm = r['client_params'] / max_params
        print(f"   {v} | {r['client_params']:>10,} | {W_norm:>6.2f} | {r['comp_time_ms']:>8.2f}")
    
    print("\n2. Communication Load (D):")
    print("   v | Shape              | Size (MB) | Ratio to v=1")
    print("   " + "-"*60)
    base_size = results[1]['smashed_size_mb']
    for v in range(1, 6):
        r = results[v]
        ratio = r['smashed_size_mb'] / base_size
        shape_str = str(tuple(r['smashed_shape']))
        print(f"   {v} | {shape_str:>18} | {r['smashed_size_mb']:>9.4f} | {ratio:>6.2f}x")
    
    print("\n3. Trade-off Analysis:")
    print("   v | W (comp) | D (comm) | W+D | Recommendation")
    print("   " + "-"*70)
    for v in range(1, 6):
        r = results[v]
        W_norm = r['client_params'] / max_params
        D_norm = r['smashed_size_mb'] / max_size
        total = W_norm + D_norm
        
        if v == 1:
            rec = "Low comp, high comm - weak devices"
        elif v == 2:
            rec = "Balanced"
        elif v == 3:
            rec = "Medium comp, medium comm"
        elif v == 4:
            rec = "High comp, low comm"
        else:
            rec = "Highest comp, lowest comm - strong devices"
        
        print(f"   {v} | {W_norm:>8.2f} | {D_norm:>8.2f} | {total:>5.2f} | {rec}")
    
    print("\n" + "="*80)
    
    return results


if __name__ == '__main__':
    results = measure_loads()
