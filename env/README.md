# Training Environment

## Overview

This module provides environment definitions, model architectures, and training scripts for the **real-world dataset experiments** in the LENS-SFL paper, supporting split federated learning comparison experiments on MNIST, Fashion-MNIST, and CIFAR-10.

## File Description

| File | Description |
|------|-------------|
| `__init__.py` | Module initialization |
| `cifar_env.py` | CIFAR-10 experiment environment configuration |
| `data_loader.py` | Data loader: supports MNIST, Fashion-MNIST, CIFAR-10 with Non-IID data partitioning |
| `measure_load.py` | Computational load measurement: simulates client and server computation overhead under different cut layers |
| `models.py` | Model definitions: includes Split ResNet-18 and other split model architectures |
| `train_cifar10.py` | CIFAR-10 standard training script |
| `train_cifar10_fast.py` | CIFAR-10 fast training script (optimized) |
| `train_cifar10_simple.py` | CIFAR-10 simplified training script |

## How to Run

```bash
# Standard training
python train_cifar10.py

# Fast training (recommended)
python train_cifar10_fast.py

# Simplified training
python train_cifar10_simple.py
```

## Dependencies

- torch, torchvision
- numpy, matplotlib
- `configs/` module from the project root
