# Real-World Dataset Comparison Experiments

## Overview

This module compares four methods on three real-world datasets (MNIST, Fashion-MNIST, CIFAR-10):

1. **SplitFed**: No incentive mechanism; clients misreport and receive the worst data allocation (cut layer 1)
2. **Multi-Tenant SFL**: Conventional incentive mechanism with free-rider problem; medium data allocation (cut layer 4)
3. **LENS-UCB**: Our proposed method with intelligent allocation and progressive optimization (cut layers 2-7)
4. **Full-Info**: Full information scenario as the theoretical upper bound; optimal data allocation (cut layer 8)

## Core Design

### Data Quality Bound to Cut Layer

Deeper cut layer → Higher client computation → Better data quality

| Cut Layer | Data Quality | Data Ratio | Used By |
|-----------|-------------|------------|---------|
| 1 | Worst | 10% | SplitFed |
| 2-3 | Low | 25-40% | LENS-UCB (early) |
| 4 | Medium | 55% | Multi-Tenant |
| 5-7 | Good | 70-90% | LENS-UCB (mid-late) |
| 8 | Optimal | 100% | Full-Info |

## File Structure

```
real_world/
├── README.md                    # This file
├── __init__.py                  # Module initialization
├── download_datasets.py         # Dataset download script
├── resnet18_split.py           # Split ResNet-18 model
├── non_iid_config.py           # Non-IID experiment configuration
├── non_iid_experiment.py       # Non-IID experiment core logic
├── data_quality_manager.py     # Data quality manager
├── trainer.py                  # Trainer for all four methods
├── visualizer.py               # Result visualization
├── run_experiments.py          # Main experiment runner
├── run_non_iid_simple.py       # Simplified Non-IID runner
├── data/                       # Dataset directory
└── results/                    # Experiment results
```

## Quick Start

### 1. Download Datasets

```bash
python download_datasets.py
```

- MNIST and Fashion-MNIST are downloaded automatically from PyTorch official sources
- CIFAR-10 is auto-detected if already present locally
- Data is saved to `./data`

### 2. Run Experiments

```bash
# Single dataset
python run_experiments.py --dataset mnist --rounds 100
python run_experiments.py --dataset fmnist --rounds 100
python run_experiments.py --dataset cifar10 --rounds 100

# All datasets
python run_experiments.py --dataset all --rounds 100

# Custom parameters
python run_experiments.py \
    --dataset mnist \
    --rounds 100 \
    --clients 10 \
    --batch_size 64 \
    --lr 0.01 \
    --output_dir ./results
```

### 3. View Results

Results are saved to `./results`:

```
results/
├── test_mnist_accuracy.png              # MNIST accuracy comparison plot
├── test_mnist_results.txt               # MNIST result summary
├── test_mnist_SplitFed.txt             # SplitFed training log
├── test_mnist_Multi-Tenant_SFL.txt     # Multi-Tenant training log
├── test_mnist_LENS-UCB.txt              # LENS-UCB training log
├── test_mnist_Full-Info.txt            # Full-Info training log
└── ... (results for other datasets)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name (mnist/fmnist/cifar10/all) | mnist |
| `--rounds` | Training rounds | 100 |
| `--clients` | Number of clients | 10 |
| `--batch_size` | Batch size | 64 |
| `--lr` | Learning rate | 0.01 |
| `--output_dir` | Output directory | ./results |

## Experiment Design

### ResNet-18 Model

- **Client model**: Front portion; computation depth determined by cut layer
- **Server model**: Rear portion; completes remaining computation and classification
- **Cut points**: 8 selectable cut points (corresponding to 8 basic blocks of ResNet-18)

### Training Strategies

- **SplitFed**: Fixed cut layer 1 (shallowest), 10% data quality, no incentive
- **Multi-Tenant SFL**: Fixed cut layer 4 (medium), 55% data quality, conventional incentive
- **LENS-UCB**: Dynamic cut layers (2-3 early, 4-5 mid, 6-7 late), 25-90% data quality, intelligent allocation
- **Full-Info**: Fixed cut layer 8 (deepest), 100% data quality, theoretical upper bound

## Notes

- **GPU recommended**: Training on CPU is significantly slower
- **Memory**: At least 8GB RAM recommended
- **Training time**: MNIST/FMNIST ~10-15 min per method; CIFAR-10 ~20-30 min per method
- **Randomness**: Results may vary slightly due to random initialization and data sampling

## Troubleshooting

- **CUDA out of memory**: Reduce `--batch_size 32`
- **Dataset download failure**: Manually download datasets to `./data`
- **Training too slow**: Reduce `--rounds 50`
