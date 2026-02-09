# LENS-SFL: Learning Contracts for Split Federated Learning under Incomplete Information

## Overview

This repository contains the source code for the LENS-SFL paper, which proposes a contract-theoretic framework for split federated learning (SFL) under incomplete information. The framework uses an Online Contract Design with Upper Confidence Bound (OCD-UCB) algorithm to optimize cut-layer selection and incentive mechanisms for heterogeneous clients.

## Project Structure

```
├── configs/                    # Global configuration
│   ├── __init__.py
│   └── config.py              # System parameters and ResNet layer configuration
│
├── core/                       # Core algorithm modules
│   ├── __init__.py
│   ├── agent.py               # Client agent (type, strategy)
│   ├── contract.py            # Contract design (optimal contract solving)
│   ├── ocd_ucb.py             # OCD-UCB online learning algorithm
│   ├── physics.py             # System physical model (cost, utility computation)
│   └── regret.py              # Regret analysis
│
├── TheoryValidation/           # Theory validation & ablation studies
│   ├── theory_validation.py   # Heatmap generation (Figures 3-5)
│   ├── regret_convergence.py  # Regret convergence experiment (Figure 6)
│   ├── ablation_studies.py    # Ablation studies (Figures 7-9)
│   └── ...
│
├── env/                        # Training environment
│   ├── models.py              # Split model definitions
│   ├── data_loader.py         # Data loader with Non-IID partitioning
│   ├── train_cifar10*.py      # CIFAR-10 training scripts
│   └── ...
│
├── real_world/                 # Real-world dataset experiments
│   ├── resnet18_split.py      # Split ResNet-18 model
│   ├── trainer.py             # Trainer for four comparison methods
│   ├── run_experiments.py     # Main experiment runner
│   └── ...
│
├── new_figure/                 # Figure generation scripts
│   ├── plot_architecture.py   # Architecture diagram
│   ├── plot_figures.py        # Heatmap figures
│   └── plot_accuracy.py       # Accuracy curves
│
├── pac_fig/                    # All paper figures (PDF + PNG)
├── results/                    # Experiment results
└── utils/                      # Utility functions
```

## Key Components

- **Contract Theory Module** (`core/contract.py`, `core/physics.py`): Optimal contract design and cost/utility models
- **Online Learning** (`core/ocd_ucb.py`, `core/regret.py`): OCD-UCB algorithm and regret analysis
- **Theory Validation** (`TheoryValidation/`): Heatmaps for optimal cut-layer, reward, and client utility distributions
- **Ablation Studies** (`TheoryValidation/ablation_studies.py`): Component-wise ablation experiments
- **Real-World Experiments** (`real_world/`): Comparison on MNIST, Fashion-MNIST, and CIFAR-10

## Quick Start

### Theory Validation

```bash
cd TheoryValidation
python theory_validation.py
python regret_convergence.py
python ablation_studies.py
```

### Real-World Experiments

```bash
cd real_world
python download_datasets.py
python run_experiments.py --dataset all --rounds 100
```

## Dependencies

- Python 3.8+
- numpy, matplotlib
- torch, torchvision (for real-world experiments)

## License

This project is for academic research purposes.
