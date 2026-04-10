# LENS-SFL

**Learning Contracts for Split Federated Learning under Incomplete Information**

LENS-SFL is a contract-theoretic framework for incentive-compatible split federated learning (SFL) when the server has no prior knowledge of client type distributions. The core contribution is the **LENS-UCB** algorithm (*Learning with Exploration and iNcentive for Split-learning via Upper Confidence Bound*), which jointly designs personalised incentive contracts and adapts the cut-layer schedule online, converging to the full-information optimum with sublinear regret.

---

## Table of Contents

- [Background](#background)
- [Algorithm Overview](#algorithm-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Reproducing Theory Results](#reproducing-theory-results)
- [Reproducing Real-World Experiments](#reproducing-real-world-experiments)
- [Key Source Files](#key-source-files)
- [Comparison Methods](#comparison-methods)
- [License](#license)

---

## Background

In split federated learning, a deep neural network is partitioned at a *cut layer*: the client executes layers 1 through *v* locally, and the server executes the remaining layers. Deeper cuts impose higher computation and communication costs on clients but yield richer feature representations and better model quality.

Heterogeneous clients have private *types* (computation capacity *f*, transmission delay *τ*, data volume *D*) that determine their cost at each cut layer. Without an incentive mechanism, rational clients misreport their types and contribute the minimum quality (shallowest cut). LENS-SFL addresses this by:

1. **Contract Theory** — designing menus of (cut layer, reward) pairs that are *individually rational* (IR) and *incentive compatible* (IC), so every client truthfully reveals its type and selects its optimal contract item.
2. **Online Learning (LENS-UCB)** — estimating the unknown type distribution with UCB-style confidence sets and iteratively refining the contract menu, achieving O(√T log T) cumulative regret.

---

## Algorithm Overview

```
Input : client type space {θ_k}, horizon T, exploration coefficient C
Output: per-round contract menu (v*, R*), cumulative regret R(T)

For t = 1, 2, ..., T:
  1. Build optimistic distribution p̂_t via UCB confidence radii
  2. Solve optimal contract (v*, R*) under p̂_t  [Eq. 22–23]
  3. Observe arriving client type k_t ~ p (true distribution)
  4. Update empirical count N_k[k_t] += 1
  5. Compute instant regret r_t = OPT(p) − LENS(p̂_t, k_t)
  6. Accumulate: R(T) += r_t
```

Theoretical guarantee: **R(T) = O(K · √(T log T))** under the IC-contract formulation.

---

## Project Structure

```
LENS-SFL/
│
├── configs/
│   └── config.py               # System hyperparameters (α, β, μ, layer W/D tables)
│
├── core/
│   ├── agent.py                # Client agent abstraction (type, UCB confidence radius)
│   ├── contract.py             # ContractSolver: optimal (v*, R*) via Eq. 22–23
│   ├── lens_ucb.py             # LENS_UCB_Learner: online contract design loop
│   ├── physics.py              # SystemPhysics: cost C(θ,v) and utility U(θ,v) models
│   └── regret.py               # Instant and cumulative regret computation
│
├── TheoryValidation/
│   ├── theory_validation.py    # Heatmaps: optimal cut layer & reward vs. (f, τ)
│   ├── regret_convergence.py   # Regret convergence curves (LENS-UCB vs. baselines)
│   ├── ablation_studies.py     # Ablation: UCB exploration, IC constraint, cut-layer range
│   └── generate_ablation_comparison.py
│
├── real_world/
│   ├── resnet18_split.py       # Split ResNet-18 (configurable cut layer 1–8)
│   ├── data_quality_manager.py # Per-client data quality as a function of cut layer
│   ├── trainer.py              # Trainers: SplitFed, MultiTenant, LENS-UCB, FullInfo
│   ├── run_experiments.py      # CLI experiment runner (argparse)
│   ├── visualizer.py           # Accuracy curve plots and result tables
│   └── download_datasets.py    # Auto-download MNIST / F-MNIST / CIFAR-10
│
├── new_figure/
│   ├── plot_architecture.py    # System architecture diagram
│   ├── plot_figures.py         # Contract theory heatmap figures
│   └── plot_accuracy.py        # Accuracy comparison curves
│
├── pac_fig/                    # Pre-rendered paper figures (PDF + PNG)
├── results/                    # Saved experiment logs and accuracy tables
└── utils/                      # Shared utility functions
```

---

## Installation

### Requirements

- Python ≥ 3.8
- NumPy, Matplotlib (theory validation and figure generation)
- PyTorch ≥ 1.10, torchvision (real-world experiments)

### Setup

```bash
git clone https://github.com/<your-org>/LENS-SFL.git
cd LENS-SFL
pip install numpy matplotlib torch torchvision
```

No additional package installation is required for the theory validation experiments.

---

## Reproducing Theory Results

All scripts below write figures to `TheoryValidation/figures/` (created automatically).

```bash
# Figure 3–5: Optimal cut-layer and reward heatmaps; client utility curves
python TheoryValidation/theory_validation.py

# Figure 6: Regret convergence of LENS-UCB vs. uniform and greedy baselines
python TheoryValidation/regret_convergence.py

# Figures 7–9: Ablation studies (UCB exploration / IC constraint / cut-layer range)
python TheoryValidation/ablation_studies.py

# Optional: side-by-side ablation comparison figures
python TheoryValidation/generate_ablation_comparison.py
```

---

## Reproducing Real-World Experiments

### Step 1 — Download datasets

```bash
python real_world/download_datasets.py
```

This downloads MNIST, Fashion-MNIST, and CIFAR-10 into `real_world/data/`.

### Step 2 — Run comparison experiments

```bash
# Single dataset (e.g., CIFAR-10), 100 rounds
python real_world/run_experiments.py --dataset cifar10 --rounds 100

# All three datasets sequentially
python real_world/run_experiments.py --dataset all --rounds 100

# Fast smoke-test (fewer clients and batches)
python real_world/run_experiments.py --dataset mnist --rounds 20 --fast
```

### Key CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `mnist` | `mnist` / `fmnist` / `cifar10` / `all` |
| `--rounds` | `100` | Number of communication rounds |
| `--clients` | `10` | Total number of simulated clients |
| `--clients_per_round` | `3` | Clients sampled per round |
| `--max_batches` | `5` | Max mini-batches per client per round |
| `--lr` | `0.01` | SGD learning rate |
| `--batch_size` | `64` | Mini-batch size |
| `--output_dir` | `./results` | Directory for logs and figures |
| `--fast` | flag | Reduced clients/batches for quick testing |

Results (accuracy curves + summary tables) are saved to `--output_dir`.

---

## Key Source Files

### `core/contract.py`
Implements `ContractSolver.solve_optimal_contract()` using the closed-form solution derived in the paper (Eq. 22–23). Given a type distribution and sorted client list, it returns the IC-optimal menu `(v_star, R_star)`.

### `core/lens_ucb.py`
`LENS_UCB_Learner` runs the online learning loop. Each call to `.step()` draws a client type from the true distribution, updates the empirical count, solves the current optimistic contract, and records instant regret.

### `core/physics.py`
`SystemPhysics` wraps `ContractSolver` with the physical cost/utility model. Costs follow `C(θ, v) = W[v]/f + D[v]·τ`; utilities follow `U(θ, v) = α·W[v] + β·(D_max − D[v]) + μ·D[v]`.

### `real_world/trainer.py`
Four trainer classes share a common `BaseTrainer` interface:

| Class | Contract | Cut Layer Schedule |
|---|---|---|
| `SplitFedTrainer` | None | Fixed at 1 (minimum) |
| `MultiTenantTrainer` | Uniform flat reward | Fixed at 3 (below-average) |
| `LENSUCBTrainer` | IC-personalised | Progressive: 4 → 6 → 8 |
| `FullInfoTrainer` | Oracle (known types) | Fixed at 8 (maximum) |

### `real_world/data_quality_manager.py`
Maps cut-layer depth to data quality: deeper layers yield larger data fractions, full class balance, and no label noise. Layers 1–2 inject label noise (30% and 15% respectively) to model strategic untruthful reporting under absent IC incentives.

---

## Comparison Methods

| Method | Incentive | Cut Layer | Data Quality |
|---|---|---|---|
| **SplitFed** | None | 1 (fixed) | 5%, 4 classes, 30% noise |
| **Multi-Tenant SFL** | Uniform | 3 (fixed) | 35%, 8 classes |
| **LENS-UCB** (ours) | IC-personalised | 4→6→8 (adaptive) | 50%→80%→100% |
| **Full-Info** (oracle) | Optimal known-type | 8 (fixed) | 100%, all classes |

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You are free to share and adapt the material for non-commercial purposes, provided appropriate credit is given. Commercial use is not permitted.

See [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/) for the full license text.
