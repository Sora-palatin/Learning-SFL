# Theory Validation & Ablation Studies

## Overview

This module contains all code for the **theory validation** and **ablation studies** in the LENS-SFL paper, verifying the correctness of the contract-theoretic model and the effectiveness of each component.

## File Description

| File | Description |
|------|-------------|
| `theory_validation.py` | Main theory validation script: generates heatmaps (optimal cut-layer distribution, reward distribution, client utility) and regret convergence curves |
| `regret_convergence.py` | LENS-UCB regret convergence experiment: validates the convergence performance of the online learning algorithm |
| `ablation_studies.py` | Ablation studies: removes data subsidy, incentive mechanism, and online learning components respectively to observe system performance changes |
| `generate_ablation_comparison.py` | Ablation comparison figure generation script |
| `ablation/` | Ablation study output directory (auto-generated) |

## How to Run

```bash
# Run all theory validations
python theory_validation.py

# Run regret convergence experiment
python regret_convergence.py

# Run ablation studies
python ablation_studies.py

# Generate ablation comparison plots
python generate_ablation_comparison.py
```

## Dependencies

- numpy, matplotlib
- `configs/` and `core/` modules from the project root
