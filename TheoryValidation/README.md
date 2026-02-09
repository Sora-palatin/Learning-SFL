# Theory Validation & Ablation Studies

## Overview

This module contains all code for the **theory validation** and **ablation studies** in the LENS-SFL paper, verifying the correctness of the contract-theoretic model and the effectiveness of each component.

## File Description

| File | Description |
|------|-------------|
| `theory_validation.py` | Main theory validation script: generates heatmaps (optimal cut-layer distribution, reward distribution, client utility) and regret convergence curves |
| `regret_convergence.py` | OCD-UCB regret convergence experiment: validates the convergence performance of the online learning algorithm |
| `ablation_studies.py` | Ablation studies: removes data subsidy, incentive mechanism, and online learning components respectively to observe system performance changes |
| `generate_ablation_comparison.py` | Ablation comparison figure generation script |
| `run_validation.bat` | Windows batch execution script |
| `ablation/` | Ablation study result figures |

## How to Run

```bash
# Run all theory validations
python theory_validation.py

# Run regret convergence experiment
python regret_convergence.py

# Run ablation studies
python ablation_studies.py

# Or run all at once via batch script
run_validation.bat
```

## Dependencies

- numpy, matplotlib
- `configs/` and `core/` modules from the project root
