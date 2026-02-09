"""
Config for SFL Contract under Information Asymmetry.
Paper: Contract Design for SFL under Information Asymmetry
"""
from typing import List, Tuple


class Config:
    """Central parameter class for simulation and real-world modes."""

    # --- Mode switch ---
    # 'SIMULATION': numerical simulation (math model, no PyTorch training)
    # 'REAL_WORLD': CIFAR-10 real training (env/)
    MODE: str = "SIMULATION"

    # --- Utility coefficients (Eq.3) ---
    # 归一化权重：ALPHA + BETA = 1
    ALPHA: float = 0.6     # computation weight
    BETA: float = 0.4      # communication weight  
    MU: float = 0.1        # data subsidy

    # --- SFL settings ---
    TOTAL_LAYERS: int = 5   # cut points, v in {1, ..., 5}
    D_MAX: float = 0.25     # max communication volume (MB, from v=1 or v=2)

    # --- UCB (Lemma 5.1) ---
    EXPLORATION_C: float = 0.5  # Reduced for faster convergence

    # --- Helpers ---
    @classmethod
    def is_simulation(cls) -> bool:
        return cls.MODE == "SIMULATION"

    @classmethod
    def is_real_world(cls) -> bool:
        return cls.MODE == "REAL_WORLD"


# ResNet-18 load profile for CIFAR-10 (REAL MEASURED DATA)
# W: Normalized computation load (0-1, based on client-side parameters)
# D: Communication load in MB (smashed data size)
# Measured from actual SplitResNet18 model on CIFAR-10 (32x32 images)
# Key insight: W increases exponentially, D decreases exponentially
RESNET_PROFILE = {
    1: {'W': 0.07, 'D': 1.33},  # Conv1: 轻计算，重传输 (Data Expansion)
    2: {'W': 0.32, 'D': 1.33},  # Layer1: 中计算，重传输 (Worst case for weak clients)
    3: {'W': 0.54, 'D': 0.67},  # Layer2: 计算加重，数据减半 (Sweet spot?)
    4: {'W': 0.76, 'D': 0.33},  # Layer3: 重计算，轻传输
    5: {'W': 1.00, 'D': 0.17},  # Layer4: 全计算，微传输
}

# 20 Heterogeneous Client Types
# f: GHz (Compute Power)
# tau: s/unit (Transmission Latency, lower is better)
# data_size: Local samples
CLIENT_TYPES = [
    {'id': 1,  'f': 0.10, 'tau': 0.5403, 'data_size': 387},  # Extremely Weak
    {'id': 2,  'f': 0.15, 'tau': 0.4593, 'data_size': 689},
    {'id': 3,  'f': 0.26, 'tau': 0.4436, 'data_size': 965},
    {'id': 4,  'f': 0.42, 'tau': 0.3759, 'data_size': 1312},
    {'id': 5,  'f': 0.70, 'tau': 0.3765, 'data_size': 1546},
    {'id': 6,  'f': 0.87, 'tau': 0.3334, 'data_size': 1846},
    {'id': 7,  'f': 1.35, 'tau': 0.3111, 'data_size': 2109},
    {'id': 8,  'f': 1.63, 'tau': 0.2453, 'data_size': 2362},
    {'id': 9,  'f': 2.02, 'tau': 0.2284, 'data_size': 2855},
    {'id': 10, 'f': 2.66, 'tau': 0.2016, 'data_size': 2991},  # Median Capability
    {'id': 11, 'f': 3.32, 'tau': 0.1878, 'data_size': 3374},
    {'id': 12, 'f': 3.82, 'tau': 0.1603, 'data_size': 3510},
    {'id': 13, 'f': 4.26, 'tau': 0.1329, 'data_size': 3826},
    {'id': 14, 'f': 4.83, 'tau': 0.1270, 'data_size': 4194},
    {'id': 15, 'f': 5.19, 'tau': 0.0948, 'data_size': 4699},
    {'id': 16, 'f': 5.93, 'tau': 0.0835, 'data_size': 4972},
    {'id': 17, 'f': 6.79, 'tau': 0.0574, 'data_size': 4905},
    {'id': 18, 'f': 8.05, 'tau': 0.0430, 'data_size': 5568},
    {'id': 19, 'f': 8.26, 'tau': 0.0228, 'data_size': 5513},
    {'id': 20, 'f': 8.96, 'tau': 0.0020, 'data_size': 5747},  # Server-Grade
]

# Real distribution (Normal distribution for 20 clients)
REAL_DISTRIBUTION = [
    0.0014, 0.0034, 0.0076, 0.0153, 0.0279, 0.0460, 0.0685, 0.0924, 0.1128, 0.1246,
    0.1246, 0.1128, 0.0924, 0.0685, 0.0460, 0.0279, 0.0153, 0.0076, 0.0034, 0.0014
]
