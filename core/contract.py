"""
Contract logic: cost, utility (Eq.1, Eq.3), and optimal contract solver.
"""
from typing import List, Tuple

from configs.config import Config


class ClientType:
    """Device type (f, tau, D_k) per Eq.1."""

    __slots__ = ("f", "tau", "D_k")

    def __init__(self, f: float, tau: float, D_k: float):
        self.f = f
        self.tau = tau
        self.D_k = D_k

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "ClientType":
        return cls(t[0], t[1], t[2])


def get_devices_from_config() -> List[ClientType]:
    """Build list of ClientType from Config.CLIENT_TYPES (weakest to strongest)."""
    return [ClientType.from_tuple(t) for t in Config.CLIENT_TYPES]


def calculate_cost(device: ClientType, v: int) -> float:
    """Cost for device at cut v (1-indexed). Uses W_LIST[v-1]/f + D_LIST[v-1]*tau."""
    W = Config.W_LIST[v - 1]
    D = Config.D_LIST[v - 1]
    return W / device.f + D * device.tau


def calculate_utility(device: ClientType, v: int) -> float:
    """Utility for device at cut v (Eq.3): alpha*W + beta*(D_MAX-D) + mu*D."""
    W = Config.W_LIST[v - 1]
    D = Config.D_LIST[v - 1]
    return (
        Config.ALPHA * W
        + Config.BETA * (Config.D_MAX - D)
        + Config.MU * D
    )


class ContractSolver:
    """Solves optimal contract (v_star, R_star) given devices and type distribution."""

    def solve_optimal_contract(
        self, devices: List[ClientType], probabilities: List[float]
    ) -> Tuple[List[int], List[float]]:
        """
        devices: sorted weakest (index 0) to strongest (index K-1).
        probabilities: type distribution p_k.
        Returns (v_star, R_star).
        """
        K = len(devices)
        L = Config.TOTAL_LAYERS
        v_star = [0] * K

        for k in range(K):
            best_v = 1
            max_obj = -float("inf")
            for v in range(1, L + 1):
                # D_k constraint: only consider v with D_LIST[v-1] <= device.D_k
                if Config.D_LIST[v - 1] > devices[k].D_k:
                    continue
                p_k = probabilities[k]
                if p_k == 0:
                    obj = -float("inf")
                else:
                    phys_cost = calculate_cost(devices[k], v)
                    rent_sum = 0.0
                    for m in range(k + 1, K):
                        p_m = probabilities[m]
                        rent_sum += (p_m / p_k) * (
                            calculate_cost(devices[k], v)
                            - calculate_cost(devices[m], v)
                        )
                    virtual_cost = phys_cost + rent_sum
                    obj = calculate_utility(devices[k], v) - virtual_cost

                if obj > max_obj:
                    max_obj = obj
                    best_v = v
            v_star[k] = best_v

        R_star = [0.0] * K
        for k in range(K):
            r = calculate_cost(devices[k], v_star[k])
            for j in range(k):
                r += (
                    calculate_cost(devices[j], v_star[j])
                    - calculate_cost(devices[j + 1], v_star[j])
                )
            R_star[k] = r

        return v_star, R_star
