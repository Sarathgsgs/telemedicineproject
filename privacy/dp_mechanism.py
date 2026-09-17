"""
privacy/dp_mechanism.py
-----------------------
Client-side Differential Privacy (DP-SGD) mechanism for FedLiverNet.
Implements L2 norm gradient/update clipping, calibrated Gaussian noise addition
(base paper Eq. 7), and analytical (epsilon, delta) privacy accounting.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


def clip_parameter_updates(
    parameter_updates: List[np.ndarray],
    clip_norm: float = 1.0
) -> Tuple[List[np.ndarray], float]:
    """
    Clips the global L2 norm of the client's parameter updates to threshold C.
    gamma = min(1.0, C / ||Delta w||_2)
    """
    total_norm_sq = sum(np.sum(np.square(p)) for p in parameter_updates)
    global_norm = float(np.sqrt(total_norm_sq))

    if global_norm > clip_norm and global_norm > 0:
        scaling_factor = clip_norm / global_norm
        clipped_updates = [p * scaling_factor for p in parameter_updates]
    else:
        clipped_updates = [p.copy() for p in parameter_updates]

    return clipped_updates, global_norm


def compute_gaussian_sigma(
    epsilon: float,
    delta: float = 1e-5,
    clip_norm: float = 1.0
) -> float:
    """
    Computes calibrated standard deviation for the Gaussian mechanism:
    sigma = (C * sqrt(2 * ln(1.25 / delta))) / epsilon
    """
    if epsilon == float("inf") or epsilon <= 0:
        return 0.0
    return float((clip_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon)


def apply_differential_privacy(
    parameter_updates: List[np.ndarray],
    epsilon: float,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Applies per-client Differential Privacy before transmitting updates to the central server.
    1. Clips L2 norm to threshold C.
    2. Adds calibrated zero-mean Gaussian noise N(0, sigma^2).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Clip norm
    clipped_updates, original_norm = clip_parameter_updates(parameter_updates, clip_norm)

    # 2. Add calibrated Gaussian noise
    if epsilon == float("inf") or epsilon is None:
        sigma = 0.0
        noisy_updates = clipped_updates
    else:
        sigma = compute_gaussian_sigma(epsilon, delta, clip_norm)
        noisy_updates = []
        for p in clipped_updates:
            noise = np.random.normal(0.0, sigma, size=p.shape).astype(p.dtype)
            noisy_updates.append(p + noise)

    dp_metadata = {
        "epsilon": epsilon if epsilon != float("inf") else "infinity",
        "delta": delta,
        "clip_norm": clip_norm,
        "original_norm": round(original_norm, 4),
        "sigma": round(sigma, 6)
    }

    return noisy_updates, dp_metadata


class PrivacyAccountant:
    """
    Analytical Privacy Accountant tracking cumulative (epsilon, delta) budget
    across successive communication rounds.
    """
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self.round_sigmas = []

    def step(self, sigma: float):
        self.round_sigmas.append(sigma)

    def get_cumulative_privacy(self, clip_norm: float = 1.0) -> Dict[str, Any]:
        if not self.round_sigmas or any(s == 0.0 for s in self.round_sigmas):
            return {"cumulative_epsilon": "infinity", "target_delta": self.target_delta, "rounds": len(self.round_sigmas)}

        # Advanced composition theorem: eps_total = sum(eps_i) or analytical moments bound
        total_variance = sum(s ** 2 for s in self.round_sigmas)
        effective_sigma = math.sqrt(total_variance) / len(self.round_sigmas)
        cum_eps = (clip_norm * math.sqrt(2.0 * math.log(1.25 / self.target_delta) * len(self.round_sigmas))) / effective_sigma

        return {
            "cumulative_epsilon": round(cum_eps, 3),
            "target_delta": self.target_delta,
            "rounds": len(self.round_sigmas)
        }
