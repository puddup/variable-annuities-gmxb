"""
Global configuration for qabit.

Contains only constants reused across multiple modules. No dynamic modification
functions are provided; users can directly assign attributes if needed.
"""

import torch

# Precision and device defaults
DEFAULT_DTYPE = torch.float32
DEFAULT_DEVICE = torch.device("cpu")

# Numerical stability thresholds (used in safe_log, d1/d2, CIR floor, etc.)
EPS = 1e-12
EPS_TIME = 1e-8
EPS_VAR = 1e-8

# Default parameters for Monte Carlo simulations
DEFAULT_DT = 1.0 / 52.0
DEFAULT_N_PATHS = 50_000


__all__ = [
    "DEFAULT_DTYPE",
    "DEFAULT_DEVICE",
    "EPS",
    "EPS_TIME",
    "EPS_VAR",
    "DEFAULT_DT",
    "DEFAULT_N_PATHS",
]
