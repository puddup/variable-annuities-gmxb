"""dynamics/sde/rates/cir.py — CIR (square-root) short rate."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from qabit.core.dynamics.sde.base import Factor


class CIRRateFactor(Factor):
    r"""CIR short rate ``dr = kappa (theta - r) dt + sigma sqrt(r) dW``.

    Full-truncation Euler with a non-negativity floor.
    """

    n_factors = 1
    n_outputs = 1

    def __init__(self, kappa=0.5, theta=0.03, sigma=0.05, r0=0.03, floor=1e-10) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0
        self.floor = floor

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), float(self.r0), device=device, dtype=dtype)

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        r_pos = value.clamp(min=0.0)
        nxt = (
            value
            + self.kappa * (self.theta - r_pos) * dt
            + self.sigma * r_pos.sqrt() * math.sqrt(dt) * z
        )
        return nxt.clamp(min=self.floor)

    def __repr__(self) -> str:
        return (
            f"CIRRateFactor(kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, r0={self.r0})"
        )


__all__ = ["CIRRateFactor"]
