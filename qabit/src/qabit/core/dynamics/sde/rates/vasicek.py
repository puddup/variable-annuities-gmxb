"""dynamics/sde/rates/vasicek.py — Vasicek (Gaussian) short rate."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from qabit.core.dynamics.sde.base import Factor


class VasicekRateFactor(Factor):
    r"""Vasicek short rate ``dr = kappa (theta - r) dt + sigma dW``.

    Advanced with the *exact* Ornstein-Uhlenbeck transition over the step, so
    accuracy does not depend on ``dt``.
    """

    n_factors = 1
    n_outputs = 1

    def __init__(self, kappa=0.5, theta=0.03, sigma=0.01, r0=0.03) -> None:
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.r0 = r0

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), float(self.r0), device=device, dtype=dtype)

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        e = math.exp(-self.kappa * dt)
        drift = self.theta * (1.0 - e)
        std = self.sigma * math.sqrt((1.0 - e * e) / (2.0 * self.kappa))
        return e * value + drift + std * z

    def __repr__(self) -> str:
        return (
            f"VasicekRateFactor(kappa={self.kappa}, theta={self.theta}, "
            f"sigma={self.sigma}, r0={self.r0})"
        )


__all__ = ["VasicekRateFactor"]
