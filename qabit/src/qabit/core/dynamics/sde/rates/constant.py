"""dynamics/sde/rates/constant.py — deterministic flat short rate."""

from __future__ import annotations

import torch
from torch import Tensor

from qabit.core.dynamics.sde.base import Factor


class ConstantRateFactor(Factor):
    r"""Deterministic flat short rate ``r(t) = r`` for all ``t``.

    Contributes **zero** Brownian drivers, so it adds nothing to the system's
    noise budget or correlation matrix.  Dropping it into a system in place of a
    :class:`VasicekRateFactor` turns the model back into a constant-rate world
    without changing anything else — useful as a control when measuring the
    impact of stochastic rates (e.g. on hedging error).

    Examples
    --------
    >>> SDESystem({"rate": ConstantRateFactor(0.05),
    ...            "equity": GeometricBrownianFactor(sigma=0.2, s0=100.0, rate="rate")})
    """

    n_factors = 0  # deterministic — no noise
    n_outputs = 1

    def __init__(self, rate: float = 0.03) -> None:
        self.rate = rate

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), float(self.rate), device=device, dtype=dtype)

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        return value  # constant — ignores noise (z is None)

    def __repr__(self) -> str:
        return f"ConstantRateFactor(rate={self.rate})"


__all__ = ["ConstantRateFactor"]
