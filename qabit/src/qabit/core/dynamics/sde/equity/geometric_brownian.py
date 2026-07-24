"""dynamics/sde/equity/geometric_brownian.py — GBM equity factor."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor

from qabit.core.dynamics.sde.base import Factor


class GeometricBrownianFactor(Factor):
    r"""Log-Euler GBM equity whose drift is read from another factor.

    ``dS / S = r_t dt + sigma dW`` with ``r_t`` taken from the factor named by
    ``rate``.  The step integrates the drift with the trapezoidal rule
    ``0.5 (r_k + r_{k+1})`` — matching the quadrature used by
    ``StochasticDiscountCurve`` so the discounted price is a discrete
    martingale.  Pass ``rate=None`` to fall back to a constant ``mu`` (plain
    GBM), handy for A/B testing against the stochastic-rate version.

    ``excess_drift`` adds a *constant* premium on top of whichever drift is
    selected, so the simulated (statistical / real-world) drift is
    ``r_t + excess_drift`` while the pricing/discount curve still uses the
    risk-neutral ``r_t``.  A non-zero ``excess_drift`` therefore makes the
    discounted price a *sub-/super-martingale* under P — i.e. it injects a
    deliberate **statistical-arbitrage** drift (an equity risk premium) that the
    near-martingale measure of :mod:`qabit.core.nn.near_martingale` removes.
    It works in every regime (flat or stochastic rate); leave it ``0`` for a
    drift-free, classic-arbitrage-free market.

    The drift function is bound **once** at construction (``self._drift``) to
    keep per-step branching out of the simulation loop.
    """

    n_factors = 1
    n_outputs = 1

    def __init__(
        self,
        sigma=0.20,
        s0=1.0,
        rate: Optional[str] = "rate",
        mu: float = 0.0,
        excess_drift: float = 0.0,
    ) -> None:
        self.sigma = sigma
        self.s0 = s0
        self.rate = rate
        self.mu = mu
        self.excess_drift = float(excess_drift)
        # Select the drift integrator once — no if-branch in the hot loop.
        self._drift = self._drift_flat_dt if rate is None else self._drift_state_dt

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), float(self.s0), device=device, dtype=dtype)

    # ── drift variants (bound at init) ──────────────────────────────────
    def _drift_flat_dt(self, prev, cur, dt) -> Union[Tensor, float]:
        return (self.mu + self.excess_drift) * dt

    def _drift_state_dt(self, prev, cur, dt) -> Union[Tensor, float]:
        r_k = prev[self.rate]
        r_k1 = cur.get(self.rate, r_k)  # end-of-step rate if already advanced
        return (0.5 * (r_k + r_k1) + self.excess_drift) * dt

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        drift = self._drift(prev, cur, dt) - 0.5 * self.sigma**2 * dt
        return value * torch.exp(drift + self.sigma * math.sqrt(dt) * z)

    def __repr__(self) -> str:
        tgt = "const-mu" if self.rate is None else f"rate='{self.rate}'"
        return f"GeometricBrownianFactor(sigma={self.sigma}, s0={self.s0}, drift={tgt})"


# Backward-compatible alias (the factor was originally named EquityFactor).
EquityFactor = GeometricBrownianFactor


__all__ = ["GeometricBrownianFactor", "EquityFactor"]
