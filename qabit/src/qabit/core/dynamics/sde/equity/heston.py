"""dynamics/sde/equity/heston.py — Heston stochastic-volatility equity factor."""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor
from dataclasses import dataclass

from qabit.core.dynamics.sde.base import Factor


@dataclass
class HestonParams:
    """Container passed to each Heston scheme step."""

    mu: float
    kappa: float
    theta: float
    xi: float
    rho: float


class HestonFactor(Factor):
    r"""Heston stochastic-volatility equity with a state-coupled drift.

    .. math::
        dS / S &= r_t \, dt + \sqrt{V}\, dW^S \\
        dV     &= \kappa (\theta - V)\, dt + \xi \sqrt{V}\, dW^V

    where the drift ``r_t`` is read from the factor named by ``rate`` (use
    ``rate=None`` for a constant ``mu``).  The leverage correlation
    ``corr(dW^S, dW^V) = rho`` is a **parameter of this factor** and is exposed
    to the system via :meth:`internal_corr`; the cross-*factor* correlation
    matrix passed to :class:`SDESystem` therefore needs only one entry per
    factor (this factor's asset driver), never the internal ``(S, V)`` block.

    Discretisation reuses the existing Heston schemes (Euler, Milstein).  The
    scheme is run **drift-free** (its ``mu`` set to 0) and the spot is then
    multiplied by ``exp(int r ds)`` over the step, integrated with the same
    trapezoidal rule the discount curve uses — keeping the discounted asset an
    exact per-step martingale, so ``E[S_T B(0,T)] = S_0`` for any correlation.

    .. note::
        ``QEScheme`` is **not** supported here: it assumes the asset normal is
        independent of the variance normal and injects ``rho`` analytically,
        double-counting the correlation already carried by the system's noise.
        Pass ``EulerScheme`` or ``HestonMilsteinScheme``.

    Outputs two columns ``[S, V]``; ``S`` is the factor's spot.
    """

    n_factors = 2  # asset + variance Brownian drivers
    n_outputs = 2  # [S, V]

    def __init__(
        self,
        kappa=2.0,
        theta=0.04,
        xi=0.30,
        rho=-0.70,
        s0=1.0,
        v0=0.04,
        rate: Optional[str] = "rate",
        mu: float = 0.0,
        excess_drift: float = 0.0,
        scheme=None,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.s0 = s0
        self.v0 = v0
        self.rate = rate
        self.mu = mu
        self.excess_drift = float(excess_drift)
        if scheme is None:
            from qabit.core.analytics.schemes.milstein import HestonMilsteinScheme

            scheme = HestonMilsteinScheme()
        self.scheme = scheme
        # Bind drift integrator once (no branch in the hot loop).
        self._drift = self._drift_flat_dt if rate is None else self._drift_state_dt
        self.p = HestonParams(
            mu=0.0, kappa=self.kappa, theta=self.theta, xi=self.xi, rho=0.0
        )

    def init(self, n_paths, device, dtype) -> Tensor:
        out = torch.empty(n_paths, 2, device=device, dtype=dtype)
        out[:, 0] = float(self.s0)
        out[:, 1] = float(self.v0)
        return out

    def internal_corr(self):
        r"""Leverage block ``corr(dW^S, dW^V) = rho`` for the system to honour."""
        import numpy as np

        return np.array([[1.0, self.rho], [self.rho, 1.0]])

    def _drift_flat_dt(self, prev, cur, dt) -> Union[Tensor, float]:
        return torch.as_tensor((self.mu + self.excess_drift) * dt, dtype=torch.float32)

    def _drift_state_dt(self, prev, cur, dt) -> Union[Tensor, float]:
        r_k = prev[self.rate]
        r_k1 = cur.get(self.rate, r_k)
        return (0.5 * (r_k + r_k1) + self.excess_drift) * dt

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        S, V = value[:, 0], value[:, 1]
        z1, z2 = z[:, 0], z[:, 1]
        # Drift-free step (mu=0); rho unused by Euler/Milstein — the system
        # noise already carries it.
        S0_next, V_next = self.scheme.step(S, V, dt, z1, z2, self.p)
        S_next = S0_next * torch.exp(self._drift(prev, cur, dt))
        return torch.stack([S_next, V_next], dim=1)

    def __repr__(self) -> str:
        tgt = "const-mu" if self.rate is None else f"rate='{self.rate}'"
        return (
            f"HestonFactor(kappa={self.kappa}, theta={self.theta}, xi={self.xi}, "
            f"rho={self.rho}, v0={self.v0}, drift={tgt}, scheme={type(self.scheme).__name__})"
        )


__all__ = ["HestonFactor"]
