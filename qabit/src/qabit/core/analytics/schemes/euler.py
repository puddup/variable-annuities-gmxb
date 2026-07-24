"""
core/analytics/schemes/euler.py
Euler-Maruyama scheme for Heston and CIR processes."""

from __future__ import annotations

import math

import torch

from qabit.core.analytics.schemes.base import SchemeBase


class HestonEulerScheme(SchemeBase):
    """Standard Euler-Maruyama discretisation for Heston."""

    def step(self, S, V, dt, z1, z2, p):
        Vp = V.clamp(min=0.0)
        sv = Vp.sqrt()
        sqrt_dt = math.sqrt(dt)

        V_next = self.fix(V + p.kappa * (p.theta - Vp) * dt + p.xi * sqrt_dt * sv * z2)
        S_next = S * torch.exp((p.mu - 0.5 * Vp) * dt + sv * sqrt_dt * z1)
        return S_next, V_next


class CIREulerScheme(SchemeBase):
    """Euler-Maruyama for the CIR SDE: dX = κ(θ(t) - X) dt + σ√X dZ."""

    def step(self, X, dt, z, p):
        Xp = X.clamp(min=0.0)
        X_next = self.fix(
            X + p.kappa * (p.theta_t - Xp) * dt + p.sigma * (Xp * dt).sqrt() * z
        )
        return X_next


__all__ = [
    "HestonEulerScheme",
    "CIREulerScheme",
]
