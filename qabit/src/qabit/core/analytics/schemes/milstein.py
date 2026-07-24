"""
core/analytics/schemes/milstein.py

Milstein scheme for Heston and CIR processes."""

from __future__ import annotations

import math

import torch

from qabit.core.analytics.schemes.base import SchemeBase


class HestonMilsteinScheme(SchemeBase):
    """Milstein correction on the Heston variance SDE."""

    def step(self, S, V, dt, z1, z2, p):
        Vp = V.clamp(min=0.0)
        sv = Vp.sqrt()
        sqrt_dt = math.sqrt(dt)

        milstein_corr = 0.25 * p.xi**2 * dt * (z2.square() - 1.0)

        V_next = self.fix(
            V + p.kappa * (p.theta - Vp) * dt + p.xi * sqrt_dt * sv * z2 + milstein_corr
        )
        S_next = S * torch.exp((p.mu - 0.5 * Vp) * dt + sv * sqrt_dt * z1)
        return S_next, V_next


class CIRMilsteinScheme(SchemeBase):
    """Milstein for CIR: dX = κ(θ(t) - X) dt + σ√X dZ.

    The Milstein correction for √X diffusion is ¼σ²dt(Z² - 1).
    """

    def step(self, X, dt, z, p):
        Xp = X.clamp(min=0.0)
        milstein_corr = 0.25 * p.sigma**2 * dt * (z.square() - 1.0)
        X_next = self.fix(
            X
            + p.kappa * (p.theta_t - Xp) * dt
            + p.sigma * (Xp * dt).sqrt() * z
            + milstein_corr
        )
        return X_next


__all__ = [
    "HestonMilsteinScheme",
    "CIRMilsteinScheme",
]
