"""core/products/derivatives/va/guarantee.py — composite guarantee floors.

New structural design (replaces the string-keyed ``GuaranteeFloor`` and the
``GuaranteeBase → GMDB/GMLB/GMAB/GMWB`` rider hierarchy).

A *guarantee floor* is a composite of elementary *schemes*; its value at an
anniversary is the element-wise maximum over its schemes, evaluated **before**
the living benefit is paid (the SU-W convention). See the framework note §3.

    G_{n^-} = max_schemes { scheme.update(G_{(n-1)^+}, A_{n^-}, n, dt) }

A step-up is exactly ``Ratchet`` whose date set is *all* anniversaries, so no
separate ``StepUp`` class is needed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor


# ── elementary schemes ────────────────────────────────────────────────────────


class GuaranteeScheme(ABC):
    """Elementary guarantee-floor update rule G_{(n-1)^+} → G_{n^-}."""

    @abstractmethod
    def update(self, G_prev: Tensor, A_now: Tensor, t: float, dt: float) -> Tensor:
        """Advance the floor one anniversary step.  Returns Tensor[N]."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class Fixed(GuaranteeScheme):
    """Constant floor (optionally scaled each step by ``multiplier``)."""

    def __init__(self, multiplier: float = 1.0) -> None:
        self.multiplier = multiplier

    def update(self, G_prev, A_now, t, dt):
        if self.multiplier == 1.0:
            return G_prev
        return G_prev * self.multiplier

    def __repr__(self) -> str:
        if self.multiplier == 1.0:
            return "<Fixed>"
        return f"<Fixed(×{self.multiplier})>"


class RollUp(GuaranteeScheme):
    """Roll-up at instantaneous rate ``delta``:  G_prev · e^{δ·dt}."""

    def __init__(self, delta: float) -> None:
        self.delta = delta

    def update(self, G_prev, A_now, t, dt):
        return G_prev * math.exp(self.delta * dt)

    def __repr__(self) -> str:
        return f"<RollUp(δ={self.delta:.2%})>"


class Ratchet(GuaranteeScheme):
    """Lock-in: max(G_prev, A_now) on ratchet dates, else unchanged.

    ``dates=None`` ratchets on **every** anniversary — i.e. a step-up.
    """

    def __init__(self, dates: Optional[List[float]] = None, tol: float = 1e-6) -> None:
        self.dates = sorted(dates) if dates is not None else None
        self.tol = tol

    def _is_ratchet_date(self, t: float) -> bool:
        if self.dates is None:
            return True
        return any(abs(t - d) < self.tol for d in self.dates)

    def update(self, G_prev, A_now, t, dt):
        if self._is_ratchet_date(t):
            return torch.maximum(G_prev, A_now)
        return G_prev

    def __repr__(self) -> str:
        if self.dates is None:
            return "<Ratchet(step-up)>"
        return f"<Ratchet(dates={self.dates})>"


class Reset(GuaranteeScheme):
    """Reset to the current account value on reset dates (may decrease)."""

    def __init__(self, dates: List[float], tol: float = 1e-6) -> None:
        self.dates = sorted(dates)
        self.tol = tol

    def _is_reset_date(self, t: float) -> bool:
        return any(abs(t - d) < self.tol for d in self.dates)

    def update(self, G_prev, A_now, t, dt):
        if self._is_reset_date(t):
            return A_now.clone()
        return G_prev

    def __repr__(self) -> str:
        return f"<Reset(dates={self.dates})>"


# ── composite floor ───────────────────────────────────────────────────────────


class GuaranteeFloor:
    """Composite floor: element-wise ``max`` over a list of schemes.

    Parameters
    ----------
    schemes : list of GuaranteeScheme
        Elementary update rules. The floor's one-step value is the maximum
        across them.
    G0 : float
        Initial level G_0 (every floor starts at the premium P).
    """

    def __init__(self, schemes: List[GuaranteeScheme], G0: float = 1.0) -> None:
        if not schemes:
            raise ValueError("GuaranteeFloor needs at least one scheme.")
        self.schemes = list(schemes)
        self.G0 = float(G0)

    def step(self, G_prev: Tensor, A_now: Tensor, t: float, dt: float) -> Tensor:
        """G_{n^-} = max_schemes update(G_{(n-1)^+}, A_{n^-}, t, dt)."""
        out = None
        for s in self.schemes:
            val = s.update(G_prev, A_now, t, dt)
            out = val if out is None else torch.maximum(out, val)
        return out

    def initial(self, A0: Tensor) -> Tensor:
        """Floor value at time 0 (G_0) broadcast to the path shape."""
        return torch.full_like(A0, self.G0)

    # convenience constructors ────────────────────────────────────────────────

    @classmethod
    def fixed(cls, G0: float) -> "GuaranteeFloor":
        return cls([Fixed()], G0=G0)

    @classmethod
    def zero(cls) -> "GuaranteeFloor":
        """Degenerate floor at 0 — account-only (Bacinello surrender / no GMDB)."""
        return cls([Fixed()], G0=0.0)

    @classmethod
    def rollup(cls, G0: float, delta: float) -> "GuaranteeFloor":
        return cls([RollUp(delta)], G0=G0)

    @classmethod
    def ratchet(cls, G0: float, dates: Optional[List[float]] = None) -> "GuaranteeFloor":
        return cls([Ratchet(dates)], G0=G0)

    def __repr__(self) -> str:
        return f"<GuaranteeFloor(G₀={self.G0}, schemes={self.schemes})>"


__all__ = [
    "GuaranteeScheme",
    "Fixed",
    "RollUp",
    "Ratchet",
    "Reset",
    "GuaranteeFloor",
]