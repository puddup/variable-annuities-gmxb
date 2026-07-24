"""Shared dynamics infrastructure: variance fixes and scheme base.

There is exactly **one** process abstraction in qabit — :class:`SDESystem`
(see :mod:`qabit.core.dynamics.sde`).  This module holds only the small,
process-agnostic pieces that the system and its discretisation schemes share:
the non-negativity ``VarianceFix`` family and the abstract :class:`SchemeBase`.
The per-factor :class:`FactorState` (which lives where the factors do) is
defined in :mod:`qabit.core.dynamics.sde.base`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor


# ── variance fixes ────────────────────────────────────────────────────────────


class VarianceFix(ABC):
    """Ensure a non-negative state variable after a discretisation step."""

    @abstractmethod
    def __call__(self, V: Tensor) -> Tensor:
        ...


class TruncationFix(VarianceFix):
    """``V⁺ = max(V, 0)``."""

    def __call__(self, V: Tensor) -> Tensor:
        return V.clamp(min=0.0)

    def __repr__(self) -> str:
        return "TruncationFix()"


class ReflectionFix(VarianceFix):
    """``V⁺ = |V|`` — reflect negatives."""

    def __call__(self, V: Tensor) -> Tensor:
        return V.abs()

    def __repr__(self) -> str:
        return "ReflectionFix()"


class AbsorptionFix(VarianceFix):
    """``V⁺ = max(V, ε)`` — absorbing barrier at a small positive value."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, V: Tensor) -> Tensor:
        return V.clamp(min=self.eps)

    def __repr__(self) -> str:
        return f"AbsorptionFix(eps={self.eps})"


# ── generic scheme base ──────────────────────────────────────────────────────


class SchemeBase(ABC):
    """Abstract base for any discretisation scheme.

    Parameters
    ----------
    fix : VarianceFix — non-negativity strategy (default: truncation).
    """

    def __init__(self, fix: Optional[VarianceFix] = None) -> None:
        self.fix = fix or TruncationFix()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fix={self.fix})"


__all__ = [
    "VarianceFix",
    "TruncationFix",
    "ReflectionFix",
    "AbsorptionFix",
    "SchemeBase",
]
