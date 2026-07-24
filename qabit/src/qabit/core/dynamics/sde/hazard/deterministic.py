"""dynamics/sde/hazard/deterministic.py — deterministic hazard factors.

One factor class per intensity law: ``ExponentialHazardFactor``,
``GompertzHazardFactor``, ``WeibullHazardFactor``.  Each binds its parameters
to the corresponding pure function in :mod:`qabit.core.analytics.hazard.deterministic`
and contributes **zero** Brownian drivers (so adds nothing to the system noise
budget or correlation matrix).

The math — closed-form intensities and the survival family derived from them —
lives entirely in :mod:`qabit.core.analytics.hazard.deterministic`.  The factors
here are thin: parameters + ``init``/``step``.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import Tensor

from qabit.core.analytics.hazard.deterministic import (
    exponential_hazard,
    gompertz_hazard,
    weibull_hazard,
)
from qabit.core.dynamics.sde.base import Factor


class _DeterministicHazardFactor(Factor):
    r"""Base class for deterministic intensity ``lambda(t) = hazard_fn(t)``.

    ``hazard_fn`` is bound at construction with the law's parameters.  All
    paths see the same intensity (zero Brownian drivers); the factor exists so
    that survival / death-time sampling, market curves, and joint simulation
    with stochastic factors can read a hazard-factor view uniformly.
    """

    n_factors = 0
    n_outputs = 1
    hazard_fn: Callable[[Tensor], Tensor]

    def _at(self, t: float) -> float:
        return float(self.hazard_fn(torch.tensor([float(t)]))[0])

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), self._at(0.0), device=device, dtype=dtype)

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        return torch.full_like(value, self._at(t + dt))


class ExponentialHazardFactor(_DeterministicHazardFactor):
    r"""Deterministic exponential (constant) hazard ``lambda(t) = lambda0``.

    Examples
    --------
    >>> SDESystem({"mort": ExponentialHazardFactor(lambda0=0.05)})
    """

    def __init__(self, lambda0: float = 0.05) -> None:
        self.lambda0 = lambda0
        self.hazard_fn = partial(exponential_hazard, lambda0=lambda0)

    def __repr__(self) -> str:
        return f"ExponentialHazardFactor(lambda0={self.lambda0})"


class GompertzHazardFactor(_DeterministicHazardFactor):
    r"""Deterministic Gompertz hazard ``lambda(t) = b^{-1} exp((age + t - m) / b)``.

    Parameters
    ----------
    m         : modal age at death.
    b         : dispersion (Gompertz scale).
    entry_age : starting age at simulation t=0 (so ``age = entry_age + t``).
    """

    def __init__(
        self, m: float = 80.0, b: float = 9.5, entry_age: float = 60.0
    ) -> None:
        self.m = m
        self.b = b
        self.entry_age = entry_age
        self.hazard_fn = partial(gompertz_hazard, m=m, b=b, entry_age=entry_age)

    def __repr__(self) -> str:
        return (
            f"GompertzHazardFactor(m={self.m}, b={self.b}, entry_age={self.entry_age})"
        )


class WeibullHazardFactor(_DeterministicHazardFactor):
    r"""Deterministic Weibull hazard ``lambda(t) = (c2/c1) ((age + t)/c1)^{c2-1}``.

    Parameters
    ----------
    c1        : Weibull scale.
    c2        : Weibull shape.
    entry_age : starting age at simulation t=0.
    """

    def __init__(
        self, c1: float = 90.43, c2: float = 10.36, entry_age: float = 60.0
    ) -> None:
        self.c1 = c1
        self.c2 = c2
        self.entry_age = entry_age
        self.hazard_fn = partial(weibull_hazard, c1=c1, c2=c2, entry_age=entry_age)

    def __repr__(self) -> str:
        return f"WeibullHazardFactor(c1={self.c1}, c2={self.c2}, entry_age={self.entry_age})"


__all__ = [
    "ExponentialHazardFactor",
    "GompertzHazardFactor",
    "WeibullHazardFactor",
]
