"""analytics/hazard/laws.py — deterministic hazard mathematics (functions only).

Closed-form intensity ``lambda(t)`` on a years-since-inception grid, plus
the survival family derived from a hazard curve by quadrature.  Everything here
is a pure function of arrays — no classes, no Monte-Carlo state.  A hazard
*factor* (see :mod:`qabit.core.dynamics.sde.credit`) holds the parameters and
calls these to get its target intensity; the survival/density helpers are shared
with the pathwise estimators in :mod:`qabit.core.analytics.hazard.pathwise`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from qabit.tools.util import as_tensor


# ── intensity lambda(t) ────────────────────────────────────────────────


def exponential_hazard(t, lambda0: float) -> Tensor:
    r"""Constant hazard ``lambda(t) = lambda0`` (memoryless lifetime)."""
    return torch.full_like(as_tensor(t), float(lambda0))


def gompertz_hazard(t, m: float, b: float, entry_age: float = 0.0) -> Tensor:
    r"""Gompertz hazard ``lambda(t) = b^{-1} exp((age + t - m) / b)``."""
    t = as_tensor(t)
    return (1.0 / b) * torch.exp((entry_age + t - m) / b)


def weibull_hazard(t, c1: float, c2: float, entry_age: float = 0.0) -> Tensor:
    r"""Weibull hazard ``lambda(t) = (c2/c1) ((age + t)/c1)^{c2-1}``."""
    t = as_tensor(t)
    age = entry_age + t
    return (c2 / c1) * (age / c1) ** (c2 - 1)


# ── survival family from a deterministic hazard curve ────────────────────────


def cumulative_hazard(dates, hazard) -> Tensor:
    r"""``Lambda(t) = \int_0^t lambda(s) ds`` by the trapezoid rule; ``Lambda(0)=0``."""
    dates, hazard = as_tensor(dates), as_tensor(hazard)
    dt = dates[1:] - dates[:-1]
    inc = 0.5 * (hazard[:-1] + hazard[1:]) * dt
    zero = torch.zeros(1, dtype=hazard.dtype, device=hazard.device)
    return torch.cat([zero, inc.cumsum(0)])


def survival(dates, hazard) -> Tensor:
    r"""``S(t) = exp(-Lambda(t))`` for a deterministic hazard curve."""
    return torch.exp(-cumulative_hazard(dates, hazard))


def density(dates, hazard) -> Tensor:
    r"""Death density ``f(t) = lambda(t) S(t)``."""
    return as_tensor(hazard) * survival(dates, hazard)


def expected_lifetime(dates, hazard) -> Tensor:
    r"""``E[tau] = \int_0^T S(t) dt`` (trapezoid)."""
    return torch.trapezoid(survival(dates, hazard), as_tensor(dates))


__all__ = [
    "exponential_hazard",
    "gompertz_hazard",
    "weibull_hazard",
    "cumulative_hazard",
    "survival",
    "density",
    "expected_lifetime",
]
