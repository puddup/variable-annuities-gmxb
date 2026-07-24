"""Statistics computed from *simulated* hazard-rate paths — pure functions.

Each takes the simulation ``dates`` ``(E,)`` and per-path hazard ``paths``
``(N, E)`` and returns a path-resolved quantity.  These are model-agnostic: they
work on any simulated intensity, whether produced by a CIR hazard factor or any
other source.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from qabit.tools.interpolation import integrate_trapezoid_paths


def pathwise_cumulative_hazard(dates: Tensor, paths: Tensor) -> Tensor:
    r"""Per-path \hat{Lambda}(t) by trapezoidal integration of the hazard. (N, E)."""
    return integrate_trapezoid_paths(dates, paths)


def pathwise_survival(dates: Tensor, paths: Tensor) -> Tensor:
    r"""Per-path S(t) = exp(-\hat{Lambda}(t)). (N, E)."""
    return torch.exp(-pathwise_cumulative_hazard(dates, paths))


def pathwise_density(dates: Tensor, paths: Tensor) -> Tensor:
    r"""Per-path death density via finite differences on survival. (N, E)."""
    sp = pathwise_survival(dates, paths)
    dt = dates[1:] - dates[:-1]
    f = torch.zeros_like(sp)
    f[:, 1:] = -(sp[:, 1:] - sp[:, :-1]) / dt.unsqueeze(0)
    return f.clamp(min=0.0)


def pathwise_expected_lifetime(dates: Tensor, paths: Tensor) -> Tensor:
    r"""Per-path E[tau] ~= \int S_i(t) dt. (N,)."""
    sp = pathwise_survival(dates, paths)
    dt = dates[1:] - dates[:-1]
    return (0.5 * (sp[:, :-1] + sp[:, 1:]) * dt.unsqueeze(0)).sum(dim=1)


def sample_tau(dates: Tensor, paths: Tensor, seed: Optional[int] = None) -> Tensor:
    r"""Death times by integrated-hazard inversion. Returns (N,).

    Draws ``E_i ~ Exp(1)`` and returns the first date where the per-path
    cumulative hazard crosses ``E_i``.  Paths that never cross are assigned a
    sentinel just past the horizon (so ``tau > T`` means "survived").
    """
    cumL = pathwise_cumulative_hazard(dates, paths)
    N = paths.shape[0]
    rng = np.random.default_rng(seed)
    E = torch.as_tensor(rng.exponential(1.0, N).astype(np.float32), device=paths.device)
    exceeded = cumL >= E.unsqueeze(1)
    first_cross = exceeded.long().argmax(dim=1)
    tau = dates[first_cross].float()
    tau[~exceeded[:, -1]] = float(dates[-1]) + 1.0
    return tau


__all__ = [
    "pathwise_cumulative_hazard",
    "pathwise_survival",
    "pathwise_density",
    "pathwise_expected_lifetime",
    "sample_tau",
]
