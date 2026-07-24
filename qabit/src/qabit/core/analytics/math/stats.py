r"""analytics/math/stats.py — :math:`F_T`-measurable path descriptors.

Pure, pathwise functionals of a simulated spot panel.  Each maps the full
per-path spot tensor ``path`` of shape ``(N, E)`` (``N`` paths, ``E`` time steps)
to a single column ``(N,)``, and computes from spot alone — no model parameters,
no dynamics knowledge — so it is a valid clustering feature in the sense of the
*Ambiguity-Averse Deep Hedging* framework (Jones et al., 2025, §3.3).

These are *analytics*, not hedging machinery: they live here so any part of the
stack (regime partitioners, diagnostics, research notebooks) can share one
implementation.  :mod:`qabit.core.nn.adversarial` imports them to build its
regime partitioners.

Scale-freeness
--------------
Every feature is used only to *cluster* paths, and the partitioners min-max
normalise each feature to :math:`[0, 1]` before clustering.  Any global positive
rescaling (e.g. the ``1/sqrt(dt)`` that annualises a return std into a volatility)
is therefore absorbed by the normalisation and need not be applied here — which is
why none of these take ``dt``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def realised_vol(path: Tensor) -> Tensor:
    r"""Per-path realised volatility — the std of log returns, ``(N,)``."""
    log_ret = torch.log(path[:, 1:] / path[:, :-1].clamp_min(1e-12))
    return log_ret.std(dim=1)


def autocorrelation(path: Tensor) -> Tensor:
    r"""Per-path lag-1 autocorrelation of log returns, ``(N,)`` in ``[-1, 1]``."""
    log_ret = torch.log(path[:, 1:] / path[:, :-1].clamp_min(1e-12))
    r = log_ret - log_ret.mean(dim=1, keepdim=True)
    num = (r[:, :-1] * r[:, 1:]).sum(dim=1)
    den = (r**2).sum(dim=1).clamp_min(1e-12)
    return num / den


def max_drawdown(path: Tensor) -> Tensor:
    r"""Per-path maximum drawdown ``(N,)`` in ``[0, 1)``."""
    cummax = path.cummax(dim=1).values
    drawdown = (cummax - path) / cummax.clamp_min(1e-12)
    return drawdown.max(dim=1).values


def mean_reversion(path: Tensor) -> Tensor:
    r"""Per-path OU mean-reversion estimate :math:`\hat\kappa` (MLE ratio form), ``(N,)``.

    The discrete-AR(1) slope :math:`\hat b` of :math:`S_{t+1}` on :math:`S_t` is
    mapped to :math:`\hat\kappa = -\log \hat b` (the ``1/dt`` factor is dropped — see
    the module note on scale-freeness).  Clamped to a stable, normalisable range.
    """
    S_prev, S_next = path[:, :-1], path[:, 1:]
    n = S_prev.shape[1]
    num = (S_next * S_prev).sum(dim=1) - (S_next.sum(dim=1) * S_prev.sum(dim=1)) / n
    den = ((S_prev**2).sum(dim=1) - S_prev.sum(dim=1) ** 2 / n).clamp_min(1e-12)
    ratio = (num / den).clamp(1e-8, 1.0 - 1e-8)
    return -torch.log(ratio)


# ─────────────────────────────────────────────────────────────────────────────
# Mortality / hazard path features
# ─────────────────────────────────────────────────────────────────────────────
# F_T-measurable functionals applied to the **stochastic hazard path** ``lambda_t``
# (the CIR-Gompertz intensity) rather than the spot.  Pair each with the source tag
# ``"mort"`` so :class:`AmbiguityAverseLoss` reads the hazard factor::
#
#     loss = AmbiguityAverseLoss.factors(
#         gm, a=20, k=20,
#         features=((integrated_hazard, "mort"), (realised_hazard_vol, "mort")),
#         inner=JonesLoss(lam=1/100), underlier_key="fund")
#
# They are valid §3.3 features: pathwise-computable, F_T-measurable, policy-free.
# Like the equity pair, they span a level/accumulation axis (integrated_hazard,
# = -log S_T) and a dispersion axis (realised_hazard_vol, the sigma_lam footprint).
# Any constant grid-step scale is absorbed by the partitioner's min-max normalise,
# so — as with the spot features — none of these take ``dt``.


def integrated_hazard(path: Tensor) -> Tensor:
    r"""Cumulative force of mortality :math:`\Lambda_T=\sum_i \lambda_{t_i}\,\Delta t`.

    Equal to :math:`-\log S_T`; the single most hedging-relevant mortality scalar.
    """
    return path[:, 1:].clamp_min(0.0).sum(dim=1) # * HAZARD_DT


def realised_hazard_vol(path: Tensor) -> Tensor:
    r"""Realised dispersion of the hazard path — the footprint of ``sigma_lam``."""
    return path[:, 1:].std(dim=1)


def mean_hazard(path: Tensor) -> Tensor:
    r"""Path-mean hazard level :math:`\overline{\lambda}` — the CIR level regime."""
    return path.mean(dim=1)


def hazard_max_jump(path: Tensor) -> Tensor:
    r"""Largest single-step hazard increment — a tail-shock regime descriptor."""
    return (path[:, 1:] - path[:, :-1]).abs().max(dim=1).values



# ─────────────────────────────────────────────────────────────────────────────
# Generic level / trend / tail path features
# ─────────────────────────────────────────────────────────────────────────────
# Source-agnostic §3.3 features: pathwise, F_T-measurable, policy-free.  Pair
# them with any factor source — ``(mean_level, "rate")`` clusters on the
# realised short-rate regime (the θ_r ambiguity axis), ``(downside_vol, SPOT)``
# adds a left-tail axis the plain realised vol misses.  As with everything in
# this module, constant scale factors are absorbed by the partitioner's
# min-max normalisation.


def mean_level(path: Tensor) -> Tensor:
    r"""Path-mean level :math:`\overline{x}` — the realised *level* regime.

    On a rate path this is the single most hedging-relevant rate scalar: the
    footprint of the mean-reversion target :math:`\theta` over the horizon.
    """
    return path.mean(dim=1)


def terminal_level(path: Tensor) -> Tensor:
    r"""Terminal value :math:`x_T` — where the path *ends*, e.g. terminal
    moneyness of the fund against a fixed guarantee."""
    return path[:, -1]


def level_change(path: Tensor) -> Tensor:
    r"""Net drift over the horizon, :math:`x_T - x_0` — the realised *trend*
    regime (separates persistent-undershoot from persistent-overshoot paths,
    which a path-mean alone can conflate)."""
    return path[:, -1] - path[:, 0]


def downside_vol(path: Tensor) -> Tensor:
    r"""Semivolatility of log returns (below-mean only), ``(N,)``.

    A left-tail dispersion axis: two paths with equal realised vol but opposite
    skew (crash-y vs melt-up) land in different clusters — exactly the split a
    guarantee hedger cares about."""
    log_ret = torch.log(path[:, 1:] / path[:, :-1].clamp_min(1e-12))
    d = (log_ret - log_ret.mean(dim=1, keepdim=True)).clamp(max=0.0)
    return (d * d).mean(dim=1).sqrt()


def realised_skew(path: Tensor) -> Tensor:
    r"""Sample skewness of log returns, ``(N,)`` — the jump/crash footprint.

    Under a pure diffusion this concentrates near 0; negative jumps push it
    strongly negative, making it a natural axis for separating jump-bearing
    paths from diffusive ones when the market generator carries jumps."""
    log_ret = torch.log(path[:, 1:] / path[:, :-1].clamp_min(1e-12))
    c = log_ret - log_ret.mean(dim=1, keepdim=True)
    m2 = (c * c).mean(dim=1).clamp_min(1e-18)
    m3 = (c * c * c).mean(dim=1)
    return m3 / m2.pow(1.5)


__all__ = [
    # spot / equity path descriptors
    "realised_vol",
    "autocorrelation",
    "max_drawdown",
    "mean_reversion",
    # generic level / trend / tail descriptors
    "mean_level",
    "terminal_level",
    "level_change",
    "downside_vol",
    "realised_skew",
    # hazard / mortality path descriptors
    "integrated_hazard",
    "realised_hazard_vol",
    "mean_hazard",
    "hazard_max_jump",
]
