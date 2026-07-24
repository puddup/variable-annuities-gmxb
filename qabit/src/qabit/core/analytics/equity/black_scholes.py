"""core/analytics/equity/black_scholes.py — BSM analytics.

All primitives (d1, d2, ncdf, npdf) are module-level functions so they
can be imported and reused by lookback, cliquet, and other analytics
without duplication.

Now supports per-path rates: ``r`` can be a float or Tensor[N].
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor
from scipy.stats import norm as _sp_norm
from scipy.optimize import brentq

from qabit.config import EPS_TIME
from qabit.tools.broadcast import auto_tensor, clamp_inputs

_ndtr = torch.special.ndtr
_ndtr(torch.tensor(0.0))  # verify available


# ── primitives — reused by all equity analytics ───────────────────────────────


@auto_tensor()
@clamp_inputs(tau=EPS_TIME)
def d1(S: Tensor, K, r: Union[float, Tensor], sigma, tau) -> Tensor:
    r"""d1 = (log(S/K) + (r + σ²/2)τ) / (σ√τ)."""
    return (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * tau.sqrt())


@auto_tensor()
@clamp_inputs(tau=EPS_TIME)
def d2(S: Tensor, K, r: Union[float, Tensor], sigma, tau) -> Tensor:
    r"""d2 = d1 - σ√τ."""
    return d1(S, K, r, sigma, tau) - sigma * tau.sqrt()


def ncdf(x: Tensor) -> Tensor:
    """Standard normal CDF N(x) — differentiable via torch.special.ndtr."""
    return _ndtr(x)


def npdf(x: Tensor) -> Tensor:
    """Standard normal PDF n(x) = exp(-x²/2) / √(2π)."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


# ── Black-Scholes-Merton price ────────────────────────────────────────────────


@auto_tensor()
@clamp_inputs(tau=EPS_TIME)
def bs_price(
    S: Tensor,
    K,
    r: Union[float, Tensor],
    sigma: Union[Tensor, float],
    tau: Union[Tensor, float],
    is_call: bool = True,
) -> Tensor:
    r"""Vectorised BSM price — differentiable w.r.t. S and sigma.

    .. math::
        C = S\,N(d_1) - K e^{-r\tau} N(d_2)

    ``r`` may be a float or per-path Tensor[N].  ``K`` may be a float or
    per-path Tensor[N] (for forward-start contracts).
    """
    d1_ = d1(S, K, r, sigma, tau)
    d2_ = d2(S, K, r, sigma, tau)
    disc = torch.exp(-r * tau)
    call = S * ncdf(d1_) - K * disc * ncdf(d2_)
    return call if is_call else call - S + K * disc


def bs_put_scalar(
    sigma: float,
    r: float,
    t: float,
    g: float = 0.0,
    k: float = 0.0,
    S0: float = 1.0,
) -> float:
    r"""Scalar BSM put for density-integration loops.

    Strike X = S0·e^{g·t}, fee k as dividend yield.
    Returns intrinsic max(X - S0, 0) at t = 0.
    """
    if t < 1e-8:
        return max(S0 * math.exp(g * t) - S0, 0.0)
    X = S0 * math.exp(g * t)
    _d1 = (math.log(S0 / X) + (r - k + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    _d2 = _d1 - sigma * math.sqrt(t)
    return X * math.exp(-r * t) * _sp_norm.cdf(-_d2) - S0 * math.exp(
        -k * t
    ) * _sp_norm.cdf(-_d1)


def implied_vol(
    price: float,
    S: float,
    K: float,
    r: float,
    tau: float,
    is_call: bool = True,
    lo: float = 1e-4,
    hi: float = 10.0,
) -> float:
    """Black-Scholes implied volatility via Brent root-find."""
    S_t = torch.tensor(S)

    def gap(sig):
        return float(bs_price(S_t, K, r, sig, tau, is_call)) - price

    try:
        return brentq(gap, lo, hi, xtol=1e-6, maxiter=100)
    except ValueError:
        return float("nan")


__all__ = [
    "d1",
    "d2",
    "ncdf",
    "npdf",
    "bs_price",
    "bs_put_scalar",
    "implied_vol",
]
