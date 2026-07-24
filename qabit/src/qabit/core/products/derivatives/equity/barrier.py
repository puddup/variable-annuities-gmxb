"""Barrier options — DAO/DAI/UAO/UAI, Rubinstein-Reiner (deterministic rates).

Vectorised: ``_barrier_cf`` now accepts per-path ``S0``, ``K`` and ``B`` tensors
(the ``K>=B`` branch becomes a per-path ``torch.where``), so a forward-start /
rolling barrier whose strike & barrier are ``alpha * S_{T_start}`` prices in one
shot with no Python loop.  ``window_start`` rebases the breach monitoring window:
default ``0.0`` reproduces the on-the-run barrier; a ForwardStart wrapper sets it
to the leg's start date so the breach test runs over ``[T_start, t]``.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.analytics.equity.black_scholes import bs_price as _vanilla, ncdf


# ── Rubinstein-Reiner analytic, fully vectorised over S0/K/B ──────────────────


def _barrier_cf(
    S0: Tensor,
    K: Union[float, Tensor],
    B: Union[float, Tensor],
    T: float,
    r: float,
    sigma: float,
    btype: str,
    is_call: bool,
) -> Tensor:
    """Continuous-monitoring barrier price (Rubinstein-Reiner 1991), per path.

    ``S0`` is ``Tensor[N]``; ``K`` and ``B`` may be float or ``Tensor[N]``;
    ``r, sigma, T`` are scalars (deterministic-rate regime).  Returns ``Tensor[N]``.
    """
    dtype, device = S0.dtype, S0.device
    K = torch.as_tensor(K, dtype=dtype, device=device).broadcast_to(S0.shape)
    B = torch.as_tensor(B, dtype=dtype, device=device).broadcast_to(S0.shape)

    mu = (r - 0.5 * sigma ** 2) / sigma ** 2
    sqT = math.sqrt(max(T, 1e-12))
    s = sigma * sqT
    df = math.exp(-r * T)

    x1 = torch.log(S0 / K) / s + (1 + mu) * s
    x2 = torch.log(S0 / B) / s + (1 + mu) * s
    y1 = torch.log(B ** 2 / (S0 * K)) / s + (1 + mu) * s
    y2 = torch.log(B / S0) / s + (1 + mu) * s
    pow_a = (B / S0) ** (2 * (mu + 1))
    pow_b = (B / S0) ** (2 * mu)

    def A(ph):
        return ph * S0 * ncdf(ph * x1) - ph * K * df * ncdf(ph * (x1 - s))

    def Bf(ph):
        return ph * S0 * ncdf(ph * x2) - ph * K * df * ncdf(ph * (x2 - s))

    def C(ph, et):
        return ph * S0 * pow_a * ncdf(et * y1) - ph * K * df * pow_b * ncdf(et * (y1 - s))

    def D(ph, et):
        return ph * S0 * pow_a * ncdf(et * y2) - ph * K * df * pow_b * ncdf(et * (y2 - s))

    KgeB = K >= B
    if btype == "dao":
        if is_call:
            return torch.where(KgeB, A(1) - C(1, 1), Bf(1) - D(1, 1))
        # put:  X>=H: A-B+C-D ;  X<H: knocked out before it can pay -> 0
        return torch.where(
            KgeB, A(-1) - Bf(-1) + C(-1, 1) - D(-1, 1), torch.zeros_like(S0)
        )
    if btype == "dai":
        van = _vanilla(S0, K, r, sigma, T, is_call)
        return van - _barrier_cf(S0, K, B, T, r, sigma, "dao", is_call)
    if btype == "uao":
        if is_call:
            # X<H: A-B+C-D (η=-1);  X>=H: knocked out before it can pay -> 0
            return torch.where(K < B, A(1) - Bf(1) + C(1, -1) - D(1, -1), torch.zeros_like(S0))
        # put:  X>=H: B-D ;  X<H: A-C   (η=-1)
        return torch.where(KgeB, Bf(-1) - D(-1, -1), A(-1) - C(-1, -1))
    if btype == "uai":
        van = _vanilla(S0, K, r, sigma, T, is_call)
        return van - _barrier_cf(S0, K, B, T, r, sigma, "uao", is_call)
    raise ValueError(btype)


# ── BarrierOption ─────────────────────────────────────────────────────────────


class BarrierOption(BaseDerivative):
    """Single-barrier knock-in/knock-out option (discrete monitoring on the grid).

    Per-path ``strike``/``barrier`` tensors are supported (forward-start legs).
    Deterministic discount curves only for ``t>0``; ``t=0`` with stochastic rates
    falls back to MC.  ``window_start`` rebases the breach-monitoring window.
    """

    _TYPES = {"dao", "dai", "uao", "uai"}

    def __init__(
        self,
        underlier,
        strike,
        barrier,
        maturity,
        barrier_type="dao",
        is_call=True,
        rebate=0.0,
        marketkeys=None,
        cost=0.0,
        window_start: float = 0.0,
    ) -> None:
        super().__init__(underlier, maturity, marketkeys, cost)
        if barrier_type not in self._TYPES:
            raise ValueError(barrier_type)
        self.strike = strike
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.is_call = is_call
        self.rebate = rebate
        self.window_start = window_start
        self.observation_dates = None

    # ── breach test over [window_start, t] ────────────────────────────────────

    def _crossed(self, paths: Tensor) -> Tensor:
        """Boolean ``Tensor[N]`` — barrier breached within the supplied window."""
        if self.barrier_type in ("dao", "dai"):
            return paths.min(dim=1).values <= self.barrier
        return paths.max(dim=1).values >= self.barrier

    def mc_payoff(self) -> Tensor:
        S = self.underlier.full_path()
        i0 = self._idx(self.window_start)
        iT = self._idx(self.maturity)            # monitor & pay at maturity — NOT
        window = S[:, i0 : iT + 1]               # at the end of the simulated horizon
        ST = S[:, iT]
        van = torch.relu(ST - self.strike) if self.is_call else torch.relu(self.strike - ST)
        crossed = self._crossed(window)
        if self.barrier_type in ("dao", "uao"):
            return torch.where(crossed, torch.full_like(van, float(self.rebate)), van)
        return torch.where(crossed, van, torch.zeros_like(van))

    # ── pricers ───────────────────────────────────────────────────────────────

    def _idx(self, t: float) -> int:
        dates = self.underlier._state().dates
        return int(torch.searchsorted(dates, torch.tensor(t, dtype=dates.dtype)).item())

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        disc = self.marketkeys.discount_curve(market)
        if t != 0.0:
            if not disc.is_flat:
                raise NotImplementedError("Barrier pricing for t>0 requires deterministic rates.")
            return self._cf_pricer(market, t)
        if disc.is_flat:
            return self._cf_pricer(market, 0.0)
        return self._mc_price_zero(market)

    def _cf_pricer(self, market, t: float) -> Tensor:
        """Per-path barrier price at ``t`` — vectorised, deterministic rates."""
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)

        tau = max(self.maturity - t, 1e-8)

        # TODO: update flatvol to stochvol
        sigma = vol.vol(0.0, tau)

        # NOTE: deterministic rate → scalar/float
        fwd_df = float(disc.fwd_df(t, self.maturity))
        r = -math.log(max(fwd_df, 1e-12)) / tau
        
        S_t = self.underlier.spot(t)
        paths = self.underlier.full_path()
        window = paths[:, self._idx(self.window_start): self._idx(t) + 1]
        crossed = self._crossed(window)

        cf = _barrier_cf(S_t, self.strike, self.barrier, tau, r, sigma,
                         self.barrier_type, self.is_call)
        if self.barrier_type in ("dao", "uao"):
            crossed_val = torch.full_like(S_t, float(self.rebate))
        else:  # already knocked-in → vanilla on the (per-path) strike
            crossed_val = _vanilla(S_t, self.strike, r, sigma, tau, self.is_call)
        return torch.where(crossed, crossed_val, cf)

    def __repr__(self):
        return (
            f"<BarrierOption({self.barrier_type}, {'call' if self.is_call else 'put'}, "
            f"T={self.maturity}, ws={self.window_start})>"
        )


# ── convenience constructors (unchanged) ──────────────────────────────────────


def DownAndOutCall(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "dao", True, **kw)
def DownAndInCall(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "dai", True, **kw)
def UpAndOutCall(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "uao", True, **kw)
def UpAndInCall(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "uai", True, **kw)
def DownAndOutPut(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "dao", False, **kw)
def DownAndInPut(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "dai", False, **kw)
def UpAndOutPut(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "uao", False, **kw)
def UpAndInPut(u, K, B, T, **kw): return BarrierOption(u, K, B, T, "uai", False, **kw)


__all__ = [
    "BarrierOption", "DownAndOutCall", "DownAndInCall", "UpAndOutCall", "UpAndInCall",
    "DownAndOutPut", "DownAndInPut", "UpAndOutPut", "UpAndInPut",
]