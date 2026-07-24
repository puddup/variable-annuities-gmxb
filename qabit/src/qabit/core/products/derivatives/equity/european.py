"""European call/put — closed-form BSM, fully vectorised for any t."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.underlying.equity.stock import Stock
from qabit.core.market.curve import MarketKeys
from qabit.core.analytics.equity.black_scholes import bs_price, d1, ncdf
from qabit.tools.util import to_path, as_dates


class EuropeanOption(BaseDerivative):
    """European call or put.

    ``_pricer`` → ``_bsm_pricer`` (closed-form, per path, any t).
    Supports deterministic and stochastic discount curves.
    """

    def __init__(
        self,
        underlier: Stock,
        strike: float,
        maturity: float,
        is_call: bool = True,
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(underlier, maturity, marketkeys, cost)
        self.strike = strike
        self.is_call = is_call
        self.observation_dates = [maturity]

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        S_T = self.underlier.spot(self.maturity)
        if self.is_call:
            return torch.relu(S_T - self.strike)
        return torch.relu(self.strike - S_T)

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        return self._bsm_pricer(market, t)

    def _bsm_pricer(self, market, t: float = 0.0) -> Tensor:
        """BSM price at valuation date t. Returns Tensor[N]."""
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)
        S_t = self.underlier.spot(t)
        N = S_t.shape[0]
        tau = max(self.maturity - t, 1e-8)

        # TODO: update flatvol to stochvol
        sigma = vol.vol(0.0, tau)
        fwd_df = to_path(disc.fwd_df(t, self.maturity), N, S_t.dtype, S_t.device)
        # per-path continuously compounded rate
        r_path = -torch.log(fwd_df.clamp(1e-12)) / tau
        return bs_price(S_t, self.strike, r_path, sigma, tau, self.is_call)

    # ── greeks ────────────────────────────────────────────────────────────────

    def delta(self, market, t=None) -> Tensor:
        """BSM delta.  Returns Tensor[N] when t is a scalar/None, else Tensor[T_val, N]."""
        single = (t is None) or isinstance(t, (int, float))
        dates = as_dates(t)
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)
        slices = []
        for t_i in dates:
            t_val = float(t_i)
            tau = max(self.maturity - t_val, 1e-8)

            # TODO: update flatvol to stochvol
            sigma = vol.vol(0.0, tau)

            S_t = self.underlier.spot(t_val)
            N = S_t.shape[0]
            fwd_df = to_path(disc.fwd_df(t_val, self.maturity), N, S_t.dtype, S_t.device)
            r_path = -torch.log(fwd_df.clamp(1e-12)) / tau
            d1_val = d1(S_t, self.strike, r_path, sigma, tau)
            delta_val = ncdf(d1_val) if self.is_call else ncdf(d1_val) - 1.0
            slices.append(delta_val)
        stacked = torch.stack(slices)
        # Match the user-facing convention used in hedging notebooks:
        # delta(mkt, t) with scalar t → Tensor[N]; delta(mkt, [t0,...]) → Tensor[T_val, N]
        return stacked[0] if single else stacked

    def __repr__(self) -> str:
        return (
            f"<EuropeanOption({'call' if self.is_call else 'put'}, "
            f"K={self.strike}, T={self.maturity})>"
        )


__all__ = [
    "EuropeanOption",
]
