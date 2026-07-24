"""Fixed-strike lookback options — Goldman-Sosin closed-form (deterministic rates)."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.underlying.equity.stock import Stock
from qabit.core.market.curve import MarketKeys
from qabit.core.analytics.equity.lookback import goldman_sosin_price


class FixedStrikeLookback(BaseDerivative):
    r"""Fixed-strike lookback call/put.  
    Call: max(S_max[ws,T]-K,0); put: max(K-S_min[ws,T],0).
    Closed-form requires deterministic rates; for t>0 with stochastic
    rates we raise.  At t=0 with stochastic rates we fall back to MC.
    """

    def __init__(
        self,
        underlier: Stock,
        strike: float,
        maturity: float,
        is_call: bool = True,
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
        window_start: float = 0.0
    ) -> None:
        super().__init__(underlier, maturity, marketkeys, cost)
        self.strike = strike
        self.is_call = is_call
        self.window_start = window_start
        self.observation_dates = None

    def _idx(self, t: float) -> int:
        dates = self.underlier._state().dates
        return int(torch.searchsorted(dates, torch.tensor(t, dtype=dates.dtype)).item())   
    
    def _extremum(self, lo: float, hi: float) -> Tensor:
        paths = self.underlier.full_path()
        w = paths[:, self._idx(lo): self._idx(hi) + 1]
        return w.max(dim=1).values if self.is_call else w.min(dim=1).values

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        S_ext = self._extremum(self.window_start, self.maturity)
        if self.is_call:
            return torch.relu(S_ext - self.strike)
        return torch.relu(self.strike - S_ext)

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        disc = self.marketkeys.discount_curve(market)
        if t != 0.0:
            if not disc.is_flat:
                raise NotImplementedError("Lookback pricing for t>0 requires deterministic rates.")
            return self._bsm_pricer(market, t)
        if disc.is_flat:
            return self._bsm_pricer(market, 0.0)
        return self._mc_price_zero(market)

    def _bsm_pricer(self, market, t: float) -> Tensor:
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)
        S_t = self.underlier.spot(t)
        tau = max(self.maturity - t, 1e-8)
        
        # TODO: update flatvol to stochvol
        sigma = vol.vol(0.0, tau)

        # NOTE: deterministic rate → scalar/float
        fwd_df = float(disc.fwd_df(t, self.maturity))
        r = -math.log(max(fwd_df, 1e-12)) / tau

        # NOTE: running extremum over [ws, t]
        S_ext = self._extremum(self.window_start, t)          
        tau_t = torch.full_like(S_t, tau)
        return goldman_sosin_price(S_t, self.strike, S_ext, r, sigma, tau_t, self.is_call)

    def __repr__(self):
        return (
            f"<FixedStrikeLookback({'call' if self.is_call else 'put'}, "
            f"K={self.strike}, T={self.maturity})>"
        )


__all__ = [
    "FixedStrikeLookback",
]
