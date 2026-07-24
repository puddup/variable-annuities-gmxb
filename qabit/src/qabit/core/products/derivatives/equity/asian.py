"""Asian (average price) options — Kemna-Vorst closed-form (geometric, flat rates)."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from scipy.stats import norm as _norm

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.market.curve import MarketKeys


class AsianOption(BaseDerivative):
    """Fixed-strike Asian call or put.

    average = "geometric"  : Kemna-Vorst closed-form (deterministic rates only).
    average = "arithmetic" : MC only.
    Both support t=0; t>0 not yet implemented.
    """

    def __init__(
        self,
        underlier,
        strike,
        maturity,
        average="arithmetic",
        is_call=True,
        marketkeys=None,
        cost=0.0,
    ) -> None:
        super().__init__(underlier, maturity, marketkeys, cost)
        self.strike = strike
        self.average = average
        self.is_call = is_call
        self.observation_dates = None

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        full = self.underlier.full_path()
        dates = self.underlier._state().dates
        iT = int(torch.searchsorted(dates, torch.tensor(self.maturity, dtype=dates.dtype)).item())
        paths = full[:, : iT + 1]                # average over [0, maturity] only
        if self.average == "arithmetic":
            avg = paths.mean(dim=1)
        else:
            avg = torch.exp(torch.log(paths.clamp(min=1e-8)).mean(dim=1))
        if self.is_call:
            return torch.relu(avg - self.strike)
        return torch.relu(self.strike - avg)

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        if t != 0.0:
            raise NotImplementedError("Asian pricing for t>0 not yet implemented.")
        if self.average == "geometric":
            disc = self.marketkeys.discount_curve(market)
            if disc.is_flat:
                return self._kv_pricer(market)
            return self._mc_price_zero(market)
        # arithmetic: always MC
        return self._mc_price_zero(market)

    def _kv_pricer(self, market) -> Tensor:
        """Kemna-Vorst geometric Asian at t=0.  Returns Tensor[N] (constant)."""
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)

        tau = self.maturity
        df_val = float(disc.df(tau))
        r = -math.log(max(df_val, 1e-12)) / tau
        sigma = float(vol.vol(self.strike, tau))

        S0 = float(self.underlier.spot(0.0)[0].item())
        sig_G = sigma / math.sqrt(3.0)
        b_G = 0.5 * (r - sigma**2 / 6.0)

        d1_ = (math.log(S0 / self.strike) + (b_G + 0.5 * sig_G**2) * tau) / (
            sig_G * math.sqrt(tau)
        )
        d2_ = d1_ - sig_G * math.sqrt(tau)
        N_cdf = _norm.cdf
        ph = 1.0 if self.is_call else -1.0
        val = ph * (
            S0 * math.exp((b_G - r) * tau) * N_cdf(ph * d1_)
            - self.strike * math.exp(-r * tau) * N_cdf(ph * d2_)
        )

        N = self.underlier._state().paths.shape[0]
        return torch.full((N,), val, dtype=torch.float32)

    def __repr__(self):
        return (
            f"<AsianOption({self.average}, "
            f"{'call' if self.is_call else 'put'}, "
            f"K={self.strike}, T={self.maturity})>"
        )


__all__ = [
    "AsianOption",
]
