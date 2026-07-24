"""Forward-start European option — BSM, fully vectorised for any t.

At T0 the strike is set K = α·S_{T0}.  At T1 the payoff is (S_{T1} − K)⁺.
Pre-T0 pricing uses the forward-start homogeneity property; post-T0
is a vanilla BSM call with the (now-known) per-path strike.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.underlying.equity.stock import Stock
from qabit.core.market.curve import MarketKeys
from qabit.core.analytics.equity.black_scholes import bs_price, ncdf, d1 as bsm_d1
from qabit.tools.util import to_path


class EuropeanForwardOption(BaseDerivative):
    r"""Forward-start European call or put.

    Supports deterministic and stochastic discount curves at any t.
    """

    def __init__(
        self,
        underlier: Stock,
        alpha: float,
        T0: float,
        T1: float,
        is_call: bool = True,
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
    ) -> None:
        if T0 >= T1:
            raise ValueError(f"T0={T0} must be < T1={T1}.")
        super().__init__(underlier, T1, marketkeys, cost)
        self.alpha = alpha
        self.T0 = T0
        self.T1 = T1
        self.is_call = is_call
        self.observation_dates = [T0, T1]

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        S_T0 = self.underlier.spot(self.T0)
        S_T1 = self.underlier.spot(self.T1)
        strike_path = self.alpha * S_T0
        if self.is_call:
            return torch.relu(S_T1 - strike_path)
        return torch.relu(strike_path - S_T1)

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        return self._bsm_pricer(market, t)

    def _bsm_pricer(self, market, t: float) -> Tensor:
        """BSM forward-start price at t.  Returns Tensor[N]."""
        disc = self.marketkeys.discount_curve(market)
        vol = self.marketkeys.volatility_surface(market)
        S_t = self.underlier.spot(t)
        N = S_t.shape[0]
        dtype, device = S_t.dtype, S_t.device

        tau = self.T1 - self.T0
        sigma = vol.vol(self.alpha, tau)

        if t >= self.T0:
            # Post-T0: strike is known per path = α·S_{T0}
            S_T0 = self.underlier.spot(self.T0)
            K_path = self.alpha * S_T0
            tau_rem = max(self.T1 - t, 1e-8)
            fwd_df = to_path(disc.fwd_df(t, self.T1), N, dtype, device)
            r_path = -torch.log(fwd_df.clamp(1e-12)) / tau_rem
            return bs_price(S_t, K_path, r_path, sigma, tau_rem, self.is_call)
        else:
            # Pre-T0: forward-start homogeneity
            fwd_df_T0_T1 = to_path(disc.fwd_df(self.T0, self.T1), N, dtype, device)
            r_path = -torch.log(fwd_df_T0_T1.clamp(1e-12)) / max(tau, 1e-8)

            d1_val = bsm_d1(
                torch.ones_like(S_t),
                self.alpha,
                r_path,
                sigma,
                tau,
            )
            sigma_t = torch.tensor(sigma, dtype=dtype, device=device)
            tau_t = torch.tensor(tau, dtype=dtype, device=device)
            d2_val = d1_val - sigma_t * tau_t.sqrt()
            # S_t scales the unit-spot call on S_{T0}/(α·S_{T0})
            call = S_t * (ncdf(d1_val) - self.alpha * fwd_df_T0_T1 * ncdf(d2_val))
            if self.is_call:
                return call
            return call - S_t * (1.0 - self.alpha * fwd_df_T0_T1)

    def __repr__(self) -> str:
        return (
            f"<EuropeanForwardOption({'call' if self.is_call else 'put'}, "
            f"α={self.alpha}, T0={self.T0}, T1={self.T1})>"
        )


__all__ = [
    "EuropeanForwardOption",
]
