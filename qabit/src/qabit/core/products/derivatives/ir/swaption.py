"""European swaption — Black's formula, per-path, works at any t under stochastic rates."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.derivatives.ir.swap import InterestRateSwap
from qabit.core.market.curve import MarketKeys


class Swaption(BaseDerivative):
    """Payer or receiver European swaption priced with Black's formula.

    The forward swap rate and PVBP/annuity are computed per path from the
    discount/projection curves, so the price is fully path-dependent under
    stochastic rates.
    """

    def __init__(
        self,
        swap: InterestRateSwap,
        expiry: float,
        strike: float,
        is_payer: bool = True,
        volatility: Optional[float] = None,
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(None, max(swap.schedule), marketkeys or swap.marketkeys, cost)
        self.swap = swap
        self.expiry = expiry
        self.strike = strike
        self.is_payer = is_payer
        self.volatility = volatility
        self.observation_dates = [expiry]

    def mc_payoff(self) -> Tensor:
        raise NotImplementedError("Swaption mc_payoff not needed — use _pricer (Black) directly.")

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        """Swaption price at valuation date t.  Returns Tensor[N]."""
        disc = self.marketkeys.discount_curve(market)
        proj = self.marketkeys.forward_curve(market) or disc

        S_t, A_t = self._forward_swap_rate(market, t, proj, disc)

        tau = max(self.expiry - t, 1e-8)
        if self.marketkeys.volatility is not None:
            sigma = float(self.marketkeys.volatility_surface(market).vol(self.strike, tau))
        elif self.volatility is not None:
            sigma = float(self.volatility)
        else:
            sigma = 0.20

        sqrt_tau = math.sqrt(tau)
        K = self.strike
        # Per-path d1, d2; S_t is Tensor[N]
        d1 = (torch.log(S_t.clamp(min=1e-12) / K) + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        ncdf = torch.special.ndtr

        if self.is_payer:
            payoff = S_t * ncdf(d1) - K * ncdf(d2)
        else:
            payoff = K * ncdf(-d2) - S_t * ncdf(-d1)

        return A_t * payoff

    def _forward_swap_rate(self, market, t: float, proj, disc) -> Tuple[Tensor, Tensor]:
        """Forward swap rate S_t and PVBP/annuity A_t — both Tensor[N]."""
        schedule = self.swap.schedule

        # Resolve N from the first DF query (Tensor or float)
        sample = proj.df(t) if t > 0 else proj.df(schedule[0])
        if isinstance(sample, Tensor):
            N = int(sample.shape[0])
            dtype = sample.dtype
            device = sample.device
        else:
            N = 1
            dtype = torch.float32
            device = torch.device("cpu")

        def ensure(x) -> Tensor:
            if isinstance(x, Tensor):
                return x.to(dtype=dtype, device=device)
            return torch.full((N,), float(x), dtype=dtype, device=device)

        # P(t, T_i): use a single curve (proj) for the forward
        # P(t, T_i) = P(0, T_i) / P(0, t) for forward DF at t for maturity T_i.
        df_t = ensure(proj.df(t)) if t > 0 else None

        P_t = {}
        for T_i in schedule:
            if T_i <= t + 1e-12:
                P_t[T_i] = torch.ones(N, dtype=dtype, device=device)
            else:
                df_Ti = ensure(proj.df(T_i))
                if df_t is not None:
                    P_t[T_i] = df_Ti / df_t.clamp(min=1e-12)
                else:
                    P_t[T_i] = df_Ti

        # PVBP / annuity uses the *discount* curve
        annuity = torch.zeros(N, dtype=dtype, device=device)
        for i in range(1, len(schedule)):
            dcf = schedule[i] - schedule[i - 1]
            if schedule[i] <= t + 1e-12:
                continue
            df_d_Ti = ensure(disc.df(schedule[i]))
            df_d_t = ensure(disc.df(t)) if t > 0 else torch.ones(N, dtype=dtype, device=device)
            P_d_t_Ti = df_d_Ti / df_d_t.clamp(min=1e-12)
            annuity = annuity + dcf * P_d_t_Ti

        S_t = (P_t[schedule[0]] - P_t[schedule[-1]]) / annuity.clamp(min=1e-12)
        return S_t, annuity

    def __repr__(self) -> str:
        return (
            f"<Swaption({'payer' if self.is_payer else 'receiver'}, "
            f"K={self.strike}, expiry={self.expiry})>"
        )


__all__ = [
    "Swaption",
]
