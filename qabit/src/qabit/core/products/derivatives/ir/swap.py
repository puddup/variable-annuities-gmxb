"""Interest rate swap — per-path pricing under stochastic rates."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.market.curve import MarketKeys


class InterestRateSwap(BaseDerivative):
    """Receiver IRS: fixed leg received, floating leg paid.

    Returns per-path PV under stochastic discount curves.

    Parameters
    ----------
    schedule     : list[float] — payment dates.
    fixed_rate   : float
    notional     : float
    marketkeys   : :class:`MarketKeys` — ``discount`` selects the OIS/CSA curve,
        ``forward`` the Libor projection curve (both read from the market's
        discount bag).  A swap has no spot underlier, so the base ``underlier``
        is ``None``.
    is_receiver  : bool
    """

    def __init__(
        self,
        schedule: List[float],
        fixed_rate: float,
        notional: float = 1.0,
        marketkeys: Optional[MarketKeys] = None,
        is_receiver: bool = True,
        cost: float = 0.0,
    ) -> None:
        super().__init__(None, max(schedule), marketkeys, cost)
        self.schedule = sorted(schedule)
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.is_receiver = is_receiver
        self.observation_dates = self.schedule

    def mc_payoff(self) -> Tensor:
        raise NotImplementedError(
            "InterestRateSwap has no path-wise payoff. Use price(market) directly."
        )

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        """PV at valuation date t.  Returns Tensor[N]."""
        if t != 0.0:
            raise NotImplementedError("IRS pricing for t>0 not yet implemented.")
        return self._pv(market)

    def _pv(self, market) -> Tensor:
        """Per-path present value of the swap."""
        disc = self.marketkeys.discount_curve(market)
        proj = self.marketkeys.forward_curve(market) or disc

        # Resolve N from the first available DF (Tensor or float)
        df0 = proj.df(self.schedule[0])
        if isinstance(df0, Tensor):
            N = int(df0.shape[0])
            dtype = df0.dtype
            device = df0.device
        else:
            N = 1
            dtype = torch.float32
            device = torch.device("cpu")

        def ensure(x) -> Tensor:
            if isinstance(x, Tensor):
                return x.to(dtype=dtype, device=device)
            return torch.full((N,), float(x), dtype=dtype, device=device)

        df_start = ensure(proj.df(self.schedule[0]))
        df_end = ensure(proj.df(self.schedule[-1]))
        floating_leg = self.notional * (df_start - df_end)

        fixed_leg = torch.zeros(N, dtype=dtype, device=device)
        for i in range(1, len(self.schedule)):
            dcf = self.schedule[i] - self.schedule[i - 1]
            df_i = ensure(disc.df(self.schedule[i]))
            fixed_leg = fixed_leg + self.notional * self.fixed_rate * dcf * df_i

        pv = floating_leg - fixed_leg if self.is_receiver else fixed_leg - floating_leg
        return pv

    def par_rate(self, market) -> float:
        from scipy.optimize import brentq

        def gap(K):
            self.fixed_rate = K
            return float(self._pv(market).mean().item())

        result = brentq(gap, -0.10, 0.50, xtol=1e-6)
        self.fixed_rate = result
        return result

    def __repr__(self):
        side = "rcvr" if self.is_receiver else "payer"
        return f"<IRS({side}, K={self.fixed_rate:.2%}, T={self.maturity})>"


__all__ = [
    "InterestRateSwap",
]
