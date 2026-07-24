"""Cliquet option — MC only at t=0 (forward valuation requires conditional simulation)."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.underlying.equity.stock import Stock
from qabit.core.market.curve import MarketKeys


class CliquetOption(BaseDerivative):
    """Cliquet (ratchet) option. Pays sum of capped local returns, globally capped.

    Only t=0 supported.  t>0 requires conditional simulation.
    """

    def __init__(
        self,
        underlier: Stock,
        schedule: List[float],
        local_cap: float = float("inf"),
        local_floor: float = float("-inf"),
        global_cap: float = float("inf"),
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(underlier, max(schedule), marketkeys, cost)
        self.schedule = sorted(schedule)
        self.local_cap = local_cap
        self.local_floor = local_floor
        self.global_cap = global_cap
        self.observation_dates = self.schedule

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        spots = self.underlier.at_dates([0.0] + self.schedule)
        local_returns = spots[:, 1:] / spots[:, :-1].clamp(min=1e-12) - 1.0
        capped = local_returns.clamp(min=self.local_floor, max=self.local_cap)
        return torch.relu(capped.sum(dim=1).clamp(max=self.global_cap))

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        if t != 0.0:
            raise NotImplementedError("Cliquet pricing for t>0 requires conditional simulation.")
        return self._mc_price_zero(market)

    def __repr__(self):
        return f"<CliquetOption(T={self.maturity}, cap={self.local_cap})>"


__all__ = [
    "CliquetOption",
]
