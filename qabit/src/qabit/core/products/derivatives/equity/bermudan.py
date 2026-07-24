"""Bermudan option — LSMC pricing (stochastic-rate compatible)."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.underlying.equity.stock import Stock
from qabit.core.market.curve import MarketKeys
from qabit.core.analytics.common.lsmc import longstaff_schwartz
from qabit.tools.util import to_path


class BermudanOption(BaseDerivative):
    """Bermudan call or put — LSMC (Longstaff-Schwartz 2001).

    Supports stochastic discount curves via per-path discount factors.
    Only t=0 currently; t>0 requires conditional simulation.
    """

    def __init__(
        self,
        underlier: Stock,
        strike: float,
        exercise_dates: List[float],
        is_call: bool = False,
        basis_degree: int = 3,
        marketkeys: Optional[MarketKeys] = None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(underlier, max(exercise_dates), marketkeys, cost)
        self.strike = strike
        self.exercise_dates = sorted(exercise_dates)
        self.is_call = is_call
        self.basis_degree = basis_degree
        self.observation_dates = self.exercise_dates

    # ── payoff ────────────────────────────────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        S_T = self.underlier.spot(self.maturity)
        if self.is_call:
            return torch.relu(S_T - self.strike)
        return torch.relu(self.strike - S_T)

    # ── pricers ───────────────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        if t != 0.0:
            raise NotImplementedError("Bermudan pricing for t>0 requires conditional LSMC.")
        return self._lsmc_pricer(market)

    def _lsmc_pricer(self, market) -> Tensor:
        """LSMC at t=0. Returns Tensor[N]."""
        disc = self.marketkeys.discount_curve(market)
        S_full = self.underlier.full_path()
        sim_state = self.underlier._state()
        sim_dates = sim_state.dates.float()

        # Indices of exercise dates within the simulation grid
        steps = [
            int(torch.searchsorted(sim_dates, torch.tensor(te)).item())
            for te in self.exercise_dates
        ]
        paths_ex = S_full[:, steps]  # (N, M)

        def payoff_fn(S: Tensor) -> Tensor:
            if self.is_call:
                return torch.relu(S - self.strike)
            return torch.relu(self.strike - S)

        # Build discount-factor grid: (N, M) (broadcast from scalars when flat)
        N, M = paths_ex.shape
        dtype, device = paths_ex.dtype, paths_ex.device
        D = torch.stack(
            [to_path(disc.df(t), N, dtype, device) for t in self.exercise_dates],
            dim=1,
        )

        return longstaff_schwartz(
            paths_ex,
            self.exercise_dates,
            payoff_fn,
            discount_grid=D,
            degree=self.basis_degree,
        )

    def __repr__(self):
        return (
            f"<BermudanOption({'call' if self.is_call else 'put'}, "
            f"K={self.strike}, dates={len(self.exercise_dates)})>"
        )


__all__ = [
    "BermudanOption",
]
