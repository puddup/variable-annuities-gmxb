"""Portfolio — coordinate simulation and pricing of a product collection."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from qabit.config import DEFAULT_DT
from qabit.core.dynamics.sde.system import SDESystem
from qabit.core.market.calendar import EventCalendar
from qabit.core.market.market import Market
from qabit.core.portfolio.result import PricingResult
from qabit.exceptions import StateError


class Portfolio:
    """Coordinate simulation and pricing of a product collection.

    Parameters
    ----------
    system : SDESystem
        The single multi-factor model driving every product's underlyings.
        Products hold dynamic views (``system["stock"]``) and read the simulated
        state after :meth:`simulate`.
    products : list
        Instruments exposing ``mc_payoff`` / ``price`` and an observation grid.
    """

    def __init__(self, system: SDESystem, products: list) -> None:
        self.system = system
        self.products = products
        self._calendar: Optional[EventCalendar] = None

    def simulate(
        self,
        n_paths,
        dt=DEFAULT_DT,
        *,
        seed=None,
        antithetic=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) -> "Portfolio":
        cal = EventCalendar.build(self.products, dt=dt)
        self._calendar = cal
        self.system.simulate(
            cal,
            n_paths,
            seed=seed,
            antithetic=antithetic,
            device=device,
            dtype=dtype,
        )
        return self

    def _require_calendar(self) -> None:
        if self._calendar is None:
            raise StateError(
                "Portfolio has not been simulated. "
                "→ call portfolio.simulate(n_paths, ...) before pricing."
            )

    def mc_payoff(self) -> Tensor:
        """Raw per-path payoffs ``(N, P)``."""
        self._require_calendar()
        return torch.stack([p.mc_payoff() for p in self.products], dim=1)

    def price(self, market: Market) -> "PricingResult":
        """Best pricer per product."""
        self._require_calendar()
        quotes = {
            f"{type(p).__name__}_{i}": float(p.price(market)[0].mean())
            for i, p in enumerate(self.products)
        }
        return PricingResult(
            quotes=quotes, products=self.products, calendar=self._calendar
        )


__all__ = ["Portfolio"]
