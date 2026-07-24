"""PricingResult — quotes and bump-based Greeks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from qabit.core.market.market import Market
from qabit.core.market.calendar import EventCalendar


@dataclass
class PricingResult:
    """Output of ``Portfolio.price(market)``."""

    quotes: Dict[str, float]
    products: list
    calendar: EventCalendar

    def _reprice(self, market: Market) -> Dict[str, float]:
        return {k: p.price(market) for k, p in zip(self.quotes, self.products)}

    def dv01(self, market, key=None, bump=1e-4):
        raise NotImplementedError

    def delta(self, market, key=None, bump=1.0):
        raise NotImplementedError

    def vega(self, market, key=None, bump=0.01):
        raise NotImplementedError

    def summary(self) -> None:
        print(f"{'Product':<35} {'Price':>10}")
        print("─" * 47)
        for k, v in self.quotes.items():
            print(f"{k:<35} {float(v):>10.5f}")


__all__ = [
    "PricingResult",
]
