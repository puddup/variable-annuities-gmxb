"""Equity underlying — Stock."""

from __future__ import annotations
from typing import Optional
from torch import Tensor
from qabit.core.products.underlying.base import PrimaryInstrument


class Stock(PrimaryInstrument):
    """Equity spot.  Handles GBM (scalar) and Heston ([S, V]) outputs."""

    observation_dates = None  # needs full path for lookback etc.

    def spot(self, t: float) -> Tensor:
        p = self._state().at(t)
        return p[:, 0] if p.dim() == 2 else p

    def full_path(self) -> Tensor:
        p = self._state().paths
        return p[:, :, 0] if p.dim() == 3 else p

    def var_path(self) -> Optional[Tensor]:
        p = self._state().paths
        return p[:, :, 1] if (p.dim() == 3 and p.shape[2] >= 2) else None

    def at_dates(self, dates: list) -> Tensor:
        p = self._state().at_dates(dates)
        return p[:, :, 0] if p.dim() == 3 else p  # Heston [S, V] → S

    def __repr__(self):
        return f"<Stock({self.process!r}), cost={self.cost}, init_state={self.init_state}>"


__all__ = [
    "Stock",
]
