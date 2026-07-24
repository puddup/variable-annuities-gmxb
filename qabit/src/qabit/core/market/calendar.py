"""EventCalendar — union of fine grid and product observation dates."""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


class EventCalendar:
    """Sorted union of a uniform fine grid and all product observation dates.

    The fine grid ensures Euler discretisation error is bounded by ``dt``.
    Observation dates are guaranteed to be exact calendar entries — no
    snapping error, no off-by-one in ``searchsorted``.

    Parameters
    ----------
    dates : Tensor, shape ``(E,)``
        Sorted, deduplicated event dates.
    """

    def __init__(self, dates: Tensor):
        self.dates = dates

    @classmethod
    def build(
        cls,
        products: list,
        dt: float = 1 / 52,
    ) -> "EventCalendar":
        """Build calendar from a list of products.

        Parameters
        ----------
        products : list of product objects
            Each must expose ``maturity: float`` and
            ``observation_dates: Optional[list[float]]``.
            ``None`` means the product needs the full fine grid to its maturity.
        dt : float
            Fine-grid step size — upper bound on all Euler steps.
        """
        T = max(p.maturity for p in products)
        obs = []
        for p in products:
            if p.observation_dates is None:
                obs.append(np.arange(0.0, p.maturity + dt, dt))
            else:
                obs.append(np.atleast_1d(p.observation_dates))

        fine = np.arange(0.0, T + dt, dt)
        all_ = np.union1d(fine, np.concatenate(obs)) if obs else fine
        return cls(torch.as_tensor(all_, dtype=torch.float64))

    @classmethod
    def from_horizon(cls, T: float, dt: float = 1 / 52) -> "EventCalendar":
        """Build a uniform calendar from 0 to T with step dt.

        The simplest constructor — no products needed.

        Parameters
        ----------
        T  : float — horizon in years.
        dt : float — step size.

        Examples
        --------
        >>> cal = EventCalendar.from_horizon(50., dt=0.1)
        >>> cal = EventCalendar.from_horizon(5., dt=1/52)
        """
        dates = np.arange(0.0, T + dt, dt)
        return cls(torch.as_tensor(dates, dtype=torch.float64))

    def __len__(self) -> int:
        return len(self.dates)

    def index_of(self, t: float) -> int:
        """Exact index of date ``t`` via binary search."""
        return int(torch.searchsorted(self.dates, torch.tensor(t)).item())

    def indices_between(self, t1: float, t2: float):
        return self.index_of(t1), self.index_of(t2) + 1

    def as_tensor(self, dtype=torch.float32, device=torch.device("cpu")) -> Tensor:
        return self.dates.to(dtype=dtype, device=device)


__all__ = [
    "EventCalendar",
]
