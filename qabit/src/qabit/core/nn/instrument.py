"""nn/instrument.py — the Hedgeable protocol and the two concrete wrappers.

The hedger consumes a list of :class:`Hedgeable`.  A Hedgeable's only job is to
produce, on the rebalancing grid, two aligned tensors ``(V_open, V_close)`` each
``(M, N)``: over holding step ``i`` you OPEN the position at ``V_open[i]`` (value
at ``t_i``) and CLOSE it at ``V_close[i]`` (value at ``t_{i+1}``).  For a fixed
instrument ``V_close[i] == V_open[i+1]`` so the per-step P&L telescopes to the
familiar ``sum_i h_i (V_{i+1} - V_i)``; for a rolling leg the two differ (you
switch legs at the reset).

Underlier-agnostic by construction: point values come from ``_pricer(market, t)``
when present (any ``BaseDerivative`` / ``ForwardStart``) else ``spot(t)`` (``Stock``,
``ZeroCouponBond``, and any future rate underlier).  So a caplet/swaption leg drops
in later with no change to the hedger.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor

from qabit.core.products.derivatives.forward_start import ForwardStart


@runtime_checkable
class Hedgeable(Protocol):
    cost: float

    def grids(self, dates: Tensor, market: object) -> Tuple[Tensor, Tensor]:
        """Return ``(V_open, V_close)``, each ``(M, N)``, M = len(dates) - 1."""
        ...


class FixedHedge:
    """Wrap a fixed instrument (underlier or fixed-strike derivative) as Hedgeable."""

    def __init__(self, inst, cost: Optional[float] = None) -> None:
        self.inst = inst
        self.cost = float(inst.cost if cost is None else cost)

    @staticmethod
    def value(inst, t: float, market) -> Tensor:
        """Undiscounted (value-at-t) of any instrument — derivative or underlier."""
        if hasattr(inst, "_pricer"):  # BaseDerivative / ForwardStart
            return inst._pricer(market, float(t))
        return inst.spot(float(t))  # Stock / ZCB / rate underlier

    def grids(self, dates: Tensor, market) -> Tuple[Tensor, Tensor]:
        vals = torch.stack(
            [self.value(self.inst, float(t), market) for t in dates]
        )  # (M+1, N)
        return vals[:-1], vals[1:]

    def __repr__(self):
        return f"<FixedHedge({self.inst!r}, cost={self.cost})>"


class ResetHedge:
    """A reset-indexed family of :class:`ForwardStart` legs (rolling ATM call / barrier / lookback).

    For holding step ``i`` the leg is ``ForwardStart(make_inner, underlier, alpha,
    T_start=dates[i], tenor)``; it is opened at ``dates[i]`` and closed at
    ``dates[i+1]``.  Each leg reuses an existing product pricer twice — no new
    analytics, no stored strike.
    """

    def __init__(
        self,
        make_inner: Callable[[Tensor, float], object],
        underlier,
        alpha: float,
        tenor: float,
        *,
        cost: float = 0.0,
    ) -> None:
        self.make_inner = make_inner
        self.underlier = underlier
        self.alpha = alpha
        self.tenor = tenor
        self.cost = float(cost)

    def grids(self, dates: Tensor, market) -> Tuple[Tensor, Tensor]:
        opens, closes = [], []
        for i in range(len(dates) - 1):
            t0 = float(dates[i])
            leg = ForwardStart(
                self.make_inner,
                self.underlier,
                self.alpha,
                t0,
                self.tenor,
                cost=self.cost,
            )
            opens.append(
                leg.value(float(dates[i]), market)
            )  # struck @ t_i, valued @ t_i
            closes.append(
                leg.value(float(dates[i + 1]), market)
            )  # struck @ t_i, valued @ t_{i+1}, aged Δt
        return torch.stack(opens), torch.stack(closes)

    def __repr__(self):
        return f"<ResetHedge(alpha={self.alpha}, tenor={self.tenor}, cost={self.cost})>"


__all__ = ["Hedgeable", "FixedHedge", "ResetHedge"]
