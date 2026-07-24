"""forward_start.py — product-agnostic forward-start wrapper.

A hedge leg is only ever valued at ``t >= T_start`` (opened at the reset, closed
one step later), so the entire pre-reset regime is dead weight.  ``ForwardStart``
therefore collapses "forward-start" to a single product-agnostic operation:

    at the reset date, set the contract level to ``alpha * underlier.spot(T_start)``
    and (for path-dependent products) start the monitoring window there; then price
    the ordinary product at ``t``.

This is identical for a European, a barrier, or a lookback — so the same wrapper
drives all of them, and is the hedging half of ``EuropeanForwardOption`` (which is
kept as a *product* for forward-start *targets*, where the ``t < T0`` branch matters).

The level is read FRESH on every call (never bound at construction), so the wrapper
is re-simulation-safe: the SDESystem may resimulate between epochs and the leg
re-prices against the new paths.  The inner product is produced by a factory, so
there is no shared mutable strike to corrupt.

The only assumption on the underlier is ``spot(t) -> Tensor[N]`` — satisfied by
``Stock`` and ``ZeroCouponBond`` today and by any future rate underlier (the factory
decides what ``level`` means: an equity strike, a cap rate, a swaption strike, …).
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative


class ForwardStart(BaseDerivative):
    """Rebase any level-based product to start at ``T_start``.

    Parameters
    ----------
    make_inner : callable ``(level: Tensor[N], t0: float) -> BaseDerivative``
        Builds the configured inner product (strike/barrier/level = ``level``,
        maturity ``t0 + tenor``, ``window_start = t0`` for path-dependent ones).
    underlier  : object exposing ``spot(t) -> Tensor[N]``.
    alpha      : moneyness multiplier — ``level = alpha * underlier.spot(T_start)``.
    T_start    : reset date (strike/window set here).
    tenor      : life of the leg; maturity is ``T_start + tenor``.
    cost       : transaction cost fraction carried by the leg.
    """

    def __init__(
        self,
        make_inner: Callable[[Tensor, float], BaseDerivative],
        underlier,
        alpha: float,
        T_start: float,
        tenor: float,
        *,
        marketkeys=None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(underlier, T_start + tenor, marketkeys, cost)
        self.make_inner = make_inner
        self.alpha = alpha
        self.T_start = T_start
        self.tenor = tenor
        self.observation_dates = [T_start, T_start + tenor]

    def _leg(self) -> BaseDerivative:
        # Tensor[N], fresh each call
        level = self.alpha * self.underlier.spot(self.T_start)   
        return self.make_inner(level, self.T_start)

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        # A hedge leg is never valued before its start; clamp guards stray calls.
        return self._leg()._pricer(market, max(float(t), self.T_start))

    def value(self, t: float, market) -> Tensor:
        """Hedgeable point value at ``t`` (``t >= T_start``).  Tensor[N]."""
        return self._pricer(market, t)

    def mc_payoff(self) -> Tensor:
        return self._leg().mc_payoff()

    def __repr__(self) -> str:
        return f"<ForwardStart(alpha={self.alpha}, T_start={self.T_start}, tenor={self.tenor})>"


__all__ = ["ForwardStart"]