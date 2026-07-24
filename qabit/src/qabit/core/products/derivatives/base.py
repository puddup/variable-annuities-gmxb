"""core/products/derivatives/base.py — Base derivative contract.

Pure per-path valuation, no flattening.

Every product implements:
    _pricer(market, t: float) → Tensor[N]   — single-date valuation
    _price_grid(market, dates: Tensor)      → Tensor[T_val, N]
    mc_payoff()              → Tensor[N]     — terminal payoff per path

The base class provides:
    price(market, t=...)     → always Tensor[T_val, N]
    _mc_price_zero(market)   → Tensor[N]     — discounted payoff at t=0,
                                                works for flat & stochastic curves.

Design rule (enforced): no method inside a product collapses the path
dimension.  All pricers return Tensor[N].  Subclasses provide their own
``_pricer`` — no fallback through MC inside the base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from qabit.core.market.curve import MarketKeys
from qabit.tools.util import as_dates, to_path


class BaseDerivative(ABC):
    """Abstract base for all derivative products.

    Subclasses must implement at least ``_pricer(market, t)`` returning
    ``Tensor[N]``.  The default ``_price_grid`` loops ``_pricer`` over
    valuation dates.  Subclasses may override ``_price_grid`` for more
    efficient grid computation.
    """

    observation_dates = None

    def __init__(
        self,
        underlier,
        maturity: float,
        marketkeys: "MarketKeys | None" = None,
        cost: float = 0.0,
    ) -> None:
        """Parameters
        ----------
        underlier   : the single :class:`~qabit.core.products.underlying.base.PrimaryInstrument`
            (``Stock`` / ``Fund`` / ``ZeroCouponBond`` / raw factor lens) the
            product is written on, or ``None`` for a curve-only product (an
            interest-rate swap or swaption reads the discount/projection curves
            straight off the market and has no spot underlier).
        maturity    : float
        marketkeys  : :class:`~qabit.core.market.curve.MarketKeys` naming *which*
            discount / volatility / hazard / forward curve the product pulls from
            the ``Market`` it is priced against.  Carried by the product itself,
            so no per-call ``discount_key`` / ``volatility_key`` is threaded
            through any pricer.  Defaults to an empty :class:`MarketKeys` (every
            selector ``None`` → the market's default curve of each kind).
        cost        : float — transaction-cost fraction (hedging extensions).
        """
        self.underlier = underlier
        self.maturity = maturity
        self.marketkeys = marketkeys if marketkeys is not None else MarketKeys()
        self.cost = cost

    # ── canonical price method ────────────────────────────────────────────────

    def price(self, market, t=None) -> Tensor:
        """Fair value at valuation date(s) t.  Always returns Tensor[T_val, N]."""
        dates = as_dates(t)
        return self._price_grid(market, dates)

    # ── pricer to be implemented by each product ──────────────────────────────

    @abstractmethod
    def _pricer(self, market, t: float) -> Tensor:
        """Fair value at single valuation date t, per path.  Must return Tensor[N]."""
        ...

    # ── default grid: loops _pricer over dates ────────────────────────────────

    def _price_grid(self, market, dates: Tensor) -> Tensor:
        """Tensor[T_val, N] — stacked per-path values for each valuation date."""
        return torch.stack([self._pricer(market, float(t_i)) for t_i in dates])

    # ── terminal payoff (for MC at t=0) ───────────────────────────────────────

    def mc_payoff(self) -> Tensor:
        """Terminal payoff per path.  Tensor[N].

        Default implementation raises — subclasses that have a single
        path-wise terminal payoff (vanilla, barrier, asian, …) override
        this.  Products with cashflow streams (swaps, variable annuities)
        do not implement it and rely on ``_pricer`` directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no single terminal payoff; use price(market) directly."
        )

    # ── optional convenience: MC price at t=0 with per-path discounting ───────

    def _mc_price_zero(self, market) -> Tensor:
        """Discounted terminal payoff at t=0 — works with flat & stochastic curves.

        The discount curve is selected by the product's own ``marketkeys`` (no
        ``discount_key`` is threaded in).  Returns Tensor[N] — no flattening.
        """
        disc = self.marketkeys.discount_curve(market)
        payoff = self.mc_payoff()  # Tensor[N]
        fwd_df = disc.fwd_df(0.0, self.maturity)  # float or Tensor[N]
        N = payoff.shape[0]
        fwd_df = to_path(fwd_df, N, payoff.dtype, payoff.device)
        return payoff * fwd_df


__all__ = [
    "BaseDerivative",
]
