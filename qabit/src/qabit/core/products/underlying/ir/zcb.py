"""Zero-coupon bond — backed by any short-rate process."""

from __future__ import annotations

from torch import Tensor

from qabit.core.products.underlying.base import PrimaryInstrument
from qabit.core.market.curve import (
    StochasticDiscountCurve as StochasticDiscountFactor,
)


class ZeroCouponBond(PrimaryInstrument):
    r"""Zero-coupon bond with maturity :math:`T_b`.

    Wraps any short-rate process — :class:`~core.dynamics.sde.ConstantRateFactor`
    or :class:`~core.dynamics.sde.VasicekRateFactor` — and builds a
    :class:`~core.market.curve.StochasticDiscountFactor` from it at construction
    time.  After simulation, the full discount curve interface is available via
    ``self.curve``:

    .. code-block:: python

        zcb.curve.df(5.)            # B(0, 5)       Tensor (N,)
        zcb.curve.fwd_df(1., 5.)    # B(1, 5)       Tensor (N,)
        zcb.curve.zero_rate(5.)     # yield to 5    Tensor (N,)
        zcb.curve.inst_fwd(2.)      # f(0, 2)       Tensor (N,)

    :meth:`spot` returns the path-wise forward discount factor

    .. math::
        P^h(t, T_b) = \frac{B^h(0, T_b)}{B^h(0, t)}
                    = \exp\!\Bigl(-\int_t^{T_b} r_s^h \, ds\Bigr)

    which is correct for both stochastic and deterministic rate processes.
    For :class:`~core.dynamics.sde.ConstantRateFactor` all paths are
    identical so the result degenerates to :math:`e^{-r(T_b - t)}`.

    Parameters
    ----------
    process    : SDESystem rate-factor view (exposes get_state())
        Short-rate process.  Any process whose paths represent :math:`r_t`.
    maturity   : float
        Bond maturity :math:`T_b` in years.
    init_state : float, optional
        Override the initial rate; forwarded to ``process.simulate``.
    cost       : float
        Transaction cost fraction.
    """

    def __init__(
        self,
        process,
        maturity: float,
        init_state: object = None,
        cost: float = 0.0,
    ) -> None:
        super().__init__(process, init_state, cost)
        self.maturity = maturity
        self.observation_dates = [maturity]
        # full discount curve interface backed by the same process
        self.curve = StochasticDiscountFactor(process)

    # ── spot: forward discount via curve ─────────────────────────────────────

    def spot(self, t: float) -> Tensor:
        r"""Path-wise :math:`P^h(t, T_b) = B^h(0, T_b) / B^h(0, t)`.

        Delegates entirely to
        :meth:`~core.market.curve.StochasticDiscountFactor.fwd_df`
        so caching and path-integral logic live in one place.
        """
        return self.curve.fwd_df(t, self.maturity)

    # ── price: expectation of B(0, T_b) ──────────────────────────────────────

    def price(self, market=None) -> float:
        r"""MC mean of :math:`B^h(0, T_b) = \exp(-\int_0^{T_b} r_s^h\,ds)`.

        ``market`` is accepted for interface compatibility with
        :class:`~core.portfolio.portfolio.Portfolio` but is intentionally
        ignored — the ZCB price is fully determined by its own process paths.
        """
        return self.curve.df(self.maturity)

    def __repr__(self) -> str:
        return f"<ZeroCouponBond(T={self.maturity}, {self.process!r})>"


__all__ = [
    "ZeroCouponBond",
]
