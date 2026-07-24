"""nn/loss.py — polymorphic hedging losses (no branching in the hedger).

Sign convention: ``pnl_before_tc`` is the hedger's profit-and-loss before
transaction costs (hedge gains minus the option liability; higher = better);
``tc`` is transaction cost per path (>= 0).  Each loss decides for itself how to
combine the two, so the hedger never branches on a loss type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class HedgingLoss(ABC):

    name = "Loss(...)"

    @abstractmethod
    def compute(
        self, pnl_before_tc: Tensor, tc: Tensor, weights: Optional[Tensor] = None
    ) -> Tensor:
        ...

    @staticmethod
    def _w(x: Tensor, weights: Optional[Tensor]) -> Tensor:
        return weights if weights is not None else torch.ones_like(x) / len(x)


class VarianceLoss(HedgingLoss):
    """Var(P&L before TC) — symmetric; converges to the minimum-variance (BS) hedge."""

    name = "Var(P&L)"

    def compute(self, pnl_before_tc, tc, weights=None):
        w = self._w(pnl_before_tc, weights)
        wmean = (w * pnl_before_tc).sum()
        return (w * (pnl_before_tc - wmean) ** 2).sum()


class MSELoss(HedgingLoss):
    """E[(P&L before TC)^2] — penalises both bias and dispersion of the replication error."""

    name = "E[P&L^2]"

    def compute(self, pnl_before_tc, tc, weights=None):
        w = self._w(pnl_before_tc, weights)
        return (w * pnl_before_tc**2).sum()


class ExpectedShortfall(HedgingLoss):
    """ES at level ``alpha`` of the net LOSS (= tc - P&L) — tail-risk minimiser.

    The asymmetric tail penalty lets the optimiser keep favourable paths, which is
    why an ES hedge can be contaminated by speculative drift bets.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.name = f"ES({alpha})"

    def compute(self, pnl_before_tc, tc, weights=None):
        loss = tc - pnl_before_tc  
        w = self._w(loss, weights)
        idx = loss.argsort(descending=True)
        sorted_loss, sorted_w = loss[idx], w[idx]
        cum_w = sorted_w.cumsum(0)
        tail = cum_w <= self.alpha
        tail[0] = True  # at least one path in the tail
        return (sorted_loss * sorted_w * tail).sum() / sorted_w[tail].sum()


class JonesLoss(HedgingLoss):
    """Var(P&L before TC) + lambda * E(TC)  [Jones et al. 2025].

    Separates hedging risk from trading cost and — unlike ES — does not reward
    statistical arbitrage. ``lam`` is the capital charge per unit of expected cost.
    """

    def __init__(self, lam: float = 1.0 / 100.0):
        self.lam = lam
        self.name = f"Jones(lam={lam:g})"

    def compute(self, pnl_before_tc, tc, weights=None):
        w = self._w(pnl_before_tc, weights)
        wmean = (w * pnl_before_tc).sum()
        var_term = (w * (pnl_before_tc - wmean) ** 2).sum()
        return var_term + self.lam * (w * tc).sum()


class EntropicLoss(HedgingLoss):
    r"""Entropic risk measure — the **negative** OCE under exponential utility.

    For the exponential utility :math:`u(x)=(1-e^{-\lambda x})/\lambda` the
    optimized certainty equivalent has the closed form (the cash ``y`` is
    eliminated analytically by cash-invariance)

    .. math::
        U(X) \;=\; \sup_y \big\{\mathbb E[u(y+X)] - y\big\}
              \;=\; -\tfrac1\lambda \log \mathbb E\!\big[e^{-\lambda X}\big],

    so :math:`-U` is a convex risk measure.  With the net wealth
    :math:`X = \text{pnl} - \text{tc}` this loss returns

    .. math::
        \rho(X) \;=\; \tfrac1\lambda \log \mathbb E_w\!\big[e^{-\lambda X}\big],

    evaluated in a numerically stable ``logsumexp`` form under the (optionally
    re-weighted) sample measure ``weights``.  **Minimising** :math:`\rho` is
    exactly **maximising** the OCE.

    This is the objective of Proposition 2.3 / 3.4 of Buehler–Murray–Pakkanen–Wood:
    fitting a trading policy ``a`` against an *empty* liability under this loss
    finds the optimal statistical-arbitrage strategy :math:`a^\*`, whose wealth
    defines the minimal-entropy (near-)martingale density
    :math:`D^\* \propto e^{-\lambda(\,a^\*\!\cdot DH - M(a^\*))}`.  It is also a
    perfectly good *hedging* loss in its own right (utility-indifference pricing).
    """

    def __init__(self, lam: float = 1.0):
        if lam <= 0:
            raise ValueError("EntropicLoss risk-aversion lam must be > 0.")
        self.lam = float(lam)
        self.name = f"Entropic(lam={lam:g})"

    def compute(self, pnl_before_tc, tc, weights=None):
        x = pnl_before_tc - tc  # net wealth (higher = better)
        w = self._w(x, weights)
        # rho = (1/lam) * log( sum_i w_i exp(-lam x_i) )
        #     = (1/lam) * logsumexp_i( log w_i - lam x_i )
        log_w = torch.log(w.clamp_min(torch.finfo(x.dtype).tiny))
        return torch.logsumexp(log_w - self.lam * x, dim=0) / self.lam



class DownsideJonesLoss(HedgingLoss):
    r"""Semivariance of **net** P&L below its mean + :math:`\lambda\,E[TC]`.

    The parameter-free lower-tail inner: the left quantiles (P5/P25) are
    optimised through the object that drives them — downside dispersion —
    while upside dispersion is free, so the hedger is never punished for
    keeping favourable paths.  Unlike ES it needs no fitted level ``alpha``
    and, like :class:`JonesLoss`, it separates hedging risk from trading cost
    (``lam`` = capital charge per unit of expected cost).

    .. math::
        \rho(X) = \mathbb E_w\big[\big((X - \bar X)^-\big)^2\big]
                 + \lambda\, \mathbb E_w[\mathrm{TC}], \qquad X = \text{pnl} - \text{tc}.
    """

    def __init__(self, lam: float = 1.0 / 100.0):
        self.lam = float(lam)
        self.name = f"DownJones(lam={lam:g})"

    def compute(self, pnl_before_tc, tc, weights=None):
        net = pnl_before_tc - tc
        w = self._w(net, weights)
        w = w / w.sum()
        mu = (w * net).sum()
        d = (net - mu).clamp(max=0.0)
        return (w * d * d).sum() + self.lam * (w * tc).sum()


__all__ = [
    "HedgingLoss",
    "VarianceLoss",
    "MSELoss",
    "ExpectedShortfall",
    "JonesLoss",
    "DownsideJonesLoss",
    "EntropicLoss"
]
