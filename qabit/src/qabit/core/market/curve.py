"""core/market/curve.py — Market curves: deterministic and stochastic.

Every curve type has a deterministic base and a stochastic subclass,
constructed symmetrically:

    DiscountCurve.flat(r)          →  StochasticDiscountCurve(process)
    ForwardCurve.flat(F0)          →  StochasticForwardCurve(process)
    HazardCurve.flat(lambda0)      →  StochasticHazardCurve(process)
    VolSurface.flat(sigma, ...)    →  (deterministic only — requires calibration)

All stochastic variants share the same interface as their deterministic
counterpart so :class:`~core.market.market.Market` holds either transparently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import weakref
from torch import Tensor

from qabit.tools.interpolation import (
    log_linear,
    linear,
    integrate_trapezoid_paths,
    to_tensor,
    maybe_scalar,
)


# ─────────────────────────────────────────────────────────────────────────────
# Discount
# ─────────────────────────────────────────────────────────────────────────────


class DiscountCurve:
    """Deterministic OIS/CSA discount curve — pillar interpolation.

    Distinct from :class:`ForwardCurve`: post-2008 multi-curve framework
    requires separate curves for projection (Libor) and discounting (OIS).

    Parameters
    ----------
    maturities       : Tensor (M,) — sorted pillar maturities.
    discount_factors : Tensor (M,) — corresponding DF values in ``(0, 1]``.

    Examples
    --------
    >>> dc = DiscountCurve.flat(0.05)
    >>> dc.df(5.)                         # 0.7788...
    >>> dc.df()                           # Tensor (M,) — all pillars
    >>> dc.fwd_df(1., 5.)                 # df(5) / df(1)
    >>> dc.shift(1e-4).df(5.)            # bumped copy for DV01
    """

    is_flat: bool = True  # deterministic curve — no path dimension

    def __init__(self, maturities: Tensor, discount_factors: Tensor) -> None:
        self.maturities = maturities.double()
        self.discount_factors = discount_factors.double()

    @classmethod
    def flat(cls, r: float, t0: float = 0.0, T: float = 100.0) -> "DiscountCurve":
        """Flat continuously-compounded curve at rate ``r`` over ``[t0, T]``."""
        mats = torch.tensor([t0, T], dtype=torch.float64)
        return cls(mats, torch.exp(-r * mats))

    def df(self, t: Optional[float] = None) -> Union[float, Tensor]:
        """Discount factor(s).  ``None`` → all pillars ``Tensor (M,)``."""
        if t is None:
            return self.discount_factors
        return float(
            log_linear(
                self.maturities,
                self.discount_factors,
                torch.tensor([t], dtype=torch.float64),
            )
        )

    def fwd_df(self, t1: float, t2: float) -> float:
        return self.df(t2) / max(self.df(t1), 1e-12)

    def zero_rate(self, t: float) -> float:
        return -math.log(max(self.df(t), 1e-12)) / max(t, 1e-12)

    def inst_fwd(self, t: float, eps: float = 1e-5) -> float:
        return -math.log(max(self.fwd_df(t, t + eps), 1e-12)) / eps


class StochasticDiscountCurve:
    r"""Path-wise discount curve from a simulated short-rate process.

    .. math::
        B^h(0, t) = \exp\!\Bigl(-\int_0^t r_s^h\,ds\Bigr)

    Cache is keyed on ``id(process.state)`` — auto-invalidates on re-simulation.
    Works with any rate process: ``ConstantRate`` degenerates to deterministic.

    Parameters
    ----------
    process : SDESystem rate-factor view (or any object exposing get_state()).

    Examples
    --------
    >>> sdc = StochasticDiscountCurve(vas_process)
    >>> sdc.df(5.)       # Tensor (N,)
    >>> sdc.df()         # Tensor (N, E) — full matrix
    >>> sdc.fwd_df(1., 5.)
    """

    is_flat: bool = False  # stochastic — path-dependent

    def __init__(self, process) -> None:
        self.process = process
        self._cache: Optional[Tensor] = None  # (N, E)  ∫r ds
        self._state_ref = None  # weakref to the cached state (ids are recycled)

    def _ensure_cache(self) -> None:
        st = self.process.get_state()
        if self._state_ref is None or self._state_ref() is not st:
            self._cache = integrate_trapezoid_paths(st.dates, st.paths)
            self._state_ref = weakref.ref(st)

    def _integral_at(self, t: float) -> Tensor:
        """Per-path cumulative ∫₀ᵗ r ds, linearly interpolated between grid
        nodes (the old behaviour snapped to the *next* node, which made any
        off-grid query — e.g. a finite-difference ``inst_fwd`` — span a whole
        simulation step)."""
        self._ensure_cache()
        d = self.process.get_state().dates
        i = int(
            torch.searchsorted(
                d, torch.tensor(float(t), dtype=d.dtype), right=True
            ).clamp(max=len(d) - 1)
        )
        if i == 0:
            return self._cache[:, 0]
        t0, t1 = float(d[i - 1]), float(d[i])
        w = 0.0 if t1 == t0 else min(max((float(t) - t0) / (t1 - t0), 0.0), 1.0)
        return self._cache[:, i - 1] * (1.0 - w) + self._cache[:, i] * w

    def df(self, t: Optional[float] = None) -> Tensor:
        """``None`` → ``(N, E)`` full matrix;  ``float`` → ``(N,)``."""
        self._ensure_cache()
        if t is None:
            return torch.exp(-self._cache)
        return torch.exp(-self._integral_at(t))

    def fwd_df(self, t1: float, t2: float) -> Tensor:
        return self.df(t2) / self.df(t1).clamp(min=1e-12)

    def zero_rate(self, t: float) -> Tensor:
        return -torch.log(self.df(t).clamp(min=1e-12)) / max(t, 1e-12)

    def inst_fwd(self, t: float, eps: float = 1e-4) -> Tensor:
        """Per-path short rate ``r_t`` — read directly off the simulated rate
        path (F_t-measurable; a constant-rate factor degenerates to the flat
        rate).  The previous finite difference of the *node-snapped* integral
        returned the coming step's average rate divided by ``eps`` — wrong
        scale and a one-step lookahead."""
        st = self.process.get_state()
        p = st.paths
        r = p[:, :, 0] if p.dim() == 3 else p
        i = int(
            torch.searchsorted(
                st.dates, torch.tensor(float(t), dtype=st.dates.dtype)
            ).clamp(max=r.shape[1] - 1)
        )
        return r[:, i]


# ─────────────────────────────────────────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────────────────────────────────────────


class VolSurface:
    """Implied Black-Scholes volatility surface σ(K, T).

    Deterministic only — a stochastic vol surface requires calibration
    and lives outside the scope of this class.

    Parameters
    ----------
    strikes    : Tensor (S,) — sorted strikes.
    maturities : Tensor (M,) — sorted maturities.
    vols       : Tensor (S, M) — implied vols on the grid.

    Examples
    --------
    >>> vs = VolSurface.flat(0.20, K_min=50., K_max=200., T_max=5.)
    >>> vs.vol(100., 1.)          # 0.20
    >>> vs.shift(0.01).vol(100., 1.)  # 0.21
    """

    def __init__(self, strikes: Tensor, maturities: Tensor, vols: Tensor) -> None:
        self.strikes = strikes.double()
        self.maturities = maturities.double()
        self.vols = vols.double()

    @classmethod
    def flat(
        cls,
        sigma: float,
        K_min: float = 0.0,
        K_max: float = 1e6,
        T_min: float = 0.0,
        T_max: float = 100.0,
    ) -> "VolSurface":
        """Flat surface — constant ``sigma`` over ``[K_min, K_max] × [T_min, T_max]``."""
        K = torch.tensor([K_min, K_max], dtype=torch.float64)
        T = torch.tensor([T_min, T_max], dtype=torch.float64)
        return cls(K, T, torch.full((2, 2), sigma, dtype=torch.float64))

    def vol(self, K: float, T: float) -> float:
        """σ(K, T) via bilinear interpolation."""
        K_t = torch.tensor([K], dtype=torch.float64).clamp(
            self.strikes[0], self.strikes[-1]
        )
        T_t = torch.tensor([T], dtype=torch.float64).clamp(
            self.maturities[0], self.maturities[-1]
        )
        vol_at_T = torch.stack(
            [
                linear(self.strikes, self.vols[:, j], K_t)
                for j in range(len(self.maturities))
            ]
        ).squeeze(-1)
        return float(linear(self.maturities, vol_at_T, T_t))


# ─────────────────────────────────────────────────────────────────────────────
# Hazard
# ─────────────────────────────────────────────────────────────────────────────


class HazardCurve:
    """Deterministic hazard/survival curve — pillar interpolation.

    Can be built from pillars, from a flat rate, or from any
    a hazard function via :meth:`from_hazard`.  The underlying
    storage is always ``(maturities, survival_factors)`` with
    log-linear interpolation — works identically for Exponential,
    Gompertz, Weibull, or any other hazard-rate shape.

    Parameters
    ----------
    maturities       : Tensor (M,) — sorted maturities.
    survival_factors : Tensor (M,) — S(T) = exp(-∫₀ᵀ λ dt).

    Examples
    --------
    >>> hc = HazardCurve.flat(0.05)
    >>> hc.survival(10.)          # exp(-0.5) ≈ 0.6065
    >>> hc = HazardCurve.from_hazard(partial(gompertz_hazard, m=80, b=9.5), T=40.)
    >>> hc.hazard_rate(10.)       # Gompertz λ(10) via interpolation
    """

    is_flat: bool = True  # deterministic — no path dimension

    def __init__(self, maturities: Tensor, survival_factors: Tensor) -> None:
        self.maturities = maturities.double()
        self.survival_factors = survival_factors.double()

    @classmethod
    def flat(cls, lambda0: float, t0: float = 0.0, T: float = 100.0) -> "HazardCurve":
        """Flat (exponential) hazard at constant rate ``lambda0``."""
        mats = torch.tensor([t0, T], dtype=torch.float64)
        return cls(mats, torch.exp(-lambda0 * mats))

    @classmethod
    def from_hazard(
        cls,
        hazard_fn,
        T: float = 100.0,
        n_pillars: int = 500,
    ) -> "HazardCurve":
        """Build a deterministic hazard curve from an intensity function.

        Parameters
        ----------
        hazard_fn : callable(t) -> Tensor
            Deterministic intensity ``lambda(t)``, e.g. one of the
            :mod:`qabit.core.analytics.hazard.laws` functions bound with its
            parameters (``functools.partial(gompertz_hazard, m=80, b=9.5)``).
        T         : float — maximum maturity.
        n_pillars : int — number of interpolation nodes.
        """
        from qabit.core.analytics.hazard import survival as _survival

        mats = torch.linspace(0.0, T, n_pillars, dtype=torch.float64)
        sv = _survival(mats, hazard_fn(mats)).double()
        return cls(mats, sv)

    def survival(self, t) -> "float | Tensor":
        """S(t) — accepts scalar or Tensor."""
        t, scalar = to_tensor(t)
        return maybe_scalar(
            log_linear(self.maturities, self.survival_factors, t), scalar
        )

    def hazard_rate(self, t, eps: float = 1e-5) -> "float | Tensor":
        """λ(t) = -∂_t log S(t) via finite difference."""
        t, scalar = to_tensor(t)
        sv_up = log_linear(
            self.maturities,
            self.survival_factors,
            (t + eps).clamp(max=float(self.maturities[-1])),
        )
        sv = log_linear(self.maturities, self.survival_factors, t)
        result = -torch.log(sv_up / sv.clamp(min=1e-12)) / eps
        return maybe_scalar(result, scalar)

    def density(self, t) -> "float | Tensor":
        """f(t) = λ(t)·S(t)."""
        t, scalar = to_tensor(t)
        result = self.hazard_rate(t) * self.survival(t)
        return maybe_scalar(
            result if isinstance(result, Tensor) else torch.tensor(result),
            scalar,
        )


class StochasticHazardCurve:
    """Path-wise hazard curve from a simulated hazard process.

    All methods preserve the path dimension — no flattening.
    ``survival(t)`` returns ``(N,)`` per-path survival, not a scalar mean.
    Use ``.survival(t).mean()`` explicitly if you want the mean.

    Parameters
    ----------
    process : hazard factor view (anything exposing get_state())

    Examples
    --------
    >>> shc = StochasticHazardCurve(cir_process)
    >>> shc.survival(10.)          # Tensor (N,) — per-path survival
    >>> shc.survival(10.).mean()   # float — mean survival across paths
    """

    is_flat: bool = False  # stochastic — path-dependent

    def __init__(self, process) -> None:
        self.process = process
        self._sv_cache: Optional[Tensor] = None  # (N, E)
        self._state_ref = None  # weakref to the cached state (ids are recycled)

    def _ensure_cache(self) -> None:
        from qabit.core.analytics.hazard import pathwise_survival

        st = self.process.get_state()
        if self._state_ref is None or self._state_ref() is not st:
            self._sv_cache = pathwise_survival(st.dates, st.paths)  # (N, E)
            self._state_ref = weakref.ref(st)

    def tau(self, n_paths=None, seed: Optional[int] = None) -> Tensor:
        """Per-path death times via integrated-hazard inversion. Returns (N,).

        ``n_paths`` is accepted for interface parity with deterministic baselines
        but ignored — the simulated paths already fix the path count.
        """
        from qabit.core.analytics.hazard import sample_tau

        st = self.process.get_state()
        return sample_tau(st.dates, st.paths, seed=seed)

    def _idx(self, t: Tensor) -> Tensor:
        st = self.process.get_state()
        return torch.searchsorted(st.dates, t.to(st.dates.dtype)).clamp(
            0, len(st.dates) - 1
        )

    def survival(self, t) -> "float | Tensor":
        """Per-path S(t).  Scalar input → (N,); Tensor input → (N, len(t))."""
        self._ensure_cache()
        t, scalar = to_tensor(t)
        idx = self._idx(t)
        if scalar:
            return self._sv_cache[:, int(idx[0])]
        return self._sv_cache[:, idx]

    def survival_path(self, t: float) -> Tensor:
        """Per-path survival at ``t``, shape ``(N,)``."""
        return self.survival(t)

    def hazard_rate(self, t, eps: float = 1e-4) -> "float | Tensor":
        """Per-path λ(t) via finite difference on per-path survival.

        Returns ``(N,)`` for scalar ``t``; ``(N, len(t))`` for Tensor ``t``.
        """
        was_scalar = isinstance(t, (int, float))
        sv_t = self.survival(t)
        # For the shifted point, always pass as same type
        if was_scalar:
            sv_up = self.survival(t + eps)
        else:
            sv_up = self.survival(t + eps)
        result = -torch.log(sv_up / sv_t.clamp(min=1e-12)) / eps
        return result

    def density(self, t) -> "float | Tensor":
        """Per-path f(t) = λ(t)·S(t).

        Returns ``(N,)`` for scalar ``t``; ``(N, len(t))`` for Tensor ``t``.
        """
        return self.hazard_rate(t) * self.survival(t)


__all__ = [
    "DiscountCurve",
    "StochasticDiscountCurve",
    "VolSurface",
    "HazardCurve",
    "StochasticHazardCurve",
    "MarketKeys",
]


# ─────────────────────────────────────────────────────────────────────────────
# Curve selection keys
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketKeys:
    r"""Selection keys for reading curves out of a :class:`~core.market.market.Market`.

    A process that needs market data — a pricing function, a hedging feature —
    rarely needs the *curves themselves*; it needs to know *which* discount /
    volatility / hazard curve to pull from the market it is handed.  Threading
    three loose ``Optional[str]`` keys through every signature is noisy and easy
    to get out of order, so they travel together as one object::

        keys = MarketKeys(discount="ois", hazard="mort")
        r    = keys.discount_curve(market).inst_fwd(t)     # per-path short rate
        lam  = keys.hazard_curve(market).hazard_rate(t)    # per-path intensity

    Resolution policy
    -----------------
    Each accessor distinguishes a *deliberately-absent* curve from a
    *configured-but-missing* one:

    * key ``None`` **and** the market has no curve of that kind  ->  ``None``
      (graceful: the caller supplies its regime-appropriate default — a zero
      hazard column on a purely financial target, a flat implied vol, …).
    * key ``None`` but the market *does* carry curves of that kind  ->  the
      market's default (first) curve, exactly as ``market.get_*(None)`` behaves.
    * key an explicit name  ->  that curve, or a ``KeyError`` if it is absent.

    So a curve that was never requested degrades silently, while a curve the
    experiment *named* but the market does not provide raises at setup — the
    asymmetry the feature park needs (a zeroed column must never be substituted
    for a curve that was explicitly asked for).
    """

    discount: Optional[str] = None
    volatility: Optional[str] = None
    hazard: Optional[str] = None
    forward: Optional[str] = None

    @staticmethod
    def _resolve(category, key, getter):
        if key is None and not category:
            return None  # deliberately absent -> graceful degradation
        return getter(key)  # default (key None) or named; raises on a missing name

    def discount_curve(self, market):
        """The selected discount curve, or ``None`` if none is configured."""
        return self._resolve(
            getattr(market, "discount", None), self.discount, market.get_discount
        )

    def forward_curve(self, market):
        """The selected projection/forward curve, or ``None`` if none is configured.

        Multi-curve projection ("Libor") curves live in the same ``discount`` bag
        as the OIS/CSA discount curve, so this selector reads from there too — a
        swap that needs a separate projection leg names it via ``forward=``; with
        ``forward=None`` it resolves to the market's default discount curve (the
        single-curve regime), exactly as the old ``forward_key`` did.
        """
        return self._resolve(
            getattr(market, "discount", None), self.forward, market.get_discount
        )

    def volatility_surface(self, market):
        """The selected volatility surface, or ``None`` if none is configured."""
        return self._resolve(
            getattr(market, "volatility", None), self.volatility, market.get_volatility
        )

    def hazard_curve(self, market):
        """The selected hazard curve, or ``None`` if none is configured."""
        return self._resolve(
            getattr(market, "hazard", None), self.hazard, market.get_hazard
        )
