"""nn/feature.py — context-bound hedging features.

A feature pulls exactly the columns it needs from a :class:`FeatureContext`
(market + target + current time/step + previous holdings).  Because everything is
read off the context rather than a fixed positional signature, adding a feature
that needs a new state variable (a forward rate, a survival probability, a second
asset) never changes any other feature's signature — the hedger stays clean and
the same feature list works for equity *or* rate underliers.

State-variable park
-------------------
Because the hedger feeds the *level* of the hedging instruments to the P&L but
**not** to the model (the policy never sees an instrument price), the model can
only be Markov if every driving state variable is exposed as a feature.  The
feature park therefore spans the full state of an :class:`SDESystem`:

    equity level            -> Spot / Moneyness
    short rate / disc. rate -> ShortRate (per-path under a stochastic curve)
    stochastic volatility   -> InstantaneousVol (sqrt of the Heston variance v_t)
    stochastic hazard       -> HazardRate / Survival (per-path under a CIR hazard)
    inventory (for costs)   -> PrevHoldings

Each state feature degrades gracefully: under a flat regime it returns the
deterministic constant (so the *same* feature list trains in every regime, and a
regime that lacks a factor simply contributes a constant column that the input
normaliser collapses to zero).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math

import torch
import weakref
from torch import Tensor

from qabit.core.market.curve import MarketKeys
from qabit.tools.util import as_column, nearest_index


@dataclass
class FeatureContext:
    """Everything a feature might read at one rebalance step."""

    market: object  # qabit Market (discount / vol / hazard curves)
    target: object  # the BaseDerivative being hedged (the liability)
    t: float  # current rebalance time
    step: int  # rebalance index
    prev_h: Tensor  # previous holdings (N, H)
    keys: MarketKeys = field(
        default_factory=MarketKeys
    )  # which market curves features read (discount / vol / hazard)

    # ── primary state accessors ─────────────────────────────────────────
    def _underlier(self):
        return self.target.underlier

    def _state(self):
        """Underlier's :class:`FactorState`.

        Uniform across a :class:`Stock`, a :class:`Fund` and a raw factor view —
        all expose ``get_state()`` (``Stock`` via the alias on
        :class:`PrimaryInstrument`).
        """
        return self._underlier().get_state()

    def spot(self) -> Tensor:
        """Per-path spot of the active underlier, shape ``(N,)``.

        Delegates to the underlier's own ``spot(t)`` — uniform across
        :class:`Stock`, :class:`Fund` and any other single-factor lens (they all
        expose it), so there is no type branch here.
        """
        return self._underlier().spot(self.t)

    def variance(self) -> Optional[Tensor]:
        """Per-path instantaneous variance ``v_t`` if the underlier carries one
        (Heston exposes ``[S, V]``; column 1).  ``None`` for a single-output (GBM)
        underlier, so callers fall back to a deterministic vol."""
        p = self._state().at(self.t)
        return p[:, 1] if (p.dim() == 2 and p.shape[1] >= 2) else None

    @property
    def tau(self) -> float:
        return max(self.target.maturity - self.t, 1e-8)

    @property
    def n_paths(self) -> int:
        return self.spot().shape[0]


class Feature(ABC):
    n_out: int = 1

    #: Target attributes this feature needs.  Empty for generic state features
    #: (``Spot``, ``ShortRate``, …) that read only the market/underlier;
    #: product-specific features list the hooks they rely on (e.g.
    #: ``("living_floor",)`` for ``GuaranteeMoneyness``).  Checked by
    #: :meth:`validate`, which raises when a feature is used on a target that
    #: cannot supply them.
    requires: Tuple[str, ...] = ()

    #: When ``True``, :meth:`validate` additionally asserts that the target
    #: actually *selects* a hazard curve — ``target.marketkeys.hazard is not
    #: None`` — which is the real precondition for a mortality feature.  Because
    #: we hold the target (and therefore its key object), this checks the live
    #: selection rather than the mere presence of some attribute name.
    requires_hazard: bool = False

    @abstractmethod
    def of(self, ctx: FeatureContext) -> Tensor:
        """Return this feature's columns, shape (N, n_out)."""

    def validate(self, target) -> None:
        """Raise ``ValueError`` if ``target`` cannot supply what this feature
        needs: any attribute named in :attr:`requires`, and — when
        :attr:`requires_hazard` is set — a hazard curve selected on the target's
        ``marketkeys``."""
        for attr in self.requires:
            if not hasattr(target, attr):
                raise ValueError(
                    f"{type(self).__name__} requires target attribute '{attr}', "
                    f"which {type(target).__name__} does not provide."
                )
        if self.requires_hazard:
            keys = getattr(target, "marketkeys", None)
            if keys is None or keys.hazard is None:
                raise ValueError(
                    f"{type(self).__name__} is a mortality feature and needs a "
                    f"hazard curve selected on the target "
                    f"(target.marketkeys.hazard); {type(target).__name__} selects "
                    f"none."
                )


# ─────────────────────────────────────────────────────────────────────────────
# Equity-level features
# ─────────────────────────────────────────────────────────────────────────────


class Moneyness(Feature):
    def of(self, ctx):
        K = getattr(ctx.target, "strike", 1.0)
        return torch.log(ctx.spot() / K).unsqueeze(1)


class Spot(Feature):
    def of(self, ctx):
        return ctx.spot().unsqueeze(1)


class TimeToMaturity(Feature):
    def of(self, ctx):
        s = ctx.spot()
        return torch.full(
            (s.shape[0], 1), float(ctx.tau), dtype=s.dtype, device=s.device
        )


# ─────────────────────────────────────────────────────────────────────────────
# Volatility-state features
# ─────────────────────────────────────────────────────────────────────────────


class Volatility(Feature):
    """Implied vol used at this step (constant across paths under a flat surface).

    Reads the surface selected by ``ctx.keys.volatility`` (so *which* surface is
    explicit, not "the first one that happens to be in the market").  With no
    surface configured it degrades to a zero column — collapsed by the input
    standardizer — so the same feature list is safe on a target priced off a
    purely stochastic-vol regime with no separate implied surface.
    """

    def of(self, ctx):
        s = ctx.spot()
        vol = ctx.keys.volatility_surface(ctx.market)
        if vol is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        K = getattr(ctx.target, "strike", 1.0)
        return as_column(float(vol.vol(float(K), ctx.tau)), s)


class InstantaneousVol(Feature):
    r"""Per-path instantaneous volatility — the stochastic-vol state feature.

    Returns ``sqrt(v_t)`` from a Heston underlier, where ``v_t`` is the
    *instantaneous variance* state (``[S, V]`` column 1).  ``sqrt(v_t)`` is an
    annualized volatility (~0.2), the **same scale** as the flat implied vol
    ``sigma`` — *not* a per-step quantity like ``sigma·sqrt(dt)`` — so the column
    has one consistent meaning across regimes.  Under a single-factor (GBM)
    underlier there is no variance state, so it falls back to the implied vol of
    the surface selected by ``ctx.keys.volatility`` (and to ``0`` if no surface
    is configured): informative exactly when vol is stochastic, a harmless
    constant otherwise.
    """

    def of(self, ctx):
        s = ctx.spot()
        v = ctx.variance()
        if v is not None:
            return torch.sqrt(v.clamp(min=0.0)).to(s.dtype).reshape(-1, 1)
        vol = ctx.keys.volatility_surface(ctx.market)
        if vol is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        K = getattr(ctx.target, "strike", 1.0)
        return as_column(float(vol.vol(float(K), ctx.tau)), s)


# ─────────────────────────────────────────────────────────────────────────────
# Rate-state features
# ─────────────────────────────────────────────────────────────────────────────


class ShortRate(Feature):
    r"""Per-path short rate ``r_t`` (the rate / discount-curve state).

    Reads the instantaneous forward off the discount curve selected by
    ``ctx.keys.discount``: per-path ``(N,)`` under a
    :class:`StochasticDiscountCurve`, a broadcast constant under a flat
    :class:`DiscountCurve`.  This is the single feature that carries the entire
    rate state — there is no bond among the (memoryless) hedging instruments, so
    rate risk is only controllable if the policy can *see* the rate here.  With
    no discount curve configured it degrades to a zero column.
    """

    def of(self, ctx):
        s = ctx.spot()
        disc = ctx.keys.discount_curve(ctx.market)
        if disc is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        return as_column(disc.inst_fwd(ctx.t), s)


# ─────────────────────────────────────────────────────────────────────────────
# Hazard-state features (for variable annuities / mortality-linked targets)
# ─────────────────────────────────────────────────────────────────────────────


class HazardRate(Feature):
    r"""Per-path hazard intensity ``lambda_t`` — the mortality state feature.

    Under a :class:`StochasticHazardCurve` this is ``(N,)`` per path and carries
    the mortality state that drives a GMxB rider's value (the channel behind
    *mortality vega*).  With no hazard curve in the market it returns ``0`` — a
    constant column the normaliser collapses, so the same feature list is safe on
    purely financial targets.
    """

    def of(self, ctx):
        s = ctx.spot()
        hz = ctx.keys.hazard_curve(ctx.market)
        if hz is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        return as_column(hz.hazard_rate(ctx.t), s)


class Survival(Feature):
    r"""Per-path survival probability ``S(t) = exp(-\int_0^t lambda\,du)``.

    Companion to :class:`HazardRate`; some riders depend on the survival level
    more directly than on the instantaneous intensity.  Returns ``1`` when no
    hazard curve is present.
    """

    def of(self, ctx):
        s = ctx.spot()
        hz = ctx.keys.hazard_curve(ctx.market)
        if hz is None:
            return torch.ones((s.shape[0], 1), dtype=s.dtype, device=s.device)
        return as_column(hz.survival(ctx.t), s)


# ─────────────────────────────────────────────────────────────────────────────
# Liability-state features — "is the liability still alive?"
#
# These flags make the policy Markov for targets whose liability can *die*
# mid-horizon (a knock-out barrier, a mortality-contingent GMxB).  Both are
# F_t-measurable by construction: the breach flag is a function of the path up
# to t, and the death flag reproduces the SAME drawn tau the pricer uses (via
# the product's fixed ``seed``), evaluated only through ``tau <= t``.  Do NOT be
# tempted to feed the underlying Exp(1) threshold (or tau itself) instead — that
# reveals the *future* death/breach time and is lookahead.
# ─────────────────────────────────────────────────────────────────────────────


class BarrierBreached(Feature):
    """1/0 flag — has the target's barrier been breached on [window_start, t]?

    Reads ``barrier`` / ``barrier_type`` / ``window_start`` off the target; on a
    target without a barrier it returns a zero column (collapsed by the input
    standardizer).  The running extreme is cumulative in t, so it is cached per
    simulated state and merely indexed at each rebalance step.
    """

    def __init__(self):
        self._sref = None  # weakref: object ids are recycled
        self._cum = None  # (N, E) running max or min over the window

    def of(self, ctx):
        s = ctx.spot()
        B = getattr(ctx.target, "barrier", None)
        if B is None:
            return torch.zeros((s.shape[0], 1), dtype=s.dtype, device=s.device)
        st = ctx._state()
        if self._sref is None or self._sref() is not st:
            p = st.paths
            S = p[:, :, 0] if p.dim() == 3 else p
            i0 = int(
                torch.searchsorted(
                    st.dates,
                    torch.tensor(
                        float(getattr(ctx.target, "window_start", 0.0)),
                        dtype=st.dates.dtype,
                    ),
                )
            )
            ext = torch.full_like(S, float("-inf"))
            if ctx.target.barrier_type in ("dao", "dai"):
                ext[:, i0:] = torch.cummin(S[:, i0:], dim=1).values
            else:
                ext[:, i0:] = torch.cummax(S[:, i0:], dim=1).values
            self._cum, self._sref = ext, weakref.ref(st)
        idx = (
            int(
                torch.searchsorted(
                    st.dates, torch.tensor(ctx.t, dtype=st.dates.dtype), right=True
                )
            )
            - 1
        )
        idx = max(idx, 0)
        ext_t = self._cum[:, idx]
        if ctx.target.barrier_type in ("dao", "dai"):
            flag = ext_t <= float(ctx.target.barrier)
        else:
            flag = ext_t >= float(ctx.target.barrier)
        return flag.to(s.dtype).reshape(-1, 1)


class Deceased(Feature):
    """1/0 flag — has the policyholder died by t?  (tau <= t, per path.)

    Reproduces the liability's death times exactly: it draws ``tau`` from the
    hazard curve selected by ``ctx.keys.hazard`` with the *target's own seed*,
    which is the same call the GMxB pricer makes (``hazard.tau(seed=self.seed)``
    is deterministic given the simulated hazard state).  For the reproduction to
    hold, ``ctx.keys.hazard`` must select the same curve the target is priced
    under (the hedger derives ``ctx.keys`` from ``target.marketkeys`` by default —
    the mortality analogue of the discount-key-must-match contract, now structural).
    Only the indicator ``tau <= t`` is exposed, so the feature is F_t-measurable.
    """

    requires_hazard = True

    def __init__(self):
        self._sref = None  # weakref: object ids are recycled
        self._tau = None

    def of(self, ctx):
        s = ctx.spot()
        hz = ctx.keys.hazard_curve(ctx.market)
        st = hz.process.get_state()
        if self._sref is None or self._sref() is not st:
            self._tau = hz.tau(seed=getattr(ctx.target, "seed", None))
            self._sref = weakref.ref(st)
        return (self._tau <= ctx.t).to(s.dtype).reshape(-1, 1)


class AnniversaryMax(Feature):
    r"""Ratchet state: log of the running anniversary high-water mark over spot.

    For a ratchet/step-up guarantee the locked-in floor is the max of the
    (fee-adjusted) account at past anniversaries, so the policy is only Markov
    if it can see that high-water mark.  Returns
    ``log( max_{a<=t, a anniversary} S_a e^{-alpha a} / (S_t e^{-alpha t}) )`` —
    i.e. the locked guarantee level relative to the current account, in log
    space.  Anniversary dates come from ``target.anniversary_schedule`` (mapped
    to grid nodes with the same ``nearest_index`` the pricer's forward pass uses);
    the fee ``alpha`` is read off the target (0 if absent).
    """

    requires = ("anniversary_schedule",)

    def of(self, ctx):
        s = ctx.spot()
        st = ctx._state()
        p = st.paths
        S = p[:, :, 0] if p.dim() == 3 else p
        alpha = float(getattr(ctx.target, "alpha", 0.0))
        anns = [float(a) for a in ctx.target.anniversary_schedule if a <= ctx.t + 1e-9] 
        # \or [float(ctx.target.anniversary_schedule[0])]
        dates = st.dates.tolist()
        idxs = [nearest_index(dates, a) for a in anns]
        levels = torch.stack(
            [S[:, i] * math.exp(-alpha * a) for a, i in zip(anns, idxs)], dim=1
        )
        hwm = levels.max(dim=1).values
        cur = (s * math.exp(-alpha * ctx.t)).clamp(min=1e-12)
        return torch.log(hwm.clamp(min=1e-12) / cur).to(s.dtype).reshape(-1, 1)


class GuaranteeMoneyness(Feature):
    r"""Exact account/guarantee state: ``[log(A_t/G_t), d1]`` (n_out = 2).

    ``A_t`` and ``G_t`` come from the product's own anniversary recursion (the
    nested :class:`_StateCache`, which calls the target's ``_paths`` forward
    pass), so this is correct for fees, GMWB withdrawals and combined
    rollup·ratchet floors — unlike :class:`Moneyness` (raw spot over premium) or
    :class:`AnniversaryMax` (ratchet HWM only).  Because ``_paths`` lives on
    :class:`GMxBAnnuity` (not on :class:`BaseDerivative`), the cache is *owned by
    this feature* and only ever reached for a target that exposes it.  The second
    column is the normalized distance

        d1 = ( log(A/G) + (r - alpha + sigma^2/2) tau ) / (sigma sqrt(tau))

    with per-path ``r`` (stochastic discount, via ``ctx.keys.discount``) and
    per-path ``sigma`` (Heston) when available — pure inductive bias: the policy
    could compose it from existing columns, but converges much faster when handed
    the BS-normalized moneyness directly, and it is the natural x-axis for
    strategy plots.
    """

    n_out = 2
    requires = ("living_floor", "_paths")

    class _StateCache:
        """One product forward pass (``target._paths``) per simulated fund state.

        Shares the *exact* account/floor recursion with the pricer — fee, living
        benefit withdrawals and the (possibly combined rollup·ratchet) floor
        steps — so the features built on it can never drift from the liability's
        own state.  Anniversary dates are mapped to grid nodes with the same
        ``nearest_index`` the pricer's ``_paths`` uses, so the indices agree.
        """

        def __init__(self):
            self._sref = None  # weakref: object ids are recycled
            self._data = None

        def _refresh(self, ctx):
            st = ctx._state()
            if self._sref is not None and self._sref() is st:
                return
            paths = ctx.target._paths(ctx.market)
            anniv = [float(a) for a in paths["anniv"]]
            p = st.paths
            S = p[:, :, 0] if p.dim() == 3 else p
            dates = st.dates.tolist()
            idxs = [nearest_index(dates, a) for a in anniv]
            S_ann = [S[:, i].clamp(min=1e-12) for i in idxs]
            self._data, self._sref = (paths, anniv, S_ann), weakref.ref(st)

        def account_and_floor(self, ctx):
            """(A_t, G_t): account grown from the last anniversary's A_post, and
            the post-benefit living floor locked at that anniversary."""
            self._refresh(ctx)
            paths, anniv, S_ann = self._data
            k = 0
            for j, a in enumerate(anniv):
                if a <= ctx.t + 1e-9:
                    k = j
            s = ctx.spot()
            alpha = float(getattr(ctx.target, "alpha", 0.0))
            A = (
                paths["A_post"][k].to(s.dtype)
                * (s / S_ann[k].to(s.dtype))
                * math.exp(-alpha * (ctx.t - anniv[k]))
            )
            G = paths["GX_p"][k].to(s.dtype)
            return A, G

    def __init__(self):
        self._cache = self._StateCache()

    def of(self, ctx):
        s = ctx.spot()
        A, G = self._cache.account_and_floor(ctx)
        x = torch.log(A.clamp(min=1e-9) / G.clamp(min=1e-9)).clamp(-6.0, 6.0)
        # per-path sigma / r where the regime carries them
        v = ctx.variance()
        if v is not None:
            sig = torch.sqrt(v.clamp(min=1e-8)).to(s.dtype)
        else:
            vol = ctx.keys.volatility_surface(ctx.market)
            sig = (
                torch.full_like(s, float(vol.vol(1.0, ctx.tau)))
                if vol is not None
                else torch.ones_like(s)  # degenerate: no vol info → benign scale
            )
        disc = ctx.keys.discount_curve(ctx.market)
        r = disc.inst_fwd(ctx.t) if disc is not None else 0.0
        r = (
            torch.as_tensor(r, dtype=s.dtype, device=s.device).expand_as(s)
            if not torch.is_tensor(r)
            else r.to(s.dtype)
        )
        alpha = float(getattr(ctx.target, "alpha", 0.0))
        tau = ctx.tau
        d1 = (x + (r - alpha + sig * sig / 2.0) * tau) / (sig * math.sqrt(tau))
        return torch.stack([x, d1.clamp(-8.0, 8.0)], dim=1)


class AnniversaryPhase(Feature):
    """Time elapsed since the last anniversary, in years.  GMxB cashflows fix at
    the anniversary dates, so holdings should sawtooth across them — a smooth MLP
    of TimeToMaturity alone cannot express that; the phase column makes the
    within-period position explicit.  Uses ``target.anniversary_schedule``
    (``t - max{a in schedule : a <= t}``).
    """

    requires = ("anniversary_schedule",)

    def of(self, ctx):
        s = ctx.spot()
        sched = ctx.target.anniversary_schedule
        last = max((float(a) for a in sched if a <= ctx.t + 1e-9), default=float(sched[0]))
        return as_column(ctx.t - last, s)


class PendingDeathBenefit(Feature):
    """1/0 — policyholder has died but the death benefit has not yet been paid.

    Death between anniversaries settles at the *anniversary following death*, so
    the pending window is ``tau <= t < pay``, where ``pay`` is the smallest
    anniversary ``>= tau`` taken from ``target.anniversary_schedule`` (matching
    the pricer's ``(tau > anniv[k-1]) & (tau <= anniv[k])`` bucketing — *not* the
    integer ``ceil(tau)``, which is only correct for a yearly schedule).
    Distinguishes the two post-death regimes :class:`Deceased` alone conflates:
    *pending* (the liability still has fund delta until the A_pre fixing at the
    next anniversary) vs *settled* (zero).  Same measurable ``tau`` reconstruction
    as :class:`Deceased`, via ``ctx.keys.hazard``.
    """

    requires = ("anniversary_schedule",)
    requires_hazard = True

    def __init__(self):
        self._sref = None  # weakref: object ids are recycled
        self._tau = None

    def of(self, ctx):
        s = ctx.spot()
        hz = ctx.keys.hazard_curve(ctx.market)
        st = hz.process.get_state()
        if self._sref is None or self._sref() is not st:
            self._tau = hz.tau(seed=getattr(ctx.target, "seed", None))
            self._sref = weakref.ref(st)
        anns = torch.as_tensor(
            [float(a) for a in ctx.target.anniversary_schedule],
            dtype=self._tau.dtype, device=self._tau.device,
        )
        idx = torch.searchsorted(anns, self._tau).clamp(max=len(anns) - 1)
        pay = anns[idx]
        flag = (self._tau <= ctx.t) & (pay > ctx.t)
        return flag.to(s.dtype).reshape(-1, 1)


class PeriodReturn(Feature):
    """Reset-indexed state for cliquet-style targets: ``[period-to-date return,
    time to next reset, accumulated capped sum]`` (n_out = 3).

    A cliquet's spot delta lives *entirely* in the current period (future periods
    are forward-start, independent of ``S_t``) — but it is gated by the outer
    ``relu`` through the running sum ``Y`` of past capped returns: once
    ``Y + (periods left)·cap < 0`` the payout is unrecoverable and the true delta
    is zero.  Neither the period return nor ``Y`` is visible to any global
    feature (``Moneyness`` is ``log(S_t/K)``), so without this column the policy
    can only track them through its own ``PrevHoldings`` recursion.

    Why three columns in one feature (not three features): they are a *single*
    Markov state for this payoff — all read off the same reset-period boundaries
    in one pass (the period return drives the delta, ``Y`` plus the remaining cap
    gates it through the ``relu``, the time-to-reset locates the period), and
    they share the spot lookups at the period endpoints.  Splitting them would
    either recompute that schedule/spot work three times or need a shared cache,
    for no modelling gain, so they are kept cohesive.  Cliquet-specific: reads
    ``target.schedule`` and ``target.local_cap`` / ``target.local_floor``
    (defaults ±inf).
    """

    n_out = 3
    requires = ("schedule",)

    def of(self, ctx):
        s = ctx.spot()
        sched = [float(x) for x in ctx.target.schedule]
        cap = float(getattr(ctx.target, "local_cap", math.inf))
        flr = float(getattr(ctx.target, "local_floor", -math.inf))
        dates = [0.0] + sched
        last = max(d for d in dates if d <= ctx.t + 1e-9)
        nxt = min((d for d in dates if d > ctx.t + 1e-9), default=dates[-1])
        und = ctx.target.underlier
        ret = s / und.spot(last).clamp(min=1e-12) - 1.0
        past = [d for d in dates if d <= last + 1e-9]
        Y = torch.zeros_like(s)
        for a, b in zip(past[:-1], past[1:]):
            Y = Y + (und.spot(b) / und.spot(a).clamp(min=1e-12) - 1.0).clamp(flr, cap)
        return torch.stack([ret, torch.full_like(s, nxt - ctx.t), Y], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Inventory feature
# ─────────────────────────────────────────────────────────────────────────────


class PrevHoldings(Feature):
    """Previous holdings — the standard state feature for cost-aware hedging."""

    def __init__(self, n_inst: int = 1):
        self.n_out = n_inst

    def of(self, ctx):
        h = ctx.prev_h
        return h if h.dim() == 2 else h.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Liveness gates — "force the book flat once the liability is dead"
#
# These are NOT features (they are not fed to the model); they are F_t-measurable
# multiplicative gates in {0, 1} for ``DeepHedger(liveness=...)``.  Each one
# *reuses the very flag a feature already computes*, so the gate can never drift
# from the measurable state the policy is shown:
#
#   knock-out barrier     -> 1 - BarrierBreached          (dead once breached)
#   cliquet unrecoverable -> PeriodReturn's Y still recoverable
#   settled GMxB death    -> 1 - (Deceased & ~PendingDeathBenefit)
#
# Why a gate and not "let the network learn it": the dead cohort is a measure-zero
# afterthought to an aggregate risk loss (its residual instrument P&L barely moves
# the objective) and a binary flag gets smeared by the ReLU stack, so a smooth MLP
# parks the dead paths near — but never at — zero, leaving the tell-tale slope and
# spread.  The true delta there is *exactly* 0 by no-arbitrage, and the condition
# is measurable, so a hard gate is the correct closed-form structure, not a hack.
# ─────────────────────────────────────────────────────────────────────────────


class KnockOutLiveness:
    """``1`` while a knock-out target is live, ``0`` once its barrier is breached.

    Reuses :class:`BarrierBreached` verbatim (same cached running extreme, same
    ``F_t`` indexing), so the gate and the breach *feature* can never disagree.
    Only meaningful for knock-*outs* (``uao`` / ``dao``): a knock-*in* is not a
    liability that dies on breach — it is *born* on breach — so for those (and for
    a barrier-less target) the gate is all-ones and changes nothing.
    """

    def __init__(self):
        self._flag = BarrierBreached()

    def __call__(self, ctx: FeatureContext) -> Tensor:
        s = ctx.spot()
        bt = getattr(ctx.target, "barrier_type", None)
        if getattr(ctx.target, "barrier", None) is None or bt not in ("uao", "dao"):
            return torch.ones((s.shape[0], 1), dtype=s.dtype, device=s.device)
        return 1.0 - self._flag.of(ctx)


class CliquetLiveness:
    """``1`` while a capped cliquet's payout is still recoverable, ``0`` once it
    is not.

    The locally-capped sum ``Y`` of realised period returns can, at best, still
    grow by ``cap`` in each remaining period (including the current one), so the
    payout is unrecoverable exactly when ``Y + (periods_left + 1)·cap <= 0`` — at
    which point every remaining spot delta is gated off through the payoff's outer
    ``relu`` and the true delta is ``0``.  ``Y`` is read from the same
    :class:`PeriodReturn` state the policy already sees (column 2), so the gate
    matches the feature.  No-op (all-ones) on a target without a ``schedule`` or
    without a finite ``local_cap`` (an uncapped payout can always recover).
    """

    def __init__(self):
        self._state = PeriodReturn()

    def __call__(self, ctx: FeatureContext) -> Tensor:
        s = ctx.spot()
        cap = float(getattr(ctx.target, "local_cap", math.inf))
        if not hasattr(ctx.target, "schedule") or not math.isfinite(cap):
            return torch.ones((s.shape[0], 1), dtype=s.dtype, device=s.device)
        sched = [float(x) for x in ctx.target.schedule]
        Y = self._state.of(ctx)[:, 2]
        periods_left = sum(1 for d in sched if d > ctx.t + 1e-9)
        alive = (Y + (periods_left + 1) * cap > 0.0).to(s.dtype)
        return alive.reshape(-1, 1)


class MortalityLiveness:
    """``1`` while a GMxB still carries fund delta, ``0`` once the death benefit
    has *settled*.

    The subtle case: death between anniversaries does **not** immediately kill the
    fund hedge.  The death benefit settles at the *following* anniversary and is
    struck on the pre-fixing account ``A_pre`` (fund-linked), so during the pending
    window ``tau <= t < pay`` the liability still moves with the fund and the book
    must stay live.  The gate therefore zeroes only the *settled* cohort
    ``pay <= t``, i.e. ``Deceased & ~PendingDeathBenefit`` — which is exactly why
    both features exist.  Gating naively on :class:`Deceased` alone would unwind
    the hedge a whole period early and *create* error.  No-op without a hazard
    curve or an ``anniversary_schedule``.
    """

    def __init__(self):
        self._dead = Deceased()
        self._pending = PendingDeathBenefit()

    def __call__(self, ctx: FeatureContext) -> Tensor:
        s = ctx.spot()
        if ctx.keys.hazard_curve(ctx.market) is None or not hasattr(
            ctx.target, "anniversary_schedule"
        ):
            return torch.ones((s.shape[0], 1), dtype=s.dtype, device=s.device)
        dead = self._dead.of(ctx)
        pending = self._pending.of(ctx)
        settled = dead * (1.0 - pending)  # tau <= t AND pay <= t
        return 1.0 - settled
    

class RealisedVol(Feature):
    """Annualised realised vol of the underlier over ``[0, t]`` (running)."""

    def __init__(self, ann: float = 1.0) -> None:
        self.ann = ann

    def of(self, ctx: FeatureContext) -> Tensor:
        st = ctx._state()
        dates = st.dates
        i = int(torch.searchsorted(dates, torch.tensor(ctx.t, dtype=dates.dtype)).item())
        path = ctx._underlier().full_path()[:, : i + 1]
        if path.shape[1] < 3:
            return torch.zeros(path.shape[0], 1)
        logret = (path[:, 1:] / path[:, :-1].clamp_min(1e-9)).log()
        dt = float(dates[1] - dates[0]) if len(dates) > 1 else 1.0
        rv = logret.std(dim=1) / (dt ** 0.5) * (self.ann ** 0.5)
        return rv.nan_to_num().unsqueeze(1)


class MaxDrawdown(Feature):
    """Max peak-to-trough drawdown of the underlier over ``[0, t]`` (running)."""

    def of(self, ctx: FeatureContext) -> Tensor:
        st = ctx._state()
        dates = st.dates
        i = int(torch.searchsorted(dates, torch.tensor(ctx.t, dtype=dates.dtype)).item())
        path = ctx._underlier().full_path()[:, : i + 1]
        if path.shape[1] < 2:
            return torch.zeros(path.shape[0], 1)
        peak = path.cummax(dim=1).values
        dd = ((peak - path) / peak.clamp_min(1e-9)).max(dim=1).values
        return dd.nan_to_num().unsqueeze(1)


class FeatureSet:
    """Assemble the (N, total_dim) model input from a list of features."""

    def __init__(self, features: List[Feature]):
        self.features = features
        self.total_dim = sum(f.n_out for f in features)

    def build(self, ctx: FeatureContext) -> Tensor:
        return torch.cat([f.of(ctx) for f in self.features], dim=1)

    def names(self) -> List[str]:
        """Per-column feature names (handy for inspecting normaliser stats)."""
        out: List[str] = []
        for f in self.features:
            base = type(f).__name__
            out += [base] if f.n_out == 1 else [f"{base}[{i}]" for i in range(f.n_out)]
        return out

    def validate(self, target) -> None:
        """Validate every feature against ``target`` (see :meth:`Feature.validate`);
        raises ``ValueError`` on the first feature whose ``requires`` the target
        cannot satisfy."""
        for f in self.features:
            f.validate(target)


__all__ = [
    "FeatureContext",
    "Feature",
    "Moneyness",
    "Spot",
    "TimeToMaturity",
    "Volatility",
    "InstantaneousVol",
    "ShortRate",
    "HazardRate",
    "Survival",
    "BarrierBreached",
    "Deceased",
    "AnniversaryMax",
    "GuaranteeMoneyness",
    "AnniversaryPhase",
    "PendingDeathBenefit",
    "PeriodReturn",
    "PrevHoldings",
    "KnockOutLiveness",
    "CliquetLiveness",
    "MortalityLiveness",
    "FeatureSet",
]
