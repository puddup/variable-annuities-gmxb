r"""nn/regret.py — regret-robust and *relative*-regret-robust hedging losses.

This module contains the robustification family built **on top of** the
Ambiguity-Averse Deep Hedger (AADH) of Jones et al. (2025), whose machinery
(partitioners, per-regime inner risks, entropic aggregation) lives in
:mod:`qabit.core.nn.adversial`.  The two modules are deliberately kept
separate: ``adversial.py`` is the faithful implementation of the published
AADH, ``regret.py`` is the extension family that generalises it.  Everything
here subclasses or composes :class:`~qabit.core.nn.adversial.AmbiguityAverseLoss`.

The family tree
---------------
Write :math:`R_i(\pi)` for the inner risk of policy :math:`\pi` in regime
(world) :math:`i`, :math:`\mu_i` for the regime masses, and :math:`R_i^*` for
the per-regime **achievable floor** (the irreducible risk of the best policy in
that regime — a per-world oracle, or the SDH itself for a do-no-harm floor).

=====================  ========================================================
loss                    objective
=====================  ========================================================
``AmbiguityAverse``     :math:`\rho_a(\{R_i\})` — entropic over **raw** risks
(AADH, adversial.py)    (:math:`a=0` ⇒ SDH; :math:`a\to\infty` ⇒ min-max).
``RegretRobust``        :math:`\rho_a(\{(R_i - R_i^*)_+\})` — the floor is
                        netted out; only *controllable* risk is aggregated.
                        ``baseline=None`` recovers the AADH **exactly**.
``RelativeRegret``      :math:`\sum_i \tilde\mu_i\,(R_i - R_i^*)_+` with the
                        floor-tilted masses
                        :math:`\tilde\mu_i \propto \mu_i / R_i^*` — every
                        regime competes in *percent of its achievable risk*.
``TiltedMixture``       :math:`\sum_i \tilde\mu_i\,R_i` — an SDH on the same
(:math:`A_w`)           floor-tilted mixture, but on **raw** risk (no floor
                        netting).  The attribution baseline: if the relative
                        regret arm only re-weights training effort, it will
                        match this arm.
``minmax`` constructor  :math:`\max_i R_i` (softmax-max, large :math:`a`,
                        zero baseline) — the group-DRO / worst-case baseline.
``Anchored``            :math:`L_{\text{robust}} + \varepsilon\,
                        L_{\text{inner}}` — the do-no-harm tie-break: when
                        regret is exhausted the arm degrades to the SDH.
=====================  ========================================================

Why regret instead of raw risk
------------------------------
The AADH tilts toward high-:math:`R_i` regimes.  That is the right move only
when regimes differ in **shape** — a different optimal hedge per regime.  On an
incomplete-market **size** spread — regimes sharing the same optimal hedge but
carrying different *irreducible* residual (jump gap risk, guarantee gap risk,
transaction-cost drag) — tilting toward the high-risk regime pours effort where
effort cannot reduce risk, only cost.  Netting out the floor,

.. math:: G_i(\pi) = \big(R_i(\pi) - R_i^*\big)_+ ,

removes the irreducible component:

* pure **size** spread → :math:`R_i \approx R_i^*` → :math:`G_i \approx 0` →
  the loss is inert (no over-trading);
* **shape** spread → :math:`G_i` large exactly in the mis-hedged regime → the
  loss reallocates capacity there.

Why *relative* regret
---------------------
Absolute regret still lets the large-scale regimes (high-σ worlds) monopolise
the gradient even when the small-scale regimes carry the largest **relative**
headroom (``rel_capacity`` in a gate table).  Dividing each regime's regret by
its own floor puts every regime on a percent-of-achievable scale.  For the
:math:`a=0` aggregation this is *exactly* a mass tilt,

.. math::
    \sum_i \mu_i \frac{(R_i - R_i^*)_+}{R_i^*}
    \;=\; \sum_i \tilde\mu_i\,(R_i - R_i^*)_+ ,
    \qquad \tilde\mu_i \propto \frac{\mu_i}{R_i^*},

which is how :class:`RelativeRegretLoss` implements it (floors in the tilt are
clipped below so a near-perfect regime cannot monopolise the gradient).

Floors are estimated quantities
-------------------------------
``regret = signal − baseline``: when the floor is estimated on finite samples
its noise can swamp the true controllable regret (the loss then chases noise —
the failure mode documented in ``docs/analysis/REGRET_VERDICT.md``).  Use
:func:`measure_floor_stats` (bootstrap SEs, paired common-random-number
measurement) and :func:`robust_floor` (James–Stein-style shrinkage plus a
noise margin) so that only regret exceeding its own estimation noise drives
training.

Usage sketch
------------
::

    stats = measure_floor_stats(sdh.hedger, gm, system, cal, rebal, market,
                                partitioner, n_paths=32_768, seed=42)
    floor = robust_floor(stats, shrink=0.1, margin_z=1.0)

    rr = RelativeRegretLoss.factors(gm, k=n_worlds, features=ORACLE_AXIS,
                                    inner=DownsideJonesLoss(), baseline=floor)
    loss = AnchoredLoss(rr, DownsideJonesLoss(), eps=0.10)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch
from torch import Tensor

from qabit.core.keys import SPOT
from qabit.core.nn.adversial import (
    AmbiguityAverseLoss,
    KMeansPartitioner,
    PathPartitioner,
    SourcedFeature,
)
from qabit.core.nn.loss import HedgingLoss, JonesLoss


# ─────────────────────────────────────────────────────────────────────────────
# Regret-robust: ambiguity aversion over (R_i − R_i*)_+
# ─────────────────────────────────────────────────────────────────────────────


class RegretRobustLoss(AmbiguityAverseLoss):
    r"""Ambiguity-averse over per-regime regret :math:`(R_i - R_i^*)_+`.

    Parameters
    ----------
    baseline
        The per-regime achievable floor :math:`R_i^*` as a length-``k`` tensor
        indexed by regime label (the labelling of the frozen partition).  Use
        :func:`measure_floor_stats` / :func:`robust_floor` to estimate it.
        ``None`` sets every floor to 0, and the loss is then **identical** to
        the plain AADH — the sense in which this class is a strict
        generalisation of :class:`AmbiguityAverseLoss`.
    relu
        ``True`` (default): only positive regret counts (beating the floor in
        an easy regime earns nothing).  ``False``: the raw gap
        :math:`R_i - R_i^*` is aggregated (a *centred* AADH).
    All other parameters are inherited from :class:`AmbiguityAverseLoss`.
    """

    def __init__(self,
                 *args,
                 baseline: Optional[Tensor] = None,
                 relu: bool = True,
                 **kwargs) -> None:
        kwargs.pop("underlier_key", None)  # deprecated, unused (kept for old call sites)
        super().__init__(*args, **kwargs)
        self.relu = bool(relu)
        self._baseline: Optional[Tensor] = None
        self.register_baseline(baseline)
        self.name = self.name.replace("AADH(", f"{type(self).__name__.replace('Loss','')}(")

    # ── the floor ─────────────────────────────────────────────────────────────

    def register_baseline(self, baseline: Optional[Tensor]) -> None:
        """Set/replace the per-regime floor :math:`R_i^*` (length-k, by label)."""
        if baseline is None:
            self._baseline = None
        else:
            b = torch.as_tensor(baseline, dtype=torch.get_default_dtype())
            if b.numel() != self.partitioner.k:
                raise ValueError(
                    f"baseline has {b.numel()} entries but the partition has "
                    f"{self.partitioner.k} regimes.")
            self._baseline = b
        self._on_floor_or_mass_change()

    def _on_floor_or_mass_change(self) -> None:
        """Hook for subclasses whose masses depend on the floor (relative regret)."""

    # ── aggregation: identical to the AADH up to netting out the floor ────────

    def _gaps(self, R: Tensor, keep: List[int]) -> Tensor:
        """Per-regime regret :math:`G_i` for the kept regimes."""
        G = R if self._baseline is None else R - self._baseline[keep].to(R)
        return torch.relu(G) if self.relu else G

    def compute(self, pnl_before_tc, tc, weights=None, *, path=None):
        R, mu_keep, keep = self._regime_risks(pnl_before_tc, tc, weights, path=path)
        if R is None:  # degenerate batch — fall back to the plain inner loss
            return self.inner.compute(pnl_before_tc, tc, weights=weights)
        G = self._gaps(R, keep)
        if self.a == 0.0:  # regime-averaged regret
            loss = (mu_keep * G).sum()
        else:              # entropic worst-case over regret (NOT raw risk)
            z = self.a * G + torch.log(mu_keep)
            loss = torch.logsumexp(z, dim=0) / self.a
        if self.diagnostics:
            self._record(loss, G, mu_keep, keep, risk=R)
        return loss

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def factors(cls,
                target,
                *,
                a: float = 0.0,
                k: int = 20,
                features: Sequence[SourcedFeature] = (),
                inner: Optional[HedgingLoss] = None,
                baseline: Optional[Tensor] = None,
                relu: bool = True,
                seed: int = 0,
                min_cluster: int = 2,
                diagnostics: bool = False,
                **kw) -> "RegretRobustLoss":
        """k-means regimes on ``(callable, source)`` features (see AADH docs)."""
        from qabit.core.analytics.math.stats import autocorrelation, realised_vol
        feats = tuple(features) or ((realised_vol, SPOT), (autocorrelation, SPOT))
        return cls(
            inner if inner is not None else JonesLoss(lam=1.0 / 100.0),
            KMeansPartitioner(feats, k=k, seed=seed),
            a=a,
            target=target,
            min_cluster=min_cluster,
            diagnostics=diagnostics,
            baseline=baseline,
            relu=relu,
            **kw,
        )

    @classmethod
    def minmax(cls,
               target,
               *,
               a: float = 1_000.0,
               k: int = 20,
               features: Sequence[SourcedFeature] = (),
               inner: Optional[HedgingLoss] = None,
               seed: int = 0,
               min_cluster: int = 2,
               diagnostics: bool = False) -> "RegretRobustLoss":
        r"""**Min-max SDH** — the zero-baseline, large-``a`` corner of the family.

        With ``baseline=None`` and ``relu=False`` the aggregated quantities are
        the raw per-regime risks, and for :math:`a` large relative to the risk
        spread the entropic aggregation is a softmax-max,

        .. math:: \tfrac1a \log \sum_i \mu_i e^{a R_i} \;\xrightarrow{a\to\infty}\; \max_i R_i,

        i.e. hedging the worst-case regime (group-DRO without floors).  Choose
        ``a`` so that ``a * (max R − min R) ≫ 1`` at the problem's risk scale.
        """
        return cls.factors(target, a=a, k=k, features=features, inner=inner,
                           baseline=None, relu=False, seed=seed,
                           min_cluster=min_cluster, diagnostics=diagnostics)


# ─────────────────────────────────────────────────────────────────────────────
# Relative regret: G_i normalised by its own floor  (⇔ a floor tilt of μ)
# ─────────────────────────────────────────────────────────────────────────────


def floor_tilt(mu: Tensor, floor: Tensor, rel_clip: float = 0.25) -> Tensor:
    r"""The floor-tilted masses :math:`\tilde\mu_i \propto \mu_i / R_i^*`.

    Floors in the denominator are clipped below at ``rel_clip`` times the
    μ-weighted mean floor, so a near-perfect regime (tiny floor) cannot
    monopolise the tilt.  Returns a normalised mass vector.
    """
    mu = mu / mu.sum().clamp_min(1e-12)
    b = floor.to(mu)
    bbar = float((mu * b).sum())
    w = mu / b.clamp_min(max(rel_clip * bbar, 1e-12))
    return w / w.sum()


class RelativeRegretLoss(RegretRobustLoss):
    r"""Regret normalised by its own floor: every regime competes in **percent
    of its achievable risk**.

    .. math::
        L(\pi) = \sum_i \mu_i \, \frac{\big(R_i(\pi) - R_i^*\big)_+}{R_i^*}
               = \sum_i \tilde\mu_i \,\big(R_i(\pi) - R_i^*\big)_+ ,
        \qquad \tilde\mu_i \propto \frac{\mu_i}{R_i^*}.

    The small-σ / rate regimes — whose *relative* capacity is typically the
    largest in a gate table even when their absolute gaps are tiny — stop
    being drowned by the vol regimes' absolute magnitudes.

    Implemented (for the ``a = 0`` aggregation, where the division commutes
    into the masses) as the floor tilt above, re-applied automatically whenever
    the partition is (re)frozen or the baseline changes.  Requires ``a = 0``:
    the entropic tilt would not commute with the division.

    ``rel_clip`` clips the tilt's denominators below at ``rel_clip`` × the
    μ-weighted mean floor (default 0.25).
    """

    def __init__(self, *args, rel_clip: float = 0.25, **kw) -> None:
        self._raw_mu: Optional[Tensor] = None  # untilted frozen masses
        self.rel_clip = float(rel_clip)
        super().__init__(*args, **kw)
        if self.a != 0.0:
            raise ValueError("RelativeRegretLoss requires a = 0 "
                             "(the floor tilt only commutes with the mean).")

    def freeze_partition(self, population=None, weights=None) -> Tensor:
        labels = super().freeze_partition(population, weights)
        self._raw_mu = self._frozen_mu
        self._on_floor_or_mass_change()
        return labels

    def _on_floor_or_mass_change(self) -> None:
        """Re-derive the tilted masses from the raw masses + current floor."""
        if getattr(self, "_raw_mu", None) is None:
            return
        if self._baseline is None:
            self._frozen_mu = self._raw_mu
        else:
            self._frozen_mu = floor_tilt(self._raw_mu, self._baseline, self.rel_clip)


# ─────────────────────────────────────────────────────────────────────────────
# A_w — SDH on a floor-tilted mixture (raw risk, tilted masses)
# ─────────────────────────────────────────────────────────────────────────────


class TiltedMixtureLoss(AmbiguityAverseLoss):
    r"""**A_w** — a pooled SDH on the floor-tilted mixture (the attribution arm).

    .. math:: L(\pi) = \sum_i \tilde\mu_i \, R_i(\pi),
        \qquad \tilde\mu_i \propto \frac{\mu_i}{R_i^*} .

    Same tilt as :class:`RelativeRegretLoss`, but applied to **raw** per-regime
    risk — no floor netting.  Its role is attribution: if a relative-regret arm
    only re-weights training effort toward the hard regimes, it will match this
    arm; any gap between the two is earned by the regret (floor-netting)
    machinery itself.

    ``floor`` here is *not* subtracted anywhere — it only shapes the tilt, so
    it may equally be a "capacity" vector (weights ∝ capacity) if that is the
    tilt being tested; pass it through :meth:`register_floor`.
    """

    def __init__(self, *args, floor: Optional[Tensor] = None,
                 rel_clip: float = 0.25, **kw) -> None:
        kw.pop("underlier_key", None)
        super().__init__(*args, **kw)
        if self.a != 0.0:
            raise ValueError("TiltedMixtureLoss is an SDH on a tilted mixture: a must be 0.")
        self.rel_clip = float(rel_clip)
        self._raw_mu: Optional[Tensor] = None
        self._floor: Optional[Tensor] = None
        self.name = f"A_w(k={self.partitioner.k}, inner={self.inner.name})"
        self.register_floor(floor)

    def register_floor(self, floor: Optional[Tensor]) -> None:
        if floor is None:
            self._floor = None
        else:
            f = torch.as_tensor(floor, dtype=torch.get_default_dtype())
            if f.numel() != self.partitioner.k:
                raise ValueError(f"floor has {f.numel()} entries for k={self.partitioner.k}.")
            self._floor = f
        self._retilt()

    def freeze_partition(self, population=None, weights=None) -> Tensor:
        labels = super().freeze_partition(population, weights)
        self._raw_mu = self._frozen_mu
        self._retilt()
        return labels

    def _retilt(self) -> None:
        if getattr(self, "_raw_mu", None) is None:
            return
        if self._floor is None:
            self._frozen_mu = self._raw_mu
        else:
            self._frozen_mu = floor_tilt(self._raw_mu, self._floor, self.rel_clip)

    @classmethod
    def factors(cls, target, *, k: int = 20,
                features: Sequence[SourcedFeature] = (),
                inner: Optional[HedgingLoss] = None,
                floor: Optional[Tensor] = None,
                rel_clip: float = 0.25,
                seed: int = 0, min_cluster: int = 2,
                diagnostics: bool = False, **kw) -> "TiltedMixtureLoss":
        from qabit.core.analytics.math.stats import autocorrelation, realised_vol
        feats = tuple(features) or ((realised_vol, SPOT), (autocorrelation, SPOT))
        return cls(inner if inner is not None else JonesLoss(lam=1.0 / 100.0),
                   KMeansPartitioner(feats, k=k, seed=seed),
                   a=0.0, target=target, min_cluster=min_cluster,
                   diagnostics=diagnostics, floor=floor, rel_clip=rel_clip, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Anchor: L = robust + ε · inner  (do-no-harm tie-break)
# ─────────────────────────────────────────────────────────────────────────────


class AnchoredLoss(HedgingLoss):
    r"""``L = L_robust + ε · L_inner`` — degrade to the SDH when regret is exhausted.

    A pure regret objective is *flat* wherever every regime sits at (or below)
    its floor: gradients vanish and the policy can drift.  The small anchor
    ``ε · inner`` breaks the tie toward the plain SDH objective on the training
    mixture, so "no exploitable regret" resolves to "be the SDH", not "be
    anything".  Forwarding methods make the wrapper transparent to trainers
    that freeze partitions or (re)register baselines.
    """

    def __init__(self, robust, anchor: HedgingLoss, eps: float = 0.10) -> None:
        self.robust, self.anchor, self.eps = robust, anchor, float(eps)
        self.name = f"{robust.name} + {eps:g}·{anchor.name}"

    def compute(self, pnl_before_tc, tc, weights=None, *, path=None):
        r = self.robust.compute(pnl_before_tc, tc, weights, path=path)
        return r + self.eps * self.anchor.compute(pnl_before_tc, tc, weights)

    # transparent forwarding ---------------------------------------------------
    def freeze_partition(self, population=None, weights=None):
        return self.robust.freeze_partition(population, weights)

    def thaw_partition(self) -> None:
        self.robust.thaw_partition()

    @property
    def partitioner(self):
        return self.robust.partitioner

    @property
    def inner(self):
        return self.robust.inner

    @property
    def diagnostics(self):
        return self.robust.diagnostics

    @property
    def last_(self):
        return self.robust.last_

    def register_baseline(self, baseline) -> None:
        self.robust.register_baseline(baseline)


# ─────────────────────────────────────────────────────────────────────────────
# Floor measurement — floors are ESTIMATES; carry their uncertainty
# ─────────────────────────────────────────────────────────────────────────────


def measure_floor_stats(hedger,
                        target,
                        system,
                        calendar,
                        rebalance_dates,
                        market,
                        partitioner: PathPartitioner,
                        *,
                        n_paths: int,
                        seed: int,
                        B: int = 200,
                        boot_seed: int = 0) -> SimpleNamespace:
    r"""Per-regime risk floor :math:`R_i^*` of a *trained* hedger, **with
    bootstrap standard errors**.

    Simulates one reference population, freezes ``partitioner`` on it (so the
    floor labelling matches the labelling the regret loss will use), scores the
    hedger's inner risk per regime, and bootstraps each regime's floor ``B``
    times.  Paired measurement — floor and later regret scored on populations
    drawn from the same generator/partition — is the variance control that
    keeps ``regret = signal − baseline`` from being dominated by baseline noise.

    Returns a namespace with ``floor``, ``se``, ``count`` (all length-k),
    ``labels``, ``pnl``, ``tc`` (per path).
    """
    system.simulate(calendar, n_paths, seed=seed)
    spot = target.underlier.full_path()
    labels = partitioner.freeze(spot, system.get_state())
    inner = getattr(hedger.loss, "inner", None) or hedger.loss
    with torch.no_grad():
        pnl, tc = hedger._pnl(target, market, rebalance_dates.double())
    k = partitioner.k
    floor, se = torch.zeros(k), torch.zeros(k)
    cnt = torch.zeros(k, dtype=torch.long)
    g = torch.Generator().manual_seed(boot_seed)
    for i in range(k):
        m = torch.where(labels == i)[0]
        cnt[i] = len(m)
        if len(m) < 2:
            continue
        floor[i] = float(inner.compute(pnl[m], tc[m]))
        bs = []
        for _ in range(B):
            j = m[torch.randint(len(m), (len(m),), generator=g)]
            bs.append(float(inner.compute(pnl[j], tc[j])))
        se[i] = float(torch.tensor(bs).std())
    return SimpleNamespace(floor=floor, se=se, count=cnt, labels=labels, pnl=pnl, tc=tc)


def robust_floor(stats: SimpleNamespace,
                 *,
                 shrink: float = 0.1,
                 margin_z: float = 1.0) -> Tensor:
    r"""A variance-controlled floor from :func:`measure_floor_stats` output.

    Two corrections, both aimed at ``noise(floor) ≪ signal(regret)``:

    * **Shrinkage** (James–Stein-style): thin regimes borrow strength from the
      mass-weighted mean floor,
      :math:`R_i^* \leftarrow (1-w)\,R_i^* + w\,\bar R^*`;
    * **Noise margin**: raise the floor by ``margin_z`` × its own bootstrap SE,
      so only regret exceeding its estimation noise drives the loss (the
      "gate in disguise": if no regime clears its margin, the loss is inert —
      evidence, not assumption, that there is no shape structure on the axis).
    """
    floor = stats.floor.clone()
    if shrink > 0:
        mu = stats.count.float() / stats.count.sum().clamp_min(1)
        floor = (1 - shrink) * floor + shrink * float((mu * floor).sum())
    if margin_z > 0:
        floor = floor + margin_z * stats.se
    return floor


def measure_regime_floor(hedger, target, system, calendar, rebalance_dates,
                         market, partitioner, *, n_paths, seed, **_ignored) -> Tensor:
    """Deprecated point-estimate floor; use :func:`measure_floor_stats`."""
    return measure_floor_stats(hedger, target, system, calendar, rebalance_dates,
                               market, partitioner, n_paths=n_paths, seed=seed, B=0).floor


__all__ = [
    "RegretRobustLoss",
    "RelativeRegretLoss",
    "TiltedMixtureLoss",
    "AnchoredLoss",
    "floor_tilt",
    "measure_floor_stats",
    "robust_floor",
    "measure_regime_floor",
]
