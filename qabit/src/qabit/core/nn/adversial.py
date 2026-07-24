r"""nn/adversarial.py — ambiguity-averse (distributionally-robust) deep hedging.

A PyTorch-native implementation of *Ambiguity-Averse Deep Hedging with Feature
Clustering* (Jones, Horvath, Reisinger, Wood, Bai & Akkari, 2025), expressed
entirely through qabit's own hedging stack so it **collaborates with the existing
:class:`~qabit.core.nn.hedger.DeepHedger`** rather than replacing it.

The idea (Algorithms 1 & 3)
---------------------------
A standard deep hedger (SDH) minimises a single risk functional under the
training measure :math:`P`.  When the *true* market may differ from the
generator — a model-uncertainty / out-of-distribution gap — minimising the
average risk leaves the hedge exposed to whichever regime the SDH under-weighted.
The ambiguity-averse deep hedger (AADH) instead

1.  **partitions each batch of paths into regimes** (Algorithm 1) using
    :math:`F_T`-measurable *path features* — realised volatility, return
    autocorrelation, maximum drawdown, a mean-reversion estimate (defined in
    :mod:`qabit.core.analytics.math.stats`) — normalised to :math:`[0,1]`
    and clustered (k-means) or quantile-bucketed;

2.  **scores the risk inside each cluster** with an ordinary
    :class:`~qabit.core.nn.loss.HedgingLoss` (variance, Jones, ES, …) — these are
    the *regime-conditional* risks :math:`R_i`;

3.  **aggregates them with an entropic ambiguity functional** over the clusters
    with weights :math:`\mu_i` (the regime masses),

    .. math::
        \rho_a(\{R_i\}) \;=\; \tfrac1a \log \sum_i \mu_i \, e^{\,a R_i},

    and minimises :math:`\rho_a`.  This is the deep-hedging objective of the paper.

Why "adversarial"
-----------------
:math:`\rho_a` is the convex (entropic) risk measure over the *finite set of
regimes*, and by convex duality it equals a **worst-case over a KL-ball of regime
measures**,

.. math::
    \rho_a(\{R_i\}) \;=\; \max_{q \ge 0,\ \sum_i q_i = 1}
        \Big\{ \textstyle\sum_i q_i R_i \;-\; \tfrac1a\, \mathrm{KL}(q \,\|\, \mu) \Big\},

whose maximiser is the **adversary's tilt** :math:`q_i \propto \mu_i e^{a R_i}`.
A fictitious adversary re-weights the regimes toward the ones the hedger handles
worst, paying a :math:`1/a` KL penalty for departing from the nominal masses
:math:`\mu`.  Two limits make the dial intuitive:

* :math:`a \to 0`  ⇒ :math:`q \to \mu`, :math:`\rho_a \to \sum_i \mu_i R_i` — the
  plain expected risk, i.e. the **SDH** (no ambiguity aversion);
* :math:`a \to \infty` ⇒ :math:`q` concentrates on :math:`\arg\max_i R_i`,
  :math:`\rho_a \to \max_i R_i` — the **fully robust min-max** hedge.

The adversary's view (weights :math:`q`, per-regime risks :math:`R_i`, the
*ambiguity premium* :math:`\rho_a - \sum_i \mu_i R_i \ge 0`) is **not** computed on
the training hot path: it is gathered only when diagnostics are switched on
(:meth:`AmbiguityAverseLoss.diagnosing`), so training carries no constant overhead.

How it collaborates with DeepHedger (no edits to the hedger)
-----------------------------------------------------------
The hedger is *loss-polymorphic*: it only ever calls ``loss.compute(pnl, tc,
weights)`` and never branches on the loss type.  The whole AADH therefore lives in
a **loss object** — :class:`AmbiguityAverseLoss` — that you hand to an ordinary
:class:`DeepHedger`::

    loss   = AmbiguityAverseLoss.by_vol(call, a=250.0, k=20, inner=JonesLoss(lam=1/100))
    hedger = DeepHedger(model, features, instruments, loss, ...)
    hedger.fit(call, system=sde, calendar=cal, rebalance_dates=rebal, market=mkt, ...)

    d = hedger.diagnose(call, system=sde, calendar=cal, rebalance_dates=rebal, market=mkt)
    with loss.diagnosing():                 # opt-in, off during training
        loss.compute(d["pnl"], d["tc"])     # score the diagnosed batch (state is live)
    loss.report()                           # the adversary's view of that batch

The clustering is :math:`F_T`-measurable and depends on the *paths*, not on the
policy — so it must see the batch that produced this step's P&L.  The hedger
resimulates fresh paths every mini-batch and calls ``loss.compute`` immediately
after ``_pnl`` on those same paths, with no resimulation in between.  So the loss
reads the **live underlier state** (``target.underlier.full_path()``) at
``compute`` time and partitions exactly the batch being scored.  The partition
carries no gradient (it is a function of the simulated state, taken under
``no_grad``); gradients flow only through the per-cluster risks :math:`R_i`.

Composes with the near-martingale reweighting: if ``DeepHedger.fit`` is given a
``path_weight_fn`` the per-path density ``w`` arrives in ``compute`` and is applied
*inside* each cluster (and used for the regime masses), so an AADH can be trained
under :math:`Q^\*` just like an SDH.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from qabit.core.analytics.math.stats import (
    autocorrelation,
    max_drawdown,
    mean_reversion,
    realised_vol,
)
from qabit.core.keys import SPOT, FactorKey, FactorKeyLike, FactorKeyMap
from qabit.core.nn.loss import HedgingLoss, JonesLoss

#: A path feature is any pathwise functional ``(N, E) -> (N,)``.
PathFeature = Callable[[Tensor], Tensor]

#: A feature paired with the factor source it reads, as a **mandatory**
#: ``(callable, source)`` pair.  ``source`` is any
#: :data:`~qabit.core.keys.FactorKeyLike` — ``"spot"`` (the hedged underlier's spot
#: path), ``"rate"`` (short rate), ``"vol:1"`` (the variance column of a Heston
#: factor), ``"mort"`` (the CIR hazard), or a ready
#: :class:`~qabit.core.keys.FactorKey`.  One feature set therefore mixes equity,
#: rate, vol and mortality clustering axes; each pair's key grabs *its own* path
#: straight from the live SDESystem state, so no separate bundle plumbing is needed.
SourcedFeature = Tuple["PathFeature", "FactorKeyLike"]

#: A resolved, row-aligned bundle of named factor paths ``{label: (N, E)}``, as
#: produced by :meth:`~qabit.core.keys.FactorKeyMap.resolve`.
PathBundle = Dict[str, Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# Partitioners — assign each path to a regime and report the regime masses
# ─────────────────────────────────────────────────────────────────────────────


class PathPartitioner(ABC):
    r"""Partition a batch of paths into ``k`` regimes from :math:`F_T`-measurable
    features.

    A partitioner maps the hedged **spot** panel ``(N, E)`` plus the live
    SDESystem ``state`` (the source any non-``spot`` feature resolves its own path
    from) to ``(labels, mu)`` where ``labels`` is a ``LongTensor[N]`` in
    ``{0, …, k-1}`` and ``mu`` is the ``Tensor[k]`` of regime masses (mean ≈
    ``1/k``).  Assignment is done under ``no_grad`` by the caller, so partitioners
    need not worry about autograd.

    Each feature is a **mandatory** ``(callable, source)`` pair: the callable is
    the pathwise functional and ``source`` is the :class:`~qabit.core.keys.FactorKey`
    it reads — ``"spot"`` (the hedged underlier), ``"rate"``, ``"vol"``, ``"mort"``,
    …  A single partitioner can therefore mix equity, rate, vol and mortality
    clustering axes; each key resolves *its own* path from the live SDESystem state
    the loss hands in, so every source is row-aligned with the scored P&L.
    """

    def __init__(self, features: Sequence[SourcedFeature], k: int) -> None:
        if k < 1:
            raise ValueError("k (number of regimes) must be >= 1.")
        if not features:
            raise ValueError("at least one (callable, source) feature is required.")
        #: the configured features as ``(callable, resolved FactorKey)`` pairs.
        self.features: List[Tuple[PathFeature, FactorKey]] = []
        for spec in features:
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise TypeError(
                    "each feature must be a (callable, source) pair, e.g. "
                    f"(realised_vol, 'spot') or (mean_level, 'rate'); got {spec!r}."
                )
            fn, source = spec
            self.features.append((fn, FactorKey.parse(source)))
        self.feature_names = [
            f"{getattr(fn, '__name__', 'feature')}@{key.label}"
            for fn, key in self.features
        ]
        self.k = int(k)

    @property
    def key_map(self) -> FactorKeyMap:
        """The distinct :class:`FactorKey`\\ s this partitioner's features read.

        One :class:`~qabit.core.keys.FactorKeyMap`, de-duplicated in declaration
        order — the loss uses it only to decide whether a live SDESystem state is
        needed (i.e. whether any feature reads a non-``spot`` source)."""
        return FactorKeyMap._dedup(key for _, key in self.features)

    @property
    def sources(self) -> List[str]:
        """Distinct factor-source labels this partitioner's features read."""
        return self.key_map.labels

    @staticmethod
    def _normalize(X: Tensor) -> Tensor:
        """Min-max each column to ``[0, 1]`` (Algorithm 1)."""
        lo = X.min(dim=0).values
        rng = (X.max(dim=0).values - lo).clamp_min(1e-8)
        return (X - lo) / rng

    @staticmethod
    def _masses(labels: Tensor, k: int, weights: Optional[Tensor]) -> Tensor:
        """Regime masses ``mu_i`` — count fractions, or weighted mass if a per-path
        ``weights`` density is supplied (so masses track the active measure)."""
        if weights is None:
            return torch.bincount(labels, minlength=k).to(torch.get_default_dtype()) / len(
                labels
            )
        w = weights / weights.sum().clamp_min(1e-12)
        mu = torch.zeros(k, dtype=w.dtype, device=w.device)
        return mu.index_add_(0, labels, w)

    @abstractmethod
    def assign(
        self, spot: Tensor, state=None, weights: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(labels[N], mu[k])`` for the batch.

        ``spot`` is the hedged underlier's spot panel and ``state`` the live
        SDESystem state any non-``spot`` features read from (``None`` for
        spot-only features)."""

    def freeze(
        self, spot: Tensor, state=None, weights: Optional[Tensor] = None
    ) -> Tensor:
        """Lock the partition rule to this population (Algorithm 1's one-off
        clustering) and return its per-path labels.  Later :meth:`assign` calls reuse
        the locked rule, so regime *identity* is stable across the mini-batches drawn
        from the population — the prerequisite for Algorithm 3's stratified sampling.

        Fitting partitioners (k-means) cache their centroids on first assignment, so
        the base implementation suffices; partitioners with a per-batch rule override
        this to cache it.
        """
        labels, _ = self.assign(spot, state, weights)
        return labels

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(features={self.feature_names}, k={self.k})>"


class QuantilePartitioner(PathPartitioner):
    r"""Equal-mass quantile buckets on a **single** feature (fast, torch-native).

    The batch is split into ``k`` buckets by the per-batch quantiles of the chosen
    feature (default realised volatility — the regime axis of the paper's Figure 2).
    Re-bucketing each batch keeps the partition matched to exactly the paths being
    scored; equal-mass buckets give nearly uniform regime masses ``mu_i ≈ 1/k``.
    """

    def __init__(
        self, feature: SourcedFeature = (realised_vol, SPOT), k: int = 20
    ) -> None:
        super().__init__([feature], k)
        self._edges: Optional[Tensor] = None  # frozen bucket edges (else per-batch)

    def _bucket_edges(self, x: Tensor) -> Tensor:
        qs = torch.linspace(0, 1, self.k + 1, device=x.device, dtype=x.dtype)[1:-1]
        return torch.quantile(x, qs)

    def freeze(self, spot, state=None, weights=None):
        fn, key = self.features[0]
        x = fn(key.resolve(state, spot_path=spot))  # (N,)
        self._edges = self._bucket_edges(x) if self.k > 1 else None
        labels, _ = self.assign(spot, state, weights)
        return labels

    def assign(self, spot, state=None, weights=None):
        fn, key = self.features[0]
        x = fn(key.resolve(state, spot_path=spot))  # (N,)
        if self.k == 1:
            labels = torch.zeros(len(x), dtype=torch.long, device=x.device)
            return labels, self._masses(labels, 1, weights)
        edges = self._edges if self._edges is not None else self._bucket_edges(x)
        labels = torch.bucketize(x, edges).clamp_(max=self.k - 1)
        return labels, self._masses(labels, self.k, weights)


class KMeansPartitioner(PathPartitioner):
    r"""k-means regimes on one or more features (Algorithm 1, faithful).

    The centroids are fitted **once** (lazily, on the first batch the loss scores)
    in the normalised feature space, then every later batch is assigned by nearest
    centroid — a torch-native, gradient-free lookup.  Fitting once (rather than per
    batch) keeps regime *identity* stable across batches, so the regime masses
    ``mu_i`` carry the paper's *global* cluster weights (rarer regimes get a smaller
    mass) instead of the uniform masses a per-batch refit would impose.

    The one-time fit uses :class:`sklearn.cluster.KMeans`.
    """

    def __init__(
        self,
        features: Sequence[SourcedFeature] = (
            (realised_vol, SPOT),
            (autocorrelation, SPOT),
        ),
        k: int = 20,
        *,
        seed: int = 0,
        n_fit: int = 20_000,
    ) -> None:
        super().__init__(features, k)
        self.seed = int(seed)
        self.n_fit = int(n_fit)
        self._centroids: Optional[Tensor] = None  # (k, F) in normalised space
        self._lo: Optional[Tensor] = None
        self._rng: Optional[Tensor] = None

    # normalisation frozen at fit time so assignment is consistent across batches
    def _normalize_fixed(self, X: Tensor) -> Tensor:
        return (X - self._lo) / self._rng

    def _fit(self, X: Tensor) -> None:
        from sklearn.cluster import KMeans

        if len(X) > self.n_fit:
            X = X[torch.randperm(len(X))[: self.n_fit]]
        self._lo = X.min(dim=0).values
        self._rng = (X.max(dim=0).values - self._lo).clamp_min(1e-8)
        Xn = self._normalize_fixed(X)
        k = min(self.k, len(Xn))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        km.fit(Xn.detach().cpu().numpy())
        self.k = k  # may shrink if fewer points than clusters
        self._centroids = torch.as_tensor(
            km.cluster_centers_, dtype=Xn.dtype, device=Xn.device
        )

    def freeze(self, spot, state=None, weights=None):
        # fit the centroids on the *whole* population (not a lazy first batch)
        self._centroids = None
        X = torch.stack(
            [fn(key.resolve(state, spot_path=spot)) for fn, key in self.features], dim=1
        )
        self._fit(X)
        labels, _ = self.assign(spot, state, weights)
        return labels

    def assign(self, spot, state=None, weights=None):
        X = torch.stack(
            [fn(key.resolve(state, spot_path=spot)) for fn, key in self.features], dim=1
        )
        if self._centroids is None:
            self._fit(X)
        Xn = self._normalize_fixed(X)
        labels = torch.cdist(Xn, self._centroids).argmin(dim=1)
        return labels, self._masses(labels, self.k, weights)


# ─────────────────────────────────────────────────────────────────────────────
# The ambiguity-averse loss (Algorithm 3) — a drop-in HedgingLoss
# ─────────────────────────────────────────────────────────────────────────────


class AmbiguityAverseLoss(HedgingLoss):
    r"""Entropic-over-regimes wrapper around any :class:`HedgingLoss`.

    ``compute`` partitions the live batch into regimes, scores the *inner* loss
    inside each, and returns the entropic aggregation
    :math:`\tfrac1a\log\sum_i \mu_i e^{a R_i}` (Algorithm 3).  With ``a = 0`` it
    is exactly the regime-averaged inner loss — i.e. the ordinary (SDH) objective —
    so the same object trains both an SDH (``a=0``) and an AADH (``a>0``).

    Build one with the convenience constructors :meth:`by_vol` / :meth:`kmeans`,
    which bind the loss to its ``target`` and wire up the partitioner::

        loss = AmbiguityAverseLoss.by_vol(call, a=250.0, k=20, inner=VarianceLoss())

    Parameters
    ----------
    inner : HedgingLoss
        The per-regime risk functional (variance, Jones, ES, entropic, …).
    partitioner : PathPartitioner
        Maps the batch to regimes (and their masses).
    a : float, default ``0.0``
        Ambiguity aversion.  ``0`` → SDH; larger → closer to the min-max robust
        hedge.  The dual KL radius is ``1/a``.
    min_cluster : int, default ``2``
        Regimes with fewer than this many paths are dropped from the aggregation
        (a variance needs ≥ 2 points); their tiny mass is renormalised away.
    diagnostics : bool, default ``False``
        When ``True``, :meth:`compute` also records the adversary's view in
        :attr:`last_` (regime risks, masses, tilt ``q``, worst regime, ambiguity
        premium).  Left ``False`` on the training hot path; flip it on for an
        evaluation pass with :meth:`diagnosing`.
    """

    def __init__(
        self,
        inner: HedgingLoss,
        partitioner: PathPartitioner,
        a: float = 0.0,
        *,
        target=None,
        min_cluster: int = 2,
        diagnostics: bool = False,
    ) -> None:
        if a < 0:
            raise ValueError("ambiguity aversion a must be >= 0 (0 = SDH).")
        self.inner = inner
        self.partitioner = partitioner
        self.a = float(a)
        self.target = target
        self.min_cluster = int(min_cluster)
        self.diagnostics = bool(diagnostics)
        self.name = f"AADH(a={a:g}, k={partitioner.k}, inner={inner.name})"
        #: global regime masses cached by ``freeze_partition`` (else per-batch)
        self._frozen_mu: Optional[Tensor] = None
        #: adversary's view from the most recent diagnostics-on ``compute`` (else empty)
        self.last_: Dict[str, object] = {}

    # ── convenience constructors (bind to a target → a DeepHedger loss) ───────

    @classmethod
    def kmeans(
        cls,
        target,
        *,
        a: float = 0.0,
        k: int = 20,
        features: Sequence[SourcedFeature] = (
            (realised_vol, SPOT),
            (autocorrelation, SPOT),
        ),
        inner: Optional[HedgingLoss] = None,
        seed: int = 42,
        min_cluster: int = 2,
        diagnostics: bool = False,
    ) -> "AmbiguityAverseLoss":
        """k-means regimes on a feature set (Algorithm 1, with global masses).

        Each feature is a ``(callable, source)`` pair; pass ``SPOT`` as the source
        to cluster on the hedged underlier (the common case)."""
        return cls(
            inner if inner is not None else JonesLoss(lam=1.0 / 100.0),
            KMeansPartitioner(features, k=k, seed=seed),
            a=a,
            target=target,
            min_cluster=min_cluster,
            diagnostics=diagnostics,
        )

    @classmethod
    def factors(
        cls,
        target,
        *,
        a: float = 0.0,
        k: int = 20,
        features: Sequence[SourcedFeature] = (
            (realised_vol, SPOT),
            (autocorrelation, SPOT),
        ),
        inner: Optional[HedgingLoss] = None,
        seed: int = 42,
        min_cluster: int = 2,
        diagnostics: bool = False,
    ) -> "AmbiguityAverseLoss":
        r"""k-means regimes on **cross-factor** features (the general case).

        Every feature is a ``(callable, source)`` pair, so one loss clusters on
        equity, rate, vol and mortality axes at once — just name each feature's
        source::

            from qabit.core.analytics.math.stats import realised_vol
            # spot realised-vol  +  hazard accumulation  +  short-rate level
            loss = AmbiguityAverseLoss.factors(
                gm, a=20, k=20,
                features=(
                    (realised_vol,      "spot"),         # reads the hedged spot
                    (integrated_hazard, "mort"),         # reads the CIR hazard factor
                    (mean_level,        "rate"),         # reads the Vasicek short rate
                ),
                inner=JonesLoss(lam=1/100))

        Each pair's key resolves its own path from the live SDESystem the ``fund``
        underlier belongs to; every source is row-aligned with the scored P&L.
        """
        return cls(
            inner if inner is not None else JonesLoss(lam=1.0 / 100.0),
            KMeansPartitioner(features, k=k, seed=seed),
            a=a,
            target=target,
            min_cluster=min_cluster,
            diagnostics=diagnostics,
        )

    # ── fixed-population pre-clustering (Algorithm 3) ─────────────────────────

    def freeze_partition(self, population=None, weights: Optional[Tensor] = None) -> Tensor:
        """Pre-cluster a **fixed** population once and cache its global regime masses.

        When the trainer reuses a fixed panel of paths (rather than re-simulating each
        step), the partition is computed once here: the partitioner locks its rule to
        the population and the global masses :math:`\\mu_i` are cached, so every
        mini-batch later drawn from the population is scored against *stable* regime
        identities and the paper's global weights.  Returns the population's per-path
        labels, which the trainer uses to draw stratified mini-batches.

        ``population`` is the fixed spot panel to cluster on; the partition is
        clustered on the **full multi-factor state** resolved from the live system,
        so features reading ``"rate"`` / ``"vol"`` / ``"mort"`` see their own factor,
        not the spot.  When omitted, the spot panel is read from the bound target.

        For the online (re-simulating) mode leave the partition thawed: each fresh
        batch is then partitioned independently.
        """
        with torch.no_grad():
            spot, state = self._live(population)
            labels = self.partitioner.freeze(spot, state, weights)
            self._frozen_mu = self.partitioner._masses(labels, self.partitioner.k, weights)
        return labels

    def thaw_partition(self) -> None:
        """Forget the frozen population; each batch is partitioned independently."""
        self._frozen_mu = None

    @property
    def partition_frozen(self) -> bool:
        return self._frozen_mu is not None

    # ── locate the live spot panel and SDE state the partition reads ──────────

    def _live(self, spot: Optional[Tensor] = None) -> Tuple[Tensor, object]:
        """Locate the ``(spot, state)`` pair the partitioner resolves its features from.

        The spot panel is ``spot`` when the trainer passes the freshly simulated
        batch, else the bound underlier's ``full_path()``.  The state is the live
        SDESystem state — which the trainer has re-pointed at this batch's rows, so
        every factor panel the partitioner reads is row-aligned with the spot and the
        P&L.  When every feature reads ``"spot"`` no system is needed and ``state`` is
        ``None`` (spot keys ignore it); this lets the same loss run in a target-free
        loop (e.g. the RVBS toy).  All key parsing/resolution lives on the
        partitioner via :class:`~qabit.core.keys.FactorKey`; this only hands it the
        spot and the live state.
        """
        key_map = self.partitioner.key_map
        if spot is None:
            if self.target is None:
                raise RuntimeError(
                    "AmbiguityAverseLoss is unbound; build it via .by_vol / .kmeans "
                    "/ .factors with a target, or pass the spot panel as `path=`."
                )
            spot = self.target.underlier.full_path()

        if all(key.is_spot for key in key_map):  # spot-only features: no system needed
            return spot, None

        process = getattr(self.target.underlier, "process", None)
        system = process._system() if hasattr(process, "_system") else None
        if system is None or system.state is None:
            named = ", ".join(k.label for k in key_map if not k.is_spot)
            raise RuntimeError(
                f"features read factor source(s) [{named}] but no live SDESystem "
                "state is reachable from the bound underlier."
            )
        return spot, system.get_state()

    def _regime_risks(self, pnl_before_tc, tc, weights=None, *, path=None):
        """Partition the live batch and score the inner loss per regime.

        Returns ``(R, mu_keep, keep)`` — the per-regime risks (carrying grad),
        the renormalised masses of the kept regimes, and the kept labels — or
        ``(None, None, None)`` for a degenerate batch (no regime reaches
        ``min_cluster`` paths), in which case the caller should fall back to
        the plain inner loss.
        """
        n = pnl_before_tc.shape[0]
        with torch.no_grad():
            spot, state = self._live(path)
            if spot.shape[0] != n:
                raise RuntimeError(
                    f"underlier path has {spot.shape[0]} paths but the batch P&L has "
                    f"{n}; the loss must be scored on the freshly simulated batch."
                )
            labels, mu = self.partitioner.assign(spot, state, weights)
            if labels.shape[0] != n:
                raise RuntimeError(
                    f"partition produced {labels.shape[0]} labels but the batch P&L "
                    f"has {n}; the live system state is not row-aligned with the "
                    "scored batch (was it re-simulated after _pnl?).")
            if self._frozen_mu is not None:  # fixed population → stable global masses
                mu = self._frozen_mu

        risks: List[Tensor] = []
        keep: List[int] = []
        for i in range(self.partitioner.k):
            mask = labels == i
            if int(mask.sum()) < self.min_cluster:
                continue
            w_i = weights[mask] if weights is not None else None
            risks.append(self.inner.compute(pnl_before_tc[mask], tc[mask], weights=w_i))
            keep.append(i)

        if not risks:
            return None, None, None
        R = torch.stack(risks)  # (K',) — carries grad
        mu_keep = mu[keep].clamp_min(1e-12)
        mu_keep = mu_keep / mu_keep.sum()  # renormalise dropped mass
        return R, mu_keep, keep

    def compute(self, pnl_before_tc, tc, weights=None, *, path=None):
        """Entropic-over-regimes loss for the batch.

        ``path`` is the (N, steps+1) **spot** panel the partition's ``"spot"``-source
        features read from.  When omitted it is taken from the bound ``target``; any
        non-spot sources (rate, vol, mort) are always resolved from the live system.
        Passing ``path`` explicitly lets the same loss be used in a target-free loop
        (e.g. the RVBS toy) as long as its features only read ``"spot"``.
        """
        R, mu_keep, keep = self._regime_risks(pnl_before_tc, tc, weights, path=path)
        if R is None:  # degenerate batch — fall back to the plain inner loss
            return self.inner.compute(pnl_before_tc, tc, weights=weights)

        if self.a == 0.0:  # SDH limit: regime-averaged risk
            loss = (mu_keep * R).sum()
        else:
            z = self.a * R + torch.log(mu_keep)
            loss = torch.logsumexp(z, dim=0) / self.a

        if self.diagnostics:
            self._record(loss, R, mu_keep, keep)
        return loss

    # ── the adversary's view (opt-in; off on the training hot path) ────────────

    def _record(self, loss: Tensor, vals: Tensor, mu_keep: Tensor,
                keep: List[int], risk: Optional[Tensor] = None) -> None:
        """Record the adversary's view of this batch in :attr:`last_`.

        ``vals`` is whatever the aggregation ran on — raw risks for the AADH,
        regret for the regret family (which also passes the raw ``risk``)."""
        with torch.no_grad():
            nominal = (mu_keep * vals).sum()
            q = (mu_keep if self.a == 0.0 else
                 torch.softmax(self.a * vals + torch.log(mu_keep), dim=0))
            self.last_ = {
                "clusters": keep,
                "risk": (vals if risk is None else risk).detach(),
                "regret": vals.detach() if risk is not None else None,
                "mu": mu_keep.detach(),
                "adversary": q.detach(),  # worst-case regime tilt q_i ∝ mu_i e^{a G_i}
                "worst": int(keep[int(vals.argmax())]),
                "ambiguity_premium": float((loss - nominal).detach()),
                "loss": float(loss.detach()),
                "n_active": len(keep),
            }

    @contextmanager
    def diagnosing(self) -> Iterator["AmbiguityAverseLoss"]:
        """Temporarily switch diagnostics recording on (for an evaluation pass)."""
        prev = self.diagnostics
        self.diagnostics = True
        try:
            yield self
        finally:
            self.diagnostics = prev

    def report(self) -> Dict[str, object]:
        """Pretty-print (and return) the adversary's view of the last diagnosed batch."""
        if not self.last_:
            print("no diagnostics recorded — run compute() inside `with loss.diagnosing():`")
            return {}
        d = self.last_
        print(f"{self.name}: loss={d['loss']:.4e}  ambiguity premium={d['ambiguity_premium']:.4e}"
              f"  worst regime={d['worst']}  active={d['n_active']}")
        return d


__all__ = [
    "PathFeature",
    "SourcedFeature",
    "PathBundle",
    "FactorKey",
    "FactorKeyMap",
    "PathPartitioner",
    "QuantilePartitioner",
    "KMeansPartitioner",
    "AmbiguityAverseLoss",
]