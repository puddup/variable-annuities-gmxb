"""nn/hedger.py — HedgingMLP and the DeepHedger training/pricing loop.

Clean, product-agnostic interface (pfhedge-style ``fit`` / ``price``):

    hedger = DeepHedger(model, features, instruments, loss, liability=...)
    history = hedger.fit(target, system=sde, calendar=cal,
                         rebalance_dates=rebal, market=mkt,
                         n_paths=..., n_epochs=..., batch_size=..., val_paths=...)
    premium = hedger.price(target, system=sde, calendar=cal,
                           rebalance_dates=rebal, market=mkt)

The hedger only ever calls ``instrument.grids(dates, market)`` — it never branches
on stock vs. option, fixed vs. rolling, equity vs. rate.  All discounting of the
*hedge* leg is applied once here, in the discounted-to-0 numéraire (per-path
``disc.df(t)``), so the products stay numéraire-agnostic and the discounted
underlying is a discrete martingale by construction of the SDESystem.

Liability hook
--------------
The short liability is a single callable ``liability(target, market) -> Tensor[N]``
returning the per-path liability **already discounted to time 0**.  The default
(``liability=None``) is ``target._mc_price_zero(market)`` — terminal
payoff × DF to 0 — which every standard derivative implements (vanilla, barrier,
asian, lookback, cliquet).  A cashflow product whose liability is a *stream* — a
GMxB variable annuity, with death / surrender / maturity benefits at many dates
and mortality applied per path — has no single terminal payoff; it values itself
by Monte Carlo through ``price(market)``.  Hedge it by choosing the pricer mode
once and passing that method as the liability::

    gmab.list_mixed()        # or gmab.list_static()  — picks the MC pricer
    DeepHedger(..., liability=lambda t, m: t.price(m)[0])

So the hedger calls the *right* Monte-Carlo function on the target without knowing
anything about it — no adapter registry, just a callable.

Because the liability never depends on the model parameters, it is computed under
``no_grad`` and detached — gradients flow only through the hedge gains and costs.
The hedge leg and the liability are discounted through the *target's* own
``marketkeys`` (``target.marketkeys.discount_curve``), so leg gains and liability
share one numéraire by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam

from qabit.tools.util import to_path, as_tensor
from qabit.core.market.curve import MarketKeys
from qabit.core.nn.feature import FeatureContext, FeatureSet
from qabit.core.nn.instrument import Hedgeable
from qabit.core.nn.loss import HedgingLoss


# ─────────────────────────────────────────────────────────────────────────────
# Model — the policy networks live in `networks.py`; re-exported here so existing
# imports (`from qabit.core.nn.hedger import HedgingMLP`) keep working.
# ─────────────────────────────────────────────────────────────────────────────

from qabit.core.nn.networks import (
    FeatureStandardizer,
    HedgingMLP,
    BoundedHedgingMLP,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deep hedger
# ─────────────────────────────────────────────────────────────────────────────


class DeepHedger:
    """Deep hedger over a list of :class:`Hedgeable` instruments."""

    def __init__(
        self,
        model: nn.Module,
        features: FeatureSet,
        instruments: List[Hedgeable],
        loss: HedgingLoss,
        *,
        liability: Optional[Callable] = None,
        batch_size: Optional[int] = None,
        base_policy: Optional[Callable] = None,
        liveness: Optional[Callable] = None,
    ) -> None:
        """``liability(target, market) -> Tensor[N]`` — the short liability, per
        path **already discounted to 0**.  Default (``None``): the product's own
        ``_mc_price_zero`` (terminal payoff × DF to 0), which every standard
        derivative implements; for a cashflow product with no terminal payoff (a
        GMxB) pass its Monte-Carlo valuation, e.g.
        ``liability=lambda t, m: t.price(m)[0]``.

        The hedge leg, the liability, and the features all read their
        discount / volatility / hazard curves through the *target's* own
        :class:`~qabit.core.market.curve.MarketKeys` (``target.marketkeys``), so
        the leg gains and the liability are guaranteed to share one numéraire and
        a mortality feature reads the same hazard curve the liability is priced
        under — no separate ``discount_key`` / ``underlier_key`` / ``market_keys``
        need be threaded in, and the single hedged underlier is read straight off
        ``target.underlier``.

        ``base_policy(ctx) -> Tensor[N, H]`` — optional structural prior added to
        the network output at every step (holdings = base + NN residual).  Use it
        to inject *known* structure the loss only weakly identifies: the analytic
        delta of a tractable liability, a knockout gate (zero book after breach), a
        cap/floor band delta for cliquets.  The network then learns only the
        correction (variance-optimal shrinkage, costs, incompleteness), which
        converges orders of magnitude faster than learning the full shape — the
        deep-hedging analogue of physics-informed residual learning.

        ``liveness(ctx) -> Tensor[N, 1] | Tensor[N, H]`` — optional **multiplicative**
        gate in ``{0, 1}`` applied to the *final* book at every step (holdings =
        liveness * (base + NN residual)).  This is the structurally-correct way to
        zero the hedge on paths whose liability has *died* — a knocked-out barrier,
        a cliquet whose capped sum can no longer recover, a GMxB whose death benefit
        has settled.  ``base_policy`` cannot do this: it is *additive*, so it can
        only cancel its own analytic contribution, never the network's residual,
        which is what leaves the tell-tale non-zero slope/spread on dead cohorts.
        The gate must be ``F_t``-measurable — built from path-to-``t`` state only
        (the liveness builders in :mod:`qabit.core.nn.feature` reuse the same
        measurable flags as :class:`BarrierBreached` / :class:`Deceased` /
        :class:`PeriodReturn`, so they introduce no look-ahead).  Because gating
        happens inside ``_pnl``, it is applied identically in training, pricing and
        :meth:`diagnose`, and the forced unwind it implies at the death step is
        charged its transaction cost like any other trade."""
        self.model = model
        self.base_policy = base_policy
        self.liveness = liveness
        self.features = features
        self.instruments = instruments
        self.loss = loss
        self.batch_size = batch_size
        self.history_: List[float] = []
        self.val_history_: List[Tuple[int, float]] = []
        # The short liability is a single callable ``f(target, market) -> (N,)``
        # giving the per-path liability discounted to 0.  Default: the product's
        # own ``_mc_price_zero`` (terminal payoff × DF to 0), which every standard
        # derivative implements.  For a cashflow product without a terminal payoff
        # (a GMxB) pass its Monte-Carlo valuation instead.
        self.liability: Callable = liability or (
            lambda target, market: target._mc_price_zero(market)
        )

    def _ctx(self, target, market, t, step, prev_h) -> FeatureContext:
        return FeatureContext(
            market,
            target,
            float(t),
            int(step),
            prev_h,
            keys=target.marketkeys,
        )

    # ── core: per-path P&L before costs, and transaction costs ────────────────

    def _pnl(
        self,
        target,
        market,
        dates: Tensor,
        *,
        return_holdings: bool = False,
        panel: Optional[Dict[str, Tensor]] = None,
        idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, ...]:
        H, M = len(self.instruments), len(dates) - 1
        costs = [inst.cost for inst in self.instruments]

        # Panel-determined quantities (instrument grids, discount factors, liability)
        # do not depend on the network.  Under a frozen population (Algorithm 3) they
        # are priced once over the whole panel (``_build_panel``) and this batch only
        # *indexes* its rows — slicing a per-path quantity is bit-identical to pricing
        # the slice, since none of these involves a cross-path reduction.  ``panel is
        # None`` (online batching, or a direct ``_pnl`` call) prices from scratch.
        if panel is not None:
            Vo = panel["Vo"][..., idx]  # (H, M, n_batch)
            Vc = panel["Vc"][..., idx]
            D = panel["D"][:, idx]  # (M+1, n_batch)
            payoff_d = None  # liability is not slice-invariant — recompute below
            N = Vo.shape[-1]
        else:
            opens, closes = [], []
            for inst in self.instruments:
                vo, vc = inst.grids(dates, market)
                opens.append(vo)
                closes.append(vc)
            Vo = torch.stack(opens)  # (H, M, N)
            Vc = torch.stack(closes)
            N = Vo.shape[-1]
            disc = target.marketkeys.discount_curve(market)
            D = torch.stack(
                [to_path(disc.df(float(t)), N, Vo.dtype, Vo.device) for t in dates]
            )  # (M+1,N)
            payoff_d = None  # computed below, after the model rollout
        Vo_d = Vo * D[:-1].unsqueeze(0)
        Vc_d = Vc * D[1:].unsqueeze(0)

        # holdings from the model, rolled forward
        prev_h = torch.zeros(N, H, dtype=Vo.dtype, device=Vo.device)
        holdings = []
        for i in range(M):
            ctx = self._ctx(target, market, dates[i], i, prev_h)
            h = self.model(self.features.build(ctx))  # (N, H)
            if self.base_policy is not None:
                h = h + self.base_policy(ctx)
            if self.liveness is not None:
                # Hard, F_t-measurable gate: a dead liability has zero delta, so the
                # book is forced flat (and unwound) once it dies.  Multiplicative —
                # so it overrides both the network residual and the base policy,
                # which an additive prior cannot.  prev_h below carries the *gated*
                # holdings, keeping PrevHoldings consistent (dead path ⇒ prev 0).
                h = h * self.liveness(ctx)
            holdings.append(h)
            prev_h = h
        Hh = torch.stack(holdings).permute(2, 0, 1)  # (H, M, N)

        # hedge gains and the short liability, both discounted to 0
        gains = (Hh * (Vc_d - Vo_d)).sum(dim=(0, 1))  # (N,)
        # The liability does not depend on the model — compute it once, detached
        # (or reuse the panel-cached value when slicing a frozen population).
        if payoff_d is None:
            with torch.no_grad():
                payoff_d = self.liability(target, market).detach()  # (N,) to 0
        pnl_before_tc = gains - payoff_d

        # transaction cost: round-trip turnover on the (discounted) traded notional
        cost_vec = torch.tensor(costs, dtype=Vo.dtype, device=Vo.device).view(H, 1, 1)
        dHh = Hh.clone()
        dHh[:, 1:, :] = (
            Hh[:, 1:, :] - Hh[:, :-1, :]
        )  # opens (Δh; step 0 is the initial build)
        tc = (cost_vec * dHh.abs() * Vo_d).sum(dim=(0, 1))
        tc = tc + (cost_vec.squeeze(-1) * Hh[:, -1, :].abs() * Vc_d[:, -1, :]).sum(
            0
        )  # final unwind

        if return_holdings:
            return pnl_before_tc, tc, Hh
        return pnl_before_tc, tc

    def _build_panel(self, target, market, dates: Tensor) -> Dict[str, Tensor]:
        """Price the model-independent, **per-path** quantities ONCE over the panel
        currently in ``system.state`` — instrument grids and discount factors — and
        slice them per mini-batch in ``_pnl``.  These involve no cross-path reduction,
        so ``cache[..., idx]`` is bit-identical to pricing the slice.

        The liability is deliberately *not* cached: a GMxB ``price`` carries internal
        Monte Carlo (mortality / surrender) whose draw depends on the batch shape, so
        ``liability(panel)[idx] != liability(slice)``.  It is also cheap (~2% of
        ``_pnl`` vs ~70% for the grids), so it stays recomputed per step — keeping the
        result identical to the uncached path.  Valid only for a frozen panel
        (``resimulate=False``)."""
        opens, closes = [], []
        for inst in self.instruments:
            vo, vc = inst.grids(dates, market)
            opens.append(vo)
            closes.append(vc)
        Vo = torch.stack(opens).detach()  # (H, M, N)
        Vc = torch.stack(closes).detach()
        N = Vo.shape[-1]
        disc = target.marketkeys.discount_curve(market)
        D = torch.stack(
            [to_path(disc.df(float(t)), N, Vo.dtype, Vo.device) for t in dates]
        ).detach()  # (M+1, N)
        return {"Vo": Vo, "Vc": Vc, "D": D}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        target,
        *,
        system,
        calendar,
        rebalance_dates,
        market,
        n_paths: int = 10_000,
        n_epochs: int = 100,
        lr: float = 1e-3,
        optimizer=Adam,
        seed: Optional[int] = None,
        batch_size: Optional[int] = None,
        resimulate: bool = True,
        stratified: bool = False,
        weight_decay: float = 0.0,
        holdings_penalty: float = 0.0,
        holdings_threshold: float = 1.0,
        grad_clip: Optional[float] = None,
        val_paths: Optional[int] = None,
        val_every: int = 10,
        val_seed: Optional[int] = None,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        lr_schedule=None,
        restore_best: bool = True,
        path_weight_fn: Optional[Callable] = None,
        verbose: bool = True,
    ) -> List[float]:
        """Train the hedge.

        Parameters
        ----------
        n_paths : int
            Paths sampled fresh **each epoch** from the true SDE — the resampling
            that makes a held-out set less essential than in fixed-dataset
            supervised learning (there is no finite set to memorise).
        batch_size : int, optional
            Mini-batch size.  ``None`` → full batch (one optimiser step over all
            ``n_paths`` per epoch).  Otherwise each epoch takes
            ``n_paths // batch_size`` steps on fresh draws.  Risk-measure losses
            are *batch statistics*, so a small batch makes the estimate (the ES
            tail especially) noisier — prefer the largest batch that fits.  Falls
            back to the ctor ``batch_size`` if not given here.
        resimulate, stratified : bool
            *How* the mini-batches are drawn — the only knob separating a standard
            deep hedge from Jones et al.'s Algorithm 3, since the loss is unchanged.

            * ``resimulate=True`` (default) — **online**: a fresh batch is simulated
              from the SDE every step.  An ambiguity-averse loss re-clusters each
              fresh batch, so its regime *identities* drift step-to-step; fine for an
              SDH, but it leaves an AADH under-fit and its holdings over-dispersed
              (the entropic term never sees a stable regime).
            * ``resimulate=False`` — **fixed population (Algorithm 3)**: simulate one
              panel of ``n_paths`` once, freeze the loss's regime partition on it
              (stable identities + the paper's global masses, via
              ``loss.freeze_partition``), then draw mini-batches by *slicing* that
              panel.  ``stratified=True`` samples each batch by the global masses so
              every regime — including rare high-vol ones — appears with its global
              weight every step.  This is what reproduces the paper's AADH.

            The mode is resolved once into a batch plan; the loop is mode-agnostic.
            ``stratified`` is ignored when ``resimulate=True``.
        weight_decay : float
            Standard L2 ridge on the network weights (passed to the optimiser).
        holdings_penalty, holdings_threshold : float
            A **hinge** penalty applied only to holdings that exceed the
            threshold: ``+ holdings_penalty * mean(relu(|h| - holdings_threshold)^2)``.
            Below the threshold (default 1.0) it is exactly zero, so a well-posed
            hedge whose optimal positions are O(1) is left unbiased; it acts only
            as a soft barrier against runaway positions.  ``history_`` records the
            bare hedging loss; only the optimiser sees the penalty.
        grad_clip : float, optional
            Max global gradient norm (``clip_grad_norm_``); ``None`` disables it.
        val_paths : int, optional
            If set, every ``val_every`` epochs the loss is evaluated (no grad,
            eval mode) on a **fixed** held-out set simulated with ``val_seed`` —
            the out-of-sample signal.  Lands in ``self.val_history_`` as
            ``(epoch, loss)`` pairs.
        patience, min_delta, restore_best : early stopping
            With ``patience`` set (and validation enabled), training stops once the
            validation loss has not improved by at least ``min_delta`` for
            ``patience`` consecutive checks; if ``restore_best`` the best-validation
            weights are reloaded before returning.  ``patience=None`` (default)
            runs all ``n_epochs``.
        path_weight_fn : callable, optional
            ``path_weight_fn(market, dates) -> Tensor[N]`` returning a per-path
            density ``D`` (mean ≈ 1) used to **re-weight the loss** — the
            near-martingale measure change ``dQ/dP`` of
            :class:`qabit.core.nn.near_martingale.NMWeighting`.  It is called
            **per mini-batch on the freshly simulated market**, so the importance
            weights track the *same* paths that produced this batch's P&L — the
            batch/epoch resampling that gives the hedge its variance reduction is
            preserved (a single precomputed weight vector, tied to one seed, would
            go stale the moment the next batch resimulates).  The weights are
            detached (a fixed measure change), normalised to a probability vector
            (``w = D / Σ D``) and passed to ``loss.compute(pnl, tc, weights=w)``;
            gradients flow only through the hedge policy.  ``None`` (default) →
            ordinary uniform-weight training under P, byte-for-byte unchanged.

        Before training, the feature set is validated against ``target`` (see
        :meth:`FeatureSet.validate`).  Curve selection is no longer cross-checked:
        the hedge leg, the liability and the features all read through the target's
        own ``marketkeys``, so a mortality feature reads the very curve the
        liability is priced under by construction.
        """
        if patience is not None and val_paths is None:
            raise ValueError("early stopping (patience) requires val_paths to be set.")
        self.features.validate(target)
        opt = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        dates = as_tensor(rebalance_dates, dtype=torch.float64)

        # One-time input standardisation: freeze the FeatureStandardizer's mean/std
        # over a warmup sample spanning every rebalance step and path (so a
        # time-varying-but-batch-constant feature — time to maturity, a
        # deterministic rate/hazard — keeps a meaningful scale).
        std = getattr(self.model, "standardizer", None)
        if isinstance(std, FeatureStandardizer) and not bool(std.calibrated):
            with torch.no_grad():
                n_cal = min(n_paths, 8192)
                system.simulate(
                    calendar, n_cal, seed=None if seed is None else seed + 424_242
                )
                mkt0 = market(system) if callable(market) else market
                zeros = torch.zeros(n_cal, len(self.instruments))
                X = torch.cat(
                    [
                        self.features.build(self._ctx(target, mkt0, dates[i], i, zeros))
                        for i in range(len(dates) - 1)
                    ],
                    dim=0,
                )  # (M * n_cal, F)
                std.fit(X)

        bs = batch_size if batch_size is not None else self.batch_size
        bs = n_paths if bs is None else min(int(bs), n_paths)

        # The batching mechanism (online vs fixed pre-clustered population) is the
        # only difference between a standard deep hedge and Algorithm 3.  Resolve it
        # once into a plan; the epoch loop below is mode-agnostic.
        plan = self._batch_plan(
            target,
            system=system,
            calendar=calendar,
            market=market,
            n_paths=n_paths,
            batch_size=bs,
            seed=seed,
            resimulate=resimulate,
            stratified=stratified,
        )
        steps_per_epoch = plan.steps_per_epoch

        do_val = val_paths is not None
        if val_seed is None:
            val_seed = None if seed is None else seed + 1_000_000  # disjoint stream

        # Population mode (Algorithm 3): the panel is frozen, so price every
        # model-independent quantity once and slice it per batch in `_pnl`.  After
        # `_batch_plan` the system still holds the full population (it was just
        # simulated and not yet sliced), so we can build the cache from it directly.
        # The validation panel is identical at every check (fixed `val_seed`), so we
        # simulate it once and reuse both its state and its priced panel instead of
        # re-simulating each check.  Online mode keeps `panel=None` (recompute).
        panel = None
        val_panel = val_state = val_idx = None
        if not resimulate:
            mkt_pop = market(system) if callable(market) else market
            panel = self._build_panel(target, mkt_pop, dates)
            if do_val:
                with torch.no_grad():
                    system.simulate(calendar, val_paths, seed=val_seed)
                    mkt_v0 = market(system) if callable(market) else market
                    val_panel = self._build_panel(target, mkt_v0, dates)
                    val_state = system.get_state()
                    val_idx = torch.arange(val_paths)

        self.history_, self.val_history_ = [], []
        best_val, best_state, stale = float("inf"), None, 0
        for epoch in range(n_epochs):
            if lr_schedule is not None:
                cur_lr = lr
                for _start_ep, _lr_val in lr_schedule:
                    if epoch >= _start_ep:
                        cur_lr = _lr_val
                for _g in opt.param_groups:
                    _g["lr"] = cur_lr
            self.model.train()
            step_losses: List[float] = []
            for s in range(steps_per_epoch):
                plan.bind(epoch, s)  # online: fresh sim | population: slice the panel
                mkt = market(system) if callable(market) else market
                # Per-batch near-martingale importance weights dQ/dP, computed on
                # *this* batch's freshly simulated paths so they re-weight exactly
                # the P&L we are about to score (preserving batch resampling).
                w = None
                if path_weight_fn is not None:
                    with torch.no_grad():
                        D = path_weight_fn(mkt, dates)
                        w = (D / D.sum().clamp_min(1e-12)).detach()
                if holdings_penalty:
                    pnl, tc, Hh = self._pnl(
                        target,
                        mkt,
                        dates,
                        return_holdings=True,
                        panel=panel,
                        idx=getattr(plan, "_idx", None),
                    )
                    base = self.loss.compute(pnl, tc, weights=w)
                    excess = (Hh.abs() - holdings_threshold).clamp(min=0.0)
                    loss = base + holdings_penalty * excess.pow(2).mean()
                else:
                    pnl, tc = self._pnl(
                        target,
                        mkt,
                        dates,
                        panel=panel,
                        idx=getattr(plan, "_idx", None),
                    )
                    base = loss = self.loss.compute(pnl, tc, weights=w)
                opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                opt.step()
                step_losses.append(float(base.item()))
            epoch_loss = sum(step_losses) / len(step_losses)
            self.history_.append(epoch_loss)

            stop = False
            if do_val and (epoch % max(1, val_every) == 0 or epoch == n_epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    if val_panel is not None:
                        system.state = val_state  # features read the fixed val paths
                        mkt_v = market(system) if callable(market) else market
                    else:
                        system.simulate(calendar, val_paths, seed=val_seed)
                        mkt_v = market(system) if callable(market) else market
                    w_v = None
                    if path_weight_fn is not None:
                        D_v = path_weight_fn(mkt_v, dates)
                        w_v = (D_v / D_v.sum().clamp_min(1e-12)).detach()
                    pnl_v, tc_v = self._pnl(
                        target, mkt_v, dates, panel=val_panel, idx=val_idx
                    )
                    v = float(self.loss.compute(pnl_v, tc_v, weights=w_v).item())
                self.val_history_.append((epoch, v))
                # early stopping on the held-out loss
                if v < best_val - min_delta:
                    best_val, stale = v, 0
                    if restore_best:
                        best_state = deepcopy(self.model.state_dict())
                else:
                    stale += 1
                    if patience is not None and stale >= patience:
                        stop = True

            if verbose and (
                stop or epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1
            ):
                msg = f"epoch {epoch:4d}  {self.loss.name}={epoch_loss:.6e}"
                if do_val and self.val_history_:
                    msg += f"  val={self.val_history_[-1][1]:.6e}"
                if stop:
                    msg += f"  ⟂ early stop (no val improvement in {patience} checks)"
                print(msg)
            if stop:
                break

        if restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history_

    def _batch_plan(
        self,
        target,
        *,
        system,
        calendar,
        market,
        n_paths,
        batch_size,
        seed,
        resimulate,
        stratified,
    ) -> "_BatchPlan":
        """Resolve the mini-batch source once (see :meth:`fit`)."""
        if resimulate:
            return _OnlineBatches(
                self,
                system,
                calendar,
                n_paths=n_paths,
                batch_size=batch_size,
                seed=seed,
            )
        return _PopulationBatches(
            self,
            target,
            system,
            calendar,
            n_paths=n_paths,
            batch_size=batch_size,
            seed=seed,
            stratified=stratified,
        )

    # ── diagnostics: per-path holdings over the rebalance grid ────────────────

    def diagnose(
        self,
        target,
        *,
        system,
        calendar,
        rebalance_dates,
        market,
        n_paths: int = 20_000,
        seed: Optional[int] = None,
        resimulate: bool = True,
        path_weight_fn: Optional[Callable] = None,
    ) -> Dict[str, Tensor]:
        """Eval-mode pass returning the trained policy's holdings and the state
        needed to split paths into cohorts for holding-evolution plots.

        ``resimulate`` (default ``True``) simulates a fresh panel of ``n_paths`` for
        this pass.  Pass ``resimulate=False`` to evaluate on the panel **already**
        in ``system.state`` (simulate it once beforehand): this is how two trained
        policies are compared on *identical* paths — e.g. plotting ``AADH − SDH``
        holdings, where any path misalignment between two independent simulations
        would swamp the small genuine difference.  ``n_paths``/``seed`` are then
        ignored.

        Returns a dict with:
            ``holdings``     (H, M, N) — holding in each instrument over time;
            ``open_times``   (M,)      — the open time of each holding step;
            ``spot_grid``    (M, N)    — underlier spot at each open time (so the
                                         holding-vs-spot plot at any step ``k`` is
                                         ``holdings[:, k, :]`` against
                                         ``spot_grid[k]``);
            ``spot_T``       (N,)      — terminal underlier spot;
            ``pnl``, ``tc``  (N,)      — per-path P&L before costs and TC;
            ``liability``    (N,)      — per-path liability discounted to 0;
            ``weights``      (N,)|None — per-path near-martingale density ``D``
                                         (mean ≈ 1) if ``path_weight_fn`` is given,
                                         for measure-aware cohort statistics.
        """
        self.model.eval()
        dates = as_tensor(rebalance_dates, dtype=torch.float64)
        with torch.no_grad():
            if resimulate:
                system.simulate(calendar, n_paths, seed=seed)
            mkt = market(system) if callable(market) else market
            pnl, tc, Hh = self._pnl(target, mkt, dates, return_holdings=True)
            N = Hh.shape[-1]
            liability = self.liability(target, mkt)
            zeros_h = torch.zeros(N, len(self.instruments))
            # spot at each *open* time t_0..t_{M-1} → (M, N), aligned with holdings
            spot_grid = torch.stack(
                [
                    self._ctx(target, mkt, dates[i], i, zeros_h).spot()
                    for i in range(len(dates) - 1)
                ]
            )
            spot_T = self._ctx(target, mkt, dates[-1], len(dates) - 1, zeros_h).spot()
            weights = path_weight_fn(mkt, dates) if path_weight_fn is not None else None
            # ITM cohort from the terminal payoff, computed while the simulated
            # state is the one we just diagnosed.  Cashflow products (a GMxB) have
            # no single payoff → None, and the caller supplies its own split.
            try:
                itm_payoff = target.mc_payoff() > 0
            except NotImplementedError:
                itm_payoff = None
        return {
            "holdings": Hh,
            "open_times": dates[:-1],
            "spot_grid": spot_grid,
            "spot_T": spot_T,
            "pnl": pnl,
            "tc": tc,
            "liability": liability,
            "weights": weights,
            "itm_payoff": itm_payoff,
        }


__all__ = [
    "FeatureStandardizer",
    "HedgingMLP",
    "BoundedHedgingMLP",
    "DeepHedger",
]


# ─────────────────────────────────────────────────────────────────────────────
# Batch plans for fit() — online resimulation vs a fixed pre-clustered population
# ─────────────────────────────────────────────────────────────────────────────
#
# Each owns "what is this step's batch, and how does it become the live system
# state".  fit() just calls plan.bind(epoch, step) and then reads the state, so the
# training loop carries no mode flag.  Both lean only on the public engine surface:
# system.simulate(...) (online) and SystemState.select(idx) (population).


class _BatchPlan(ABC):
    """Supplies one mini-batch per training step by binding it as the live state."""

    steps_per_epoch: int

    @abstractmethod
    def bind(self, epoch: int, step: int) -> None:
        """Make this step's batch the live underlier state the rollout reads."""


class _OnlineBatches(_BatchPlan):
    """Online — re-simulate a fresh batch every step (standard deep hedging).

    The loss (if ambiguity-averse) re-clusters each fresh batch, so the partition is
    thawed here to make that explicit.
    """

    def __init__(
        self, hedger: "DeepHedger", system, calendar, *, n_paths, batch_size, seed
    ) -> None:
        self._system, self._cal = system, calendar
        self._bs, self._seed = batch_size, seed
        self.steps_per_epoch = max(1, n_paths // batch_size)
        thaw = getattr(hedger.loss, "thaw_partition", None)
        if callable(thaw):
            thaw()

    def bind(self, epoch: int, step: int) -> None:
        step_seed = (
            None
            if self._seed is None
            else self._seed + epoch * self.steps_per_epoch + step
        )
        self._system.simulate(self._cal, self._bs, seed=step_seed)


class _PopulationBatches(_BatchPlan):
    """Fixed population (Algorithm 3) — simulate one panel, freeze the regime
    partition on it, then draw mini-batches by slicing it.

    ``stratified`` draws each batch by the frozen global regime masses, so every
    regime appears with its global weight every step; otherwise the draw is uniform.
    A loss without ``freeze_partition`` (a plain SDH loss) simply gets uniform draws.
    """

    def __init__(
        self,
        hedger: "DeepHedger",
        target,
        system,
        calendar,
        *,
        n_paths,
        batch_size,
        seed,
        stratified,
    ) -> None:
        self._system = system
        self._bs, self._n, self._stratified = batch_size, n_paths, stratified
        self._gen = torch.Generator().manual_seed(0 if seed is None else int(seed))
        self.steps_per_epoch = max(1, n_paths // batch_size)
        # one fixed population, simulated once and kept for the whole run
        system.simulate(calendar, n_paths, seed=seed)
        self._population = system.get_state()
        self._groups, self._mu = self._freeze(hedger, target)

    @staticmethod
    def _freeze(hedger: "DeepHedger", target):
        """Pre-cluster the population once via the loss, returning per-regime index
        groups and the global masses (or ``(None, None)`` for a non-regime loss).

        We hand the loss the spot panel (pins the row count and the ``"spot"``
        feature source); the loss itself resolves any *named-factor* sources
        (rate / vol / mort) from the live system state, which is this same freshly
        simulated population — so a cross-factor or mortality partition clusters on
        its own factor, not the spot.  This is why the loss must read the system,
        not just the tensor we pass here.
        """
        freeze = getattr(hedger.loss, "freeze_partition", None)
        if not callable(freeze):
            return None, None
        spot = target.underlier.full_path()
        labels = freeze(spot)
        k = int(labels.max().item()) + 1
        groups = [torch.where(labels == i)[0] for i in range(k)]
        mu = torch.tensor([len(g) / len(labels) for g in groups])
        return groups, mu

    def bind(self, epoch: int, step: int) -> None:
        # Draw once; keep the indices so `_pnl` can slice the cached panel by the
        # *same* rows (no extra RNG advance — the generator stream is unchanged).
        self._idx = self._draw()
        self._system.state = self._population.select(self._idx)

    def _draw(self) -> Tensor:
        if not (self._stratified and self._groups):
            return torch.randint(self._n, (self._bs,), generator=self._gen)
        return torch.cat(
            [
                g[
                    torch.randint(
                        len(g), (max(int(self._bs * m), 1),), generator=self._gen
                    )
                ]
                for g, m in zip(self._groups, self._mu)
                if len(g) > 0
            ]
        )
