"""core/products/derivatives/va/variable_annuity.py — GMxB annuity framework.

Three-floor design (living G^X, death G^D, surrender G^S), single management
fee ``alpha`` on the account, single flat surrender charge ``p_S``.  No
excess-withdrawal penalty (β); no GMIB; no θ toggle (Bacinello vs Fontana
surrender is the choice of G^S).  See the framework note for the full spec.

Per-path valuation throughout — discount factors are never flattened.

Mortality is supplied through the :class:`~core.market.market.Market` (its
``hazard`` curves), selected by ``marketkeys.hazard`` exactly as discounting
is selected by ``marketkeys.discount``; death times are drawn reproducibly from the
instance ``seed`` fixed at construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor

from qabit.core.products.derivatives.base import BaseDerivative
from qabit.core.products.derivatives.va.guarantee import GuaranteeFloor
from qabit.core.market.curve import MarketKeys
from qabit.tools.util import to_path, nearest_index


class GMxBAnnuity(BaseDerivative, ABC):
    """Abstract variable annuity with GMxB guarantees.

    Holds all valuation machinery (account + floor forward pass, static MC,
    mixed LSMC, fair fee) and the uniform death/surrender cashflow defaults.
    Concrete products supply ``living_cashflow`` and ``terminal_cashflow``.
    """

    observation_dates = None

    def __init__(
        self,
        fund,
        living_floor: GuaranteeFloor,
        death_floor: Optional[GuaranteeFloor] = None,
        surrender_floor: Optional[GuaranteeFloor] = None,

        anniversary_schedule: Optional[List[float]] = None,
        surrender_schedule: Optional[List[float]] = None,

        premium: float = 1.0,
        maturity: float = 10.0,
        alpha: float = 0.0,
        p_S: float = 0.0,

        marketkeys: Optional[MarketKeys] = None,
        seed: Optional[int] = None,

    ) -> None:
        super().__init__(fund, maturity, marketkeys, cost=0.0)
        self.fund = fund
        self.premium = float(premium)
        self.alpha = float(alpha)
        self.p_S = float(p_S)
        self.seed = seed

        # death_floor   default -> Fixed(0)  (no death guarantee: max(A,0))
        # surrender_floor default -> Fixed(0) (account-only / Bacinello)
        self.living_floor = living_floor
        self.death_floor = death_floor or GuaranteeFloor.zero()
        self.surrender_floor = surrender_floor or GuaranteeFloor.zero()

        self._use_mixed = False
        self._mixed_kwargs = {}

        N = int(round(maturity))
        # The forward pass treats the grid's first node as inception (t=0, where
        # A_post = premium) and its last node as maturity (t=N, the terminal
        # cashflow).  Whatever interior anniversaries are supplied, we therefore
        # always pin 0 and N into the grid.  (The default below is the full
        # integer-year grid 0..N.)
        if anniversary_schedule is None:
            self.anniversary_schedule = [float(k) for k in range(0, N + 1)]
        else:
            self.anniversary_schedule = sorted(float(t) for t in anniversary_schedule)

        if surrender_schedule is not None:
            self.surrender_schedule = sorted(float(s) for s in surrender_schedule)
        else:
            self.surrender_schedule = [float(k) for k in range(1, N)]  # 1..N-1

    # ── abstract product hooks ────────────────────────────────────────────────

    @abstractmethod
    def living_cashflow(self, t: float, A_pre: Tensor, G_X_pre: Tensor) -> Tensor:
        """Contractual living benefit ΔB^L at anniversary t.  Tensor[N].

        This single quantity is both (a) the amount withdrawn from the account
        and subtracted from every floor in the forward pass, and (b) the
        survival cashflow paid to a living policyholder.  Zero off-schedule.
        """

    @abstractmethod
    def terminal_cashflow(self, A_post: Tensor, G_X_post: Tensor) -> Tensor:
        """Terminal survival cashflow at maturity N (excl. the last living amount)."""

    # ── uniform cashflows (no penalty on death / terminal) ────────────────────

    def death_cashflow(self, A_pre: Tensor, G_D_pre: Tensor) -> Tensor:
        return torch.maximum(A_pre, G_D_pre)

    def surrender_cashflow(self, t: float, A_pre: Tensor, G_S_pre: Tensor) -> Tensor:
        """Homogeneous charge p_S on the elected maximum."""
        return (1.0 - self.p_S) * torch.maximum(A_pre, G_S_pre)

    # ── anniversary grid & path forward pass ──────────────────────────────────

    def _paths(self, market):
        """Forward pass: account (A_pre, A_post) and the three floors at each
        anniversary, all on the integer-year grid.  All floors are reduced by
        the common living benefit; only the account bears the fee α.

        Returns a dict of lists indexed by anniversary position k=0..N.
        """
        st = self.fund.get_state()
        S = st.paths.float()
        if S.dim() == 3:
            S = S[:, :, 0]
        sim_dates = st.dates.float().tolist()
        Np = S.shape[0]
        dtype, device = S.dtype, S.device

        idxs = [nearest_index(sim_dates, float(t)) for t in self.anniversary_schedule]
        S0 = S[:, idxs[0]].clamp(min=1e-12)
        # fee-adjusted gross growth factor per unit premium at each anniversary
        growth = [
            (S[:, idx] / S0) * torch.exp(torch.tensor(-self.alpha * float(t)))
            for t, idx in zip(self.anniversary_schedule, idxs)
        ]

        A_pre = [None] * len(self.anniversary_schedule)
        A_post = [None] * len(self.anniversary_schedule)
        GX_m = [None] * len(self.anniversary_schedule)
        GD_m = [None] * len(self.anniversary_schedule)
        GS_m = [None] * len(self.anniversary_schedule)
        GX_p = [None] * len(self.anniversary_schedule)
        GD_p = [None] * len(self.anniversary_schedule)
        GS_p = [None] * len(self.anniversary_schedule)

        P0 = torch.full((Np,), self.premium, dtype=dtype, device=device)
        A_post[0] = P0.clone()
        GX_p[0] = self.living_floor.initial(P0)
        GD_p[0] = self.death_floor.initial(P0)
        GS_p[0] = self.surrender_floor.initial(P0)

        for k in range(1, len(self.anniversary_schedule)):
            n = float(self.anniversary_schedule[k])
            dt = float(self.anniversary_schedule[k] - self.anniversary_schedule[k - 1])
            grow = growth[k] / growth[k - 1].clamp(min=1e-12)
            A_pre[k] = (A_post[k - 1] * grow).clamp(min=0.0)

            GX_m[k] = self.living_floor.step(GX_p[k - 1], A_pre[k], n, dt)
            GD_m[k] = self.death_floor.step(GD_p[k - 1], A_pre[k], n, dt)
            GS_m[k] = self.surrender_floor.step(GS_p[k - 1], A_pre[k], n, dt)

            # the living benefit reduces the account and *all* floors equally
            dBL = self.living_cashflow(n, A_pre[k], GX_m[k])  # Tensor[N]
            A_post[k] = (A_pre[k] - dBL).clamp(min=0.0)
            GX_p[k] = GX_m[k] - dBL
            GD_p[k] = GD_m[k] - dBL
            GS_p[k] = GS_m[k] - dBL

        return {
            "anniv": self.anniversary_schedule,
            "A_pre": A_pre,
            "A_post": A_post,
            "GX_m": GX_m,
            "GD_m": GD_m,
            "GS_m": GS_m,
            "GX_p": GX_p,
            "Np": Np,
            "dtype": dtype,
            "device": device,
        }

    # ── pricing dispatch ──────────────────────────────────────────────────────

    def _pricer(self, market, t: float = 0.0) -> Tensor:
        dates = torch.tensor([t], dtype=torch.float32)
        return self._price_grid(market, dates)[0]

    def _price_grid(self, market, dates):
        if self._use_mixed:
            return self._mixed_price_grid(market, dates)
        return self._static_price_grid(market, dates)

    def list_mixed(self, **kwargs):
        self._use_mixed = True
        self._mixed_kwargs = kwargs

    def list_static(self):
        self._use_mixed = False

    # ── static (ordinary Monte Carlo) ─────────────────────────────────────────

    def _static_price_grid(self, market, dates):
        pp = self._paths(market)
        disc = self.marketkeys.discount_curve(market)
        Np, dtype, device = pp["Np"], pp["dtype"], pp["device"]
        anniv = pp["anniv"]
        N = int(round(self.maturity))
        hazard = self.marketkeys.hazard_curve(market)
        tau = hazard.tau(Np, seed=self.seed) if hazard is not None else None

        df = {n: to_path(disc.df(float(n)), Np, dtype, device) for n in anniv}

        T_val = dates.shape[0]
        grid = torch.zeros(T_val, Np, device=device, dtype=dtype)

        for vi, t_i in enumerate(dates.tolist()):
            df_i = to_path(disc.df(float(t_i)), Np, dtype, device).clamp(min=1e-12)
            val = torch.zeros(Np, device=device, dtype=dtype)
            for k, n in enumerate(anniv):
                if n == 0 or n < t_i - 1e-8:
                    continue
                fwd = df[n] / df_i
                is_terminal = n == N
                if is_terminal:
                    surv_cf = self.living_cashflow(
                        float(n), pp["A_pre"][k], pp["GX_m"][k]
                    ) + self.terminal_cashflow(pp["A_post"][k], pp["GX_p"][k])
                else:
                    surv_cf = self.living_cashflow(
                        float(n), pp["A_pre"][k], pp["GX_m"][k]
                    )

                if tau is not None:
                    alive = (tau > n).to(dtype)
                    val = val + fwd * surv_cf * alive
                    died = ((tau > anniv[k - 1]) & (tau <= n)).to(dtype)
                    death_cf = self.death_cashflow(pp["A_pre"][k], pp["GD_m"][k])
                    val = val + fwd * death_cf * died
                else:
                    val = val + fwd * surv_cf
            grid[vi] = val
        return grid

    # ── mixed (Least-Squares Monte Carlo) ─────────────────────────────────────

    def _mixed_price_grid(self, market, dates):
        """Mixed valuation by Least-Squares Monte Carlo (Longstaff–Schwartz).

        The mixed contract is the static contract *plus* a surrender option, so
        its value must dominate the static value and decrease monotonically in
        the surrender charge p_S, converging to the static value as p_S grows.
        We enforce both structurally by valuing the surrender option as an
        explicit non-negative premium on top of the static value:

          1. Regress the realised **contractual** continuation Ĉ_n (no surrender,
             penalty-free) on the state at each admissible date — independent of
             p_S.
          2. The holder surrenders at the first admissible date where the
             surrender value exceeds Ĉ_n; the gain (surrender − Ĉ_n) ≥ 0 is
             discounted and added to the static value.

        Booking the gain against the regressed continuation Ĉ_n (rather than the
        realised path continuation) is what guarantees V_mixed ≥ V_static: a
        noisy continuation estimate can no longer book a value-destroying
        surrender below the static stream.  Because the gain is non-negative and
        the surrender value is monotone decreasing in p_S, so is the option
        premium — hence the whole table is monotone with no dip below static.
        """
        from qabit.core.analytics.common.lsmc import _poly_basis, _ols

        pp = self._paths(market)
        disc = self.marketkeys.discount_curve(market)
        Np, dtype, device = pp["Np"], pp["dtype"], pp["device"]
        anniv = pp["anniv"]
        N = int(round(self.maturity))
        hazard = self.marketkeys.hazard_curve(market)
        tau = hazard.tau(Np, seed=self.seed) if hazard is not None else None
        # Degree 2 is the default: with a single continuation feature (A/P, the
        # other state variables being deterministic in these settings) a low
        # order keeps the continuation estimate smooth.  Dominance/monotonicity
        # no longer depend on the degree — they are structural (see above) — so
        # the degree only affects how closely the option premium is estimated.
        degree = self._mixed_kwargs.get("degree", 2)
        min_paths = self._mixed_kwargs.get("min_paths", 50)

        df = {n: to_path(disc.df(float(n)), Np, dtype, device) for n in anniv}

        # contractual per-anniversary cashflows (survival + death), no surrender
        cf = [torch.zeros(Np, device=device, dtype=dtype) for _ in anniv]
        for k, n in enumerate(anniv):
            if n == 0:
                continue
            is_terminal = n == N
            if is_terminal:
                surv_cf = self.living_cashflow(
                    float(n), pp["A_pre"][k], pp["GX_m"][k]
                ) + self.terminal_cashflow(pp["A_post"][k], pp["GX_p"][k])
            else:
                surv_cf = self.living_cashflow(float(n), pp["A_pre"][k], pp["GX_m"][k])
            if tau is not None:
                surv_cf = surv_cf * (tau > n).to(dtype)
                death_cf = self.death_cashflow(pp["A_pre"][k], pp["GD_m"][k])
                death_cf = death_cf * ((tau > anniv[k - 1]) & (tau <= n)).to(dtype)
                cf[k] = surv_cf + death_cf
            else:
                cf[k] = surv_cf

        # PV at time 0 of cf[k], then suffix sums → continuation PV at each date
        pv0 = [cf[k] * df[n] for k, n in enumerate(anniv)]
        suffix = [None] * len(anniv)
        acc = torch.zeros(Np, device=device, dtype=dtype)
        for k in range(len(anniv) - 1, -1, -1):
            acc = acc + pv0[k]
            suffix[k] = acc.clone()

        surr_set = {int(round(s)) for s in self.surrender_schedule}

        def _features(A_pre_k, GS_m_k):
            feats = [A_pre_k / self.premium]
            if float(GS_m_k.abs().max()) > 1e-9:  # non-degenerate G^S only
                feats.append(GS_m_k / self.premium)
            return torch.stack(feats, dim=1)

        # ── pass 1: penalty-independent continuation regression Ĉ_n ───────────
        chat = {}
        for k, n in enumerate(anniv):
            if n not in surr_set:
                continue
            alive = (
                (tau > n)
                if tau is not None
                else torch.ones(Np, dtype=torch.bool, device=device)
            )
            if int(alive.sum()) < min_paths:
                continue
            Y = (suffix[k] / df[n].clamp(min=1e-12))[alive]
            beta = _ols(
                _poly_basis(
                    _features(pp["A_pre"][k][alive], pp["GS_m"][k][alive]),
                    degree=degree,
                ),
                Y,
            )
            C = (
                _poly_basis(_features(pp["A_pre"][k], pp["GS_m"][k]), degree=degree)
                @ beta
            )
            chat[k] = C

        # ── pass 2: surrender-option premium ──────────────────────────────────
        # Fix each path's surrender date as the FIRST admissible date where
        # surrendering at zero charge beats continuation (chronological, so —
        # unlike an argmax over dates — it does not preferentially select the
        # date where the regression most underestimates continuation). That date
        # is charge-independent; the premium is the gain there re-valued at the
        # actual charge.  Fixed date + surrender value monotone in p_S ⇒ premium
        # ≥ 0 (dominance) and monotone decreasing in p_S, with no cross-date
        # noise and no penalty-dependent stopping.
        """
        ks = [k for k in range(len(anniv)) if k in chat]
        chosen = torch.full((Np,), -1, dtype=torch.long, device=device)
        for slot, k in enumerate(ks):
            n = anniv[k]
            alive = (
                (tau > n)
                if tau is not None
                else torch.ones(Np, dtype=torch.bool, device=device)
            )
            sv0 = self.surrender_cashflow(
                float(n), pp["A_pre"][k], pp["GS_m"][k]
            )
            crosses = alive & (sv0 > chat[k]) & (chosen < 0)
            chosen = torch.where(crosses, torch.full_like(chosen, slot), chosen)

        prem0 = torch.zeros(Np, device=device, dtype=dtype)
        for slot, k in enumerate(ks):
            sel = chosen == slot
            if bool(sel.any()):
                n = anniv[k]
                svp = self.surrender_cashflow(float(n), pp["A_pre"][k], pp["GS_m"][k])
                gain = (svp - chat[k]).clamp(min=0.0) * df[n]
                prem0 = torch.where(sel, gain, prem0)
        self._last_surrendered = chosen >= 0
        """
        # Fix each path's surrender date as the FIRST admissible date where
        # surrendering at ZERO charge beats the regressed continuation.
        # This date is charge-independent by construction.
        ks = [k for k in range(len(anniv)) if k in chat]
        chosen = torch.full((Np,), -1, dtype=torch.long, device=device)
        for slot, k in enumerate(ks):
            n = anniv[k]
            alive = (
                (tau > n)
                if tau is not None
                else torch.ones(Np, dtype=torch.bool, device=device)
            )
            # Use unpenalised surrender value: max(A, G^S)
            sv0_unpenalised = torch.maximum(pp["A_pre"][k], pp["GS_m"][k])
            crosses = alive & (sv0_unpenalised > chat[k]) & (chosen < 0)
            chosen = torch.where(crosses, torch.full_like(chosen, slot), chosen)

        prem0 = torch.zeros(Np, device=device, dtype=dtype)
        for slot, k in enumerate(ks):
            sel = chosen == slot
            if bool(sel.any()):
                n = anniv[k]
                # Actual surrender value with the current penalty
                svp = self.surrender_cashflow(float(n), pp["A_pre"][k], pp["GS_m"][k])
                gain = (svp - chat[k]).clamp(min=0.0) * df[n]
                prem0 = torch.where(sel, gain, prem0)
        self._last_surrendered = chosen >= 0

        # ── value = static stream + surrender-option premium ──────────────────
        T_val = dates.shape[0]
        grid = torch.zeros(T_val, Np, device=device, dtype=dtype)
        for vi, t_i in enumerate(dates.tolist()):
            df_i = to_path(disc.df(float(t_i)), Np, dtype, device).clamp(min=1e-12)
            val = prem0 / df_i  # premium PV carried to the valuation date
            for k, n in enumerate(anniv):
                if n == 0 or n < t_i - 1e-8:
                    continue
                val = val + (df[n] / df_i) * cf[k]
            grid[vi] = val

        return grid

    # ── fair fee ──────────────────────────────────────────────────────────────

    def fair_fee(self, market, lo=1e-5, hi=0.30, tol=1e-4) -> float:
        from scipy.optimize import brentq

        def residual(fee):
            self.alpha = fee
            return float(self.price(market)[0].mean()) - self.premium

        try:
            return brentq(residual, lo, hi, xtol=tol, maxiter=60)
        except ValueError:
            return float("nan")

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}(P={self.premium}, T={self.maturity}, "
            f"α={self.alpha:.2%}, p_S={self.p_S:.2%})>"
        )


# ── concrete products ──────────────────────────────────────────────────────────


class GMABAnnuity(GMxBAnnuity):
    """Guaranteed Minimum Accumulation Benefit.

    living = 0 throughout; terminal = max(A_T, G^X_T); death = max(A, G^D).
    A separable GMDB is just an independent ``death_floor``.
    """

    label = "GMAB"

    def living_cashflow(self, t, A_pre, G_X_pre):
        return torch.zeros_like(A_pre)

    def terminal_cashflow(self, A_post, G_X_post):
        return torch.maximum(A_post, G_X_post)


class GMWBAnnuity(GMxBAnnuity):
    """Guaranteed Minimum Withdrawal Benefit.

    living = g at each withdrawal anniversary; terminal = g + max(A_{N^+}, G^X_{N^+}).
    """

    label = "GMWB"

    def __init__(
        self,
        *args,
        withdrawal: Optional[float] = None,
        withdrawal_schedule: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        N = int(round(self.maturity))
        if withdrawal_schedule is None:
            withdrawal_schedule = [float(k) for k in range(1, N + 1)]
        self.withdrawal_schedule = sorted(float(s) for s in withdrawal_schedule)
        self.g = float(withdrawal) if withdrawal is not None else self.premium / N

    def _on_schedule(self, t):
        return any(abs(t - s) < 1e-6 for s in self.withdrawal_schedule)

    def living_cashflow(self, t, A_pre, G_X_pre):
        if self._on_schedule(t):
            return torch.full_like(A_pre, self.g)
        return torch.zeros_like(A_pre)

    def terminal_cashflow(self, A_post, G_X_post):
        return torch.maximum(A_post, G_X_post)


class GMDBAnnuity(GMxBAnnuity):
    """Guaranteed Minimum Death Benefit.

    A *pure death* rider: there is no living/accumulation guarantee, so a holder
    who survives to maturity receives only the account ``A_N`` (the floor never
    bites on survival).  The guarantee applies **only on death**, where the
    benefit is ``max(A_τ, G^D_τ)`` paid at the anniversary following the drawn
    death time τ.

    Construct it with a non-trivial ``death_floor`` (the death guarantee — a
    return-of-premium ``Fixed``, a ``RollUp`` roll-up, or a ``Ratchet`` lock-in)
    and leave ``living_floor`` at the degenerate zero floor::

        GMDBAnnuity(fund=fund,
                    living_floor=GuaranteeFloor.zero(),
                    death_floor=GuaranteeFloor.rollup(P, delta=0.03),
                    premium=P, maturity=T,
                    marketkeys=MarketKeys(discount="ois", hazard="mort"), ...)

    With no hazard key set (``marketkeys.hazard=None``) and no hazard curve in
    the market, the death leg is
    never triggered and the contract degenerates to the bare account — so a GMDB
    is only meaningful under a mortality model, which is exactly the regime where
    the ``HazardRate`` / ``Survival`` features and (optionally) a stochastic
    hazard factor carry the mortality state the hedge must react to.
    """

    label = "GMDB"

    def living_cashflow(self, t, A_pre, G_X_pre):
        return torch.zeros_like(A_pre)

    def terminal_cashflow(self, A_post, G_X_post):
        # Survival to maturity pays the account only — the guarantee is a *death*
        # benefit, captured entirely by ``death_cashflow`` along the path.
        return A_post


__all__ = [
    "GMxBAnnuity",
    "GMABAnnuity",
    "GMWBAnnuity",
    "GMDBAnnuity",
]