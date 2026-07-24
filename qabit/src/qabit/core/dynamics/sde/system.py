"""dynamics/sde/system.py — SDESystem: the one and only process abstraction.

``SDESystem`` simulates any number of :class:`Factor` objects (rates, equities,
hazards, …) in a single explicit time-stepping loop, with **correlated noise**
*and* **state coupling** (one factor reading another's current value).  It is the
only stochastic-process class in qabit — there is no ``BaseProcess`` and no
separate noise-only "correlated processes" engine.

Example
-------
Dict construction (explicit names; products reference factors by name)::

    sde = SDESystem(
        {
            "rate": VasicekRateFactor(kappa=0.5, theta=0.03, sigma=0.01, r0=0.03),
            "stock": GeometricBrownianFactor(sigma=0.2, s0=100.0, rate="rate"),
            "vol": HestonFactor(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
                                s0=100.0, v0=0.04, rate="rate"),
        },
        correlation=[[1.0,  0.2,  0.1],   # rate-stock-vol asset cross-correlation
                     [0.2,  1.0,  0.0],
                     [0.1,  0.0,  1.0]],
    )
    cal = EventCalendar.from_horizon(T=5.0, dt=0.01)
    paths = sde.simulate(cal, n_paths=10_000)   # (N, E, total_outputs)
    sde["rate"]      # dynamic view → StochasticDiscountCurve(sde["rate"])

Correlation convention
----------------------
The ``correlation`` matrix is **factor-level**: one row/column per *stochastic*
factor (``n_factors > 0``, declaration order), giving the correlation between
each pair's primary (asset/rate) Brownian driver.  A factor's *internal* driver
correlation — e.g. Heston's ``corr(dW^S, dW^V) = rho`` — is supplied by the
factor itself via :meth:`Factor.internal_corr` and need not appear in this
matrix.  Deterministic factors (``ConstantRateFactor``, deterministic hazard)
carry no driver and are absent from the matrix.

Update order is declaration order (Gauss-Seidel): declare the short rate before
the equity that reads it, so the equity integrates the trapezoidal rate average
over the step — the same quadrature ``StochasticDiscountCurve`` uses — making the
discounted asset a discrete martingale.

Calendar
--------
``simulate`` always takes an :class:`~qabit.core.market.calendar.EventCalendar`.
The calendar is the single source of truth for the date grid: build it from a
list of products (:meth:`EventCalendar.build`) or directly from a horizon
(:meth:`EventCalendar.from_horizon`).  This guarantees observation dates are
exact entries (no ``searchsorted`` snapping error) and that the Euler step is
bounded by the calendar's fine ``dt``.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from qabit.core.dynamics.sde.base import Factor, FactorState, SystemState, _FactorView
from qabit.exceptions import ShapeError, StateError
from qabit.tools.random import NumberGenerator, PseudoRandom
from qabit.core.analytics.math import cholesky_psd


class SDESystem:
    r"""A bundle of state-coupled, noise-coupled factors simulated as one model.

    Parameters
    ----------
    factors : dict[str, Factor]
        The named factors.  Declaration order matters (Gauss-Seidel update).
    correlation : array, optional
        Factor-level correlation matrix over the stochastic factors (see module
        docstring).  Identity if omitted.
    num_gen : NumberGenerator, optional
        RNG strategy for noise (default :class:`PseudoRandom`).
    device, dtype : torch device / dtype
    """

    def __init__(
        self,
        factors: Dict[str, Factor],
        correlation=None,
        *,
        num_gen: Optional[NumberGenerator] = None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ) -> None:
        if not isinstance(factors, dict):
            raise TypeError("factors must be a dictionary of {name: Factor}")
        if not all(isinstance(f, Factor) for f in factors.values()):
            raise TypeError("all values in factors must be Factor instances")
        self.factors: Dict[str, Factor] = factors
        self._order: List[str] = list(self.factors)
        self.num_gen = num_gen or PseudoRandom()
        self.device = device
        self.dtype = dtype
        self.state: Optional[SystemState] = None

        # Noise budget and per-factor driver offsets (deterministic add 0).
        self.n_factors = sum(f.n_factors for f in self.factors.values())
        self.n_outputs = sum(f.n_outputs for f in self.factors.values())
        off, self._offsets = 0, {}
        for name, f in self.factors.items():
            self._offsets[name] = off
            off += f.n_factors

        self._views = {name: _FactorView(self, name) for name in self.factors}
        self._L = self._build_cholesky(correlation)

    # ── construction helpers ────────────────────────────────────────────

    def _build_cholesky(self, correlation_matrix) -> Optional[Tensor]:
        """Expand the factor-level matrix + per-factor internal blocks to a
        driver-level correlation matrix, then return its lower Cholesky factor."""
        if self.n_factors == 0:
            return None

        # ── stochastic factors only ─────────────────────────────────────
        stoch = [(n, f) for n, f in self.factors.items() if f.n_factors > 0]

        # M  = driver-level correlation matrix (F × F)
        # C  = user-supplied cross-factor correlation matrix (n_stoch × n_stoch)
        # F  = total number of individual Brownian drivers
        F = self.n_factors
        M = torch.zeros(F, F, dtype=self.dtype, device=self.device)

        # ── diagonal blocks: each factor's own driver correlation ──────
        offset = 0  # running driver index
        start = {}  # first driver index per factor
        for name, factor in stoch:
            start[name] = offset
            block = factor.internal_corr()
            if block is None:
                # None → identity
                block = torch.eye(
                    factor.n_factors, dtype=self.dtype, device=self.device
                )
            else:
                block = torch.as_tensor(block, dtype=self.dtype, device=self.device)

            if block.shape != (factor.n_factors, factor.n_factors):
                raise ShapeError(
                    f"Factor '{name}'.internal_corr() must be "
                    f"({factor.n_factors},{factor.n_factors}); got {tuple(block.shape)}."
                )

            end = offset + factor.n_factors
            # place on diagonal
            M[offset:end, offset:end] = block
            offset = end

        if correlation_matrix is not None:
            C = torch.as_tensor(
                correlation_matrix, dtype=self.dtype, device=self.device
            )
            S = len(stoch)
            if C.shape != (S, S):
                raise ShapeError(
                    f"correlation must be ({S},{S}) — one row per stochastic factor "
                    f"{[n for n, _ in stoch]}; got {tuple(C.shape)}."
                )
            for i, (name_i, _) in enumerate(stoch):
                for j, (name_j, _) in enumerate(stoch):
                    if i != j:
                        M[start[name_i], start[name_j]] = C[i, j]

        return cholesky_psd(M)

    # ── introspection ───────────────────────────────────────────────────

    @property
    def factor_names(self) -> List[str]:
        return list(self.factors)

    def __getitem__(self, key: Union[str, int]) -> _FactorView:
        if isinstance(key, int):
            key = self._order[key]
        if key not in self._views:
            raise KeyError(f"No factor '{key}'. Available: {self._order}.")
        return self._views[key]

    def __len__(self) -> int:
        return len(self.factors)

    def get_state(self) -> SystemState:
        if self.state is None:
            raise StateError("SDESystem has no state — run simulate() first.")
        return self.state

    def factor_state(self, name: str) -> FactorState:
        if self.state is None:
            raise StateError("SDESystem has no state — run simulate() first.")
        return self.state.factor_state(name)

    def __repr__(self) -> str:
        body = ", ".join(f"{k}:{v!r}" for k, v in self.factors.items())
        return f"<SDESystem({{{body}}})>"

    # ── simulation ──────────────────────────────────────────────────────
    def simulate(
        self,
        calendar,
        n_paths: int = 10_000,
        *,
        z=None,
        antithetic: bool = False,
        seed: Optional[int] = None,
        generator=None,
        device=None,
        dtype=None,
    ) -> Tensor:
        """Simulate and return the composite path tensor ``(N, E, total_outputs)``.

        Parameters
        ----------
        calendar : EventCalendar
            Date grid (sorted, deduplicated).  Build via
            :meth:`EventCalendar.build` from a product list or
            :meth:`EventCalendar.from_horizon` for a plain horizon/dt grid.
        n_paths : int
            Number of Monte-Carlo paths.

        Other Parameters
        ----------------
        z, antithetic, seed, generator, device, dtype : standard simulation knobs.

        Notes
        -----
        Per-factor results are read by name afterwards (``sde["rate"]``).
        """
        from qabit.core.market.calendar import EventCalendar

        if not isinstance(calendar, EventCalendar):
            raise TypeError(
                "SDESystem.simulate() takes an EventCalendar.  Build one with "
                "EventCalendar.from_horizon(T, dt) or EventCalendar.build(products, dt)."
            )

        dev = device or self.device
        dtp = dtype or self.dtype
        dates = calendar.as_tensor(dtype=dtp, device=dev)

        if generator is None and seed is not None:
            generator = torch.Generator(device=dev)
            generator.manual_seed(seed)

        n_steps = len(dates) - 1
        dt_vec = (dates[1:] - dates[:-1]).to(dtp)
        z_corr = self._resolve_noise(
            z, n_paths, n_steps, antithetic, generator, dev, dtp
        )

        buffers: Dict[str, Tensor] = {}
        current: Dict[str, Tensor] = {}
        for name, f in self.factors.items():
            v0 = f.init(n_paths, dev, dtp)
            if f.n_outputs == 1:
                buf = torch.empty(n_paths, n_steps + 1, device=dev, dtype=dtp)
                buf[:, 0] = v0 if v0.dim() == 1 else v0[:, 0]
            else:
                buf = torch.empty(
                    n_paths, n_steps + 1, f.n_outputs, device=dev, dtype=dtp
                )
                buf[:, 0, :] = v0
            buffers[name] = buf
            current[name] = v0

        for i in range(n_steps):
            dt_i = float(dt_vec[i])
            t = float(dates[i])
            prev = {name: self.factors[name].spot_of(v) for name, v in current.items()}
            cur_spot: Dict[str, Tensor] = {}
            new_vals: Dict[str, Tensor] = {}
            for name, f in self.factors.items():
                z_i = self._slice_noise(z_corr, name, i, f)
                nv = f.step(current[name], prev, cur_spot, dt_i, t, z_i)
                new_vals[name] = nv
                cur_spot[name] = f.spot_of(nv)
            current = new_vals
            for name, f in self.factors.items():
                if f.n_outputs == 1:
                    buffers[name][:, i + 1] = (
                        current[name]
                        if current[name].dim() == 1
                        else current[name][:, 0]
                    )
                else:
                    buffers[name][:, i + 1, :] = current[name]

        self._assemble_state(buffers, dates, n_paths)
        return self.state.paths

    # ── helpers ─────────────────────────────────────────────────────────
    def _resolve_noise(self, z, n_paths, n_steps, antithetic, generator, dev, dtp):
        if self.n_factors == 0:
            return None
        if z is not None:  # already-correlated increments injected for testing
            if z.dim() == 2:
                z = z.unsqueeze(-1)
            if z.shape[-1] != self.n_factors:
                raise ShapeError(
                    f"Injected noise has {z.shape[-1]} factors; system expects {self.n_factors}."
                )
            return z
        z_raw = self.num_gen.draw(
            n_paths,
            n_steps,
            self.n_factors,
            antithetic=antithetic,
            generator=generator,
            device=dev,
            dtype=dtp,
        )
        if z_raw.dim() == 2:
            z_raw = z_raw.unsqueeze(-1)
        return z_raw @ self._L.T

    def _slice_noise(self, z_corr, name, i, f) -> Optional[Tensor]:
        if f.n_factors == 0 or z_corr is None:
            return None
        off = self._offsets[name]
        if f.n_factors == 1:
            return z_corr[:, i, off]
        return z_corr[:, i, off : off + f.n_factors]

    def _assemble_state(self, buffers, dates, n_paths) -> None:
        factor_states = {
            name: FactorState(buf, dates, n_paths) for name, buf in buffers.items()
        }
        cols = [
            (buf.unsqueeze(-1) if buf.dim() == 2 else buf)
            for buf in (buffers[name] for name in self.factors)
        ]
        composite = torch.cat(cols, dim=2)
        self.state = SystemState(composite, dates, n_paths, factor_states, self._views)

    def clone(self) -> "SDESystem":
        new = copy.copy(self)
        new.state = None
        new._views = {name: _FactorView(new, name) for name in new.factors}
        return new


__all__ = ["SDESystem"]
