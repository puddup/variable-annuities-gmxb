"""dynamics/sde/base.py — the Factor protocol, per-factor state, and composite state.

A :class:`Factor` is one named component of an
:class:`~qabit.core.dynamics.sde.system.SDESystem`.  The system advances every
factor in a single explicit time-stepping loop; each factor may read the state
of the others (state coupling) and consumes its own slice of jointly correlated
increments (noise coupling).

State access is **by name** — there is no "primary"/column-0 convention.  Each
factor exposes a dynamic :class:`_FactorView` (``system["rate"]``) that always
reflects the latest ``simulate`` and plugs straight into ``Stock`` and the
stochastic curves.  The system supports any mix of factors: many rates, many
equities, or none of one kind — dependencies are expressed purely via names.

:class:`FactorState` is the immutable per-factor snapshot (``paths``, ``dates``,
``n_paths``) returned by ``system['name'].get_state()``; :class:`SystemState`
is the composite snapshot of the whole system.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor

from qabit.exceptions import StateError


# ─────────────────────────────────────────────────────────────────────────────
# Per-factor immutable simulation state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FactorState:
    """Immutable snapshot of one factor's simulation.

    All instruments read from here via the factor's dynamic view.
    """

    paths: Tensor  # (N, E) or (N, E, n_outputs)
    dates: Tensor  # (E,)
    n_paths: int

    def _idx(self, t: float) -> int:
        return int(torch.searchsorted(self.dates, torch.tensor(t)).item())

    def at(self, t: float) -> Tensor:
        return self.paths[:, self._idx(t)]

    def between(self, t1: float, t2: float) -> Tensor:
        i, j = self._idx(t1), self._idx(t2) + 1
        return self.paths[:, i:j]

    def at_dates(self, dates: list) -> Tensor:
        return self.paths[:, [self._idx(t) for t in dates]]

    def terminal(self) -> Tensor:
        return self.paths[:, -1]


# ─────────────────────────────────────────────────────────────────────────────
# Factor protocol
# ─────────────────────────────────────────────────────────────────────────────


class Factor(ABC):
    """One named component of an :class:`SDESystem`.

    A factor owns its model parameters and knows how to (a) produce its initial
    value and (b) advance itself by one time step given correlated noise and
    read-access to the other factors' states.

    Attributes
    ----------
    n_factors : int
        Number of independent Brownian drivers this factor consumes (the width
        of the noise slice it is handed).  ``0`` for deterministic factors.
    n_outputs : int
        Number of state columns this factor contributes (``1`` for a scalar
        like a short rate, ``2`` for e.g. Heston ``[S, V]``).
    """

    n_factors: int = 1
    n_outputs: int = 1

    def internal_corr(self) -> Optional["object"]:
        """Correlation among *this factor's own* Brownian drivers.

        Returns an ``(n_factors, n_factors)`` array, or ``None`` for the identity
        (the common single-driver case).  A multi-driver factor that couples its
        own drivers — e.g. Heston's leverage ``corr(dW^S, dW^V) = rho`` — returns
        its internal block here, so the cross-*factor* correlation matrix passed
        to :class:`SDESystem` only needs one entry per factor.
        """
        return None

    @abstractmethod
    def init(self, n_paths: int, device, dtype) -> Tensor:
        """Return the initial value, shape ``(N,)`` if ``n_outputs == 1`` else
        ``(N, n_outputs)``."""

    @abstractmethod
    def step(
        self,
        value: Tensor,
        prev: Mapping[str, Tensor],
        cur: Mapping[str, Tensor],
        dt: float,
        t: float,
        z: Optional[Tensor],
    ) -> Tensor:
        """Advance this factor by one step.

        Parameters
        ----------
        value : Tensor
            This factor's current value (``(N,)`` or ``(N, n_outputs)``).
        prev : mapping name -> Tensor
            Start-of-step spot of *all* factors (shape ``(N,)``; output column 0
            for multi-output factors).
        cur : mapping name -> Tensor
            End-of-step spot of factors already advanced this step (declaration
            order).  Lets a factor read a freshly updated driver.
        dt, t : float
            Step size and start time of the step.
        z : Tensor | None
            Correlated standard-normal increment(s): ``(N,)`` when
            ``n_factors == 1`` else ``(N, n_factors)``; ``None`` if deterministic.
        """

    def spot_of(self, value: Tensor) -> Tensor:
        """Scalar 'spot' view of this factor's value (output column 0)."""
        return value[:, 0] if value.dim() == 2 else value


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic per-factor view (so Stock(system["eq"]) and curves always see latest)
# ─────────────────────────────────────────────────────────────────────────────


class _FactorView:
    """Read-only, process-like proxy to one factor's *latest* state.

    Holds a weak reference to the owning :class:`SDESystem`, so ``get_state()``
    always returns the most recent ``simulate`` result — re-simulating the
    system transparently updates every view.  Implements just enough of the
    process surface (``get_state`` / ``state``) for ``Stock`` and the
    stochastic curve builders.
    """

    def __init__(self, system: "object", name: str) -> None:
        self._system_ref = weakref.ref(system)
        self.name = name

    def _system(self):
        sys = self._system_ref()
        if sys is None:
            raise StateError(
                f"The SDESystem owning factor '{self.name}' no longer exists."
            )
        return sys

    def get_state(self) -> FactorState:
        return self._system().factor_state(self.name)

    @property
    def state(self) -> Optional[FactorState]:
        try:
            return self.get_state()
        except StateError:
            return None

    def spot(self, t: float) -> Tensor:
        """Per-path spot at ``t``, shape ``(N,)`` — the uniform underlier accessor
        shared with :class:`Stock` and :class:`Fund` (a multi-output factor such
        as Heston ``[S, V]`` returns column 0).  Lets a feature read ``spot`` off
        a raw factor view (a GMxB fund passed as ``sde['equity']``) the same way
        it does off a ``Stock`` lens, with no type branch."""
        p = self.get_state().at(t)
        return p[:, 0] if p.dim() == 2 else p

    def simulate(self, *args, **kwargs):  # pragma: no cover - guard rail
        raise TypeError(
            "Simulate the SDESystem, not an individual factor view: "
            "call system.simulate(calendar, n_paths, ...), then read system['name']."
        )

    def __repr__(self) -> str:
        return f"<FactorView('{self.name}')>"


# ─────────────────────────────────────────────────────────────────────────────
# Composite state
# ─────────────────────────────────────────────────────────────────────────────


class SystemState:
    """Composite simulation state of an :class:`SDESystem`.

    Behaves like a :class:`FactorState` (``paths``, ``dates``, ``n_paths``
    plus ``at``/``between``/``terminal``) over a declaration-order stack of all
    factor outputs, **and** is subscriptable by factor name, returning the
    system's dynamic view::

        system.state.paths        # (N, E, total_outputs), declaration order
        system.state['rate']      # dynamic _FactorView (preferred: system['rate'])
    """

    def __init__(
        self,
        paths: Tensor,
        dates: Tensor,
        n_paths: int,
        factor_states: Dict[str, FactorState],
        views: Dict[str, _FactorView],
    ) -> None:
        self.paths = paths
        self.dates = dates
        self.n_paths = n_paths
        self._factor_states = factor_states
        self._views = views

    # ── FactorState-compatible accessors (over the stacked tensor) ──
    def _idx(self, t: float) -> int:
        return int(torch.searchsorted(self.dates, torch.tensor(t)).item())

    def at(self, t: float) -> Tensor:
        return self.paths[:, self._idx(t)]

    def between(self, t1: float, t2: float) -> Tensor:
        i, j = self._idx(t1), self._idx(t2) + 1
        return self.paths[:, i:j]

    def at_dates(self, dates: list) -> Tensor:
        return self.paths[:, [self._idx(t) for t in dates]]

    def terminal(self) -> Tensor:
        return self.paths[:, -1]

    # ── factor access by name ───────────────────────────────────────────
    def __getitem__(self, key: str) -> _FactorView:
        if key not in self._views:
            raise KeyError(f"No factor '{key}'. Available: {list(self._views)}.")
        return self._views[key]

    def __contains__(self, key: str) -> bool:
        return key in self._views

    def keys(self):
        return self._views.keys()

    def factor_state(self, key: str) -> FactorState:
        """Raw :class:`FactorState` of one factor (paths ``(N, E[, k])``)."""
        return self._factor_states[key]

    def select(self, idx: Tensor) -> "SystemState":
        """Return a state restricted to a subset of paths (advanced indexing on the
        path axis), sharing the dynamic views.

        Used by :meth:`~qabit.core.nn.hedger.DeepHedger.fit` in fixed-population mode
        to slice one pre-simulated panel into mini-batches: re-pointing
        ``system.state`` at ``population.select(idx)`` makes every downstream reader
        (the underlier views, the features, the loss's ``full_path()``) see just that
        mini-batch, with no re-simulation.
        """
        n = int(idx.numel())
        factor_states = {
            name: FactorState(fs.paths[idx], fs.dates, n)
            for name, fs in self._factor_states.items()
        }
        return SystemState(self.paths[idx], self.dates, n, factor_states, self._views)


__all__ = [
    "FactorState",
    "Factor",
    "SystemState",
]
