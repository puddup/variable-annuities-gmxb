"""PrimaryInstrument — base for all underlying instruments."""

from __future__ import annotations
from torch import Tensor


class PrimaryInstrument:
    """Lens over a process's :class:`FactorState`.

    No paths stored here — all data lives on ``self.process.state``.
    Simulation is driven by :class:`~qabit.core.dynamics.sde.SDESystem`
    (or :class:`~qabit.core.portfolio.portfolio.Portfolio`), not by the
    instrument itself: ``system.simulate(calendar, n_paths, ...)``, then
    products read ``self.process.get_state()`` on demand.

    Parameters
    ----------
    process    : SDESystem factor view (exposes ``get_state()``).
    init_state : object, optional — kept for subclass-specific overrides.
    cost       : float — transaction cost fraction (hedging extensions).
    """

    # Products must declare these for EventCalendar.build()
    observation_dates = None  # None → full fine grid to maturity
    maturity: float = 0.0

    def __init__(self, process, init_state=None, cost=0.0):
        self.process = process
        self.init_state = init_state
        self.cost = cost

    def _state(self):
        return self.process.get_state()

    def get_state(self):
        """Underlier's :class:`FactorState` — the uniform accessor shared with
        ``Fund`` and a raw factor view (so features need no type-specific path)."""
        return self.process.get_state()

    def spot(self, t: float) -> Tensor:
        """Market value at date ``t``, shape ``(N,)``."""
        raise NotImplementedError


__all__ = [
    "PrimaryInstrument",
]