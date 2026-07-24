"""core/products/underlying/composite/fund.py — Fund (composite primary instrument).

A Fund is a weighted basket of already-simulated processes. Its value at
time ``t`` is the weighted sum of the component paths::

    F_t = Σ_i  w_i · (S^i_t / S^i_0)  ·  F_0

The weights are constant-mix (rebalanced continuously) so the fund path
is the product of the weighted returns, not the sum. In the log-return
convention the continuous-mix fund has dynamics::

    dF / F = Σ_i w_i (dS^i / S^i)

which, discretized, gives::

    F_{k+1} / F_k = Σ_i w_i (S^i_{k+1} / S^i_k)

Example
-------
.. code-block:: python

    from qabit.core.products.underlying.composite.fund import Fund

    fund = Fund({'equity': sde['equity'], 'bond': sde['bond']},
                weights={'equity': 0.60, 'bond': 0.40},
                f0=100.0)

    # After SDESystem.simulate(...):
    fund_paths = fund.get_state()      # FactorState with .paths, .dates

The Fund exposes the same ``.get_state()`` interface as an SDESystem factor view, so
it can be passed directly to VariableAnnuity as ``equity_process``.
"""

from __future__ import annotations

from typing import Dict

import torch

from qabit.core.dynamics.sde.base import FactorState


class Fund:
    """Constant-mix fund — composite of simulated processes.

    Parameters
    ----------
    components : dict[str, process]  — named component processes (already simulated).
    weights    : dict[str, float]    — portfolio weights (must sum to 1).
    f0         : float               — initial fund value (default: 1.0).
    """

    n_factors = 1  # looks like a single-factor process to downstream consumers

    def __init__(
        self,
        components: Dict[str, object],
        weights: Dict[str, float],
        f0: float = 1.0,
    ) -> None:
        if set(components) != set(weights):
            raise ValueError(
                f"component keys {set(components)} ≠ weight keys {set(weights)}"
            )
        w_sum = sum(weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1; got {w_sum:.6f}")
        self.components = components
        self.weights = weights
        self.f0 = f0
        self._state = None

    def build(self) -> "Fund":
        """Build the fund paths from the already-simulated component paths.

        Must be called after the underlying SDESystem has been simulated.
        """
        # Collect component paths and dates
        comp_data = {}
        dates = None
        for name, proc in self.components.items():
            st = proc.get_state()
            S = st.paths.float()
            if S.dim() == 3:
                S = S[:, :, 0]
            comp_data[name] = S
            if dates is None:
                dates = st.dates.float()

        N, E = list(comp_data.values())[0].shape

        # Build fund via constant-mix: F_k+1 / F_k = Σ w_i (S^i_{k+1} / S^i_k)
        fund_paths = torch.ones(N, E, device=dates.device, dtype=dates.dtype)
        fund_paths[:, 0] = self.f0

        for k in range(E - 1):
            mix_return = torch.zeros(N, device=dates.device, dtype=dates.dtype)
            for name, S in comp_data.items():
                w = self.weights[name]
                mix_return = mix_return + w * (S[:, k + 1] / S[:, k].clamp(min=1e-12))
            fund_paths[:, k + 1] = fund_paths[:, k] * mix_return

        self._state = FactorState(paths=fund_paths, dates=dates, n_paths=N)
        return self

    def get_state(self) -> FactorState:
        """Return FactorState with fund paths."""
        if self._state is None:
            self.build()
        return self._state

    def spot(self, t: float) -> Tensor:
        """Fund value at date ``t``, shape ``(N,)``.

        Mirrors :meth:`Stock.spot` so every underlier — ``Stock``, ``Fund`` or a
        raw single-factor view — exposes the *same* ``spot(t)`` accessor.  This is
        what lets :meth:`FeatureContext.spot` delegate uniformly instead of
        branching on whether the underlier happens to carry a ``spot`` lens.
        """
        p = self.get_state().at(t)
        return p[:, 0] if p.dim() == 2 else p

    def __repr__(self) -> str:
        parts = ", ".join(f"{n}:{w:.0%}" for n, w in self.weights.items())
        return f"<Fund({parts}, F0={self.f0})>"


__all__ = [
    "Fund",
]
