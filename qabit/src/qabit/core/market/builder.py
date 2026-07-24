"""builder.py — assemble a Market from a simulated SDESystem.

The system is the single source of simulated dynamics; each market curve is a
lens over one of its factors (referenced by name), with optional flat overrides
for anything not simulated (e.g. a flat vol surface).
"""

from __future__ import annotations

from typing import Dict, Tuple

from qabit.core.dynamics.sde import SDESystem
from qabit.core.market.curve import StochasticDiscountCurve, StochasticHazardCurve
from qabit.core.market.market import Market


def build_market(
    system: SDESystem,
    mapping: Dict[str, Tuple[str, str]],  # market_name -> (factor_name, category)
    **flat_overrides: Dict[str, dict],
) -> Market:
    """Build a :class:`Market` from a simulated :class:`SDESystem`.

    Parameters
    ----------
    system : SDESystem
        An already-simulated system; curves read its per-factor views.
    mapping : dict
        Maps each market curve name to ``(factor_name, category)`` where
        ``category`` is ``"discount"`` or ``"hazard"``.  ``system[factor_name]``
        supplies the path-wise state.
    flat_overrides : dict, optional
        Extra non-simulated curves, e.g. ``volatility={"spx": VolSurface.flat(0.2)}``
        or ``discount={"ois": DiscountCurve.flat(0.03)}`` (these take precedence).

    Examples
    --------
    >>> sde = SDESystem({"rate": vasicek_factor(), "stock": gbm_factor(rate="rate")})
    >>> sde.simulate(T=1.0, dt=1/52, n_paths=50_000, seed=0)
    >>> mkt = build_market(sde, {"ois": ("rate", "discount")},
    ...                    volatility={"v": VolSurface.flat(0.2)})
    """
    discount: dict = {}
    hazard: dict = {}
    volatility: dict = {}

    for name, (factor_name, category) in mapping.items():
        view = system[factor_name]
        if category == "discount":
            discount[name] = StochasticDiscountCurve(view)
        elif category == "hazard":
            hazard[name] = StochasticHazardCurve(view)
        else:
            raise ValueError(
                f"Unknown category '{category}'. Use 'discount' or 'hazard'."
            )

    for cat, curves in flat_overrides.items():
        target = {"discount": discount, "hazard": hazard, "volatility": volatility}.get(
            cat
        )
        if target is None:
            raise ValueError(f"Unknown flat override category '{cat}'.")
        target.update(curves)

    return Market(discount=discount, volatility=volatility, hazard=hazard)


__all__ = ["build_market"]
