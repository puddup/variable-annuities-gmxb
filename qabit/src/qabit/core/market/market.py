"""market.py — Market: pure pricing context container."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Union


# ── Curve protocols (simplistic, for type hints) ───────────────────────


class DiscountCurveLike(Protocol):
    def df(self, t: float) -> Union[float, "Tensor"]:
        ...

    @property
    def is_flat(self) -> bool:
        ...


class HazardCurveLike(Protocol):
    def survival(self, t: float) -> Union[float, "Tensor"]:
        ...

    @property
    def is_flat(self) -> bool:
        ...


class VolSurfaceLike(Protocol):
    def vol(self, t: float, K: float) -> float:
        ...

    @property
    def is_flat(self) -> bool:
        ...


# ── Market ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Market:
    """Unified pricing context passed to every ``product.price(market)``.

    The market is a pure container – it is completely agnostic about
    whether a curve is deterministic or stochastic.  Construction is
    done manually or via :func:`~core.market.builder.build_market`.

    Examples
    --------
    >>> mkt = Market(discount={"ois": DiscountCurve.flat(0.05)})
    >>> mkt = Market(
    ...     discount={"ois": DiscountCurve.flat(0.05)},
    ...     volatility={"spx": VolSurface.flat(0.20)},
    ... )
    """

    discount: Dict[str, DiscountCurveLike] = field(default_factory=dict)
    volatility: Dict[str, VolSurfaceLike] = field(default_factory=dict)
    hazard: Dict[str, HazardCurveLike] = field(default_factory=dict)

    # ── accessors ──────────────────────────────────────────────────────

    def _get(self, category: dict, key: Optional[str], label: str):
        if not category:
            raise KeyError(f"Market has no {label} curves.")
        if key is None:
            return next(iter(category.values()))
        if key not in category:
            raise KeyError(f"No {label} curve '{key}'. Available: {list(category)}.")
        return category[key]

    def get_discount(self, key: Optional[str] = None) -> DiscountCurveLike:
        return self._get(self.discount, key, "discount")

    def get_volatility(self, key: Optional[str] = None) -> VolSurfaceLike:
        return self._get(self.volatility, key, "volatility")

    def get_hazard(self, key: Optional[str] = None) -> HazardCurveLike:
        return self._get(self.hazard, key, "hazard")


__all__ = [
    "DiscountCurveLike",
    "HazardCurveLike",
    "VolSurfaceLike",
    "Market",
]
