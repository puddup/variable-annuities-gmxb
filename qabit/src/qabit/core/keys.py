"""core/keys.py — one place for *resolver keys*: small, immutable selectors that
name *which* resource a class should read and know how to fetch it.

Two containers in qabit hand named resources to the rest of the stack:

* a :class:`~qabit.core.market.market.Market` — a bag of named *curves*
  (discount / volatility / hazard);
* an :class:`~qabit.core.dynamics.sde.system.SDESystem` — a bag of named *factor
  paths* (``"rate"``, ``"vol"``, ``"mort"``, the hedged underlier's ``"spot"``).

Rather than thread loose ``Optional[str]`` keys (``hazard_key``,
``underlier_key``, ``rate_key``, …) through every signature — noisy, easy to get
out of order, and resolved by ad-hoc string parsing scattered across modules —
a caller carries one small *key object* that owns the resolution:

* :class:`~qabit.core.market.curve.MarketKeys` (re-exported here) selects curves
  from a ``Market``::

      keys = MarketKeys(discount="ois", hazard="mort")
      r    = keys.discount_curve(market).inst_fwd(t)

* :class:`FactorKey` / :class:`FactorKeyMap` (defined here) select factor paths
  from an ``SDESystem`` state — the factor-side analogue::

      key  = FactorKey.parse("vol:1")              # variance column of "vol"
      var  = key.resolve(system.get_state())       # (N, E) panel

      keys = FactorKeyMap.of("spot", "rate", "vol:1")
      bundle = keys.resolve(system.get_state(), spot_path=spot)   # {name: (N,E)}

The two families are deliberately parallel (a frozen value type + a ``resolve``
that takes the container) so every class in qabit reads market data and factor
paths through the *same* shape of object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

from torch import Tensor

from qabit.core.market.curve import MarketKeys

#: The canonical name of the hedged underlier's spot panel — the one path that is
#: not a named SDESystem factor but the underlier lens the loss/feature reads.
SPOT = "spot"

#: What :meth:`FactorKey.parse` accepts: a ready key, a ``"name"`` / ``"name:col"``
#: string, or a ``(name, col)`` pair.
FactorKeyLike = Union["FactorKey", str, Tuple[str, int]]


@dataclass(frozen=True)
class FactorKey:
    r"""A name (+ output column) reference to **one** path panel in an SDESystem state.

    The factor-side analogue of a single :class:`MarketKeys` selector: it names
    *which* simulated path to read and knows how to resolve itself, so callers pass
    one small object instead of a raw string plus the ``name:col`` parsing and
    column-slicing logic that used to live in each consumer.

    * ``FactorKey("rate")``    → the short-rate factor, ``(N, E)``;
    * ``FactorKey("vol", 1)``  → column 1 (variance) of a Heston ``[S, V]`` factor;
    * ``FactorKey.spot()``     → the hedged underlier's spot panel (a sentinel name,
      supplied by the caller as ``spot_path`` rather than read from the system).
    """

    name: str
    col: int = 0

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def spot(cls) -> "FactorKey":
        """The sentinel key for the hedged underlier's spot panel."""
        return cls(SPOT, 0)

    @classmethod
    def parse(cls, spec: FactorKeyLike) -> "FactorKey":
        """Normalise any :data:`FactorKeyLike` to a :class:`FactorKey`.

        Accepts an existing key (returned as-is), a ``"name"`` or ``"name:col"``
        string, or a ``(name, col)`` pair.
        """
        if isinstance(spec, FactorKey):
            return spec
        if isinstance(spec, tuple):
            name, col = spec
            return cls(str(name), int(col))
        name, _, col = str(spec).partition(":")
        return cls(name, int(col) if col != "" else 0)

    # ── identity / display ─────────────────────────────────────────────────────
    @property
    def is_spot(self) -> bool:
        return self.name == SPOT

    @property
    def label(self) -> str:
        """Stable string form (``"vol:1"``; bare ``"vol"`` for column 0)."""
        return self.name if self.col == 0 else f"{self.name}:{self.col}"

    # ── resolution ─────────────────────────────────────────────────────────────
    def resolve(self, state, *, spot_path: Optional[Tensor] = None) -> Tensor:
        """Return this key's path panel ``(N, E)`` from an SDESystem ``state``.

        ``state`` is a :class:`~qabit.core.dynamics.sde.base.SystemState` (anything
        exposing ``factor_state(name)``).  The spot sentinel returns ``spot_path``;
        a named key reads the factor's panel and slices its output column.
        """
        if self.is_spot:
            if spot_path is None:
                raise RuntimeError(
                    "FactorKey.spot() needs a `spot_path` (the hedged underlier's "
                    "panel); none was supplied."
                )
            return spot_path
        try:
            fs = state.factor_state(self.name)
        except KeyError:
            available = list(getattr(state, "_factor_states", {})) or "unknown"
            raise KeyError(
                f"FactorKey('{self.name}') names a factor not in the SDESystem "
                f"(available: {available})."
            )
        p = fs.paths
        if p.dim() == 3:  # multi-output factor (e.g. Heston [S, V])
            return p[:, :, self.col]
        return p

    def __repr__(self) -> str:
        return f"FactorKey('{self.label}')"


@dataclass(frozen=True)
class FactorKeyMap:
    """An ordered, de-duplicated bundle of :class:`FactorKey`\\ s.

    The factor-side analogue of carrying several :class:`MarketKeys` selectors at
    once: it resolves them all against one state in a single call, returning the
    row-aligned ``{label: (N, E)}`` bundle a partitioner reads.  Generalises the
    old free-function ``resolve_bundle`` into a value type that travels with the
    object that needs the paths.
    """

    keys: Tuple[FactorKey, ...] = ()

    @classmethod
    def of(cls, *specs: FactorKeyLike) -> "FactorKeyMap":
        """Build from any mix of :data:`FactorKeyLike`, de-duplicated in order."""
        return cls._dedup(FactorKey.parse(s) for s in specs)

    @classmethod
    def _dedup(cls, keys: Iterable[FactorKey]) -> "FactorKeyMap":
        seen: set = set()
        out: List[FactorKey] = []
        for k in keys:
            if k.label not in seen:
                seen.add(k.label)
                out.append(k)
        return cls(tuple(out))

    @property
    def labels(self) -> List[str]:
        return [k.label for k in self.keys]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def resolve(self, state, *, spot_path: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Resolve every key against ``state`` → ``{label: (N, E)}``.

        The spot sentinel is filled from ``spot_path``; named keys are read from the
        live system state (which the caller has already pointed at the batch being
        scored, so every panel is row-aligned).
        """
        out: Dict[str, Tensor] = {}
        for key in self.keys:
            if key.is_spot and spot_path is None:
                continue  # caller has no spot panel to offer; skip gracefully
            out[key.label] = key.resolve(state, spot_path=spot_path)
        return out


__all__ = [
    "SPOT",
    "FactorKey",
    "FactorKeyLike",
    "FactorKeyMap",
    "MarketKeys",
]
