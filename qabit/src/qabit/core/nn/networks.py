"""nn/networks.py — the hedging-policy network skeletons.

The function-approximator side of deep hedging, kept apart from the
training/pricing loop in :mod:`qabit.core.nn.hedger`:

* :class:`FeatureStandardizer` — a frozen, calibrated input standardisation;
* :class:`HedgingMLP`          — the general policy net (linear head, optional
                                 ``max_position`` tanh bound);
* :class:`BoundedHedgingMLP`   — a sibling with a smooth sigmoid head that confines
                                 each holding to a fixed interval, for hedging a
                                 single option whose delta lives in a known range.

:class:`DeepHedger` accepts any ``nn.Module`` mapping ``(N, F)`` features to
``(N, H)`` hedge ratios, so these are conveniences, not requirements.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────


class FeatureStandardizer(nn.Module):
    r"""Fixed affine input standardizer ``(x - mean) / std``.

    Calibrated **once** over a sample spanning all rebalance steps *and* paths
    (see :meth:`DeepHedger.fit`), so a feature that is constant across the batch
    at each step but varies over time — ``TimeToMaturity`` (0…T), a deterministic
    rate or hazard — keeps a meaningful scale and stays informative.

    Deliberately **not** ``BatchNorm1d``.  BatchNorm standardises each feature by
    the *current batch* at one step, which is pathological here: a per-step-constant
    column has ~0 batch variance, so (a) in training it is mapped to 0 and the
    policy never sees the clock, and (b) at eval its running variance has collapsed,
    so the column is divided by ≈√eps and explodes — a 6-year ``TimeToMaturity``
    blowing the holdings up to ~1e6 (the exact zig-zag-to-millions failure on the
    long-dated GMxB, where the bug is amplified by the long horizon).  A frozen
    standardizer removes both: no train/eval mismatch, no batch coupling, and a
    constant-in-time column (std≈0 → set to 1) simply passes through as ``x-mean``.
    """

    def __init__(self, n_in: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        self.eps = eps

    @torch.no_grad()
    def fit(self, X: Tensor) -> None:
        """Freeze mean/std from a sample ``X`` of shape ``(K, n_in)``."""
        m = X.mean(dim=0)
        s = X.std(dim=0)
        s = torch.where(
            s < self.eps, torch.ones_like(s), s
        )  # constant col → pass-through
        self.mean.copy_(m)
        self.std.copy_(s)
        self.calibrated.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class HedgingMLP(nn.Module):
    """Plain MLP mapping (N, F) features to (N, H) hedge ratios.

    With ``normalize=True`` (default) the inputs pass through a
    :class:`FeatureStandardizer` first — a frozen, calibrated standardisation so
    heterogeneous columns (time-to-maturity, log-moneyness, vol, holdings, a short
    rate) reach the first Linear on a common scale.  :meth:`DeepHedger.fit`
    calibrates it once before training.  This replaces ``BatchNorm1d``, whose
    train/eval mismatch on per-step-constant features (time-to-maturity, a
    deterministic rate/hazard) is what blew the long-dated GMxB holdings up to ~1e6
    — see :class:`FeatureStandardizer`.

    With ``max_position`` set, each output is squashed through
    ``max_position * tanh(raw / max_position)`` so ``|holding| <= max_position`` by
    construction.  With the standardizer in place the holdings are already O(1) and
    smooth, so this is an optional belt-and-braces bound (default ``None`` = linear
    head), not the fix.
    """

    def __init__(
        self,
        n_in: int,
        n_hedges: int = 1,
        n_units: int = 32,
        n_layers: int = 4,
        *,
        normalize: bool = True,
        max_position: Optional[float] = None,
        seed: int = 0
    ):
        super().__init__()
        self.max_position = max_position
        self.standardizer = FeatureStandardizer(n_in) if normalize else nn.Identity()
        
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            layers: List[nn.Module] = []
            d = n_in
            for _ in range(n_layers):
                layers += [nn.Linear(d, n_units), nn.ReLU()]
                d = n_units
            layers += [nn.Linear(d, n_hedges)]
            self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(self.standardizer(x))
        if self.max_position is not None:
            m = float(self.max_position)
            return m * torch.tanh(out / m)
        return out


class BoundedHedgingMLP(HedgingMLP):
    r""":class:`HedgingMLP` with a smooth **sigmoid** output head, confining each
    holding to a fixed interval ``[lo, hi]`` via ``lo + (hi - lo) · sigmoid(raw)``.

    Use this when hedging a *single* option whose delta lives in a known range — a
    long call or put, whose delta is in ``[0, 1]`` (the default).  There the bound is
    the right inductive bias: the linear head of :class:`HedgingMLP` sits on top of a
    ReLU stack, so it is piecewise-linear and can both overshoot the interval (a call
    delta drifting past 1, or below 0) and leak the backbone's kinks into the
    holding-vs-spot curve.  The sigmoid squashes the policy into a smooth, monotone
    S-curve inside ``[lo, hi]`` — the clean BS-delta shape of Jones et al.'s Figure 1,
    where a narrow linear-head net otherwise produces a visibly kinked policy and a
    jagged ``AADH − SDH`` difference.

    It is a *specialisation*, not a better default: a general hedge (short positions,
    multi-instrument books, near-martingale reweighting) needs the unbounded
    :class:`HedgingMLP`.  Defaults here (``n_units=100, n_layers=3``) match the paper's
    illustrative network; widen/deepen as needed.

    Parameters
    ----------
    bounds : (float, float)
        ``(lo, hi)`` holding interval.  Default ``(0.0, 1.0)`` = a long call/put delta.
    """

    def __init__(
        self,
        n_in: int,
        n_hedges: int = 1,
        n_units: int = 100,
        n_layers: int = 3,
        *,
        bounds: Tuple[float, float] = (0.0, 1.0),
        normalize: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            n_in, n_hedges, n_units=n_units, n_layers=n_layers,
            normalize=normalize, max_position=None, seed=seed,
        )
        lo, hi = bounds
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))

    def forward(self, x: Tensor) -> Tensor:
        raw = self.net(self.standardizer(x))
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)


__all__ = [
    "FeatureStandardizer",
    "HedgingMLP",
    "BoundedHedgingMLP",
]
