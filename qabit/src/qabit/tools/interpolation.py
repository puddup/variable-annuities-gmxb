"""tools/interpolation.py — 1-D interpolation on Tensors."""

from __future__ import annotations
import torch
from torch import Tensor


def linear(x: Tensor, y: Tensor, xq: Tensor) -> Tensor:
    xq = torch.as_tensor(xq, dtype=y.dtype, device=y.device)
    idx = torch.searchsorted(x.contiguous(), xq.contiguous()).clamp(1, len(x) - 1)
    lo, hi = idx - 1, idx
    t = (xq - x[lo]) / (x[hi] - x[lo]).clamp(min=1e-12)
    return y[lo] + t * (y[hi] - y[lo])


def log_linear(x: Tensor, y: Tensor, xq: Tensor) -> Tensor:
    return torch.exp(linear(x, torch.log(y.clamp(min=1e-12)), xq))


def integrate_trapezoid_paths(x: Tensor, paths: Tensor) -> Tensor:
    """Per-path cumulative ∫ paths dt.  Returns (N, E), result[:,0]=0."""
    dx = (x[1:] - x[:-1]).to(paths.dtype).unsqueeze(0)
    inc = 0.5 * (paths[:, :-1] + paths[:, 1:]) * dx
    return torch.cat(
        [
            torch.zeros(paths.shape[0], 1, dtype=paths.dtype, device=paths.device),
            inc.cumsum(dim=1),
        ],
        dim=1,
    )


# ── scalar/tensor coercion helpers ────────────────────────────────────────────


def to_tensor(t, dtype=torch.float64) -> tuple:
    """Coerce scalar or array-like to ``(Tensor, was_scalar)``."""
    scalar = isinstance(t, (int, float))
    return torch.atleast_1d(torch.as_tensor(t, dtype=dtype)), scalar


def maybe_scalar(result: torch.Tensor, was_scalar: bool):
    """Return ``float`` if input was scalar, else ``Tensor``."""
    return float(result[0]) if was_scalar else result


__all__ = [
    "linear",
    "log_linear",
    "integrate_trapezoid_paths",
    "to_tensor",
    "maybe_scalar",
]
