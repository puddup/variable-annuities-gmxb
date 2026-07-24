"""tools/util.py — Tensor coercion, date normalisation, generic helpers.

Provides low-level, reusable utilities for tensor conversion without imposing
any particular calling convention. These functions are the building blocks
for higher-level decorators in tools.broadcast.

Consolidates:
  - HazardBaseline analytics
  - df_to_scalar / df_to_path
  - _as_dates
  - guarantee helpers
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple

import torch
from torch import Tensor

from qabit.config import EPS

# Type aliases for clarity
Number = Union[int, float, complex]
TensorLike = Union[Number, List[Number], Tensor, "np.ndarray"]


# ── type coercion (core) ─────────────────────────────────────────────────────


def as_tensor(
    t: TensorLike,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    ensure_1d: bool = False,
) -> Tensor:
    """
    Coerce input to a torch.Tensor with specified dtype and device.

    Args:
        t: Input scalar, list, numpy array, or tensor
        dtype: Target dtype (default: float32)
        device: Target device (default: CPU if None)
        ensure_1d: If True, reshape to 1D (squeezes 0-dim tensors)

    Returns:
        Coerced tensor

    Examples:
        >>> as_tensor(5.0)
        tensor(5.)
        >>> as_tensor([1, 2, 3], ensure_1d=True)
        tensor([1., 2., 3.])
    """
    device = device or torch.device("cpu")

    if isinstance(t, Tensor):
        result = t.to(dtype=dtype, device=device)
    else:
        import numpy as np

        arr = np.asarray(t, dtype=np.float32 if dtype == torch.float32 else None)
        result = torch.as_tensor(arr, dtype=dtype, device=device)

    if ensure_1d and result.ndim == 0:
        result = result.reshape(1)

    return result


def to_scalar(
    x: Union[Tensor, Number],
    method: str = "item",
    default: Optional[float] = None,
) -> float:
    """
    Extract a Python float from a scalar tensor or number.

    Args:
        x: Input value
        method: How to handle multi-element tensors:
            - "item": Raises error if not scalar
            - "mean": Returns mean of all elements
            - "first": Returns first element
            - "sum": Returns sum of all elements
        default: Return this if x is None

    Returns:
        Python float

    Examples:
        >>> to_scalar(torch.tensor(5.0))
        5.0
        >>> to_scalar(torch.tensor([1.0, 2.0, 3.0]), method="mean")
        2.0
    """
    if x is None:
        if default is not None:
            return float(default)
        raise ValueError("Cannot convert None to scalar")

    if isinstance(x, Tensor):
        if x.numel() == 1:
            return float(x.item())

        if method == "mean":
            return float(x.mean())
        elif method == "first":
            return float(x.flatten()[0])
        elif method == "sum":
            return float(x.sum())
        elif method == "item":
            raise ValueError(f"Tensor has {x.numel()} elements, cannot extract single item")
        else:
            raise ValueError(f"Unknown method: {method}")

    return float(x)


def to_path(
    x: Union[Tensor, Number, None],
    n_paths: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    allow_broadcast: bool = True,
) -> Tensor:
    """
    Expand a scalar or short tensor to shape (n_paths,).

    Designed for simulation paths where:
    - Scalar → constant array of length n_paths
    - 1D tensor of length 1 → constant array
    - 1D tensor of length n_paths → passed through
    - 1D tensor of other length → raises error (unless allow_broadcast=True)

    Args:
        x: Input value (scalar, tensor, or None)
        n_paths: Number of simulation paths
        dtype: Output dtype
        device: Output device
        allow_broadcast: If True, broadcast shorter tensors via repetition

    Returns:
        Tensor of shape (n_paths,)

    Examples:
        >>> to_path(0.05, 1000).shape
        torch.Size([1000])
        >>> to_path(torch.tensor([0.05, 0.06]), 1000)  # mismatched length
        ValueError (unless allow_broadcast=True)
    """
    device = device or torch.device("cpu")

    if x is None:
        raise ValueError("Cannot convert None to path tensor")

    if isinstance(x, Tensor):
        x_flat = x.flatten()

        if x_flat.shape[0] == n_paths:
            return x_flat.to(dtype=dtype, device=device)
        elif x_flat.shape[0] == 1:
            return torch.full((n_paths,), x_flat[0].item(), dtype=dtype, device=device)
        elif allow_broadcast and x_flat.shape[0] < n_paths:
            repeats = (n_paths + x_flat.shape[0] - 1) // x_flat.shape[0]
            repeated = x_flat.repeat(repeats)[:n_paths]
            return repeated.to(dtype=dtype, device=device)
        else:
            raise ValueError(
                f"Cannot convert tensor of length {x_flat.shape[0]} to path length {n_paths}"
            )

    scalar_val = to_scalar(x)
    return torch.full((n_paths,), scalar_val, dtype=dtype, device=device)


def as_column(value: Union[Tensor, Number], ref: Tensor) -> Tensor:
    """Coerce a scalar or ``(N,)`` tensor to an ``(N, 1)`` column on ``ref``'s
    dtype/device — the canonical "one feature column" shape.

    A scalar broadcasts to a constant column of length ``ref.shape[0]``; a tensor
    is moved onto ``ref``'s dtype/device and reshaped to ``(-1, 1)`` (so a
    ``(N,)`` or already-``(N, 1)`` tensor both pass through cleanly).

    Examples
    --------
    >>> as_column(0.2, ref).shape          # ref is (N, ...)
    torch.Size([N, 1])
    >>> as_column(torch.zeros(N), ref).shape
    torch.Size([N, 1])
    """
    if isinstance(value, Tensor):
        return value.to(dtype=ref.dtype, device=ref.device).reshape(-1, 1)
    return torch.full((ref.shape[0], 1), float(value), dtype=ref.dtype, device=ref.device)


# ── date normalisation ────────────────────────────────────────────────────────


def as_dates(
    t: Optional[Union[Number, List[Number], Tensor]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Convert t to 1-D tensor of valuation dates.

    Args:
        t: None (→ [0.0]), scalar (→ [t]), list/tensor (→ flattened)
        dtype: Output dtype
        device: Output device

    Returns:
        1D tensor of dates

    Examples:
        >>> as_dates(None)
        tensor([0.])
        >>> as_dates(2.5)
        tensor([2.5])
        >>> as_dates([1.0, 2.0, 3.0])
        tensor([1., 2., 3.])
    """
    device = device or torch.device("cpu")

    if t is None:
        return torch.tensor([0.0], dtype=dtype, device=device)

    tensor_t = as_tensor(t, dtype=dtype, device=device)
    return tensor_t.reshape(-1)


def date_grid(
    start: float,
    end: float,
    steps: int,
    include_end: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create a grid of dates from start to end.

    Utility for creating time grids for PDE solvers or simulation.

    Args:
        start: Start time
        end: End time
        steps: Number of steps
        include_end: If True, includes end date
        dtype: Output dtype
        device: Output device

    Returns:
        1D tensor of dates

    Example:
        >>> date_grid(0, 1, 4)
        tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
    """
    device = device or torch.device("cpu")
    n_points = steps + 1 if include_end else steps
    return torch.linspace(start, end, n_points, dtype=dtype, device=device)


# ── generic helpers ───────────────────────────────────────────────────────────


def zeros_like(ref: Tensor, *args, **kwargs) -> Tensor:
    """Create zeros tensor with same shape/dtype/device as reference."""
    return torch.zeros_like(ref, *args, **kwargs)


def ones_like(ref: Tensor, *args, **kwargs) -> Tensor:
    """Create ones tensor with same shape/dtype/device as reference."""
    return torch.ones_like(ref, *args, **kwargs)


def full_like(ref: Tensor, fill_value: float, *args, **kwargs) -> Tensor:
    """Create filled tensor with same shape/dtype/device as reference."""
    return torch.full_like(ref, fill_value, *args, **kwargs)


def at_time(t: float, target: float, tol: float = 1e-6) -> bool:
    """Check if t equals target within tolerance."""
    return abs(t - target) < tol


def at_any(t: float, schedule: List[float], tol: float = 1e-6) -> bool:
    """Check if t matches any value in schedule within tolerance."""
    return any(abs(t - s) < tol for s in schedule)


def nearest_index(dates: Union[Tensor, List[float]], t: float) -> int:
    """Index of the entry in ``dates`` closest to time ``t``."""
    if isinstance(dates, Tensor):
        dates = dates.flatten().tolist()
    return min(range(len(dates)), key=lambda i: abs(dates[i] - t))


def broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    """Check if two shapes are broadcastable."""
    try:
        torch.broadcast_shapes(shape1, shape2)
        return True
    except RuntimeError:
        return False


# ── safe numerical guards ────────────────────────────────────────────────────


def safe_log(x: Tensor, epsilon: float = EPS) -> Tensor:
    """Safe log with clamping to prevent log(0) or log(negative)."""
    return torch.log(torch.clamp(x, min=epsilon))


def safe_sqrt(x: Tensor, epsilon: float = EPS) -> Tensor:
    """Safe sqrt with clamping to prevent sqrt(negative)."""
    return torch.sqrt(torch.clamp(x, min=epsilon))


def safe_divide(num: Tensor, den: Tensor, epsilon: float = EPS) -> Tensor:
    """Safe division with denominator clamping."""
    return num / torch.clamp(den, min=epsilon)


# ── export ────────────────────────────────────────────────────────────────────

__all__ = [
    "as_tensor",
    "to_scalar",
    "to_path",
    "as_column",
    "as_dates",
    "date_grid",
    "zeros_like",
    "ones_like",
    "full_like",
    "at_time",
    "at_any",
    "nearest_index",
    "broadcastable",
    "safe_log",
    "safe_sqrt",
    "safe_divide",
]