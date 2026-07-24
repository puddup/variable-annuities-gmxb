"""tools/broadcast.py — Decorators for automatic tensor coercion and clamping.

Separates infrastructure concerns (dtype promotion, device placement, safe
clamping) from mathematical content so that analytic functions can be written
as pure expressions of their formulae.

Public API
----------
``auto_tensor(*arg_names)``
    Coerce named scalar/float arguments to Tensors matching a reference
    argument's dtype and device. Returns scalar when all inputs are scalar.

``clamp_inputs(**clamp_specs)``
    Apply ``.clamp(min=..., max=...)`` to named Tensor arguments before the
    body runs.

``validate_inputs(**validators)``
    Validate tensor properties (positivity, finiteness, shape) before
    execution.

Example
-------
.. code-block:: python

    @auto_tensor
    @clamp_inputs(tau=1e-8, sigma=(1e-6, None))
    @validate_inputs(S="positive", K="positive")
    def d1(S, K, r, sigma, tau):
        return (torch.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * tau.sqrt())
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Optional, Union, Tuple, Dict, Any, Callable

import torch
from torch import Tensor

from qabit.exceptions import ShapeError


# ─────────────────────────────────────────────────────────────────────────────
# auto_tensor
# ─────────────────────────────────────────────────────────────────────────────


def auto_tensor(default_dtype=torch.float32, default_device="cpu"):
    """
    Convert specified arguments to tensors. Output matches input type.

    Core rule: if ANY input is a tensor, return a tensor; if ALL inputs are
    scalars, return a scalar.

    Examples
    --------
    .. code-block:: python

        @auto_tensor
        def price(S, K, r, sigma, tau):
            return S * torch.exp(-r * tau) - K

        price(100, 90, 0.05, 0.2, 1.0)                            # -> 8.21
        price(torch.tensor([100, 110]), 90, 0.05, 0.2, 1.0)       # -> tensor([8.21, 18.21])
    """

    def decorator(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments

            # Find reference tensor for device (ignore its dtype)
            ref_device = default_device
            for v in a.values():
                if isinstance(v, Tensor):
                    ref_device = v.device
                    break

            # Convert scalars to tensors with default_dtype and ref_device
            for name, val in a.items():
                if isinstance(val, (int, float)):
                    a[name] = torch.tensor(val, dtype=default_dtype, device=ref_device)
                # Optionally, we could also convert tensors to default_dtype? Leave as is.

            return fn(**a)

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# clamp_inputs
# ─────────────────────────────────────────────────────────────────────────────


def clamp_inputs(
    **clamp_specs: Union[float, Tuple[Optional[float], Optional[float]]],
):
    """
    Apply clamping to named Tensor arguments before the function body runs.

    Args:
        **clamp_specs: Mapping of parameter_name → clamp specification.
            - Single float: clamp(min=value)
            - Tuple of (min, max): clamp both bounds (None for no bound)

    Example:
        >>> @clamp_inputs(tau=1e-8, sigma=1e-6)  # min clamping only
        ... def d1(S, K, r, sigma, tau): ...

        >>> @clamp_inputs(vol=(-1.0, 1.0))  # min and max
        ... def clipped_vol(vol): ...

        >>> @clamp_inputs(vol=(None, 1.0))  # max clamping only
        ... def capped_vol(vol): ...

    Note:
        Should be applied AFTER auto_tensor so clamping sees coerced tensors.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = dict(bound.arguments)

            for name, spec in clamp_specs.items():
                if name not in a:
                    continue

                val = a[name]
                if not isinstance(val, Tensor):
                    continue

                if isinstance(spec, (int, float)):
                    a[name] = val.clamp(min=spec)
                elif isinstance(spec, tuple) and len(spec) == 2:
                    min_val, max_val = spec
                    if min_val is not None and max_val is not None:
                        a[name] = val.clamp(min=min_val, max=max_val)
                    elif min_val is not None:
                        a[name] = val.clamp(min=min_val)
                    elif max_val is not None:
                        a[name] = val.clamp(max=max_val)
                else:
                    raise ValueError(
                        f"Invalid clamp spec for {name}: {spec}. Expected float or (min, max) tuple"
                    )

            return fn(**a)

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# validate_inputs
# ─────────────────────────────────────────────────────────────────────────────


def validate_inputs(**validators: Union[str, Tuple[str, ...]]):
    """
    Validate tensor properties before function execution.

    Args:
        **validators: Mapping of parameter_name → validation rule(s).
            Supported rules:
            - "positive": All values > 0
            - "nonnegative": All values >= 0
            - "finite": All values are finite (not inf or nan)
            - "not_nan": No NaN values
            - "1d": Tensor must be 1-dimensional
            - "2d": Tensor must be 2-dimensional
            - "ndim=N": Tensor must have N dimensions
            - "shape=(...)" : Tensor must have exact shape
            - "broadcastable_with=name" : Must broadcast with another parameter

    Example:
        >>> @validate_inputs(S="positive", K="positive", sigma=("positive", "finite"))
        ... def black_scholes(S, K, sigma): ...

    Note:
        Should be applied AFTER auto_tensor to validate coerced tensors.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = dict(bound.arguments)

            for name, rules in validators.items():
                if name not in a:
                    continue

                val = a[name]

                if not isinstance(val, Tensor):
                    continue

                if isinstance(rules, str):
                    rules = [rules]

                for rule in rules:
                    _apply_validation_rule(name, val, rule, a)

            return fn(**a)

        return wrapper

    return decorator


def _apply_validation_rule(name: str, val: Tensor, rule: str, all_args: Dict[str, Any]) -> None:
    """Helper to apply a single validation rule."""
    if rule == "positive":
        if not (val > 0).all():
            raise ValueError(
                f"{name} must be positive, got range [{val.min().item()}, {val.max().item()}]"
            )

    elif rule == "nonnegative":
        if not (val >= 0).all():
            raise ValueError(f"{name} must be non-negative, got min={val.min().item()}")

    elif rule == "finite":
        if not torch.isfinite(val).all():
            raise ValueError(f"{name} contains non-finite values (inf or nan)")

    elif rule == "not_nan":
        if torch.isnan(val).any():
            raise ValueError(f"{name} contains NaN values")

    elif rule == "1d":
        if val.ndim != 1:
            raise ShapeError(
                f"{name} must be 1-dimensional, got shape {tuple(val.shape)}. "
                "→ reshape or squeeze the input to a 1-D tensor."
            )

    elif rule == "2d":
        if val.ndim != 2:
            raise ShapeError(
                f"{name} must be 2-dimensional, got shape {tuple(val.shape)}. "
                "→ reshape the input to a 2-D tensor."
            )

    elif rule.startswith("ndim="):
        expected = int(rule.split("=")[1])
        if val.ndim != expected:
            raise ShapeError(
                f"{name} must be {expected}D, got shape {tuple(val.shape)}. "
                f"→ reshape the input to {expected}-D."
            )

    elif rule.startswith("shape="):
        shape_str = rule.split("=")[1].strip()
        import ast

        expected_shape = ast.literal_eval(shape_str)
        if not isinstance(expected_shape, tuple):
            expected_shape = (expected_shape,)
        if tuple(val.shape) != expected_shape:
            raise ShapeError(
                f"{name} must have shape {expected_shape}, got "
                f"{tuple(val.shape)}. → reshape the input to match."
            )

    elif rule.startswith("broadcastable_with="):
        other_name = rule.split("=")[1]
        if other_name not in all_args:
            raise ValueError(f"Parameter '{other_name}' not found for broadcast check")

        other = all_args[other_name]
        if not isinstance(other, Tensor):
            return

        try:
            torch.broadcast_shapes(val.shape, other.shape)
        except RuntimeError as e:
            raise ShapeError(
                f"{name} shape {tuple(val.shape)} cannot broadcast with "
                f"{other_name} shape {tuple(other.shape)}. → align dimensions "
                "or expand one of the inputs."
            ) from e

    else:
        warnings.warn(
            f"Unknown validation rule: '{rule}' for parameter '{name}'",
            UserWarning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# compose_decorators
# ─────────────────────────────────────────────────────────────────────────────


def compose_decorators(*decorators):
    """
    Compose multiple decorators into one for cleaner syntax.

    Args:
        *decorators: Decorators to compose (applied from right to left)

    Example:
        >>> analytics_decorators = compose_decorators(
        ...     auto_tensor("K", "r", "sigma", "tau"),
        ...     clamp_inputs(tau=1e-8),
        ...     validate_inputs(S="positive", K="positive"),
        ... )
        ...
        >>> @analytics_decorators
        ... def d1(S, K, r, sigma, tau):
        ...     return (torch.log(S / K) + (r + 0.5*sigma**2) * tau) / (sigma * tau.sqrt())
    """

    def decorator(fn):
        for dec in reversed(decorators):
            fn = dec(fn)
        return fn

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "auto_tensor",
    "clamp_inputs",
    "validate_inputs",
    "compose_decorators",
]
