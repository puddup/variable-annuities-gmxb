"""Utilities: tensor coercion, decorators, interpolation, randomness."""

from qabit.tools.interpolation import (
    linear,
    log_linear,
    integrate_trapezoid_paths,
    to_tensor,
    maybe_scalar,
)
from qabit.tools.util import (
    as_tensor,
    to_scalar,
    to_path,
    as_dates,
    date_grid,
    zeros_like,
    ones_like,
    full_like,
    at_time,
    at_any,
    broadcastable,
    safe_log,
    safe_sqrt,
    safe_divide,
)
from qabit.tools.broadcast import (
    auto_tensor,
    clamp_inputs,
    validate_inputs,
    compose_decorators,
)
from qabit.tools.decorators import requires_state

__all__ = [
    "linear",
    "log_linear",
    "integrate_trapezoid_paths",
    "to_tensor",
    "maybe_scalar",
    "as_tensor",
    "to_scalar",
    "to_path",
    "as_dates",
    "date_grid",
    "zeros_like",
    "ones_like",
    "full_like",
    "at_time",
    "at_any",
    "broadcastable",
    "safe_log",
    "safe_sqrt",
    "safe_divide",
    "auto_tensor",
    "clamp_inputs",
    "validate_inputs",
    "compose_decorators",
    "requires_state",
]
