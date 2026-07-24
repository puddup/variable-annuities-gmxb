"""Decorators for qabit."""

from __future__ import annotations

import functools

from qabit.exceptions import StateError


def requires_state(func):
    """
    Decorator for methods that require ``self.state`` to be populated.

    Apply to methods of Monte Carlo processes that depend on a prior call
    to :py:meth:`simulate`.

    Raises
    ------
    StateError
        If ``self.state`` is ``None`` when the decorated method is called.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "state", None) is None:
            raise StateError(
                f"{type(self).__name__}.{func.__name__}() requires "
                "that simulate() has been run. "
                "→ call <process>.simulate(dates, n_paths, ...) first."
            )
        return func(self, *args, **kwargs)

    return wrapper


__all__ = [
    "requires_state",
]
