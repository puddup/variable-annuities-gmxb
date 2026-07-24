"""
Exception hierarchy for qabit.
"""


class QabitError(Exception):
    """Base exception for all library errors."""


class ShapeError(QabitError):
    """Invalid tensor dimensions or shapes."""


class StateError(QabitError):
    """Operation requiring a pre-simulated process."""


class CalibrationError(QabitError):
    """Failure in model calibration."""


__all__ = [
    "QabitError",
    "ShapeError",
    "StateError",
    "CalibrationError",
]
