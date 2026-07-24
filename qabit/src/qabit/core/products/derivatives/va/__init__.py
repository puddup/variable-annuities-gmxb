"""Variable Annuity products and composite GMxB guarantee floors."""

from qabit.core.products.derivatives.va.guarantee import (
    GuaranteeScheme,
    Fixed,
    RollUp,
    Ratchet,
    Reset,
    GuaranteeFloor,
)
from qabit.core.products.derivatives.va.variable_annuity import (
    GMxBAnnuity,
    GMABAnnuity,
    GMWBAnnuity,
    GMDBAnnuity,
)

__all__ = [
    "GuaranteeScheme",
    "Fixed",
    "RollUp",
    "Ratchet",
    "Reset",
    "GuaranteeFloor",
    "GMxBAnnuity",
    "GMABAnnuity",
    "GMWBAnnuity",
    "GMDBAnnuity",
]