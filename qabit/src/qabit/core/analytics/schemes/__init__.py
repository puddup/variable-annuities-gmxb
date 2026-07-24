"""Discretisation schemes — mathematical machinery for MC processes.

Concrete scheme implementations live here in analytics/ because they are
mathematical algorithms (Euler-Maruyama, Milstein, QE, Broadie-Kaya).
The abstract bases (SchemeBase, HestonScheme, CIRScheme) and their param
containers live with their respective MC process modules.

Usage
-----
    from qabit.core.analytics.schemes import (
        # Heston
        HestonEulerScheme, HestonMilsteinScheme, BroadieKayaScheme,
        # CIR
        CIREulerScheme, CIRMilsteinScheme,
    )
"""

from qabit.core.analytics.schemes.euler import HestonEulerScheme, CIREulerScheme
from qabit.core.analytics.schemes.milstein import HestonMilsteinScheme, CIRMilsteinScheme
from qabit.core.analytics.schemes.broadie_kaya import BroadieKayaScheme

__all__ = [
    "HestonEulerScheme",
    "HestonMilsteinScheme",
    "BroadieKayaScheme",
    "CIREulerScheme",
    "CIRMilsteinScheme",
]
