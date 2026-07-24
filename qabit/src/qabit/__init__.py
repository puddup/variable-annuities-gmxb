"""qabit — PyTorch-native Monte Carlo for quantitative finance.

This module re-exports the public API so the common factory classes are
reachable directly as ``qabit.<Name>`` (e.g. ``qabit.VasicekRateFactor``)
regardless of the process they are imported in.

Why this matters: the notebooks run independent seeds in separate worker
processes (joblib/loky). Each worker re-imports ``qabit`` from scratch, so any
names must resolve at import time here — runtime ``setattr`` patches applied in
the parent process do not cross the process boundary.
"""

from qabit.config import *  # noqa: F401,F403  (re-export config if present)

# --- SDE factors ----------------------------------------------------------
from qabit.core.dynamics.sde.rates import (  # noqa: F401
    ConstantRateFactor,
    VasicekRateFactor,
    CIRRateFactor,
)
from qabit.core.dynamics.sde.equity.geometric_brownian import (  # noqa: F401
    GeometricBrownianFactor,
    EquityFactor,
)
from qabit.core.dynamics.sde.equity.heston import HestonFactor  # noqa: F401
from qabit.core.dynamics.sde.hazard import (  # noqa: F401
    ExponentialHazardFactor,
    GompertzHazardFactor,
    WeibullHazardFactor,
    CIRExponentialHazardFactor,
    CIRGompertzHazardFactor,
    CIRWeibullHazardFactor,
)
from qabit.core.dynamics.sde.system import SDESystem  # noqa: F401

# --- Market ---------------------------------------------------------------
from qabit.core.market.market import Market  # noqa: F401
from qabit.core.market.curve import (  # noqa: F401
    VolSurface,
    DiscountCurve,
    StochasticDiscountCurve,
    StochasticHazardCurve,
    MarketKeys,
)
from qabit.core.market.calendar import EventCalendar  # noqa: F401

# --- Products -------------------------------------------------------------
from qabit.core.products.underlying.equity.stock import Stock  # noqa: F401
from qabit.core.products.derivatives.equity.european import EuropeanOption  # noqa: F401
from qabit.core.products.derivatives.equity.barrier import BarrierOption  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "ConstantRateFactor",
    "VasicekRateFactor",
    "CIRRateFactor",
    "GeometricBrownianFactor",
    "EquityFactor",
    "HestonFactor",
    "ExponentialHazardFactor",
    "GompertzHazardFactor",
    "WeibullHazardFactor",
    "CIRExponentialHazardFactor",
    "CIRGompertzHazardFactor",
    "CIRWeibullHazardFactor",
    "SDESystem",
    "Market",
    "VolSurface",
    "DiscountCurve",
    "StochasticDiscountCurve",
    "StochasticHazardCurve",
    "MarketKeys",
    "EventCalendar",
    "Stock",
    "EuropeanOption",
    "BarrierOption",
    "__version__",
]