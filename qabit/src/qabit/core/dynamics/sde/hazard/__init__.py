"""Credit / mortality intensity factors for SDESystem.

One factor class per (model × hazard law) combination:

* Deterministic hazard (zero Brownian drivers):
  :class:`ExponentialHazardFactor`, :class:`GompertzHazardFactor`,
  :class:`WeibullHazardFactor`.

* CIR-stochastic hazard reverting to that law:
  :class:`CIRExponentialHazardFactor`, :class:`CIRGompertzHazardFactor`,
  :class:`CIRWeibullHazardFactor`.

The math for the laws and the survival family lives in
:mod:`qabit.core.analytics.hazard`.
"""

from qabit.core.dynamics.sde.hazard.deterministic import (
    ExponentialHazardFactor,
    GompertzHazardFactor,
    WeibullHazardFactor,
)
from qabit.core.dynamics.sde.hazard.cir import (
    CIRExponentialHazardFactor,
    CIRGompertzHazardFactor,
    CIRWeibullHazardFactor,
)

__all__ = [
    "ExponentialHazardFactor",
    "GompertzHazardFactor",
    "WeibullHazardFactor",
    "CIRExponentialHazardFactor",
    "CIRGompertzHazardFactor",
    "CIRWeibullHazardFactor"
]
