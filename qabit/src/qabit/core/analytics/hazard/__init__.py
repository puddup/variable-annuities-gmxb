"""Mortality / hazard **mathematics** — pure functions, no simulation, no classes.

* ``deterministic`` — deterministic intensity (exponential, Gompertz, Weibull)
  and the survival family (cumulative hazard, survival, density, E[tau]).
* ``pathwise``      — the same statistics estimated from *simulated* hazard paths,
  plus death-time sampling.

Stepping a hazard SDE lives in :mod:`qabit.core.dynamics.sde.hazard`; the factor
there holds the parameters and calls these functions for its target intensity.
"""

from qabit.core.analytics.hazard.deterministic import (
    exponential_hazard,
    gompertz_hazard,
    weibull_hazard,
    cumulative_hazard,
    survival,
    density,
    expected_lifetime,
)
from qabit.core.analytics.hazard.pathwise import (
    pathwise_cumulative_hazard,
    pathwise_survival,
    pathwise_density,
    pathwise_expected_lifetime,
    sample_tau,
)

__all__ = [
    "exponential_hazard",
    "gompertz_hazard",
    "weibull_hazard",
    "cumulative_hazard",
    "survival",
    "density",
    "expected_lifetime",
    "pathwise_cumulative_hazard",
    "pathwise_survival",
    "pathwise_density",
    "pathwise_expected_lifetime",
    "sample_tau",
]
