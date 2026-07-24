"""dynamics/sde/hazard/cir.py — CIR-stochastic hazard factors.

One factor class per intensity law: ``CIRExponentialHazardFactor``,
``CIRGompertzHazardFactor``, ``CIRWeibullHazardFactor``.

Each models

.. math:: d\\lambda = \\kappa(\\hat\\mu(t) - \\lambda)\\,dt + \\sigma_\\lambda \\sqrt{\\lambda}\\,dZ

with the deterministic target :math:`\\hat\\mu(t)` set by the corresponding law
from :mod:`qabit.core.analytics.hazard.deterministic` (``exponential_hazard``,
``gompertz_hazard``, ``weibull_hazard``).  The CIR discretisation scheme
(default: Euler with an absorbing floor) lives in
:mod:`qabit.core.analytics.schemes` — the factor only carries the parameters
and the scheme instance.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor
from dataclasses import dataclass

from qabit.config import EPS_VAR
from qabit.core.analytics.hazard.deterministic import (
    exponential_hazard,
    gompertz_hazard,
    weibull_hazard,
)
from qabit.core.dynamics.sde.base import Factor
from qabit.core.analytics.schemes.euler import CIREulerScheme
from qabit.core.dynamics.base import AbsorptionFix


@dataclass
class CIRParams:
    """Container passed to each CIR scheme step."""

    kappa: float
    theta_t: float  # possibly time-varying target
    sigma: float



class _CIRHazardFactor(Factor):
    r"""Base class for CIR stochastic hazard reverting to a deterministic target.

    Subclasses set ``hazard_fn`` at construction with the target law's
    parameters; this class handles the CIR step.
    """

    n_factors = 1
    n_outputs = 1
    hazard_fn: Callable[[Tensor], Tensor]

    def __init__(
        self,
        kappa: float = 0.5,
        sigma_lam: float = 0.03,
        scheme=None,
    ) -> None:
        self.kappa = kappa
        self.sigma_lam = sigma_lam
        self.scheme = (
            scheme
            if scheme is not None
            else CIREulerScheme(fix=AbsorptionFix(eps=EPS_VAR))
        )

    def _at(self, t: float) -> float:
        return float(self.hazard_fn(torch.tensor([float(t)]))[0])

    def _cir_step(self, lam: Tensor, dt: float, z: Tensor, target) -> Tensor:
        """One CIR-Euler step of the hazard toward ``target``.

        The single discretisation entry point: it builds the :class:`CIRParams`
        the scheme expects and delegates to ``self.scheme.step``.  ``target`` may be
        a scalar (deterministic law, broadcast over paths) or an ``(N,)`` per-path
        tensor (a trend-shifted target — see
        :class:`~qabit.core.dynamics.sde.hazard.trend_cir.TrendedCIRHazardFactor`),
        so a trended hazard reuses this instead of re-implementing the Euler update.
        """
        p = CIRParams(kappa=self.kappa, theta_t=target, sigma=self.sigma_lam)
        return self.scheme.step(lam, dt, z, p)

    def init(self, n_paths, device, dtype) -> Tensor:
        return torch.full((n_paths,), self._at(0.0), device=device, dtype=dtype)

    def step(self, value, prev, cur, dt, t, z) -> Tensor:
        return self._cir_step(value, dt, z, self._at(t))


class CIRExponentialHazardFactor(_CIRHazardFactor):
    r"""CIR hazard reverting to a constant exponential target ``lambda0``."""

    def __init__(
        self,
        lambda0: float = 0.05,
        kappa: float = 0.5,
        sigma_lam: float = 0.03,
        scheme=None,
    ) -> None:
        super().__init__(kappa=kappa, sigma_lam=sigma_lam, scheme=scheme)
        self.lambda0 = lambda0
        self.hazard_fn = partial(exponential_hazard, lambda0=lambda0)

    def __repr__(self) -> str:
        return (
            f"CIRExponentialHazardFactor(lambda0={self.lambda0}, "
            f"kappa={self.kappa}, sigma_lam={self.sigma_lam})"
        )


class CIRGompertzHazardFactor(_CIRHazardFactor):
    r"""CIR hazard reverting to a Gompertz target."""

    def __init__(
        self,
        m: float = 80.0,
        b: float = 9.5,
        entry_age: float = 60.0,
        kappa: float = 0.5,
        sigma_lam: float = 0.025,
        scheme=None,
    ) -> None:
        super().__init__(kappa=kappa, sigma_lam=sigma_lam, scheme=scheme)
        self.m = m
        self.b = b
        self.entry_age = entry_age
        self.hazard_fn = partial(gompertz_hazard, m=m, b=b, entry_age=entry_age)

    def __repr__(self) -> str:
        return (
            f"CIRGompertzHazardFactor(m={self.m}, b={self.b}, "
            f"entry_age={self.entry_age}, kappa={self.kappa}, "
            f"sigma_lam={self.sigma_lam})"
        )


class CIRWeibullHazardFactor(_CIRHazardFactor):
    r"""CIR hazard reverting to a Weibull target."""

    def __init__(
        self,
        c1: float = 90.43,
        c2: float = 10.36,
        entry_age: float = 60.0,
        kappa: float = 0.5,
        sigma_lam: float = 0.02,
        scheme=None,
    ) -> None:
        super().__init__(kappa=kappa, sigma_lam=sigma_lam, scheme=scheme)
        self.c1 = c1
        self.c2 = c2
        self.entry_age = entry_age
        self.hazard_fn = partial(weibull_hazard, c1=c1, c2=c2, entry_age=entry_age)

    def __repr__(self) -> str:
        return (
            f"CIRWeibullHazardFactor(c1={self.c1}, c2={self.c2}, "
            f"entry_age={self.entry_age}, kappa={self.kappa}, "
            f"sigma_lam={self.sigma_lam})"
        )


__all__ = [
    "CIRExponentialHazardFactor",
    "CIRGompertzHazardFactor",
    "CIRWeibullHazardFactor",
]
