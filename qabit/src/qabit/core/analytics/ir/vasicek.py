"""Vasicek affine bond pricing and yield curve."""

from __future__ import annotations
import math
import numpy as np


def _AB(kappa: float, theta: float, sigma: float, tau: float):
    """Vasicek affine coefficients B(tau), A(tau)."""
    if tau < 1e-10:
        return 0.0, 1.0
    B = (1.0 - math.exp(-kappa * tau)) / kappa
    A = math.exp(
        (theta - 0.5 * sigma**2 / kappa**2) * (B - tau)
        - sigma**2 * B**2 / (4.0 * kappa)
    )
    return B, A


def vasicek_zcb(
    r0: float, kappa: float, theta: float, sigma: float, tau: float
) -> float:
    """P(0, T) = A(T)·exp(-B(T)·r₀) — Vasicek closed form."""
    B, A = _AB(kappa, theta, sigma, tau)
    return A * math.exp(-B * r0)


def vasicek_yield_curve(
    r0: float, kappa: float, theta: float, sigma: float, maturities
) -> np.ndarray:
    """Zero yields R(T) = -log P(0,T)/T for a grid of maturities."""
    mats = np.atleast_1d(maturities)
    return np.array(
        [
            -math.log(max(vasicek_zcb(r0, kappa, theta, sigma, t), 1e-12))
            / max(t, 1e-12)
            for t in mats
        ]
    )


def vasicek_bond_option(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T_opt: float,
    T_bond: float,
    K: float,
    is_call: bool = True,
) -> float:
    """Jamshidian (1989) option on Vasicek ZCB."""
    if T_opt >= T_bond:
        raise ValueError("T_opt must be < T_bond.")
    B_s, A_s = _AB(kappa, theta, sigma, T_bond - T_opt)
    sig_p = sigma * B_s * math.sqrt((1 - math.exp(-2 * kappa * T_opt)) / (2 * kappa))
    P_T = vasicek_zcb(r0, kappa, theta, sigma, T_bond)
    P_s = vasicek_zcb(r0, kappa, theta, sigma, T_opt)
    from scipy.stats import norm

    h = math.log(P_T / (K * P_s)) / sig_p + 0.5 * sig_p
    if is_call:
        return P_T * norm.cdf(h) - K * P_s * norm.cdf(h - sig_p)
    return K * P_s * norm.cdf(-(h - sig_p)) - P_T * norm.cdf(-h)


__all__ = [
    "vasicek_zcb",
    "vasicek_yield_curve",
    "vasicek_bond_option",
]
