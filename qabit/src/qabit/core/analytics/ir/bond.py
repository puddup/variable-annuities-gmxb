"""core/analytics/ir/bond.py — Interest rate analytics — bond, swap, forward rates."""

from __future__ import annotations
import math


def zcb_price(r: float, maturity: float) -> float:
    """exp(-r · T) — flat-rate ZCB."""
    return math.exp(-r * maturity)


def fwd_rate(discount_curve, t1: float, t2: float) -> float:
    """Forward rate from t1 to t2: -log(df(t2)/df(t1)) / (t2-t1)."""
    df1 = discount_curve.df(t1)
    df2 = discount_curve.df(t2)
    return -math.log(max(df2 / max(df1, 1e-12), 1e-12)) / max(t2 - t1, 1e-12)


def inst_fwd(discount_curve, t: float, eps: float = 1e-5) -> float:
    """Instantaneous forward rate at t."""
    return fwd_rate(discount_curve, t, t + eps)


def zero_rate(discount_curve, t: float) -> float:
    """Continuously-compounded zero rate to t."""
    return -math.log(max(discount_curve.df(t), 1e-12)) / max(t, 1e-12)


def annuity(discount_curve, schedule: list) -> float:
    """Σ df(T_i) · δ_i for a fixed-leg schedule."""
    s = sorted(schedule)
    return sum(discount_curve.df(s[i]) * (s[i] - s[i - 1]) for i in range(1, len(s)))


def par_swap_rate(discount_curve, schedule: list) -> float:
    """Fair fixed rate of a par swap given fixed-leg schedule."""
    s = sorted(schedule)
    pv_float = discount_curve.df(s[0]) - discount_curve.df(s[-1])
    ann = annuity(discount_curve, s)
    return pv_float / max(ann, 1e-12)


def duration(discount_curve, cashflows: list, dates: list) -> float:
    """Modified duration: Σ t_i · cf_i · df(t_i) / P."""
    P = sum(cf * discount_curve.df(t) for cf, t in zip(cashflows, dates))
    D = sum(t * cf * discount_curve.df(t) for cf, t in zip(cashflows, dates))
    return D / max(P, 1e-12)


def convexity(discount_curve, cashflows: list, dates: list) -> float:
    """Convexity: Σ t_i² · cf_i · df(t_i) / P."""
    P = sum(cf * discount_curve.df(t) for cf, t in zip(cashflows, dates))
    C = sum(t**2 * cf * discount_curve.df(t) for cf, t in zip(cashflows, dates))
    return C / max(P, 1e-12)


__all__ = [
    "zcb_price",
    "fwd_rate",
    "inst_fwd",
    "zero_rate",
    "annuity",
    "par_swap_rate",
    "duration",
    "convexity",
]
