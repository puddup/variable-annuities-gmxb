"""core/analytics/equity/lookback.py — Goldman-Sosin (1979) fixed and floating lookback formulas."""

from __future__ import annotations

import torch
from torch import Tensor
from qabit.tools.broadcast import auto_tensor, clamp_inputs
from qabit.core.analytics.equity.black_scholes import d1, d2, ncdf, npdf


@auto_tensor()
@clamp_inputs(tau=1e-8, S_ext=1e-12)
def goldman_sosin_price(
    S: Tensor,
    K: float,
    S_ext: Tensor,
    r: float,
    sigma: float,
    tau: Tensor,
    is_call: bool = True,
) -> Tensor:
    r"""Fixed-strike lookback call or put — Goldman-Sosin (1979).

    Call: :math:`\max(S_{\max}(0,T) - K, 0)` using running maximum.
    Put:  :math:`\max(K - S_{\min}(0,T), 0)` using running minimum.

    Parameters
    ----------
    S       : Tensor ``(N,)`` — current spot.
    K       : float — strike.
    S_ext   : Tensor ``(N,)`` — running max (call) or running min (put).
    r, sigma, tau — standard parameters.
    is_call : bool
    """
    if isinstance(tau, (int, float)):
        tau = torch.full_like(S, tau)
    tau = tau.clamp(min=1e-8)
    sig_t = sigma * tau.sqrt()

    if is_call:
        # call: S_ext is running maximum
        d1_s = d1(S, K, r, sigma, tau)
        d2_s = d2(S, K, r, sigma, tau)
        S_over_ext = S / S_ext
        d1_m = d1(S_over_ext * K, K, r, sigma, tau)
        d2_m = d2(S_over_ext * K, K, r, sigma, tau)
        price_0 = S * (  # S_max < K
            ncdf(d1_s) + sig_t * (d1_s * ncdf(d1_s) + npdf(d1_s))
        ) - K * ncdf(d2_s)
        price_1 = (  # S_max >= K
            S * (ncdf(d1_m) + sig_t * (d1_m * ncdf(d1_m) + npdf(d1_m)))
            - K
            + S_ext * (1.0 - ncdf(d2_m))
        )
        return torch.where(S_ext < K, price_0, price_1)
    else:
        # put: S_ext is running minimum, mirror the call formula
        d1_s = d1(K / S * K, K, r, sigma, tau)
        d2_s = d2(K / S * K, K, r, sigma, tau)
        S_over_ext = S / S_ext
        d1_m = d1(S_over_ext * K, K, r, sigma, tau)
        d2_m = d2(S_over_ext * K, K, r, sigma, tau)
        price_0 = K * ncdf(-d2_s) - S * (  # S_min > K
            ncdf(-d1_s) + sig_t * (d1_s * ncdf(-d1_s) + npdf(d1_s))
        )
        price_1 = (  # S_min <= K
            K
            - S_ext * (1.0 - ncdf(-d2_m))
            - S * (ncdf(-d1_m) + sig_t * (d1_m * ncdf(-d1_m) + npdf(d1_m)))
        )
        return torch.where(S_ext > K, price_0, price_1)


__all__ = [
    "goldman_sosin_price",
]
