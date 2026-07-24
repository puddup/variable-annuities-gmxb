"""core/analytics/common/lsmc.py — Least Squares Monte Carlo (Longstaff-Schwartz 2001).

Vectorised with per-path discount factors.  Accepts a ``discount_grid``
Tensor of shape ``(N, M)`` so it works for stochastic discount curves.

Reference
---------
Longstaff, F., Schwartz, E. (2001). Valuing American Options by Simulation.
Review of Financial Studies 14(1): 113–147.

Bacinello, A. et al. (2010). Regression-based algorithms for life insurance
contracts with surrender guarantees. Quantitative Finance 10(9): 1077–1090.
"""

from __future__ import annotations

from typing import Callable, List

import torch
from torch import Tensor

from qabit.config import EPS
from qabit.exceptions import ShapeError


# ── basis functions ───────────────────────────────────────────────────────────


def _poly_basis(X: Tensor, degree: int = 3) -> Tensor:
    cols = [torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)]
    for k in range(X.shape[1]):
        for d in range(1, degree + 1):
            cols.append(X[:, k : k + 1] ** d)
    return torch.cat(cols, dim=1)


def _laguerre_basis(X: Tensor, degree: int = 3) -> Tensor:
    L = [
        lambda x: torch.ones_like(x),
        lambda x: 1.0 - x,
        lambda x: 1.0 - 2.0 * x + 0.5 * x**2,
        lambda x: 1.0 - 3.0 * x + 1.5 * x**2 - x**3 / 6.0,
        lambda x: 1.0 - 4.0 * x + 3.0 * x**2 - x**3 * 2.0 / 3.0 + x**4 / 24.0,
    ]
    cols = [torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)]
    for k in range(X.shape[1]):
        xk = X[:, k : k + 1].clamp(min=0.0)
        for d in range(1, min(degree + 1, len(L))):
            cols.append(L[d](xk))
    return torch.cat(cols, dim=1)


def _ols(phi: Tensor, y: Tensor) -> Tensor:
    # TODO: pseudo-inverse flag
    # (torch.linalg.pinv(phi) @ y.unsqueeze(1)).squeeze(1)
    return torch.linalg.lstsq(phi, y.unsqueeze(1)).solution.squeeze(1)


# ── vectorised LSMC engine ────────────────────────────────────────────────────


def longstaff_schwartz(
    paths: Tensor,
    exercise_times: List[float],
    payoff_fn: Callable[[Tensor], Tensor],
    discount_grid: Tensor,
    basis: str = "poly",
    degree: int = 3,
    min_itm: int = 10,
) -> Tensor:
    """Least Squares Monte Carlo for Bermudan-style options.

    Parameters
    ----------
    paths : torch.Tensor
        Underlying values at each exercise date, shape ``(N, E)`` where
        ``N`` is the number of Monte Carlo paths and ``E = len(exercise_times)``.
    exercise_times : list of float
        Exercise dates in years. Used for dimensional sanity checks against
        ``paths`` and ``discount_grid``.
    payoff_fn : callable
        Immediate undiscounted exercise payoff. Takes a Tensor of shape
        ``(N,)`` and returns a Tensor of shape ``(N,)``.
    discount_grid : torch.Tensor
        Per-path discount factors from ``t=0`` to each exercise date.
        Shape ``(N, E)`` or ``(E,)``; a 1-D tensor is broadcast over paths.
    basis : {"poly", "laguerre"}, optional
        Regression basis. Default is ``"poly"``.
    degree : int, optional
        Polynomial degree. Default is 3.
    min_itm : int, optional
        Minimum number of in-the-money paths required to run the regression
        at a given exercise date. Default is 10.

    Returns
    -------
    torch.Tensor
        Per-path present value at ``t=0`` with shape ``(N,)``. The option
        price is the sample mean.

    References
    ----------
    Longstaff, F., Schwartz, E. (2001). Valuing American Options by Simulation.
    *Review of Financial Studies*, 14(1), 113-147.
    """
    if paths.dim() != 2:
        raise ShapeError(
            f"paths must be 2-D (n_paths, n_exercise_dates), got shape "
            f"{tuple(paths.shape)}. -> pre-index the simulation grid to "
            "exercise dates before calling."
        )

    N, E = paths.shape
    if E != len(exercise_times):
        raise ShapeError(
            f"paths has {E} columns but {len(exercise_times)} exercise "
            "dates were given. -> ensure the second dimension of `paths` "
            "matches `len(exercise_times)`."
        )
    dtype = paths.dtype
    device = paths.device

    # --- Validate discount_grid ---
    if discount_grid.dim() == 1:
        if discount_grid.shape[0] != E:
            raise ShapeError(
                f"discount_grid has length {discount_grid.shape[0]} but {E} "
                "exercise dates were requested. -> resize discount_grid to "
                f"match the number of exercise dates ({E})."
            )
        discount_grid = (
            discount_grid.to(dtype=dtype, device=device).unsqueeze(0).expand(N, -1)
        )
    elif discount_grid.dim() == 2:
        if discount_grid.shape != (N, E):
            raise ShapeError(
                f"discount_grid shape {tuple(discount_grid.shape)} does not "
                f"match expected ({N}, {E}). -> provide a tensor shaped "
                "(n_paths, n_exercise_dates)."
            )
        discount_grid = discount_grid.to(dtype=dtype, device=device)
    else:
        raise ShapeError(
            f"discount_grid must be 1D or 2D, got {discount_grid.dim()}D. "
            "-> pass a tensor of shape (E,) or (N, E)."
        )

    paths_ex = paths

    basis_fn = _poly_basis if basis == "poly" else _laguerre_basis

    # --- Initialise ---
    payoff_terminal = payoff_fn(paths_ex[:, -1]).clone()
    exercise_idx = torch.full((N,), E - 1, dtype=torch.long, device=device)

    # --- Backward induction ---
    for s in range(E - 2, -1, -1):
        spot_s = paths_ex[:, s]
        intr = payoff_fn(spot_s)

        D_0_to_ex = discount_grid[torch.arange(N, device=device), exercise_idx]
        D_0_to_s = discount_grid[:, s]
        fwd_df = D_0_to_ex / D_0_to_s.clamp(min=EPS)

        cont_value = payoff_terminal * fwd_df
        itm = intr > 0

        if itm.sum() >= min_itm:
            X_itm = spot_s[itm].unsqueeze(1)
            phi_itm = basis_fn(X_itm, degree)
            beta = _ols(phi_itm, cont_value[itm])
            continuation_est = phi_itm @ beta

            exercise_now = intr[itm] > continuation_est
            itm_indices = torch.where(itm)[0]
            ex_indices = itm_indices[exercise_now]

            payoff_terminal[ex_indices] = intr[itm][exercise_now]
            exercise_idx[ex_indices] = s

    final_df = discount_grid[torch.arange(N, device=device), exercise_idx]
    return payoff_terminal * final_df


__all__ = [
    "longstaff_schwartz",
]
