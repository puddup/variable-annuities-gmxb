"""Core linear algebra helpers."""

import torch
from torch import Tensor
from qabit.exceptions import ShapeError


def cholesky_psd(M: Tensor, repair_tol: float = 1e-2) -> Tensor:
    """Lower Cholesky of ``M``, repairing rounding-level non-PSD matrices.

    Factorisation is done in float64 for robustness, then cast back.  A user
    correlation matrix of "nice" round numbers is often infinitesimally
    non-PSD; if the most-negative eigenvalue is within ``repair_tol`` it is
    clipped to the nearest PSD matrix (standard practice).  A substantively
    non-PSD matrix (worse than ``repair_tol``) raises, since that signals a
    real modelling error.
    """
    out_dtype, dev = M.dtype, M.device
    Md = M.double()
    try:
        return torch.linalg.cholesky(Md).to(out_dtype)
    except RuntimeError:
        w, Q = torch.linalg.eigh(Md)
        min_eig = float(w.min())
        if min_eig < -repair_tol:
            raise ShapeError(
                "Combined driver correlation is not positive-semidefinite "
                f"(smallest eigenvalue {min_eig:.4f}). Check the factor-level "
                "matrix together with each factor's internal_corr()."
            )
        w_clipped = w.clamp(min=1e-10)
        repaired = (Q * w_clipped) @ Q.T
        d = repaired.diagonal().sqrt()
        repaired = repaired / torch.outer(d, d)  # restore unit diagonal
        return torch.linalg.cholesky(repaired).to(out_dtype)


__all__ = [
    "cholesky_psd",
]
