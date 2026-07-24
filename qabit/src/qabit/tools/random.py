"""tools/random.py — RNG utilities: pseudo-random and quasi-random.

Number generators
─────────────────
``NumberGenerator``      Abstract base.
``PseudoRandom``         Standard torch.randn (default, backward-compat).
``SobolGenerator``       Scrambled Sobol' quasi-random via inverse-normal CDF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


# ── number-generator abstraction ─────────────────────────────────────────────


class NumberGenerator(ABC):
    """Strategy for producing N(0,1) variates of shape (n_paths, n_steps, [n_factors]).

    Every concrete generator must implement ``draw``.  The base class
    accepts ``antithetic`` so that callers don't need to special-case it.
    """

    @abstractmethod
    def draw(
        self,
        n_paths: int,
        n_steps: int,
        n_factors: int = 1,
        *,
        antithetic: bool = False,
        generator: Optional[torch.Generator] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return shape (n_paths, n_steps) if n_factors==1, else (n_paths, n_steps, n_factors)."""

    @staticmethod
    def _apply_antithetic(z: Tensor) -> Tensor:
        """Double along dim-0 via antithetic mirroring."""
        return torch.cat([z, -z], dim=0)


class PseudoRandom(NumberGenerator):
    """Standard pseudo-random N(0,1) — wraps ``torch.randn``."""

    def draw(
        self,
        n_paths,
        n_steps,
        n_factors=1,
        *,
        antithetic=False,
        generator=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        shape = (n_paths, n_steps) if n_factors == 1 else (n_paths, n_steps, n_factors)
        if antithetic:
            if n_paths % 2:
                raise ValueError(f"antithetic requires even n_paths; got {n_paths}.")
            half = (n_paths // 2, *shape[1:])
            z = torch.randn(*half, generator=generator, device=device, dtype=dtype)
            return self._apply_antithetic(z)
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)

    def __repr__(self):
        return "PseudoRandom()"


class SobolGenerator(NumberGenerator):
    """Scrambled Sobol' sequence → inverse-normal CDF.

    Parameters
    ----------
    scramble : bool
        Owen-scrambled Sobol'.  Default ``True``.
    seed : int | None
        Scramble seed for reproducibility.
    """

    def __init__(self, scramble: bool = True, seed: Optional[int] = None):
        self.scramble = scramble
        self.seed = seed

    def draw(
        self,
        n_paths,
        n_steps,
        n_factors=1,
        *,
        antithetic=False,
        generator=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        n_dims = n_steps * n_factors
        eff_paths = n_paths // 2 if antithetic else n_paths
        if antithetic and n_paths % 2:
            raise ValueError(f"antithetic requires even n_paths; got {n_paths}.")
        engine = torch.quasirandom.SobolEngine(n_dims, scramble=self.scramble, seed=self.seed)
        u = engine.draw(eff_paths, dtype=dtype).to(device).clamp(1e-6, 1 - 1e-6)
        z = torch.special.ndtri(u)  # (eff_paths, n_dims)
        if n_factors == 1:
            z = z.reshape(eff_paths, n_steps)
        else:
            z = z.reshape(eff_paths, n_steps, n_factors)
        if antithetic:
            z = self._apply_antithetic(z)
        return z

    def __repr__(self):
        return f"SobolGenerator(scramble={self.scramble}, seed={self.seed})"


__all__ = [
    "NumberGenerator",
    "PseudoRandom",
    "SobolGenerator",
]
