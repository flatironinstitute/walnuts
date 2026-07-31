from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)


T = TypeVar("T")


@dataclass
class WarmupInfo(Generic[T]):
    """
    Warmup output from a single chain, parameterized by the array type.

    Attributes
    ----------
    stepsize : float
        The adapted step size from warmup.
    inv_metric : Optional[np.ndarray]
        The diagonal inverse mass matrix estimated during warmup, or ```None`` if not saved.
    warmup_draws : Optional[T]
        The warmup draws, or ``None`` if not saved.
    """
    stepsize: float
    inv_metric: Optional[np.ndarray]
    warmup_draws: Optional[T]
