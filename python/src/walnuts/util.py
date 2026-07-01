from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import numpy as np


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)


T = TypeVar("T")


@dataclass
class WarmupInfo(Generic[T]):
    stepsize: float
    inv_metric: Optional[np.ndarray]
    warmup_draws: Optional[T]
