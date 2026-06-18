import numpy as np


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)
