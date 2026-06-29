import ctypes
from typing import TYPE_CHECKING, List, Union

import numpy as np

from .ffi import _ffi_ess, _ffi_mcse, _ffi_r_hat, _raise_for_error

if TYPE_CHECKING:
    from .stan import StanOutputBase


class Summarizer:
    def __init__(self, draws: Union[List[np.ndarray], List["StanOutputBase"]]):
        if hasattr(draws[0], "parameters"):  # StanOutputBase
            draws = [c.data for c in draws]
        self.stacked_draws = np.concat(draws)
        self.num_draws, self.num_params = self.stacked_draws.shape

        self.lengths = np.array([c.shape[0] for c in draws], dtype=np.int32)
        self.num_chains = len(draws)

    # TODO: mean, variance, stddev, and quantiles: I don't think any of these really need FFI?
    def mean(self):
        return np.mean(self.stacked_draws, axis=0)

    def ess(self) -> np.ndarray:
        out = np.zeros((self.num_params,))
        err = ctypes.pointer(ctypes.c_void_p())
        rc = _ffi_ess(
            self.stacked_draws,
            self.num_draws,
            self.num_params,
            self.lengths,
            self.num_chains,
            out,
            err,
        )
        _raise_for_error(rc, err)
        return out

    def r_hat(self) -> np.ndarray:
        out = np.zeros((self.num_params,))
        err = ctypes.pointer(ctypes.c_void_p())
        rc = _ffi_r_hat(
            self.stacked_draws,
            self.num_draws,
            self.num_params,
            self.lengths,
            self.num_chains,
            out,
            err,
        )
        _raise_for_error(rc, err)
        return out

    def mcse(self) -> np.ndarray:
        out = np.zeros((self.num_params,))
        err = ctypes.pointer(ctypes.c_void_p())
        rc = _ffi_mcse(
            self.stacked_draws,
            self.num_draws,
            self.num_params,
            self.lengths,
            self.num_chains,
            out,
            err,
        )
        _raise_for_error(rc, err)
        return out


def ess(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).ess()


def r_hat(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).r_hat()


def mcse(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).mcse()
