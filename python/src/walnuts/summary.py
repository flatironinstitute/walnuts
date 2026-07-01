from typing import TYPE_CHECKING, List, Union

import numpy as np

from ._ffi import _ffi_ess, _ffi_mcse, _ffi_r_hat

if TYPE_CHECKING:
    from .stan import StanOutputBase


class Summarizer:
    def __init__(self, draws: Union[List[np.ndarray], List["StanOutputBase"]]):
        if hasattr(draws[0], "parameters"):  # StanOutputBase
            draws = [c.data for c in draws]
        self._stacked = np.concat(draws)
        self._num_draws, self._num_params = self._stacked.shape

        self._lengths = np.array([c.shape[0] for c in draws], dtype=np.int32)
        self._num_chains = len(draws)

    # I don't think any of these really need FFI?
    def mean(self):
        return np.mean(self._stacked, axis=0)

    def variance(self):
        return np.var(self._stacked, axis=0, ddof=1)

    def standard_deviation(self):
        return np.std(self._stacked, axis=0, ddof=1)

    def ess(self) -> np.ndarray:
        out = np.zeros((self._num_params,))
        _ffi_ess(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )

        return out

    def r_hat(self) -> np.ndarray:
        out = np.zeros((self._num_params,))
        _ffi_r_hat(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )
        return out

    def mcse(self) -> np.ndarray:
        out = np.zeros((self._num_params,))
        _ffi_mcse(
            self._stacked,
            self._num_draws,
            self._num_params,
            self._lengths,
            self._num_chains,
            out,
        )
        return out


def ess(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).ess()


def r_hat(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).r_hat()


def mcse(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).mcse()


def mean(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).mean()


def variance(draws: Union[List[np.ndarray], List["StanOutputBase"]]) -> np.ndarray:
    return Summarizer(draws).variance()


def standard_deviation(
    draws: Union[List[np.ndarray], List["StanOutputBase"]]
) -> np.ndarray:
    return Summarizer(draws).standard_deviation()
