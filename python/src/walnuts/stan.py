import ctypes
import os
from typing import Any, Dict, List, Mapping, Optional, Union

import bridgestan
import numpy as np
import stanio

from ._ffi import _ffi_sample_bridgestan, raise_for_error
from .util import WarmupInfo, rand_u32

StanData = Union[str, os.PathLike, Mapping[str, Any]]


class StanOutputBase:
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutput.get` method, or by using the object as a dictionary.
    """

    def __init__(self, parameters: List[str], data: np.ndarray):
        self.raw_parameters = parameters
        self._params = stanio.parse_header(",".join(parameters))
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """The underlying draws from the Stan model."""
        return self._data

    @property
    def parameters(self) -> List[str]:
        """The names of the parameters in the Stan model."""
        return list(self._params.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """Extract a parameter from the Stan output."""
        return self.get(key)

    def get(self, key: str) -> np.ndarray:
        """
        Extract a parameter from the Stan output.
        Synonym for ``obj[key]``.

        Parameters
        ----------
        key : str
            name of the parameter to extract

        Returns
        -------
        np.ndarray
            The parameter values. Shape depends
            on the Stan type and algorithm used.
        """
        return self._params[key].extract_reshape(self._data)

    def __repr__(self) -> str:
        return f"StanOutput(parameters={repr(self.raw_parameters)}, data={repr(self.data)})"

    def __str__(self) -> str:
        p = "\n\t".join(self.parameters)
        return f"StanOutput with parameters:\n\t{p}"


class StanOutput(StanOutputBase):
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutput.get` method, or by using the object as a dictionary.
    """

    def __init__(
        self,
        parameters: List[str],
        data: np.ndarray,
        warmup: WarmupInfo[StanOutputBase],
    ):
        super().__init__(parameters, data)
        self.warmup = warmup

    def create_inits(
        self, *, chains: int = 4, seed: Optional[int] = None
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Create a dictionary of parameters suitable for initializing a new Stan run.

        Parameters
        ----------
        chains : int, optional
            Number of chains needed, by default 4
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.

        Returns
        -------
        Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
            A dictionary of parameters, or a list of dictionaries if
            chains > 1.
        """
        if self._data.ndim == 1:
            return {
                name: var.extract_reshape(self._data)
                for name, var in self._params.items()
            }

        data = self._data.reshape((-1, self._data.shape[-1]))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(data.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = data[idxs[0]]
            return {
                name: var.extract_reshape(draw) for name, var in self._params.items()
            }
        return [
            {name: var.extract_reshape(data[idx]) for name, var in self._params.items()}
            for idx in idxs
        ]


def encode_stan_json(data: Union[str, os.PathLike, Mapping[str, Any]]) -> bytes:
    """Turn the provided data into something we can send to C++."""
    if isinstance(data, os.PathLike):
        return os.fspath(data).encode()
    if isinstance(data, str):
        return data.encode()
    return stanio.dump_stan_json(data).encode()


def _encode_stan_inits(inits, chains, seed) -> Union[bytes, None]:
    inits_encoded = None
    if inits is not None:
        if isinstance(inits, StanOutput):
            inits = inits.create_inits(chains=chains, seed=seed)

        if isinstance(inits, list):
            inits_encoded = _sep.join(encode_stan_json(init) for init in inits)
        else:
            inits_encoded = encode_stan_json(inits)
    return inits_encoded


def walnuts_stan(
    model: bridgestan.StanModel,
    *,
    num_chains: int = 4,
    inits: Union[StanData, List[StanData], None] = None,
    seed: Optional[int] = None,
    id: int = 1,
    init_radius: float = 2.0,
    init_inv_metric: Optional[np.ndarray] = None,
    save_inv_metric: bool = False,
    min_warmup_iter: int = 50,
    max_warmup_iter: int = 1000,
    min_sampling_iter: int = 50,
    max_sampling_iter: int = 1000,
    max_trajectory_doublings: int = 5,
    max_step_halvings: int = 5,
    min_micro_steps: int = 1,
    max_hamiltonian_error: float = 0.5,
    step_size_converge_tol: float = 0.1,
    mass_converge_tol: float = 1.0,
    rhat_converge_tol: float = 1.01,
    mass_init_count: float = 4.0,
    mass_additive_smoothing: float = 1e-5,
    max_macro_steps_target: float = 15.0,
    step_size_init: float = 1.0,
    step_accept_rate_target: float = 0.8,
    step_learning_rate: float = 0.05,
    step_gradient_decay: float = 0.8,
    step_sq_gradient_decay: float = 0.9,
    step_stabilization: float = 1e-4,
    step_learn_rate_decay: float = 0.5,
    save_warmup: bool = False,
    refresh: int = 0,
) -> list[StanOutput]:
    # these are checked here because they're sizes for "out"
    if num_chains < 1:
        raise ValueError("num_chains must be at least 1")
    if max_warmup_iter < 0:
        raise ValueError("max_warmup_iter must be non-negative")
    if max_sampling_iter < 1:
        raise ValueError("max_sampling_iter must be at least 1")

    seed = seed or rand_u32()

    if model.model_version() < (2, 9, 0):
        # TODO assert bridgestan is at least 2.9
        pass

    model_params = model.param_unc_num()
    param_names = model.param_names(include_tp=True, include_gq=True)

    num_params = len(param_names)
    num_draws = max_sampling_iter + max_warmup_iter * save_warmup
    out = np.zeros((num_chains, num_draws, num_params), dtype=np.float64)

    metric_size = (model_params,)
    if init_inv_metric is not None:
        if init_inv_metric.shape == metric_size:
            init_inv_metric = np.repeat(init_inv_metric[np.newaxis], num_chains, axis=0)
        elif init_inv_metric.shape == (num_chains, *metric_size):
            pass
        else:
            raise ValueError(
                f"Invalid initial metric size. Expected a {metric_size} "
                f"or {(num_chains, *metric_size)} matrix."
            )

    inv_metric_out = None
    stepsize_out = np.zeros(num_chains, dtype=np.float64)
    if save_inv_metric:
        inv_metric_out = np.zeros((num_chains, *metric_size), dtype=np.float64)

    lengths_out = np.zeros((num_chains * 2,), dtype=np.int32)

    err = ctypes.pointer(ctypes.c_void_p())
    rc = _ffi_sample_bridgestan(
        model.lib_path.encode(),  # TODO: alternative that doesn't require double instantiation?
        model.data.encode(),
        model.seed,
        _encode_stan_inits(inits, num_chains, seed),
        num_chains,
        seed,
        id,
        init_radius,
        init_inv_metric,
        min_warmup_iter,
        max_warmup_iter,
        min_sampling_iter,
        max_sampling_iter,
        max_trajectory_doublings,
        max_step_halvings,
        min_micro_steps,
        max_hamiltonian_error,
        step_size_converge_tol,
        mass_converge_tol,
        rhat_converge_tol,
        mass_init_count,
        mass_additive_smoothing,
        max_macro_steps_target,
        step_size_init,
        step_accept_rate_target,
        step_learning_rate,
        step_gradient_decay,
        step_sq_gradient_decay,
        step_stabilization,
        step_learn_rate_decay,
        save_warmup,
        refresh,
        out,
        out.size,
        lengths_out,
        stepsize_out,
        inv_metric_out,
        err,
    )
    raise_for_error(rc, err)

    outputs = []
    for i in range(num_chains):
        warmup_written = lengths_out[i]
        samples_written = lengths_out[i + num_chains]

        warmup_output = None
        if save_warmup:
            warmup_output = StanOutputBase(param_names, out[i, 0:warmup_written, :])

        warmup_info = WarmupInfo(
            stepsize_out[i],
            inv_metric_out[i] if inv_metric_out is not None else None,
            warmup_output,
        )

        out_chain = out[i, warmup_written : warmup_written + samples_written, :]
        output_chain = StanOutput(param_names, out_chain, warmup_info)
        output_chain.stepsize = stepsize_out[i]

        outputs.append(output_chain)

    return outputs
