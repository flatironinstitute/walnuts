import ctypes
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union, Callable

import numpy as np
from numpy.ctypeslib import ndpointer


# ctypes helpers
def wrapped_ndptr(*args, **kwargs):
    """
    A version of np.ctypeslib.ndpointer
    which allows None (passed as NULL)
    """
    base = ndpointer(*args, **kwargs)

    def from_param(_cls, obj):
        if obj is None:
            return obj
        return base.from_param(obj)

    return type(base.__name__, (base,), {"from_param": classmethod(from_param)})


double_array = ndpointer(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
int_array = ndpointer(dtype=ctypes.c_int, flags=("C_CONTIGUOUS"))
nullable_double_array = wrapped_ndptr(dtype=ctypes.c_double, flags=("C_CONTIGUOUS"))
err_ptr = ctypes.POINTER(ctypes.c_void_p)

logp_cfunc_type = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_size_t,  # size
    ctypes.POINTER(ctypes.c_double),  # theta
    ctypes.POINTER(ctypes.c_double),  # grad
    ctypes.POINTER(ctypes.c_double),  # lp
    ctypes.c_void_p,  # data
)

_exception_types = [RuntimeError, ValueError, KeyboardInterrupt]


def _raise_for_error(rc: int, err):
    if rc != 0:
        if err.contents:
            msg = _get_error_msg(err.contents).decode("utf-8")
            exception_type = _get_error_type(err.contents)
            _free_error(err.contents)
            exn = _exception_types[exception_type]
            raise exn(msg)
        else:
            raise RuntimeError(f"Unknown error, function returned code {rc}")


_common_argtypes_suffix = [
    ctypes.c_size_t,  # num_chains
    ctypes.c_uint,  # seed
    ctypes.c_uint,  # id
    ctypes.c_double,  # init_radius
    nullable_double_array,  # metric init in
    ctypes.c_int,  # min_warmup_iter
    ctypes.c_int,  # max_warmup_iter
    ctypes.c_int,  # min_sampling_iter
    ctypes.c_int,  # max_sampling_iter
    ctypes.c_int,  # max_trajectory_doublings
    ctypes.c_int,  # max_step_halvings
    ctypes.c_int,  # min_micro_steps
    ctypes.c_double,  # max_hamiltonian_error
    ctypes.c_double,  # step_size_converge_tol
    ctypes.c_double,  # mass_converge_tol
    ctypes.c_double,  # rhat_converge_tol
    ctypes.c_double,  # mass_init_count
    ctypes.c_double,  # mass_additive_smoothing
    ctypes.c_double,  # max_macro_steps_target
    ctypes.c_double,  # step_size_init
    ctypes.c_double,  # step_accept_rate_target
    ctypes.c_double,  # step_learning_rate
    ctypes.c_double,  # step_gradient_decay
    ctypes.c_double,  # step_sq_gradient_decay
    ctypes.c_double,  # step_stabilization
    ctypes.c_double,  # step_learn_rate_decay
    ctypes.c_bool,  # save_warmup
    ctypes.c_int,  # refresh
    double_array,
    ctypes.c_size_t,  # buffer size
    int_array,  # final lengths
    nullable_double_array,  # stepsize out
    nullable_double_array,  # metric out
    err_ptr,
]


HERE = Path(__file__).parent
try:
    # TEMPORARY: replace with importlib_resources or similar in wheel build
    # c.f. https://github.com/scikit-build/scikit-build-core
    _libwalnuts_src = HERE / "walnutpy.cpp"
    _libwalnuts = HERE / "libwalnuts.so"
    if not _libwalnuts.exists() or (
        os.path.getmtime(_libwalnuts_src) > os.path.getmtime(_libwalnuts)
    ):
        print("(re)-building libwalnuts")
        import subprocess

        subprocess.run(
            [
                "c++",
                "-I",
                "include/",
                "-I",
                "thirdparty/bridgestan/",
                "-I",
                "build/_deps/eigen/",
                "python/walnutpy.cpp",
                "-std=c++20",
                "-shared",
                "-fPIC",
                "-pthread",
                "-o",
                "python/libwalnuts.so",
                "-O3",
                "-flto",
            ],
            cwd=HERE.parent.absolute(),
        )
    # load compiled library
    _lib = ctypes.CDLL(os.fspath(_libwalnuts.absolute()))

    _ffi_sample_cfunc = _lib.walnutpie_sample_cfunc
    _ffi_sample_cfunc.restype = ctypes.c_int
    _ffi_sample_cfunc.argtypes = [
        logp_cfunc_type,  # callback
        ctypes.c_void_p,  # data pointer
        ctypes.c_int,  # num_params
        nullable_double_array,  # inits
    ] + _common_argtypes_suffix

    _ffi_sample_bridgestan = _lib.walnutpie_sample_bridgestan
    _ffi_sample_bridgestan.restype = ctypes.c_int
    _ffi_sample_bridgestan.argtypes = [
        ctypes.c_char_p,  # model so
        ctypes.c_char_p,  # model data
        ctypes.c_uint,  # model seed
        ctypes.c_char_p,  # inits
    ] + _common_argtypes_suffix

    _get_error_msg = _lib.walnutpie_get_error_message
    _get_error_msg.restype = ctypes.c_char_p
    _get_error_msg.argtypes = [ctypes.c_void_p]
    _get_error_type = _lib.walnutpie_get_error_type
    _get_error_type.restype = ctypes.c_int  # really enum
    _get_error_type.argtypes = [ctypes.c_void_p]
    _free_error = _lib.walnutpie_destroy_error
    _free_error.restype = None
    _free_error.argtypes = [ctypes.c_void_p]

    # TODO
    # _get_separator = _lib.walnutpie_separator_char
    # _get_separator.restype = ctypes.c_char
    # _get_separator.argtypes = []
    # _sep = _get_separator()

except Exception as e:
    raise ImportError("Failed to load libwalnuts") from e


def rand_u32():
    """Generate a random 32-bit unsigned integer."""
    return np.random.randint(0, 2**32 - 1, dtype=np.uint32)


# TODO opportunistically use numba for this function?
@logp_cfunc_type
def logp_c_trampoline(size, buf, grad, lp, logp_ptr):
    x = np.ctypeslib.as_array(buf, (size,))
    g = np.ctypeslib.as_array(grad, (size,))
    logp = ctypes.cast(logp_ptr, ctypes.POINTER(ctypes.py_object))
    try:
        lp[0], g[:] = logp.contents.value(x)
    except Exception:
        return 1
    return 0


def walnuts_pyfunc(
    logp: Union[
        Callable[[np.ndarray], tuple[float, np.ndarray]], "numba.core.ccallback.CFunc"
    ],
    num_params: Optional[int] = None,
    inits: Optional[np.ndarray] = None,
    *,
    num_chains: int = 4,
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
):
    if num_params is None:
        if inits is None:
            raise ValueError("must specify at least one of num_params or inits")
        init_shape = inits.shape
        if len(init_shape) == 2:
            num_params = init_shape[1]
        else:
            num_params = init_shape[0]

    # these are checked here because they're sizes for "out"
    if num_chains < 1:
        raise ValueError("num_chains must be at least 1")
    if max_warmup_iter < 0:
        raise ValueError("max_warmup_iter must be non-negative")
    if max_sampling_iter < 1:
        raise ValueError("max_sampling_iter must be at least 1")
    if num_params < 1:
        raise ValueError("num_params must be at least 1")

    seed = seed or rand_u32()

    num_draws = max_sampling_iter + max_warmup_iter * save_warmup
    out = np.full((num_chains, num_draws, num_params), np.nan, dtype=np.float64)

    if inits is not None:
        if inits.shape == (num_params,):
            inits = np.repeat(inits[np.newaxis], num_chains, axis=0)
        elif inits.shape == (num_chains, num_params):
            pass
        else:
            raise ValueError(
                f"Invalid inits size. Expected a {(num_params,)} "
                f"or {(num_chains, num_params)} matrix."
            )

    if init_inv_metric is not None:
        if init_inv_metric.shape == (num_params,):
            init_inv_metric = np.repeat(init_inv_metric[np.newaxis], num_chains, axis=0)
        elif init_inv_metric.shape == (num_chains, num_params):
            pass
        else:
            raise ValueError(
                f"Invalid initial metric size. Expected a {(num_params,)} "
                f"or {(num_chains, num_params)} matrix."
            )

    inv_metric_out = None
    stepsize_out = np.zeros(num_chains, dtype=np.float64)
    if save_inv_metric:
        inv_metric_out = np.zeros((num_chains, num_params), dtype=np.float64)

    lengths_out = np.zeros((num_chains,), dtype=np.int32)

    if hasattr(logp, "ctypes"):
        # numba's @cfunc decorator, which should generate very fast code
        logp_c = logp.ctypes
        logp_c_data = None
    # elif jax: # TODO
    else:
        # if we just have a generic python function, best we can do is wrap it
        logp_c = logp_c_trampoline
        logp_c_data = ctypes.byref(ctypes.py_object(logp))

    err = ctypes.pointer(ctypes.c_void_p())
    rc = _ffi_sample_cfunc(
        logp_c,
        logp_c_data,
        num_params,
        inits,
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
    _raise_for_error(rc, err)

    outputs = []
    for i in range(num_chains):
        output_chain = out[i, 0 : lengths_out[i], :]
        # TODO
        # output_chain.stepsize = stepsize_out[i]
        # if inv_metric_out is not None:
        #     output_chain.inv_metric = inv_metric_out[i]
        # else:
        #     output_chain.inv_metric = None
        outputs.append(output_chain)

    return outputs


# BridgeStan support

# TODO actually use print_callback in underlying call
print_callback_type = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t, ctypes.c_bool
)


@print_callback_type
def print_callback(msg, size, is_error):
    print(
        ctypes.string_at(msg, size).decode("utf-8"),
        file=sys.stderr if is_error else sys.stdout,
    )


import bridgestan
import stanio

StanData = Union[str, os.PathLike, Mapping[str, Any]]


class StanOutput:
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutput.get` method, or by using the object as a dictionary.
    """

    stepsize: Optional[np.ndarray]
    inv_metric: Optional[np.ndarray]
    hessian: Optional[np.ndarray]

    def __init__(self, parameters: List[str], data: np.ndarray):
        self.raw_parameters = parameters
        self._params = stanio.parse_header(",".join(parameters))
        self._data = data
        self.inv_metric = None
        self.stepsize = None

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


# TODO also allow inits from a StanOutput?
def encode_stan_json(data: Union[str, os.PathLike, Mapping[str, Any]]) -> bytes:
    """Turn the provided data into something we can send to C++."""
    if isinstance(data, os.PathLike):
        return os.fspath(data).encode()
    if isinstance(data, str):
        return data.encode()
    return stanio.dump_stan_json(data).encode()


def _encode_stan_inits(inits, chains, seed):
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
):
    # these are checked here because they're sizes for "out"
    if num_chains < 1:
        raise ValueError("num_chains must be at least 1")
    if max_warmup_iter < 0:
        raise ValueError("max_warmup_iter must be non-negative")
    if max_sampling_iter < 1:
        raise ValueError("max_sampling_iter must be at least 1")

    seed = seed or rand_u32()

    # TODO assert bridgestan is at least 2.9
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

    lengths_out = np.zeros((num_chains,), dtype=np.int32)

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
    _raise_for_error(rc, err)

    outputs = []
    for i in range(num_chains):
        out_chain = out[i, 0 : lengths_out[i], :]
        output_chain = StanOutput(param_names, out_chain)
        output_chain.stepsize = stepsize_out[i]
        if inv_metric_out is not None:
            output_chain.inv_metric = inv_metric_out[i]
        else:
            output_chain.inv_metric = None
        outputs.append(output_chain)

    return outputs
