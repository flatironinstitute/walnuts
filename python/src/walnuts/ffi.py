import ctypes
import importlib.resources
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import numpy as np
from numpy.ctypeslib import ndpointer

from .util import rand_u32


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


_common_sampling_argtypes = [
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

_common_summary_argtypes = [
    double_array,  # draws in num_draws by num_params stacked array
    ctypes.c_int,  # num draws
    ctypes.c_int,  # num params
    int_array,  # chain lengths
    ctypes.c_int,  # num chains
    double_array,  # output (size num_params)
    err_ptr,
]



try:
    # this is only relevant to scikit-build-core's editable mode support
    # by trying to import this file as a CPython extension, we trigger
    # a rebuild. The import then fails, since it is just a generic shared
    # object, but that's fine.
    importlib.resources.files("walnuts.libwalnutpie")
except ImportError:
    pass

# NB: in almost all cases, these paths will end up resolving to the same place.
# editable installs are the primary exception
_HERE = Path(__file__).parent
# TODO: the following is primarily useful for editable installs, but will currently only
# work for editable installs on platforms which use .so for shared objects (namely Linux)
_INSTALL_PATH = (importlib.resources.files("walnuts") / "libwalnutpie.so").parent
_PATHS = [_HERE, _INSTALL_PATH]

try:
    # load compiled library
    _lib = None
    _exceptions = []
    for path in _PATHS:
        try:
            _lib = np.ctypeslib.load_library("libwalnutpie", path)
        except Exception as e:
            _exceptions.append(e)
    if _lib is None:
        raise ImportError(f"Failed to load libwalnutpie from {_PATHS}: {_exceptions}")
    _ffi_sample_cfunc = _lib.walnutpie_sample_cfunc
    _ffi_sample_cfunc.restype = ctypes.c_int
    _ffi_sample_cfunc.argtypes = [
        logp_cfunc_type,  # callback
        ctypes.c_void_p,  # data pointer
        ctypes.c_int,  # num_params
        nullable_double_array,  # inits
    ] + _common_sampling_argtypes

    _ffi_sample_bridgestan = _lib.walnutpie_sample_bridgestan
    _ffi_sample_bridgestan.restype = ctypes.c_int
    _ffi_sample_bridgestan.argtypes = [
        ctypes.c_char_p,  # model so
        ctypes.c_char_p,  # model data
        ctypes.c_uint,  # model seed
        ctypes.c_char_p,  # inits
    ] + _common_sampling_argtypes

    _ffi_ess = _lib.walnutpie_ess
    _ffi_ess.restype = ctypes.c_int
    _ffi_ess.argtypes = _common_summary_argtypes
    _ffi_r_hat = _lib.walnutpie_r_hat
    _ffi_r_hat.restype = ctypes.c_int
    _ffi_r_hat.argtypes = _common_summary_argtypes
    _ffi_mcse = _lib.walnutpie_mcse
    _ffi_mcse.restype = ctypes.c_int
    _ffi_mcse.argtypes = _common_summary_argtypes

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
    raise ImportError("Failed to load libwalnutpie") from e


@logp_cfunc_type
def logp_c_trampoline(size, buf, grad, lp, logp_ptr):
    x = np.ctypeslib.as_array(buf, (size,))
    g = np.ctypeslib.as_array(grad, (size,))
    logp = ctypes.cast(logp_ptr, ctypes.POINTER(ctypes.py_object))
    try:
        lp[0], g[:] = logp.contents.value(x)
    except Exception as e:
        print(e)
        return 1
    return 0


T = TypeVar("T")


@dataclass
class WarmupInfo(Generic[T]):
    stepsize: float
    inv_metric: Optional[np.ndarray]
    warmup_draws: Optional[T]


# Wrapper around ndarray that lets us set extra attributes
# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
class WalnutsOutputArray(np.ndarray):
    def __new__(cls, input_array, warmup: WarmupInfo[np.ndarray]):
        obj = np.asarray(input_array).view(cls)

        obj.warmup = warmup
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.warmup = getattr(obj, "warmup", None)


def walnuts_pyfunc(
    logp: Union[
        Callable[[np.ndarray], tuple[float, np.ndarray]],
        "numba.core.ccallback.CFunc",
        tuple[ctypes.CFUNCTYPE, Any],
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
) -> list[WalnutsOutputArray]:
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

    lengths_out = np.zeros((num_chains * 2,), dtype=np.int32)

    if hasattr(logp, "ctypes"):
        # numba's @cfunc decorator, which should generate very fast code
        logp_c = logp.ctypes
        logp_c_data = None
    # elif jax: # TODO
    elif isinstance(logp, tuple):
        logp_c = logp[0]
        logp_c_data = ctypes.byref(logp[1]) if logp[1] is not None else None
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
        warmup_written = lengths_out[i]
        samples_written = lengths_out[i + num_chains]

        warmup_info = WarmupInfo(
            stepsize=stepsize_out[i],
            inv_metric=inv_metric_out[i] if inv_metric_out is not None else None,
            warmup_draws=(
                out[i, 0 : warmup_written + samples_written, :] if save_warmup else None
            ),
        )

        output_chain = WalnutsOutputArray(
            out[i, warmup_written : warmup_written + samples_written, :], warmup_info
        )
        outputs.append(output_chain)

    return outputs


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
