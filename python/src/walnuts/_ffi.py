from contextlib import ExitStack
import atexit
import ctypes
import importlib.resources
import sys

from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer


# loading the library
# mostly taken from the guide at https://scikit-build-core.readthedocs.io/en/latest/guide/faqs.html#shipping-a-library-to-load-with-ctypes,
# with some inspiration from FINUFFT as well

try:
    # this is only relevant to scikit-build-core's editable mode support
    # by trying to import this file as a CPython extension, we trigger
    # a rebuild. The import then fails, since it is just a generic shared
    # object, but that's fine.
    importlib.resources.files("walnuts.libwalnutpie")
except ImportError:
    pass

# NB: in almost all cases, these paths will end up resolving to the same place.
# editable installs are the primary exception, as well as usages of `zipimport`
_HERE = Path(__file__).parent

# in some configurations, the package is loaded from a zip file or somewhere else not realized on the file system.
# the as_file function will give us a (possibly temporary) file path, which needs to be cleaned up later
_files = ExitStack()
atexit.register(_files.close)
_suffix = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
_INSTALL_PATH = _files.enter_context(
    importlib.resources.as_file(
        importlib.resources.files("walnuts") / f"libwalnutpie{_suffix}"
    )
).parent

_PATHS = set([_HERE, _INSTALL_PATH])

_lib = None
_exceptions = []
for path in _PATHS:
    try:
        _lib = np.ctypeslib.load_library("libwalnutpie", path)
    except Exception as e:
        _exceptions.append(e)

if _lib is None:
    raise ImportError(f"Failed to load libwalnutpie from {_PATHS}: {_exceptions}")


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

# getting function pointers

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


_exception_types = [RuntimeError, ValueError, KeyboardInterrupt]


def raise_for_error(rc: int, err):
    if rc != 0:
        if err.contents:
            msg = _get_error_msg(err.contents).decode("utf-8")
            exception_type = _get_error_type(err.contents)
            _free_error(err.contents)
            exn = _exception_types[exception_type]
            raise exn(msg)
        else:
            raise RuntimeError(f"Unknown error, function returned code {rc}")
