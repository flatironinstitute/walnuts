import ctypes
from typing import Any, Callable, Optional, Union

import numpy as np

from ._ffi import _ffi_sample_cfunc, logp_cfunc_type
from .util import WarmupInfo, rand_u32


# Wrapper around ndarray that lets us set extra attributes
# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
class WalnutsOutputArray(np.ndarray):

    warmup: WarmupInfo[np.ndarray]  #: TODO doc

    def __new__(cls, input_array, warmup: WarmupInfo[np.ndarray]):
        obj = np.asarray(input_array).view(cls)

        obj.warmup = warmup
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.warmup = getattr(obj, "warmup", None)


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
    """
    _summary_

    Parameters
    ----------
    logp : Union[ Callable[[np.ndarray], tuple[float, np.ndarray]], &quot;numba.core.ccallback.CFunc&quot;, tuple[ctypes.CFUNCTYPE, Any], ]
        _description_
    num_params : Optional[int], optional
        _description_, by default None
    inits : Optional[np.ndarray], optional
        _description_, by default None
    num_chains : int, optional
        _description_, by default 4
    seed : Optional[int], optional
        _description_, by default None
    id : int, optional
        _description_, by default 1
    init_radius : float, optional
        _description_, by default 2.0
    init_inv_metric : Optional[np.ndarray], optional
        _description_, by default None
    save_inv_metric : bool, optional
        _description_, by default False
    min_warmup_iter : int, optional
        _description_, by default 50
    max_warmup_iter : int, optional
        _description_, by default 1000
    min_sampling_iter : int, optional
        _description_, by default 50
    max_sampling_iter : int, optional
        _description_, by default 1000
    max_trajectory_doublings : int, optional
        _description_, by default 5
    max_step_halvings : int, optional
        _description_, by default 5
    min_micro_steps : int, optional
        _description_, by default 1
    max_hamiltonian_error : float, optional
        _description_, by default 0.5
    step_size_converge_tol : float, optional
        _description_, by default 0.1
    mass_converge_tol : float, optional
        _description_, by default 1.0
    rhat_converge_tol : float, optional
        _description_, by default 1.01
    mass_init_count : float, optional
        _description_, by default 4.0
    mass_additive_smoothing : float, optional
        _description_, by default 1e-5
    max_macro_steps_target : float, optional
        _description_, by default 15.0
    step_size_init : float, optional
        _description_, by default 1.0
    step_accept_rate_target : float, optional
        _description_, by default 0.8
    step_learning_rate : float, optional
        _description_, by default 0.05
    step_gradient_decay : float, optional
        _description_, by default 0.8
    step_sq_gradient_decay : float, optional
        _description_, by default 0.9
    step_stabilization : float, optional
        _description_, by default 1e-4
    step_learn_rate_decay : float, optional
        _description_, by default 0.5
    save_warmup : bool, optional
        _description_, by default False
    refresh : int, optional
        _description_, by default 0

    Returns
    -------
    list[WalnutsOutputArray]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
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
    elif isinstance(logp, tuple):
        logp_c = logp[0]
        logp_c_data = ctypes.byref(logp[1]) if logp[1] is not None else None
    else:
        # if we just have a generic python function, best we can do is wrap it
        # TODO: does a faster path exist for JAX?
        logp_c = logp_c_trampoline
        logp_c_data = ctypes.byref(ctypes.py_object(logp))

    _ffi_sample_cfunc(
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
    )

    outputs = []
    for i in range(num_chains):
        warmup_written = lengths_out[i]
        samples_written = lengths_out[i + num_chains]

        warmup_info = WarmupInfo(
            stepsize=stepsize_out[i],
            inv_metric=inv_metric_out[i] if inv_metric_out is not None else None,
            warmup_draws=(out[i, 0:warmup_written, :] if save_warmup else None),
        )

        output_chain = WalnutsOutputArray(
            out[i, warmup_written : warmup_written + samples_written, :], warmup_info
        )
        outputs.append(output_chain)

    return outputs
