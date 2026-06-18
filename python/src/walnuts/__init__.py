from .ffi import logp_cfunc_type, walnuts_pyfunc

try:
    from .stan import walnuts_stan
except ImportError as e:

    def walnuts_stan(*args, **kwargs):
        raise RuntimeError(
            "Failed to load walnuts_stan. Is bridgestan installed and working?"
        ) from e


__all__ = ["walnuts_stan", "walnuts_pyfunc", "logp_cfunc_type"]
__version__ = "0.0.1"
