import haiku as hk
import jax.numpy as jnp
import xarray
from graphcast import gencast, dpm_solver_plus_plus_2s, casting

class PatchedSampler(dpm_solver_plus_plus_2s.Sampler):
    def __init__(self, denoiser_fn, **kwargs):
        super().__init__(denoiser_fn, **kwargs)

    def __call__(self, inputs, targets_template, forcings=None, **kwargs):
        # Patch: wrap original call to use jnp.array(0) as lower bound
        original_fori_loop = hk.fori_loop  # store original

        def patched_fori_loop(*args, **kwargs):
            # Handle both positional and keyword argument styles
            if len(args) == 4:
                lower, upper, body_fun, init_val = args
            elif len(args) == 2 and "body_fun" in kwargs and "init_val" in kwargs:
                # Mixed positional/keyword style — seen in GraphCast
                lower, upper = args
                body_fun = kwargs["body_fun"]
                init_val = kwargs["init_val"]
            else:
                raise ValueError(f"Unexpected fori_loop args: args={args}, kwargs={kwargs}")

            lower = jnp.array(lower)
            return original_fori_loop(lower, upper, body_fun, init_val)

        # Monkey-patch hk.fori_loop locally just for this call, so any other code using Haiku outside this method is unaffected
        original_fori_loop = hk.fori_loop
        try:
            hk.fori_loop = patched_fori_loop
            return super().__call__(inputs, targets_template, forcings, **kwargs)
        finally:
            hk.fori_loop = original_fori_loop  # restore original


    
class PatchedGenCast(gencast.GenCast):
    def __call__(self, inputs, targets_template, forcings=None, **kwargs):
        if self._sampler is None:
            self._sampler = PatchedSampler(
                self._preconditioned_denoiser, **self._sampler_config
            )
        return self._sampler(inputs, targets_template, forcings, **kwargs)