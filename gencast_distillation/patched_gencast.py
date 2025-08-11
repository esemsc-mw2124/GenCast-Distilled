import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import xarray
import inspect
from graphcast import gencast, dpm_solver_plus_plus_2s, casting, xarray_jax, samplers_utils as utils

try:
    tree_map = jax.tree.map          # JAX >= 0.4.25 (incl. 0.6+)
except AttributeError:
    from jax import tree_util as _tu  # Any JAX version
    tree_map = _tu.tree_map

def zeros_like_structure(template):
    """Preserve xarray structure when present; otherwise PyTree of JAX arrays."""
    try:
        import xarray as xr
        return xr.zeros_like(template)
    except Exception:
        return tree_map(lambda a: jnp.zeros_like(jnp.asarray(a)), template)

class PatchedSampler(dpm_solver_plus_plus_2s.Sampler):
    def __init__(self, denoiser_fn, **kwargs):
        super().__init__(denoiser_fn, **kwargs)
        # Ensure the attribute our patch uses is present, regardless of base naming.
        if not hasattr(self, "_preconditioned_denoiser"):
            self._preconditioned_denoiser = (
                getattr(self, "_denoiser", None)
                or getattr(self, "_denoiser_fn", None)
                or denoiser_fn
            )

    def __call__(self, inputs, targets_template, forcings=None, **kwargs):
        noise_levels = jnp.asarray(self._noise_levels)
        assert noise_levels.ndim == 1 and noise_levels.shape[0] >= 2
        num_steps = int(noise_levels.shape[0] - 1)

        per_step_churn_rates = getattr(self, "_per_step_churn_rates", None)
        if per_step_churn_rates is None:
            per_step_churn_rates = jnp.zeros((num_steps,), dtype=noise_levels.dtype)
        else:
            per_step_churn_rates = jnp.asarray(per_step_churn_rates, dtype=noise_levels.dtype)
            assert per_step_churn_rates.shape == (num_steps,)

        # ---- Denoiser: build then remat (Option 1) ----
        def den_factory():
            den_call = self._preconditioned_denoiser  # guaranteed by __init__
            sig = inspect.signature(den_call)
            param_names = list(sig.parameters.keys())
            if param_names and param_names[0] == "self":
                param_names = param_names[1:]

            # Stable signature for JAX: (noise_level, x, i)
            def den(noise_level, x, i):
                call_kwargs = {}

                # inputs/x mapping
                if "x" in param_names:
                    call_kwargs["x"] = x
                if "inputs" in param_names:
                    call_kwargs["inputs"] = inputs
                if "forcings" in param_names:
                    call_kwargs["forcings"] = forcings

                # noise arg(s)
                if "noise_level" in param_names:
                    call_kwargs["noise_level"] = noise_level
                elif "sigma" in param_names:
                    call_kwargs["sigma"] = noise_level
                elif "noise_levels" in param_names:
                    call_kwargs["noise_levels"] = noise_levels

                # loop index if requested
                if "i" in param_names:
                    call_kwargs["i"] = i

                return den_call(**call_kwargs)

            return den

        den = hk.remat(den_factory())

        # ---- Helpers ----
        def init_noise(template):
            return noise_levels[0] * utils.spherical_white_noise_like(template)

        eps = jnp.finfo(noise_levels.dtype if jnp.issubdtype(noise_levels.dtype, jnp.floating)
                        else jnp.float32).eps

        def body_fn(i, x):
            x = utils.tree_where(i == 0, x + init_noise(x), x)

            noise_level = noise_levels[i]
            next_noise_level = noise_levels[i + 1]

            if self._stochastic_churn:
                x, noise_level = utils.apply_stochastic_churn(
                    x, noise_level,
                    stochastic_churn_rate=per_step_churn_rates[i],
                    noise_level_inflation_factor=self._noise_level_inflation_factor,
                )

            mid_noise_level = jnp.sqrt(noise_level * next_noise_level)
            mid_over_current = mid_noise_level / jnp.maximum(noise_level, eps)
            next_over_current = next_noise_level / jnp.maximum(noise_level, eps)

            x_denoised = den(noise_level, x, i)
            x_mid = mid_over_current * x + (1.0 - mid_over_current) * x_denoised
            x_mid_denoised = den(mid_noise_level, x_mid, i)
            x_next = next_over_current * x + (1.0 - next_over_current) * x_mid_denoised

            return utils.tree_where(next_noise_level == 0, x_denoised, x_next)

        noise_init = zeros_like_structure(targets_template)
        return hk.fori_loop(0, num_steps, body_fn, init_val=noise_init)




class PatchedGenCast(gencast.GenCast):
    def __call__(self, inputs, targets_template, forcings=None, **kwargs):
        if self._sampler is None:
            self._sampler = PatchedSampler(self._preconditioned_denoiser, **self._sampler_config)
        return self._sampler(inputs, targets_template, forcings, **kwargs)
