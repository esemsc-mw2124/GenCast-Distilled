import jax
import jax.numpy as jnp
from graphcast import xarray_tree

def _unwrap(da):
    # Get raw JAX arrays from xarray_jax wrappers
    return getattr(da.data, "jax_array", da.data)

def var_mse_masked(s, t):
    sd = _unwrap(s)
    td = _unwrap(t)
    mask = jnp.isfinite(sd) & jnp.isfinite(td)
    # Zero out invalid sites; count only valid ones
    diff = jnp.where(mask, sd - td, 0.0)
    denom = jnp.maximum(mask.sum(dtype=diff.dtype), jnp.array(1.0, dtype=diff.dtype))
    return jnp.sum(diff * diff) / denom

def denoiser_l2_loss(student_pred, teacher_pred):
    per_var = xarray_tree.map_structure(var_mse_masked, student_pred, teacher_pred)

    # Flatten dict-of-dicts of scalars, like you already do
    def _leaves(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from _leaves(v)
            else:
                yield v

    vals = list(_leaves(per_var))
    return jnp.mean(jnp.stack(vals))
