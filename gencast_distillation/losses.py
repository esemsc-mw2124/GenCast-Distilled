import jax
import jax.numpy as jnp
from graphcast import xarray_tree

def _unwrap(da):
    return getattr(da.data, "jax_array", da.data)

def var_mse_masked(s, t):
    sd = _unwrap(s)
    td = _unwrap(t)

    # Boolean mask of valid elements
    m = jnp.isfinite(sd) & jnp.isfinite(td)

    # Replace NaNs/Infs before arithmetic to keep the backward pass clean
    sd = jnp.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)
    td = jnp.nan_to_num(td, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute only with sanitized values, then mask multiplicatively
    diff = (sd - td) * m.astype(sd.dtype)

    denom = jnp.maximum(jnp.sum(m, dtype=sd.dtype), jnp.asarray(1.0, sd.dtype))
    return jnp.sum(diff * diff) / denom

def denoiser_l2_loss(student_pred, teacher_pred):
    per_var = xarray_tree.map_structure(var_mse_masked, student_pred, teacher_pred)
    def _leaves(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from _leaves(v)
            else:
                yield v
    vals = list(_leaves(per_var))
    return jnp.mean(jnp.stack(vals))