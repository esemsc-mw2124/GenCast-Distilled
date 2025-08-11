import jax
import jax.numpy as jnp
from graphcast import xarray_tree

def var_mse(s, t):
    # s and t are xarray.DataArray backed by xarray_jax.JaxArrayWrapper
    # Pull out the raw jax arrays, not the wrappers.
    sd = getattr(s.data, "jax_array", s.data)
    td = getattr(t.data, "jax_array", t.data)
    return jnp.mean((sd - td) ** 2)

def denoiser_l2_loss(student_pred, teacher_pred):
    per_var_mse = xarray_tree.map_structure(var_mse, student_pred, teacher_pred)
    # per_var_mse is a nested dict of JAX scalars; flatten then average
    def _leaves(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from _leaves(v)
            else:
                yield v
    return jnp.mean(jnp.stack(list(_leaves(per_var_mse))))