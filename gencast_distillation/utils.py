import jax
import jax.tree_util
import jax.numpy as jnp
import xarray as xr
from graphcast import xarray_jax
import numpy as np

def to_jax(ds: xr.Dataset) -> xr.Dataset:
    """Convert DATA VARS to JAX-backed arrays; leave coords/indexes alone."""
    data_vars = {}
    for name, da in ds.data_vars.items():
        # Make sure we have an in-memory array (compute if it's Dask)
        arr = da.data
        if hasattr(arr, "compute"):        # Dask arrays have .compute()
            arr = arr.compute()
        # Convert to JAX, then wrap
        jax_arr = jnp.asarray(np.asarray(arr))
        wrapped = xarray_jax.JaxArrayWrapper(jax_arr)

        # Rebuild as a DataArray with dims & coords so xarray is happy
        data_vars[name] = xr.DataArray(
            wrapped,
            dims=da.dims,          # <-- crucial: keep the original dims
            coords=da.coords,      # keep coords (don’t convert to JAX)
            attrs=da.attrs,
            name=name,
        )

    # Keep original coords/attrs at the dataset level
    return xr.Dataset(data_vars=data_vars, coords=ds.coords, attrs=ds.attrs)

def copy_pytree(pytree):
    """Performs a deep copy of a JAX pytree."""
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, pytree)


def dummy_dataset_iterator(inputs, targets_template, forcings, batch_size=1):
    while True:
        yield {
            "inputs": inputs,
            "targets_template": targets_template,
            "forcings": forcings,
        }


# def finite_report(tree, name, limit=10):
#     print(f"=== {name} finite report ===")
#     flat = []

#     def _collect(k, v):
#         arr = getattr(v.data, "jax_array", v.data)
#         n = arr.size
#         nfinite = int(jnp.isfinite(arr).sum())
#         flat.append((k, n, nfinite))

#     # walk nested dict structure, collecting stats
#     def _walk(prefix, d):
#         for k, v in d.items():
#             key = f"{prefix}/{k}" if prefix else k
#             if isinstance(v, dict):
#                 _walk(key, v)
#             else:
#                 _collect(key, v)

#     _walk("", tree)
#     flat.sort(key=lambda x: x[2])
#     for k, n, nf in flat[:limit]:
#         print(f"{k:60s} finite={nf:8d}/{n:8d}")


def iterator(
    ds: xr.Dataset,
    chunk_size: int,
    data_utils,
    distillation_model,
    time_dim: str = "time",
    target_lead_times=slice("12h", "12h"),
):
    """
    Iterate over dataset in chunks along the time dimension and yield
    (inputs, targets, forcings) tuples after preprocessing.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    chunk_size : int
        Number of timesteps per iteration.
    data_utils : module or object
        Must have extract_inputs_targets_forcings(batch, ..., target_lead_times=...)
    distillation_model : object
        Must have a task_config attribute (dict-like or Namespace-like).
    time_dim : str
        Name of the time dimension.
    target_lead_times : slice or other
        Lead times to pass to extract_inputs_targets_forcings.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset has no dimension '{time_dim}'.")

    n_time = ds.sizes[time_dim]
    for start in range(0, n_time - (n_time % chunk_size)):
        end = min(start + chunk_size, n_time)
        batch = ds.isel({time_dim: slice(start, end)})

        # Preprocessing & extraction
        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            batch,
            target_lead_times=target_lead_times,
            **getattr(distillation_model, "task_config", {}).__dict__,
        )

        inputs = to_jax(inputs)
        targets = to_jax(targets)
        forcings = to_jax(forcings)

        yield {
            "inputs": inputs,
            "targets": targets,
            "forcings": forcings,
        }