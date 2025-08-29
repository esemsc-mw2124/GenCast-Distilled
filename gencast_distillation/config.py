import xarray as xr
import numpy as np
import pandas as pd
import gcsfs
import io
from gencast_distillation.get_data import fetch_data, process_data

# Your GCS bucket base path
bucket_base = "gs://gencast-distillation-bucket"

# GCS-compatible filesystem
fs = gcsfs.GCSFileSystem()

def _to_timedelta_hours_if_needed(ds: xr.Dataset) -> xr.Dataset:
    # Find a time-like coord if present
    time_name = next((n for n in ("time", "forecast_time", "valid_time") if n in ds.coords), None)
    if time_name is None:
        return ds  # nothing to convert

    t = ds[time_name]
    # Convert int/float hours → pandas Timedelta
    if np.issubdtype(t.dtype, np.integer) or np.issubdtype(t.dtype, np.floating):
        ds = ds.assign_coords({time_name: pd.to_timedelta(t.values, unit="h")})
        # Clean attrs/encoding that can confuse later ops
        try:
            ds[time_name].attrs = {}
        except Exception:
            pass
        enc = dict(getattr(ds[time_name], "encoding", {}) or {})
        enc.pop("dtype", None)
        ds[time_name].encoding = enc
    return ds

def open_gcs_netcdf(file_path: str) -> xr.Dataset:
    with fs.open(file_path, mode="rb") as f:
        ds = xr.load_dataset(f, decode_timedelta=False).compute()
    return _to_timedelta_hours_if_needed(ds)


# Load normalization datasets
normalization_data = {
    "diffs_stddev_by_level": open_gcs_netcdf(f"{bucket_base}/normalization_data/gencast_stats_diffs_stddev_by_level.nc"),
    "mean_by_level": open_gcs_netcdf(f"{bucket_base}/normalization_data/gencast_stats_mean_by_level.nc"),
    "stddev_by_level": open_gcs_netcdf(f"{bucket_base}/normalization_data/gencast_stats_stddev_by_level.nc"),
    "min_by_level": open_gcs_netcdf(f"{bucket_base}/normalization_data/gencast_stats_min_by_level.nc"),
}

# Load evaluation input
example_data = open_gcs_netcdf(f"{bucket_base}/data/era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc")

if not isinstance(example_data["time"], xr.DataArray):
    example_data["time"] = xr.DataArray(example_data["time"])

training_data = fetch_data()
training_data = process_data(training_data)
# training_data = to_timedelta_hours_if_needed(training_data)

# Load model weights
weights_path = f"{bucket_base}/gencast_weights/gencast_params_GenCast 1p0deg Mini _2019.npz"
with fs.open(weights_path, 'rb') as f:
    model_weights = io.BytesIO(f.read())
