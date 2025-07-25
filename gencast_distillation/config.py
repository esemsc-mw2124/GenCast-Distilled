# import os
# import xarray as xr

# base_dir = os.path.dirname(__file__)  # gets the dir of config.py
# norm_dir = os.path.join(base_dir, "normalization_data")
# data_dir = os.path.join(base_dir, "data")
# model_weights_dir = os.path.join(base_dir, "gencast_weights")

# normalization_data = {
#     "diffs_stddev_by_level": xr.open_dataset(os.path.join(norm_dir, "gencast_stats_diffs_stddev_by_level.nc")).load(),
#     "mean_by_level": xr.open_dataset(os.path.join(norm_dir, "gencast_stats_mean_by_level.nc")).load(),
#     "stddev_by_level": xr.open_dataset(os.path.join(norm_dir, "gencast_stats_stddev_by_level.nc")).load(),
#     "min_by_level": xr.open_dataset(os.path.join(norm_dir, "gencast_stats_min_by_level.nc")).load(),
# }

# example_data = xr.load_dataset(os.path.join(data_dir, "era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc"), decode_timedelta=False).load()

# model_weights_path = os.path.join(model_weights_dir, "gencast_params_GenCast 1p0deg Mini _2019.npz")


import xarray as xr
import numpy as np
import gcsfs
import io

# Your GCS bucket base path
bucket_base = "gs://gencast-distillation-bucket"

# GCS-compatible filesystem
fs = gcsfs.GCSFileSystem()

# Load normalization datasets
normalization_data = {
    "diffs_stddev_by_level": xr.open_dataset(f"{bucket_base}/normalization_data/gencast_stats_diffs_stddev_by_level.nc", engine='h5netcdf', backend_kwargs={'fs': fs}).load(),
    "mean_by_level": xr.open_dataset(f"{bucket_base}/normalization_data/gencast_stats_mean_by_level.nc", engine='h5netcdf', backend_kwargs={'fs': fs}).load(),
    "stddev_by_level": xr.open_dataset(f"{bucket_base}/normalization_data/gencast_stats_stddev_by_level.nc", engine='h5netcdf', backend_kwargs={'fs': fs}).load(),
    "min_by_level": xr.open_dataset(f"{bucket_base}/normalization_data/gencast_stats_min_by_level.nc", engine='h5netcdf', backend_kwargs={'fs': fs}).load(),
}

# Load example input
example_data = xr.open_dataset(f"{bucket_base}/data/era5_date-2019-03-29_res-1.0_levels-13_steps-01.nc", decode_timedelta=False, engine='h5netcdf', backend_kwargs={'fs': fs}).load()

# Load .npz weights (manually using gcsfs)
weights_path = f"{bucket_base}/gencast_weights/gencast_params_GenCast 1p0deg Mini _2019.npz"
with fs.open(weights_path, 'rb') as f:
    model_weights_path = np.load(io.BytesIO(f.read()), allow_pickle=True)
