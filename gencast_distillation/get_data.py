import xarray as xr
# import xesmf as xe
import numpy as np

TEMPLATE_VAR_ORDER = [
    "geopotential_at_surface", "land_sea_mask",
    "2m_temperature", "mean_sea_level_pressure",
    "10m_v_component_of_wind", "10m_u_component_of_wind",
    "total_precipitation_6hr", "toa_incident_solar_radiation",
    "temperature", "geopotential",
    "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity",
    "sea_surface_temperature", "total_precipitation_12hr"
]

EXPECTED_COORDS = {"lon", "lat", "level", "datetime", "time"}
CANONICAL_DIM_ORDER = ("batch", "time", "level", "lat", "lon")

def fetch_data(url = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"):
    ds = xr.open_zarr(
        url,
        consolidated=True,                     # try True first; fall back to False if needed
        storage_options={"token": "anon"}      # <— force anonymous access
    )
    return ds

# Bilinear downsampling, this doesn't work on TPU VM
# def regrid_data(ds, weights_file="era5_to_1deg_bilinear.nc", reuse_weights=True):
#     # --- Accept either latitude/longitude or lat/lon ---
#     rename = {}
#     if "lat" not in ds.coords and "latitude" in ds.coords:
#         rename["latitude"] = "lat"
#     if "lon" not in ds.coords and "longitude" in ds.coords:
#         rename["longitude"] = "lon"
#     if rename:
#         ds = ds.rename(rename)

#     # Sanity check
#     if "lat" not in ds.coords or "lon" not in ds.coords:
#         raise KeyError("Dataset must have 'lat' and 'lon' coordinates for regridding.")

#     # Normalize longitude to [0, 360) and sort
#     if (ds["lon"].min() < 0).item():
#         ds = ds.assign_coords(lon=(ds["lon"] % 360)).sortby("lon")

#     # Ensure latitude ascending (south->north) for consistency
#     if (ds["lat"][0] > ds["lat"][-1]).item():
#         ds = ds.sortby("lat")

#     # 1) Define source and 1.0° target grids
#     src = xr.Dataset({"lat": ds["lat"], "lon": ds["lon"]})
#     tgt = xr.Dataset(
#         {
#             "lat": (["lat"], np.arange(-90, 90 + 1, 1.0)),   # -90..90 inclusive
#             "lon": (["lon"], np.arange(0, 360, 1.0)),        # 0..359
#         }
#     )
#     # 2) Build the regridder using grid-only datasets (robust even if 'time' is absent)
#     regridder = xe.Regridder(
#         src, tgt,
#         method="bilinear",
#         periodic=True,
#         filename=weights_file,
#         reuse_weights=reuse_weights,
#     )

#     # 3) Apply to all variables with (lat, lon); pass through others unchanged
#     regridded = xr.Dataset(coords={})
#     # copy over common coords if present
#     for coord in ["batch", "time", "datetime", "level"]:
#         if coord in ds.coords or coord in ds.dims:
#             regridded = regridded.assign_coords({coord: ds[coord]})

#     for name, da in ds.data_vars.items():
#         if {"lat", "lon"}.issubset(da.dims):
#             regridded[name] = regridder(da, keep_attrs=True)
#         else:
#             regridded[name] = da

#     # Attach target horizontal coords explicitly
#     regridded = regridded.assign_coords(lat=tgt["lat"], lon=tgt["lon"])
#     return regridded

def regrid_data(ds, dlat=1.0, dlon=1.0):
    # --- Normalize coord names first ---
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # Sanity check
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise KeyError("Dataset must have 'lat' and 'lon' coords for regridding")

    # Target grid
    new_lat = np.arange(-90, 90 + dlat, dlat)
    new_lon = np.arange(0, 360, dlon)

    # Interpolate to regular grid
    return ds.interp(lat=new_lat, lon=new_lon)

def _transpose_like(da: xr.DataArray) -> xr.DataArray:
    order = [d for d in CANONICAL_DIM_ORDER if d in da.dims]
    rest  = [d for d in da.dims if d not in order]
    return da.transpose(*(order + rest))

def align_like_template(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    ds = ds.copy()

    # 0) Normalize coord names that sometimes differ
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:   rename["latitude"]  = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:  rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # 1) Ensure batch dim exists
    if "batch" not in ds.dims:
        ds = ds.expand_dims(batch=1)

    # 2) Drop 'batch' as a coordinate (keep it only as a dimension)
    if "batch" in ds.coords:
        ds = ds.drop_vars("batch")  # <-- fix for your error

    # 3) Make datetime (batch,time) and change 'time' coord to timedelta64[ns]
    if "time" not in ds.dims:
        raise ValueError("Input must have a 'time' dimension.")

    # Ensure original 'time' values are datetime64[ns]
    t1d = xr.DataArray(ds["time"].values, dims=("time",)).astype("datetime64[ns]")

    # 2-D datetime coord (batch, time)
    B, T = ds.sizes["batch"], ds.sizes["time"]
    dt2d = np.broadcast_to(t1d.values[np.newaxis, :], (B, T))
    ds = ds.assign_coords(datetime=(("batch", "time"), dt2d.astype("datetime64[ns]")))

    # Replace time coord values with timedelta since first timestamp
    origin = t1d.isel(time=0).values
    td = (t1d - origin).astype("timedelta64[ns]")
    ds = ds.assign_coords(time=("time", td.values))

    # 4) Coord dtypes to match template
    if "lat" in ds.coords:   ds = ds.assign_coords(lat=ds["lat"].astype("float32"))
    if "lon" in ds.coords:   ds = ds.assign_coords(lon=ds["lon"].astype("float32"))
    if "level" in ds.coords: ds = ds.assign_coords(level=ds["level"].astype("int32"))

    # 5) Variable shapes + dtypes
    new_vars = {}
    for name, da in ds.data_vars.items():
        # Surface fields -> (lat, lon)  (squeeze batch if present and size==1)
        if {"lat","lon"}.issubset(da.dims) and "time" not in da.dims and "level" not in da.dims:
            if "batch" in da.dims and ds.sizes["batch"] == 1:
                da = da.isel(batch=0, drop=True)  # drops the coord if present
            da = da.transpose("lat", "lon")

        # 2D time-varying -> (batch, time, lat, lon)
        if "time" in da.dims and "level" not in da.dims:
            da = da.transpose("batch", "time", "lat", "lon")

        # 3D level-dependent -> (batch, time, level, lat, lon)
        if "time" in da.dims and "level" in da.dims:
            da = da.transpose("batch", "time", "level", "lat", "lon")

        # Cast to float32 (no computation with dask)
        if da.dtype != np.float32:
            da = da.astype("float32")

        # Safety transpose
        da = _transpose_like(da)
        new_vars[name] = da

    ds = xr.Dataset(new_vars, coords=ds.coords, attrs=ds.attrs)

    # 6) Variable ordering: template first, then any extras (preserve their order)
    ordered = {k: ds[k] for k in TEMPLATE_VAR_ORDER if k in ds.data_vars}
    for k in ds.data_vars:
        if k not in ordered:
            ordered[k] = ds[k]
    ds = xr.Dataset(ordered, coords=ds.coords, attrs=ds.attrs)

    # 7) Report anything missing / extra
    missing_coords = sorted(list(EXPECTED_COORDS - set(ds.coords)))
    missing_vars   = [v for v in TEMPLATE_VAR_ORDER if v not in ds.data_vars]
    report = {
        "dims": dict(ds.sizes),
        "coord_dtypes": {k: str(ds[k].dtype) for k in ["lat","lon","level","time","datetime"] if k in ds.coords},
        "missing_coords": missing_coords,
        "missing_vars": missing_vars,
        "extra_vars": [v for v in ds.data_vars if v not in TEMPLATE_VAR_ORDER],
        "surface_vars": [v for v, da in ds.data_vars.items() if set(da.dims) == {"lat","lon"}],
        "time2d_vars":  [v for v, da in ds.data_vars.items() if set(da.dims) == {"batch","time","lat","lon"}],
        "level3d_vars": [v for v, da in ds.data_vars.items() if set(da.dims) == {"batch","time","level","lat","lon"}],
    }
    return ds, report

def process_data(ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    # 1) Keep only variables we care about; record which are missing
    wanted  = [v for v in TEMPLATE_VAR_ORDER if v in ds.data_vars]
    missing = [v for v in TEMPLATE_VAR_ORDER if v not in ds.data_vars]
    ds = ds[wanted]

    # 2) Regrid to 1.0° grid
    ds = regrid_data(ds)

    # 3) Downsample to 12-hourly (assuming 6-hourly input)
    if "time" in ds.dims:
        ds = ds.isel(time=slice(0, None, 2))

    # 4) Conform names, dims, dtypes, ordering, and report
    ds, info = align_like_template(ds)
    info["missing_vars_in_source"] = missing
    return ds#, info