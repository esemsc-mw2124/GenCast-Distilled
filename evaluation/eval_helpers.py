import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from plotting_helpers import select, scale, plot_data

def compute_pooled_crps(preds, truth, radii_km=[120, 241, 481, 962, 1922, 3828]):
    """
    Compute CRPS at multiple spatial pooling scales.
    preds:  [sample, time, lat, lon]
    truth:  [time, lat, lon]
    Returns dict[radius] -> pooled CRPS(time).
    """
    results = {}
    for r in radii_km:
        print(f"   → Pooling at radius {r} km ...")
        pooled_preds = spatial_pooling(preds.mean(dim="sample"), radius_km=r)
        pooled_truth = spatial_pooling(truth, radius_km=r)

        # Expand pooled_preds back to ensemble-like shape with 1 sample
        pooled_crps = compute_crps(pooled_preds.expand_dims("sample"), pooled_truth)
        results[r] = pooled_crps.mean(dim=("lat", "lon"))  # Time series
    return results

def spatial_pooling(data: xr.DataArray, radius_km=120):
    """
    Apply spatial pooling (local averaging) over lat/lon only.
    Keeps time and level dimensions intact.
    """
    # Approximate degrees per km
    deg_per_km = 1.0 / 111.0
    window = max(1, int(radius_km * deg_per_km))

    # Find spatial dimensions
    spatial_dims = [d for d in data.dims if d in ["lat", "lon"]]
    assert len(spatial_dims) == 2, "Expected data with lat/lon dimensions."

    # Create size tuple matching input rank
    size = [1] * data.ndim  # Default: no smoothing
    for i, dim in enumerate(data.dims):
        if dim in spatial_dims:
            size[i] = window

    # Apply smoothing only over lat/lon dims
    pooled_array = uniform_filter(data.values, size=tuple(size), mode="nearest")

    # Return as DataArray with same coords
    return xr.DataArray(pooled_array, dims=data.dims, coords=data.coords)


def compute_crps(predictions: xr.DataArray, targets: xr.DataArray, bias_corrected=True):
    """
    Compute CRPS for ensemble forecasts following GenCast / ECMWF style.
    predictions: [sample, time, lat, lon, (level)]
    targets:     [time, lat, lon, (level)]
    Returns: per-grid CRPS with same dims as targets.
    """
    preds2 = predictions.rename({"sample": "sample2"})
    n = predictions.sizes["sample"]
    n2 = (n - 1) if bias_corrected else n
    
    # Mean absolute error (ensemble vs truth)
    mae = np.abs(predictions - targets).mean(dim="sample")
    
    # Ensemble spread term
    spread = np.abs(predictions - preds2).mean(dim=("sample", "sample2"))
    
    # Final CRPS formula
    crps = mae - 0.5 * spread
    return crps


def latitude_weighted_mean(data: xr.DataArray, lat_name="lat"):
    """Compute latitude-weighted mean for global skill scores."""
    weights = np.cos(np.deg2rad(data[lat_name]))
    return data.weighted(weights).mean(dim=lat_name)


def evaluate(predictions: xr.Dataset, eval_targets: xr.Dataset, variable="2m_temperature", level=None):
    """
    GenCast-style evaluation pipeline with numerical comparison output:
    - Per-grid CRPS
    - Global CRPS (unweighted only)
    - Spread-skill ratio
    - Global CRPS vs lead time (only unweighted)
    - Pooled CRPS (select one resolution)
    """
    print("Selecting variable:", variable)
    preds = select(predictions, variable, level)
    truth = select(eval_targets, variable, level)

    # ---- Per-grid CRPS ----
    print("Computing CRPS per grid cell...")
    crps_field = compute_crps(preds, truth)

    # ---- Global CRPS (Unweighted Only) ----
    print("Computing global CRPS (unweighted)...")
    global_crps_unweighted = crps_field.mean(dim=("lat", "lon"))
    crps_score = float(global_crps_unweighted.mean().values)
    print(f"Global CRPS (unweighted): {crps_score:.5f}")

    # ---- Spread / Skill Diagnostic ----
    print("Computing spread/skill ratio for calibration check...")
    ensemble_mean = preds.mean(dim="sample")
    rmse = np.sqrt(((ensemble_mean - truth) ** 2).mean(dim=("lat", "lon")))
    spread = preds.std(dim="sample").mean(dim=("lat", "lon"))
    spread_skill_ratio = float((spread / rmse).mean().values)
    rmse_score = float(rmse.mean().values)
    spread_score = float(spread.mean().values)

    # ---- Visualization: CRPS Evolution Map ----
    print("Visualizing CRPS evolution over time...")
    crps_scaled = scale(crps_field, robust=True, center=None)
    plot_data({"CRPS per grid": crps_scaled}, f"CRPS Evolution - {variable}")

    # ---- Visualization: Global CRPS vs Lead Time ----
    print("Plotting global CRPS vs lead time...")
    plt.figure(figsize=(8, 5))
    plt.plot(
        global_crps_unweighted.time / np.timedelta64(1, 'h') / 24,
        global_crps_unweighted,
        '--',
        label="Unweighted"
    )
    plt.xlabel("Lead time (days)")
    plt.ylabel("Global CRPS")
    plt.title(f"Global CRPS vs Lead Time ({variable})")
    plt.legend()
    plt.grid()
    plt.show()

    # ---- Spread vs RMSE Plot ----
    print("Plotting calibration check: spread vs RMSE...")
    plt.figure(figsize=(8, 5))
    plt.plot(rmse.time / np.timedelta64(1, 'h') / 24, rmse, label="RMSE (Ensemble Mean)", linewidth=2)
    plt.plot(spread.time / np.timedelta64(1, 'h') / 24, spread, label="Ensemble Spread", linewidth=2)
    plt.xlabel("Lead time (days)")
    plt.ylabel("K (Temperature Units)")
    plt.title(f"Spread vs RMSE - Calibration ({variable})")
    plt.legend()
    plt.grid()
    plt.show()

    # ---- Pooled CRPS at One Radius ----
    selected_radius = 962  # You can change this if desired
    print(f"Computing spatially pooled CRPS at {selected_radius} km radius...")
    pooled_preds = spatial_pooling(preds.mean(dim="sample"), radius_km=selected_radius)
    pooled_truth = spatial_pooling(truth, radius_km=selected_radius)
    pooled_crps = compute_crps(pooled_preds.expand_dims("sample"), pooled_truth)
    pooled_crps_timeseries = pooled_crps.mean(dim=("lat", "lon"))
    pooled_crps_score = float(pooled_crps_timeseries.mean().values)

    # ---- Pooled CRPS Plot ----
    print(f"Plotting pooled CRPS vs lead time ({selected_radius} km)...")
    plt.figure(figsize=(8, 5))
    plt.plot(
        pooled_crps_timeseries.time / np.timedelta64(1, 'h') / 24,
        pooled_crps_timeseries,
        label=f"{selected_radius} km pooling"
    )
    plt.xlabel("Lead time (days)")
    plt.ylabel("Pooled CRPS")
    plt.title(f"Pooled CRPS vs Lead Time ({variable})")
    plt.legend()
    plt.grid()
    plt.show()

    # ---- Histogram of CRPS ----
    print("Plotting CRPS distribution across grid...")
    plt.figure(figsize=(7, 5))
    crps_field_stack = crps_field.stack(grid=("lat", "lon")).mean(dim="time")
    plt.hist(crps_field_stack, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.xlabel("CRPS")
    plt.ylabel("Number of Grid Points")
    plt.title(f"CRPS Distribution Across Grid ({variable})")
    plt.grid()
    plt.show()

    print("Evaluation complete.")
    print("====================")

    return {
        "global_crps": crps_score,
        "rmse": rmse_score,
        "spread": spread_score,
        "spread_skill_ratio": spread_skill_ratio,
        "pooled_crps_962km": pooled_crps_score
    }

