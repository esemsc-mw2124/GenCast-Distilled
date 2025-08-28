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
    Full GenCast-style evaluation pipeline:
    - Per-grid CRPS
    - Global CRPS (lat-weighted + unweighted)
    - Spread-skill calibration
    - Global CRPS vs lead time
    - CRPS histograms
    - Multi-scale spatially pooled CRPS vs lead time (GenCast Fig. 6 style)
    """
    print("Selecting variable:", variable)
    preds = select(predictions, variable, level)
    truth = select(eval_targets, variable, level)

    # ---- Per-grid CRPS ----
    print("Computing CRPS per grid cell...")
    crps_field = compute_crps(preds, truth)

    # ---- Global CRPS Scores ----
    print("Computing global CRPS (unweighted and lat-weighted)...")
    global_crps_unweighted = crps_field.mean(dim=("lat", "lon"))
    global_crps_weighted = latitude_weighted_mean(crps_field, lat_name="lat").mean(dim="lon")

    print(f"Global CRPS (unweighted): {float(global_crps_unweighted.mean().values):.5f}")
    print(f"Global CRPS (lat-weighted): {float(global_crps_weighted.mean().values):.5f}")

    # ---- Spread / Skill Diagnostic ----
    print("Computing spread/skill ratio for calibration check...")
    ensemble_mean = preds.mean(dim="sample")
    rmse = np.sqrt(((ensemble_mean - truth) ** 2).mean(dim=("lat", "lon")))
    spread = preds.std(dim="sample").mean(dim=("lat", "lon"))
    spread_skill_ratio = spread / rmse

    # ---- Visualization: CRPS Map ----
    print("Visualizing CRPS evolution over time...")
    crps_scaled = scale(crps_field, robust=True, center=None)
    plot_data({"CRPS per grid": crps_scaled}, f"CRPS Evolution - {variable}")

    # ---- Visualization: Global CRPS vs Lead Time ----
    print("Plotting global CRPS vs lead time...")
    plt.figure(figsize=(8, 5))
    plt.plot(
        global_crps_weighted.time / np.timedelta64(1, 'h') / 24,
        global_crps_weighted,
        label="Lat-weighted"
    )
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

    # ---- Visualization: Spread vs RMSE ----
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

    # ---- Multi-Scale Pooled CRPS ----
    print("Computing spatially pooled CRPS at multiple radii...")
    radii_km = [120, 241, 481, 962, 1922, 3828]
    pooled_results = compute_pooled_crps(preds, truth, radii_km=radii_km)

    # ---- Visualization: Pooled CRPS vs Lead Time ----
    print("Plotting pooled CRPS vs lead time...")
    plt.figure(figsize=(10, 6))
    for r in radii_km:
        plt.plot(
            pooled_results[r].time / np.timedelta64(1, 'h') / 24,
            pooled_results[r],
            label=f"{r} km pooling"
        )
    plt.xlabel("Lead time (days)")
    plt.ylabel("Pooled CRPS")
    plt.title(f"Multi-Scale Pooled CRPS vs Lead Time ({variable})")
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

    print("GenCast-style evaluation complete.")
    return
