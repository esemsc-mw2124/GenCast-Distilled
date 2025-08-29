import jax
import haiku as hk
from graphcast import gencast, normalization, nan_cleaning
from graphcast import xarray_tree, xarray_jax

def construct_wrapped_gencast(
    sampler_config,
    task_config,
    denoiser_architecture_config,
    noise_config,
    noise_encoder_config,
    norm_data,
):
    """Constructs and wraps the GenCast Predictor."""
    predictor = gencast.GenCast(
        sampler_config=sampler_config,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
    )

    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=norm_data["diffs_stddev_by_level"],
        mean_by_level=norm_data["mean_by_level"],
        stddev_by_level=norm_data["stddev_by_level"],
    )

    predictor = nan_cleaning.NaNCleaner(
        predictor=predictor,
        reintroduce_nans=True,
        fill_value=norm_data["min_by_level"],
        var_to_clean="sea_surface_temperature",
    )

    return predictor


def build_transforms(
    sampler_config,
    task_config,
    denoiser_architecture_config,
    noise_config,
    noise_encoder_config,
    norm_data,
):
    """Builds the transformed forward and loss functions."""

    def run_forward_fn(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast(
            sampler_config,
            task_config,
            denoiser_architecture_config,
            noise_config,
            noise_encoder_config,
            norm_data,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def loss_fn(inputs, targets, forcings):
        predictor = construct_wrapped_gencast(
            sampler_config,
            task_config,
            denoiser_architecture_config,
            noise_config,
            noise_encoder_config,
            norm_data,
        )
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    return hk.transform_with_state(run_forward_fn), hk.transform_with_state(loss_fn)


def build_run_forward_pmap(
    params,
    state,
    sampler_config,
    task_config,
    denoiser_architecture_config,
    noise_config,
    noise_encoder_config,
    norm_data,
):
    """Builds a pmapped forward function using provided params and configs."""
    # Build transforms
    run_forward, _ = build_transforms(
        sampler_config,
        task_config,
        denoiser_architecture_config,
        noise_config,
        noise_encoder_config,
        norm_data,
    )

    # Compile forward pass
    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )

    # Parallelize across devices
    return xarray_jax.pmap(run_forward_jitted, dim="sample")
