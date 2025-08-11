import jax
import jax.numpy as jnp
from gencast_distillation import model, config, training, utils
from graphcast import data_utils
import xarray as xr
import numpy as np

# def setup_gpu():
#     """Setup and verify GPU configuration"""
#     # Force GPU backend
#     jax.config.update('jax_platform_name', 'gpu')
    
#     # Verify GPU is available
#     try:
#         devices = jax.devices()
#         gpu_devices = [d for d in devices if d.device_kind == 'gpu']
#         if not gpu_devices:
#             print("⚠ Warning: No GPU devices found, falling back to CPU")
#         else:
#             print(f"✓ Using GPU backend with {len(gpu_devices)} GPU(s)")
#             print(f"  GPU devices: {[str(d) for d in gpu_devices]}")
#     except Exception as e:
#         print(f"⚠ Warning: GPU setup issue: {e}")
    
#     print(f"JAX backend: {jax.default_backend()}")

def to_f32(ds):
    return ds.map(lambda v: v.astype(np.float32) if hasattr(v, "dtype") and
                  np.issubdtype(v.dtype, np.floating) else v)


def main():
    # Setup GPU first
    # setup_gpu()

    jax.config.update("jax_default_matmul_precision", "bfloat16")
    
    # Load config, weights, norm stats
    ckpt_path = config.weights_path
    _config = "dummy"
    norm_data = {
        k: (v.astype(np.float32) if hasattr(v, "astype") else np.asarray(v, np.float32))
        for k, v in config.normalization_data.items()
    }
    example_batch = config.example_data


    # print(f"example_batch.time.dtype: {example_batch.time.dtype}")
    # print(f"example_batch.time.values[:10]: {example_batch.time.values[:10]}")
    # print(f"example_batch.time.attrs: {example_batch.time.attrs}")
    # print(f"example_batch.time.encoding: {getattr(example_batch.time, 'encoding', {})}")
    # print(f"example_batch.dims: {example_batch.dims}")
    # print(f"example_batch.sizes: {example_batch.sizes}")

    # Load model and data
    print("Loading distillation model...")
    distillation_model = model.GenCastDistillationModel(config.model_weights, _config, norm_data)


    print("Extracting inputs, targets, and forcings...")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", "12h"),
        **distillation_model.task_config.__dict__,
    )

    targets_template = targets * jnp.nan

    inputs = to_f32(inputs)
    targets_template = to_f32(targets_template)
    forcings = to_f32(forcings)

    # # Initialize student
    # print("Initializing student model...")

    # temp_config = distillation_model.task_config.__dict__.copy()
    # temp_config['input_duration'] = '12h'  # or even '6h'

    # print("Extracting inputs, targets, and forcings...")
    # temp_config = distillation_model.task_config.__dict__.copy()
    # temp_config['input_duration'] = '24h'  # Back to 24h

    # inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    #     example_batch,
    #     target_lead_times=slice("24h", "24h"),  # Change target to 24h instead of 12h
    #     **temp_config,
    # )

    # # Add these debug prints:
    # print(f"AFTER EXTRACTION:")
    # print(f"Inputs time shape: {inputs.time.shape}")
    # print(f"Inputs time values: {inputs.time.values}")
    # print(f"Inputs variables: {list(inputs.data_vars.keys())}")
    # print(f"Targets time shape: {targets.time.shape}")
    # print(f"Forcings time shape: {forcings.time.shape}")

    # # Check specific variables in inputs:
    # for var_name in list(inputs.data_vars.keys())[:3]:  # Check first 3 variables
    #     var = inputs[var_name]
    #     print(f"Variable '{var_name}' shape: {var.shape}")
    #     if 'time' in var.dims:
    #         print(f"  - time dimension size: {var.sizes.get('time', 'N/A')}")

    # targets_template = targets * jnp.nan

    # Initialize teacher
    print("Initializing teacher model...")
    distillation_model.init_teacher()

    # Initialize student
    print("Initializing student model...")
    distillation_model.init_student(
        rng=0,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings
    )

    # Create a dummy dataset iterator
    print("Creating dummy dataset iterator...")
    iterator = utils.dummy_dataset_iterator(inputs, targets_template, forcings)

    # Train
    print("Starting training...")
    trained_model = training.train_model(distillation_model, iterator, num_steps=10)

    # Save
    print("Saving trained model...")
    training.save_model("student_ckpt.pkl", trained_model)

if __name__ == "__main__":
    main()