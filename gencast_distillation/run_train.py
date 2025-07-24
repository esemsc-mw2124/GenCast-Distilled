import jax
import jax.numpy as jnp
from gencast_distillation import model, config, training
from graphcast import data_utils

def setup_gpu():
    """Setup and verify GPU configuration"""
    # Force GPU backend
    jax.config.update('jax_platform_name', 'gpu')
    
    # Verify GPU is available
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.device_kind == 'gpu']
        if not gpu_devices:
            print("⚠ Warning: No GPU devices found, falling back to CPU")
        else:
            print(f"✓ Using GPU backend with {len(gpu_devices)} GPU(s)")
            print(f"  GPU devices: {[str(d) for d in gpu_devices]}")
    except Exception as e:
        print(f"⚠ Warning: GPU setup issue: {e}")
    
    print(f"JAX backend: {jax.default_backend()}")

def main():
    # Setup GPU first
    # setup_gpu()
    
    # Load config, weights, norm stats
    ckpt_path = config.model_weights_path
    _config = "dummy"
    norm_data = config.normalization_data
    example_batch = config.example_data

    # Load model and data
    print("Loading distillation model...")
    distillation_model = model.GenCastDistillationModel(ckpt_path, _config, norm_data)

    print("Extracting inputs, targets, and forcings...")
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("12h", "12h"),
        **distillation_model.task_config.__dict__,
    )

    targets_template = targets * jnp.nan

    # Initialize student
    print("Initializing student model...")
    distillation_model.init_student(
        rng=42,
        inputs=inputs,
        targets_template=targets_template,
        forcings=forcings,
    )

    # print("Params:", jax.tree.map(lambda x: x.shape, distillation_model.student_params))

    # Inititialize teacher
    model.init_teacher()

if __name__ == "__main__":
    main()