import jax
import jax.numpy as jnp
import haiku as hk
import optax
from gencast_distillation import model, config, training, utils
from graphcast import data_utils
import xarray as xr
import numpy as np

from dataclasses import replace

def to_f32(ds):
    """
    Converts all floating-point arrays in the dataset to float32 dtype.
    """
    return ds.map(lambda v: v.astype(np.float32) if hasattr(v, "dtype") and np.issubdtype(v.dtype, np.floating) else v)


def main():
    # TODO: remove
    # jax.config.update("jax_default_matmul_precision", "bfloat16")
    # import warnings
    # warnings.filterwarnings("error", message="overflow encountered in cast")
    # np.seterr(over='raise')
    
    # Load config, weights, norm stats
    ckpt_path = config.weights_path
    _config = "dummy"  # TODO: remove
    norm_data = {
        k: (v.astype(np.float32) if hasattr(v, "astype") else np.asarray(v, np.float32))
        for k, v in config.normalization_data.items()
    }
    # example_batch = config.example_data
  
    # Load model and data
    print("Loading distillation model...")
    distillation_model = model.GenCastDistillationModel(config.model_weights, _config, norm_data)

    iterator = utils.iterator(config.training_data, chunk_size=3,  # TODO: change ot training data
                        data_utils=data_utils,
                        distillation_model=distillation_model)
    
    batch = next(iterator)
    inputs = batch["inputs"]
    targets = batch["targets"]
    forcings = batch["forcings"]

    # print("Extracting inputs, targets, and forcings...")
    # inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
    #     example_batch,
    #     target_lead_times=slice("12h", "12h"),
    #     **distillation_model.task_config.__dict__,
    # )

    targets_template = targets * jnp.nan

    # inputs = to_f32(inputs)
    # targets_template = to_f32(targets_template)
    # forcings = to_f32(forcings)

    distillation_model.teacher_sampler_config = replace(
    distillation_model.teacher_sampler_config,
    num_noise_levels=2,  # 1 step (levels-1)
)

    distillation_model.student_sampler_config = replace(
        distillation_model.student_sampler_config,
        num_noise_levels=1,  # 0 steps (levels-1) – minimal test
    )

    print("FINAL teacher levels:", distillation_model.teacher_sampler_config.num_noise_levels)
    print("FINAL student levels:", distillation_model.student_sampler_config.num_noise_levels)


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

    # # Create a dummy dataset iterator
    # print("Creating dummy dataset iterator...")
    # iterator = utils.dummy_dataset_iterator(inputs, targets, forcings)

    # Train
    print("Starting training...")
    trained_model = training.train_model(distillation_model, iterator, num_steps=10, log_every=1)

    # Save
    print("Saving trained model...")
    training.save_model("student_ckpt.pkl", trained_model)

if __name__ == "__main__":
    main()

