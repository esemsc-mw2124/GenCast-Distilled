import jax
import jax.numpy as jnp
import haiku as hk
import optax
from gencast_distillation import model, config, training, utils
from graphcast import data_utils
import xarray as xr
import numpy as np

from dataclasses import replace


def main():
    
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

    iterator = utils.iterator(config.training_data, chunk_size=3,
                        data_utils=data_utils,
                        distillation_model=distillation_model)
    
    batch = next(iterator)
    inputs = batch["inputs"]
    targets = batch["targets"]
    forcings = batch["forcings"]

    targets_template = targets * jnp.nan

    distillation_model.teacher_sampler_config = replace(
    distillation_model.teacher_sampler_config,
    num_noise_levels=2,  
)

    distillation_model.student_sampler_config = replace(
        distillation_model.student_sampler_config,
        num_noise_levels=1, 
    )


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

    # Train
    print("Starting training...")
    trained_model = training.train_model(distillation_model, iterator, num_steps=20, log_every=1)

    # Save
    print("Saving trained model...")
    training.save_model("student_ckpt.pkl", trained_model)

if __name__ == "__main__":
    main()

