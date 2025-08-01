import haiku as hk
import jax
import xarray
import jax.numpy as jnp
from graphcast import gencast, checkpoint, normalization, nan_cleaning
from gencast_distillation.patched_gencast import PatchedGenCast
from gencast_distillation import utils
from flax.struct import dataclass
import optax
from typing import Any

class GenCastDistillationModel:
    def __init__(self, ckpt_path, config, normalization_data):
        # Ensure GPU backend is configured
        # jax.config.update('jax_platform_name', 'gpu')

        # Load the teacher from checkpoint
        with open(ckpt_path, "rb") as f:
            ckpt = checkpoint.load(f, gencast.CheckPoint)

        # Configure checkpoint for GPU compatibility
        # ckpt = self._configure_checkpoint_for_gpu(ckpt)

        self.teacher_params = ckpt.params
        self.task_config = ckpt.task_config
        self.sampler_config = ckpt.sampler_config
        self.noise_config = ckpt.noise_config
        self.noise_encoder_config = ckpt.noise_encoder_config
        self.denoiser_architecture_config = ckpt.denoiser_architecture_config

        self.student_params = None  # will be initialized later
        self.config = config
        self.norm = normalization_data

        self.init_teacher()


    def _construct_wrapped_gencast(self, freeze=False):
        """Wrap GenCast with normalization, NaN cleaning, etc."""
        predictor = PatchedGenCast(
            task_config=self.task_config,
            denoiser_architecture_config=self.denoiser_architecture_config,
            sampler_config=self.sampler_config,
            noise_config=self.noise_config,
            noise_encoder_config=self.noise_encoder_config,
        )

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=self.norm["diffs_stddev_by_level"],
            mean_by_level=self.norm["mean_by_level"],
            stddev_by_level=self.norm["stddev_by_level"],
        )

        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=self.norm["min_by_level"],
            var_to_clean='sea_surface_temperature',
        )
        
        return predictor

    def init_student(self, rng, inputs, targets_template, forcings):
        """Initializes student model using real training data."""

        def student_forward_fn(i, t, f):
            predictor = self._construct_wrapped_gencast(freeze=False)
            return predictor(i, targets_template=t, forcings=f)

        # Transform the model
        self._student_transformed = hk.transform_with_state(student_forward_fn)

        # Student starts with identical weights but fresh internal state
        init_params = utils.copy_pytree(self.teacher_params)
        self.student_params = init_params

        # Initialize model state (e.g., batchnorm stats)
        self.student_state = self._student_transformed.init(
            jax.random.PRNGKey(rng),
            inputs,
            targets_template,
            forcings,
        )[1]  # [1] to get state, not params since we're manually setting params

        # Initialize Optax optimizer
        optimizer = optax.adam(learning_rate=1e-4)
        opt_state = optimizer.init(init_params)

        # Update train state (new structure)
        self.train_state = TrainState(
            step=0,
            params=init_params,
            opt_state=opt_state,
            ema_params=utils.copy_pytree(init_params),
            num_sample_steps=self.student_sampling_steps,
            model_state=self.student_state,
        )

        # Save the optimizer object itself if needed for update calls later
        self.optimizer = optimizer

    
    def init_teacher(self):
        """Wraps and stores the teacher model using checkpoint params."""
        self.teacher_model = self._construct_wrapped_gencast(freeze=True)
        self.teacher_sampling_steps = self.sampler_config.num_noise_levels

    def _apply_teacher(self, inputs, targets_template, forcings):
        return self.teacher_model(inputs, targets_template=targets_template, forcings=forcings)



    def _configure_checkpoint_for_gpu(self, ckpt):
        """
        Configure checkpoint for GPU inference by setting appropriate attention type.
        This follows the official GraphCast documentation for GPU compatibility.
        """
        
        def configure_sparse_transformer_config(config_obj, config_name=""):
            """Configure sparse transformer settings for GPU"""
            if hasattr(config_obj, 'sparse_transformer_config'):
                sparse_config = config_obj.sparse_transformer_config
                
                # Set GPU-compatible attention settings
                sparse_config.attention_type = "triblockdiag_mha"
                sparse_config.mask_type = "full"
                
                print(f"✓ Configured {config_name} sparse_transformer_config for GPU")
                return True
            return False
        
        # Configure denoiser architecture config
        if hasattr(ckpt, 'denoiser_architecture_config'):
            configure_sparse_transformer_config(
                ckpt.denoiser_architecture_config, 
                "denoiser_architecture_config"
            )
        
        # Configure any other architecture configs that might exist
        for attr_name in dir(ckpt):
            if 'architecture_config' in attr_name and not attr_name.startswith('_'):
                attr_value = getattr(ckpt, attr_name)
                if hasattr(attr_value, '__dict__'):
                    configure_sparse_transformer_config(attr_value, attr_name)
        
        return ckpt

@dataclass
class TrainState:
    step: int
    params: Any  # model parameters
    opt_state: optax.OptState  # optimizer state
    ema_params: Any
    num_sample_steps: int
    model_state: Any
