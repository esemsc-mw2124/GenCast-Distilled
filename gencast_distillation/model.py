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
import gcsfs
import io
import numpy as np
import os
from dataclasses import replace

def _to_f16_xr(obj):
    # Keep xarray objects as xarray; just change dtype
    if isinstance(obj, (xarray.DataArray, xarray.Dataset)):
        return obj.astype(np.float16)
    # Fallback for numpy arrays / scalars
    return np.asarray(obj, dtype=np.float16)

class GenCastDistillationModel:
    def __init__(self, ckpt_data, config, normalization_data):
        # Ensure GPU backend is configured
        # jax.config.update('jax_platform_name', 'gpu')

        # Load the teacher from checkpoint
        ckpt = checkpoint.load(ckpt_data, gencast.CheckPoint)

        # Configure checkpoint for GPU compatibility
        # ckpt = self._configure_checkpoint_for_gpu(ckpt)

        self.teacher_params = ckpt.params
        self.task_config = ckpt.task_config
        self.sampler_config = ckpt.sampler_config

        teacher_steps = int(os.getenv("GENCAST_TEACHER_STEPS", "6"))
        student_steps = int(os.getenv("GENCAST_STUDENT_STEPS", str(max(1, teacher_steps // 2))))

        self.teacher_sampler_config = replace(
            self.sampler_config,
            num_noise_levels=min(self.sampler_config.num_noise_levels, teacher_steps),
        )
        self.student_sampler_config = replace(
            self.teacher_sampler_config,
            num_noise_levels=min(self.teacher_sampler_config.num_noise_levels, student_steps),
)


        self.noise_config = ckpt.noise_config
        self.noise_encoder_config = ckpt.noise_encoder_config
        self.denoiser_architecture_config = ckpt.denoiser_architecture_config

        self.student_params = None  # will be initialized later
        self.config = config
        self.norm = normalization_data

        self.teacher_sampling_steps = self.sampler_config.num_noise_levels
        self.student_sampling_steps = getattr(
            self.config,
            "student_sampling_steps",
            max(1, self.teacher_sampling_steps // 2)  # default: half of teacher
        )


        # self.teacher_sampler_config = self.sampler_config
        # self.student_sampler_config = replace(
        #     self.teacher_sampler_config,
        #     num_noise_levels=self.student_sampling_steps
        # )

        self.init_teacher()

    def _construct_wrapped_gencast(self, freeze=False, use_student=True):
        predictor = gencast.GenCast(
            task_config=self.task_config,
            denoiser_architecture_config=self.denoiser_architecture_config,
            sampler_config=self.student_sampler_config if use_student else self.teacher_sampler_config,
            noise_config=self.noise_config,
            noise_encoder_config=self.noise_encoder_config,
        )

        # ensure float16 but keep xarray types intact
        # norm_f16 = {
        #     "diffs_stddev_by_level": _to_f16_xr(self.norm["diffs_stddev_by_level"]),
        #     "mean_by_level":         _to_f16_xr(self.norm["mean_by_level"]),
        #     "stddev_by_level":       _to_f16_xr(self.norm["stddev_by_level"]),
        #     "min_by_level":          _to_f16_xr(self.norm["min_by_level"]),
        # }

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=self.norm["diffs_stddev_by_level"],
            mean_by_level=self.norm["mean_by_level"],
            stddev_by_level=self.norm["stddev_by_level"],
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=False,
            fill_value=self.norm["min_by_level"],   # stays xarray, now float16
            var_to_clean="sea_surface_temperature",
        )
        return predictor

    def init_student(self, rng, inputs, targets_template, forcings):
        def student_forward_fn(i, t, f):
            predictor = self._construct_wrapped_gencast(freeze=False, use_student=True)
            return predictor(i, targets_template=t, forcings=f)
        self._student_transformed = hk.transform_with_state(student_forward_fn)

        init_params = utils.copy_pytree(self.teacher_params)
        self.student_params = init_params

        _, self.student_state = self._student_transformed.init(
            jax.random.PRNGKey(rng), inputs, targets_template, forcings
        )

        # optimizer = optax.chain(
        #     optax.clip_by_global_norm(1_000.0),
        #     utils.sanitize_nan_inf(),
        #     optax.adam(learning_rate=1e-4),
        # )
        optimizer = optax.adam(learning_rate=1e-4)
        opt_state = optimizer.init(init_params)

        self.train_state = TrainState(
            step=0,
            params=init_params,
            opt_state=opt_state,
            ema_params=init_params,
            num_sample_steps=self.student_sampler_config.num_noise_levels,
            model_state=self.student_state,
        )
        self.optimizer = optimizer

    
    def init_teacher(self):
        """Wraps and stores the teacher model using checkpoint params."""
        def teacher_forward_fn(inputs, targets_template, forcings):
            predictor = self._construct_wrapped_gencast(
                freeze=True, use_student=False
            )
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        self.teacher_transformed = hk.transform_with_state(teacher_forward_fn)
        self.teacher_sampling_steps = self.teacher_sampler_config.num_noise_levels


    def _apply_teacher(self, inputs, targets_template, forcings, rng):
        preds, _state = self.teacher_transformed.apply(
            self.teacher_params,
            self.teacher_state if hasattr(self, "teacher_state") else {},  # <<< use if present
            rng,
            inputs,
            targets_template,
            forcings,
        )
        return preds


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
