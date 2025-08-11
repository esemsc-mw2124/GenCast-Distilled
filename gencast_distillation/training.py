import jax
import jax.numpy as jnp
import optax
from functools import partial
from tqdm import trange
import pickle

from gencast_distillation.losses import denoiser_l2_loss
from gencast_distillation.model import TrainState


def make_training_step(student_apply, teacher_apply_fn, optimizer):
    """
    student_apply: (params, state, rng, inputs, targets_template, forcings) -> (pred, new_state)
    teacher_apply_fn: Haiku-transformed .apply (params, state, rng, inputs, targets_template, forcings) -> (pred, new_state)
    """
    
    @partial(jax.jit, donate_argnums=(0,))
    def step(state: TrainState, batch, rng, teacher_params, teacher_state):
        def loss_fn(params, model_state):
            rng_s, rng_t = jax.random.split(rng)

            # Student forward
            student_pred, new_model_state = student_apply(
                params,
                model_state,
                rng_s,
                batch["inputs"],
                batch["targets_template"],
                batch["forcings"],
            )

            # Teacher forward (NO closures; pass weights/state explicitly)
            teacher_pred, _ = teacher_apply_fn(
                teacher_params,
                teacher_state,
                rng_t,
                batch["inputs"],
                batch["targets_template"],
                batch["forcings"],
            )

            loss = denoiser_l2_loss(student_pred, teacher_pred)
            return loss, new_model_state

        (loss, new_model_state), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params, state.model_state)

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_ema = optax.incremental_update(new_params, state.ema_params, step_size=0.001)

        new_state = TrainState(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            ema_params=new_ema,
            model_state=new_model_state,
            num_sample_steps=state.num_sample_steps,
        )
        return new_state, loss

    return step

def train_model(model, dataset_iterator, num_steps=1000, log_every=100, save_every=500, save_path="student_ckpt.pkl"):
    rng = jax.random.PRNGKey(42)
    state = model.train_state

    student_apply = model._student_transformed.apply
    teacher_apply_fn = model.teacher_transformed.apply  # pure apply
    teacher_state = getattr(model, "teacher_state", {})  # {} if stateless

    step = make_training_step(student_apply, teacher_apply_fn, model.optimizer)

    for step_idx in trange(num_steps):
        rng, step_rng = jax.random.split(rng)
        batch = next(dataset_iterator)
        state, loss = step(
            state,
            batch,
            step_rng,
            model.teacher_params,       # <-- pass params
            teacher_state,              # <-- pass state (can be {})
        )
        model.train_state = state

        if step_idx % log_every == 0:
            print(f"Step {step_idx} | Loss: {float(loss):.6f}")

        if step_idx % save_every == 0 and step_idx > 0:
            save_model(f"{save_path.replace('.pkl', '')}_step{step_idx}.pkl", model)

    return model


def save_model(path, model):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": model.train_state.params,
                "ema_params": model.train_state.ema_params,
                "state": model.train_state.model_state,
            },
            f,
        )
