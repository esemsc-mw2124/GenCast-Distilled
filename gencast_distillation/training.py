import jax
import jax.numpy as jnp
from gencast_distillation.losses import denoiser_l2_loss
from gencast_distillation.utils import dummy_dataset_iterator
import numpy as np
from tqdm import trange
import pickle

@jax.jit
def apply_teacher(model, inputs, targets_template, forcings):
    return model._apply_teacher(inputs, targets_template, forcings)

@jax.jit
def training_step(model: GenCastDistillationModel, state: TrainState, batch, rng):
    """One training step for distillation."""

    def loss_fn(params):
        # Run student forward
        student_pred, new_state = model._student_transformed.apply(
            params,
            state.model_state,
            rng,
            batch["inputs"],
            batch["targets_template"],
            batch["forcings"],
        )

        # Run teacher forward
        teacher_pred = apply_teacher(
            model,
            batch["inputs"],
            batch["targets_template"],
            batch["forcings"]
        )

        # L2 loss between student and teacher predictions
        loss = denoiser_l2_loss(student_pred, teacher_pred)
        return loss, new_state

    # Compute gradients
    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target)

    # Apply gradients
    new_optimizer = state.optimizer.apply_gradient(grads)

    # EMA update
    new_ema_params = jax.tree_map(
        lambda ema, new: 0.999 * ema + 0.001 * new,
        state.ema_params,
        new_optimizer.target,
    )

    # Update state
    new_state = TrainState(
        step=state.step + 1,
        optimizer=new_optimizer,
        ema_params=new_ema_params,
        model_state=new_model_state,
        num_sample_steps=state.num_sample_steps,
    )

    return new_state, loss


def train_model(model, dataset_iterator, num_steps=1000, log_every=100, save_every=500, save_path="student_ckpt.pkl"):
    rng = jax.random.PRNGKey(42)
    state = model.train_state

    for step in trange(num_steps):
        rng, step_rng = jax.random.split(rng)
        batch = next(dataset_iterator)
        state, loss = training_step(model, state, batch, step_rng)
        model.train_state = state

        if step % log_every == 0:
            print(f"Step {step} | Loss: {loss:.6f}")

        if step % save_every == 0 and step > 0:
            save_model(f"{save_path.replace('.pkl', '')}_step{step}.pkl", model)

    return model


def save_model(path, model):
    with open(path, "wb") as f:
        pickle.dump({
            "params": model.train_state.optimizer.target,
            "ema_params": model.train_state.ema_params,
            "state": model.train_state.model_state,
        }, f)
