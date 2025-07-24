import jax
import jax.tree_util

def copy_pytree(pytree):
    """Performs a deep copy of a JAX pytree."""
    return jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, pytree)


def dummy_dataset_iterator(inputs, targets_template, forcings, batch_size=1):
    while True:
        yield {
            "inputs": inputs,
            "targets_template": targets_template,
            "forcings": forcings,
        }
