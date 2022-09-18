"""Script for helper functions."""
from typing import Callable

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader

from .model import Model


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype) 


def comp_loss_accuracy(model: Model, loss: Callable, data_generator: DataLoader):
    """Computes loss and accuray for provided model and data."""

    num_targets = len(data_generator.dataset.classes)

    running_loss = 0.0
    running_accuracy = 0.0
    running_counter = 0.0

    for images, targets in data_generator:
        targets = one_hot(targets, num_targets)

        images = jnp.atleast_2d(images)
        targets = jnp.atleast_2d(targets)

        preds = model.forward(images)

        # Compute accuracy
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)
        batch_accuracy = float(jnp.sum(target_class == predicted_class))

        # Compute loss
        batch_loss = loss(targets, preds)

        # Accumulating stats
        running_loss += batch_loss * len(images)
        running_accuracy += batch_accuracy
        running_counter += len(images)

    total_loss = running_loss / running_counter 
    total_accuracy = running_accuracy / running_counter

    return float(total_loss), float(total_accuracy)
