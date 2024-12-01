"""Script holds classes for different loss types."""

import jax
import jax.numpy as jnp
from jax.nn import log_softmax
from jax.scipy.special import logsumexp


class Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, target: jax.Array, pred: jax.Array) -> jax.Array:
        raise NotImplementedError


class MSELoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, target: jax.Array, pred: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(target - pred))


class CrossEntropyLoss(Loss):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, target: jax.Array, logits: jax.Array) -> jax.Array:
        log_probs = log_softmax(logits)
        return -jnp.mean(jnp.sum(target * log_probs, axis=-1))
