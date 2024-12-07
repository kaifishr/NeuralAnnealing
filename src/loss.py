import jax
import jax.numpy as jnp
from jax.nn import log_softmax
from functools import partial


class Criterion:
    def __init__(self) -> None:
        pass

    def __call__(self, **kwargs) -> jax.Array:
        raise NotImplementedError


class MSELoss(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, target: jax.Array, pred: jax.Array) -> jax.Array:
        return jnp.mean(jnp.square(target - pred))


class CrossEntropyLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, target: jax.Array, logits: jax.Array) -> jax.Array:
        log_probs = log_softmax(logits)
        return -jnp.mean(jnp.sum(target * log_probs, axis=-1))


class MaxScore(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, score: jax.Array) -> jax.Array:
        return -score
