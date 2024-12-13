import jax
import jax.numpy as jnp
from functools import partial


class Criterion:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _symlog(score: jax.Array) -> jax.Array:
        return jnp.sign(score) * jnp.log(jnp.abs(score) + 1.0)

    def __call__(self) -> jax.Array:
        raise NotImplementedError


class MSELoss(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, target: jax.Array, logits: jax.Array) -> jax.Array:
        score = jnp.mean(jnp.square(target - logits))
        return self._symlog(score=score)


class CrossEntropyLoss(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, target: jax.Array, logits: jax.Array) -> jax.Array:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        score = -jnp.mean(jnp.sum(target * log_probs, axis=-1))
        return self._symlog(score=score)


class MaxScore(Criterion):

    def __init__(self) -> None:
        super().__init__()

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, score: jax.Array) -> jax.Array:
        score = -1.0 * score
        return self._symlog(score)
