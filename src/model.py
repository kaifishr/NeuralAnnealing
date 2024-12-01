import jax
import jax.numpy as jnp

from jaxlib.xla_extension import ArrayImpl
from jax.scipy.special import logsumexp


def init_params(sizes: list, key: ArrayImpl):
    keys = jax.random.split(key, len(sizes))
    return [
        _init_params(fan_in=fan_in, fan_out=fan_out, key=key)
        for fan_in, fan_out, key in zip(sizes[:-1], sizes[1:], keys)
    ]


def _init_params(fan_in: int, fan_out: int, key: ArrayImpl):
    w_key, _ = jax.random.split(key)
    scale = jnp.sqrt(2.0 / fan_in)
    w = scale * jax.random.normal(w_key, (fan_out, fan_in))
    b = jnp.zeros(shape=(fan_out,))
    return w, b


def relu(x: ArrayImpl) -> ArrayImpl:
    """Rectified Linear Unit activation function [0, inf]"""
    return jnp.maximum(0.0, x)


def heaviside(x: ArrayImpl) -> ArrayImpl:
    """Heaviside activation function [0, 1]."""
    return jnp.heaviside(x, 0.5)


def sign(x: ArrayImpl) -> ArrayImpl:
    """Heaviside activation function [-1, 1]."""
    return jnp.sign(x)


def predict(params: ArrayImpl, data: ArrayImpl):
    """Per-example forward method."""
    out = data
    out = jax.lax.stop_gradient(out)

    *layers, last = params

    for w, b in layers:
        out = jnp.dot(w, out) + b
        # out = heaviside(out)
        # out = sign(out)
        out = relu(out)

    w, b = last
    logits = jnp.dot(w, out) + b

    return logits


# mlp = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
mlp = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
