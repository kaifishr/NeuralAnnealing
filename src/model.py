import jax
import jax.numpy as jnp

from jaxlib.xla_extension import ArrayImpl


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


def predict(params: ArrayImpl, data: ArrayImpl):
    """Single-sample forward method."""
    out = data
    out = jax.lax.stop_gradient(out)

    *layers, last = params

    for w, b in layers:
        out = jnp.dot(w, out) + b
        out = jax.nn.relu(out)
        # out = jnp.heaviside(out, 0.0)
        # out = jnp.sign(out)

    w, b = last
    logits = jnp.dot(w, out) + b

    return logits


mlp = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
