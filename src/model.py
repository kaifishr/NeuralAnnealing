import copy

import jax
import jax.numpy as jnp

from jaxlib.xla_extension import ArrayImpl


class Model:
    """Multilayer perceptron."""

    def __init__(self, config: dict):

        layer_sizes = config["layer_sizes"]
        self.params = self.init_params(layer_sizes, jax.random.PRNGKey(0))
        self.params_tmp = copy.deepcopy(self.params)

        # Make a batched version of the "predict" function using "vmap".
        self._forward = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
        # self._forward = jax.vmap(predict, in_axes=(None, 0))

    def init_params(self, sizes: list, key: ArrayImpl):
        """Initialize all layers for a fully-connected neural network with sizes 'sizes'"""
        keys = jax.random.split(key, len(sizes))
        return [
            self._init_params(fan_in=fan_in, fan_out=fan_out, key=key)
            for fan_in, fan_out, key in zip(sizes[:-1], sizes[1:], keys)
        ]

    @staticmethod
    def _init_params(fan_in: int, fan_out: int, key: ArrayImpl):
        w_key, _ = jax.random.split(key)
        scale = jnp.sqrt(2.0 / fan_in)
        w = scale * jax.random.normal(w_key, (fan_out, fan_in))
        b = jnp.zeros(shape=(fan_out,))
        return w, b

    # def step(self, x: ArrayImpl, y: ArrayImpl, temperature: float):
    #     pass

    def forward(self, x: ArrayImpl):
        return self._forward(self.params, x)

    def forward_tmp(self, x: ArrayImpl):
        return self._forward(self.params_tmp, x)


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

    *layers, last = params

    for w, b in layers:
        out = jnp.dot(w, out) + b
        # out = heaviside(out)
        # out = sign(out)
        out = relu(out)

    w, b = last
    logits = jnp.dot(w, out) + b

    return logits
