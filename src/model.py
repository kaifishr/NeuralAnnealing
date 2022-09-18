"""Fully connected neural networks.
"""
from typing import Tuple

import jax.numpy as jnp
import jax

from jaxlib.xla_extension import DeviceArray


class Model:
    """Multilayer perceptron."""

    def __init__(self, config: dict):

        layer_sizes = config["layer_sizes"]
        params_type = config["params_type"]

        # Set parameter initialization according to parameter type.
        if params_type == "float":
            self._random_layer_params = self._random_float_params
        elif params_type == "binary":
            self._random_layer_params = self._random_binary_params
        elif params_type == "trinary":
            self._random_layer_params = self._random_trinary_params

        self.params = self.init_params(layer_sizes, jax.random.PRNGKey(0))

        # Make a batched version of the "predict" function using "vmap".
        self._forward = jax.jit(jax.vmap(predict, in_axes=(None, 0)))
        # self._forward = jax.vmap(predict, in_axes=(None, 0))

    def init_params(self, sizes: list, key: DeviceArray):
        """Initialize all layers for a fully-connected neural network with sizes 'sizes'"""
        keys = jax.random.split(key, len(sizes))
        return [self._random_layer_params(fan_in, fan_out, key) for fan_in, fan_out, key in zip(sizes[:-1], sizes[1:], keys)]

    @staticmethod
    def _random_float_params(fan_in: int, fan_out: int, key: DeviceArray):
        """A helper function to randomly initialize weights and biases for a dense neural network layer"""
        w_key, _ = jax.random.split(key)
        scale = jnp.sqrt(2.0 / fan_in)
        w = scale * jax.random.normal(w_key, (fan_out, fan_in)) 
        b = jnp.zeros(shape=(fan_out, ))
        return w, b

    @staticmethod
    def _random_binary_params(fan_in: int, fan_out: int, key: DeviceArray) -> Tuple[DeviceArray]:
        """Randomly initializes binary [0, 1] weights and biases."""
        w_key, b_key = jax.random.split(key)
        w = jax.random.randint(w_key, (fan_out, fan_in), minval=0, maxval=2).astype(jnp.float32)
        b = jax.random.randint(b_key, (fan_out, ), minval=0, maxval=2).astype(jnp.float32)
        return w, b

    @staticmethod
    def _random_trinary_params(fan_in: int, fan_out: int, key: DeviceArray) -> Tuple[DeviceArray]:
        """Randomly initializes trinary [-1, 0, 1] weights and biases."""
        w_key, b_key = jax.random.split(key)
        w = jax.random.randint(w_key, (fan_out, fan_in), minval=-1, maxval=2).astype(jnp.float32)
        b = jax.random.randint(b_key, (fan_out, ), minval=-1, maxval=2).astype(jnp.float32)
        return w, b

    def step(self, x: DeviceArray, y: DeviceArray, temperature: float):
        """Performs single optimization step."""

    def forward(self, x: DeviceArray):
        return self._forward(self.params, x)


def relu(x: DeviceArray) -> DeviceArray:
    """Rectified Linear Unit activation function."""
    return jnp.maximum(0.0, x)


def heaviside(x: DeviceArray) -> DeviceArray:
    """Heaviside activation function."""
    return jnp.heaviside(x, 0.5)


def sign(x: DeviceArray) -> DeviceArray:
    """Heaviside activation function returning -1 or 1."""
    return jnp.sign(x) 


def predict(params: DeviceArray, image: DeviceArray):
    """Per-example forward method."""
    activations = image

    for w, b in params[:-1]:
        out = jnp.dot(w, activations) + b
        # activations = heaviside(out)
        # activations = sign(out)
        activations = relu(out)

    w, b = params[-1]
    out = jnp.dot(w, activations) + b

    # Normalize model output.
    # This is important for discrete networks to work.

    out = (out - jnp.min(out)) / (jnp.max(out) - jnp.min(out) + 1e-5)

    # out = out / out.sum()
    out = jax.nn.softmax(out)

    return out 
