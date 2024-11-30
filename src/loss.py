"""Script holds classes for different loss types."""

import jax.numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from jax.scipy.special import logsumexp


class Loss(object):
    """Abstract base class for losses."""

    def __init__(
        self,
    ) -> None:
        """Initializes base class."""

    def __call__(self, target: ArrayImpl, pred: ArrayImpl) -> ArrayImpl:
        """Computes mean squared error loss."""
        raise NotImplementedError


class MSELoss(Loss):

    def __init__(
        self,
    ) -> None:
        """Initializes mean squared error loss."""
        super().__init__()

    def __call__(self, target: ArrayImpl, pred: ArrayImpl) -> ArrayImpl:
        """Computes mean squared error loss.

        Args:
            target: Ground truth labels.
            pred: Model predictions.
        """
        return jnp.mean(jnp.square(target - pred))


class CrossEntropyLoss(Loss):

    def __init__(
        self,
    ) -> None:
        """Initializes cross entropy loss."""
        super().__init__()

    def __call__(self, target: ArrayImpl, pred: ArrayImpl) -> ArrayImpl:
        """Computes cross entropy loss.

        Args:
            preds: Model predictions.
            targets: Ground truth labels.
        """
        return -jnp.mean((pred - logsumexp(pred)) * target)
