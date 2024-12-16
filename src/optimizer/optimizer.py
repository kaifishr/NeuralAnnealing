import pathlib
import pickle

import jax
import jax.numpy as jnp
from typing import List, Optional, Any
from functools import partial

from src.loss import Criterion
from src.scheduler import Scheduler
from src.model import init_params
from src.custom_types import Params
from src.logger import TensorboardLogger


class Optimizer:

    def __init__(
        self,
        model,
        criterion: Criterion,
        scheduler: Scheduler,
        dataset,
        config: dict[str, Any],
    ):
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.dataset = dataset
        self.config = config

        self.key = jax.random.PRNGKey(seed=config["seed"])
        layer_dims = (
            [config["dim_input"]] + config["dim_hidden"] + [config["dim_output"]]
        )
        self.params = init_params(key=self.key, dims=layer_dims)
        self.params_new = self._copy_params(params=self.params)

        self.momentum = config["momentum"]
        self.temp_start = config["temp_start"]
        self.temp_final = config["temp_final"]
        self.perturbation_prob = config["perturbation_prob"]
        self.perturbation_size = config["perturbation_size"]

        self.test_stats_every_n_iter = config["test_stats_every_n_iter"]
        self.train_stats_every_n_iter = config["train_stats_every_n_iter"]

        self.output_dir = pathlib.Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = TensorboardLogger()

    @staticmethod
    @jax.jit
    def _update(
        params: Params, params_new: Params, momentum: Optional[float] = None
    ) -> list[tuple[jax.Array]]:
        if momentum is not None or momentum > 0.:
            eta = momentum
            return [
                (eta * w + (1.0 - eta) * w_new, eta * b + (1.0 - eta) * b_new)
                for (w, b), (w_new, b_new) in zip(params, params_new)
            ]
        return params_new

    @partial(jax.jit, static_argnames=["self"])
    def _perturb(
        self,
        key: jax.Array,
        params: List[jax.Array],
        perturbation_prob: float,
        perturbation_size: float,
    ) -> List[jax.Array]:
        return [
            (
                self._perturb_params(key, w, perturbation_prob, perturbation_size),
                self._perturb_params(key, b, perturbation_prob, perturbation_size),
            )
            for w, b in params
        ]

    @staticmethod
    @jax.jit
    def _perturb_params(
        key: jax.Array, x: jax.Array, prob: float, size: float
    ) -> jax.Array:
        key, subkey1, subkey2 = jax.random.split(key, num=3)
        mask = jax.random.uniform(key=subkey1, shape=x.shape) < prob
        perturbation = jax.random.uniform(
            key=subkey2, shape=x.shape, minval=-size, maxval=size
        )
        x = x + mask * perturbation
        return x

    @staticmethod
    @jax.jit
    def _copy_params(params: list[tuple[jax.Array, jax.Array]]):
        return [(jnp.array(w), jnp.array(b)) for w, b in params]

    def _save_params(self) -> None:
        with open(self.output_dir / "ckpt.pkl", "wb") as fp:
            pickle.dump(self.params, fp)

    def _step(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()
