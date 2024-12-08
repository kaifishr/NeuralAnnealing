import pathlib
import pickle
import math
import random
import time

from functools import partial

import numpy
import jax
import jax.numpy as jnp
from typing import List, Optional, Any

from .dataloader import DataStore
from .loss import Criterion
from .scheduler import Scheduler
from .utils import comp_loss_accuracy, one_hot
from .model import init_params
from .custom_types import Params
from .logger import TensorboardLogger


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

        # Parameters for simulated annealing.
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
        if momentum is not None:
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

    def run(
        self
    ):
        raise NotImplementedError()


class DLOptimizer(Optimizer):

    def __init__(
        self,
        model,
        dataset: DataStore,
        criterion: Criterion,
        scheduler: Scheduler,
        config: dict,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            dataset=dataset,
            config=config,
        )
        self.training_generator = dataset.get_training_dataloader()
        self.test_generator = dataset.get_test_dataloader()
        self.dim_outputs = config["dim_output"]

    def _step(self, x, y, temp: float) -> tuple[float, float, jax.Array]:
        target = one_hot(y, self.dim_outputs)

        logits = self.model(self.params, x)
        loss_old = self.criterion(target=target, logits=logits)

        self.params_new = self._copy_params(params=self.params)
        self.params_new = self._perturb(
            key=self.key,
            params=self.params_new,
            perturbation_prob=self.perturbation_prob,
            perturbation_size=self.perturbation_size,
        )
        logits = self.model(self.params_new, x)
        loss_new = self.criterion(target=target, logits=logits)

        diff_loss = loss_new - loss_old
        if diff_loss < 0.0:
            self.params = self._update(
                params=self.params,
                params_new=self.params_new,
                momentum=self.momentum,
            )
        elif random.random() < math.exp(-diff_loss / temp):
            self.params = self._update(
                params=self.params,
                params_new=self.params_new,
                momentum=self.momentum,
            )

        return loss_new, diff_loss, logits

    def run(self):

        temp = self.temp_start
        iteration = 0

        while temp > self.temp_final:
            
            running_loss = 0.0
            running_accuracy = 0.0
            running_counter = 0

            start_time = time.time()

            for x, y in self.training_generator:
                loss, diff_loss, logits = self._step(x=x, y=y, temp=temp)
                batch_accuracy = float(jnp.sum(y == jnp.argmax(logits, axis=1)))
                running_loss += float(loss)
                running_accuracy += batch_accuracy
                running_counter += len(x)

            iteration_time = time.time() - start_time

            if (iteration + 1) % self.train_stats_every_n_iter == 0:
                stats = {
                    "train/running_loss": running_loss / running_counter,
                    "train/running_accuracy": running_accuracy / running_counter,
                    "train/time_per_iteration": iteration_time,
                    "train/temperature": temp,
                    "train/diff_loss": diff_loss,
                    "train/exp_argument": diff_loss / temp,
                }
                self.logger.write(stats=stats, iteration=iteration)

            if (iteration + 1) % self.test_stats_every_n_iter == 0:
                self._save_params()
                self._write_full_eval(iteration=iteration)

            iteration += 1
            temp = self.scheduler(temp, iteration)

        self._save_params()
        self._write_full_eval(iteration=iteration)

    def _write_full_eval(self, iteration: int) -> None:
        train_loss, train_accuracy = comp_loss_accuracy(
            self.model,
            self.params_new,
            self.criterion,
            self.training_generator,
        )
        test_loss, test_accuracy = comp_loss_accuracy(
            self.model,
            self.params_new,
            self.criterion,
            self.test_generator,
        )
        stats = {
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "test/loss": test_loss,
            "test/accuracy": test_accuracy,
        }
        self.logger.write(stats=stats, iteration=iteration)


class RLOptimizer(Optimizer):

    def __init__(
        self,
        model,
        rl_dataset,
        criterion: Criterion,
        scheduler: Scheduler,
        config: dict,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            scheduler=scheduler,
            dataset=rl_dataset,
            config=config,
        )
        self.num_rollouts = config["num_rollouts"]
 
    def _step(self, temp: float) -> tuple[float, float]:
        self.key, subkey = jax.random.split(key=self.key)
        reward = self.dataset.rollout(
            key=subkey,
            model=self.model,
            params=self.params,
            num_rollouts=self.num_rollouts,
        )
        score_old = self.criterion(reward)

        self.params_new = self._copy_params(params=self.params)
        self.params_new = self._perturb(
            key=self.key,
            params=self.params_new,
            perturbation_prob=self.perturbation_prob,
            perturbation_size=self.perturbation_size,
        )
        reward = self.dataset.rollout(
            key=subkey,
            model=self.model,
            params=self.params_new,
            num_rollouts=self.num_rollouts,
        )
        score_new = self.criterion(reward)

        diff_score = score_new - score_old
        if diff_score < 0.0:
            self.params = self._update(
                params=self.params,
                params_new=self.params_new,
                momentum=self.momentum,
            )
        elif random.random() < math.exp(-diff_score / temp):
            self.params = self._update(
                params=self.params,
                params_new=self.params_new,
                momentum=self.momentum,
            )

        return reward, diff_score

    def run(self):

        temp = self.temp_start
        iteration = 0

        while temp > self.temp_final:
            start_time = time.time()
            running_iter = 0
            running_reward = 0.0

            reward, diff_score = self._step(temp=temp)

            running_reward += reward
            running_iter += 1

            iteration_time = time.time() - start_time

            if (iteration + 1) % self.train_stats_every_n_iter == 0:
                stats = {
                    "train/rollouts_per_second": self.num_rollouts / iteration_time,
                    "train/running_reward": running_reward,
                    "train/time_per_iteration": iteration_time,
                    "train/temperature": temp,
                    "train/diff_score": diff_score,
                    "train/exp_argument": diff_score / temp,
                }
                self.logger.write(stats=stats, iteration=iteration)

            if (iteration + 1) % self.test_stats_every_n_iter == 0:
                self.key, subkey = jax.random.split(key=self.key)
                self._write_full_eval(
                    key=subkey,
                    model=self.model,
                    params=self.params,
                    iteration=iteration,
                )
                self._save_params()

            iteration += 1
            temp = self.scheduler(temp, iteration)

        self.key, subkey = jax.random.split(key=self.key)
        self._write_full_eval(
            key=subkey,
            model=self.model,
            params=self.params,
            iteration=iteration,
        )
        self._save_params()
        self.writer.close()

    def _write_full_eval(
        self, key: jax.Array, model, params: Params, iteration: int
    ) -> None:
        num_test_episodes = 20
        rewards = []
        for _ in range(num_test_episodes):
            key, subkey = jax.random.split(key=key)
            reward = self.dataset.rollout(key=subkey, model=model, params=params)
            rewards.append(float(reward))
        reward_mean = float(numpy.array(rewards).mean())
        reward_std = float(numpy.array(rewards).std())
        stats = {
            "test/avg_reward": reward_mean,
            "test/std_reward": reward_std,
        }
        self.logger.write(stats=stats, iteration=iteration)
        print(f"{iteration = } {reward_mean = } {reward_std = }")
