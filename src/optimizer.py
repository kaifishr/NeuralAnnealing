import math
import random
import time

from jax.nn import log_softmax
import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional

from .dataloader import DataServer
from .loss import Criterion
from .scheduler import Scheduler
from .utils import comp_loss_accuracy, one_hot
from .model import init_params


Params = list[tuple[jax.Array, jax.Array]]


class Optimizer:

    def __init__(
        self,
        model,
        criterion: Criterion,
        scheduler: Scheduler,
        data: DataServer,
        config: dict,
    ):

        self.model = model
        self.loss = criterion
        self.scheduler = scheduler
        self.data = data

        self.gamma = config["gamma"]
        self.momentum = config["momentum"]
        self.temp_initial = config["temp_initial"]
        self.temp_final = config["temp_final"]
        self.perturbation_prob = config["perturbation_prob"]
        self.perturbation_size = config["perturbation_size"]
        self.stats_every_n_epochs = config["stats_every_n_epochs"]
        self.num_targets = config["num_targets"]

        self.training_generator = data.get_training_dataloader()
        self.test_generator = data.get_test_dataloader()

        self.key = jax.random.PRNGKey(seed=config["seed"])
        self.key, subkey = jax.random.split(self.key)

        self.params = init_params(config["layer_sizes"], key=subkey)
        self.params_new = self._copy_params(params=self.params)

        self.writer = SummaryWriter()

    @staticmethod
    def _loss(target, logits):
        log_probs = log_softmax(logits)
        return -jnp.mean(jnp.sum(target * log_probs, axis=-1))

    @staticmethod
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

    def run(self):

        temp = self.temp_initial
        epoch = 0

        while temp > self.temp_final:

            start_time = time.time()

            running_iter = 0
            running_loss = 0.0
            running_accuracy = 0.0
            running_counter = 0
            num_accept = 0
            num_permut = 0

            for x, y in self.training_generator:
                target = one_hot(y, self.num_targets)

                logits = self.model(self.params, x)
                ## loss_old = self.loss(target=target, logits=logits)
                loss_old = self._loss(target=target, logits=logits)

                self.params_new = self._copy_params(params=self.params)
                self.params_new = self.perturb(params=self.params_new)
                logits = self.model(self.params_new, x)
                ## loss_new = self.loss(target=target, logits=logits)
                loss_new = self._loss(target=target, logits=logits)

                delta_loss = loss_new - loss_old

                if delta_loss < 0.0:
                    ## self.params = self.params_new
                    self.params = self._update(
                        params=self.params,
                        params_new=self.params_new,
                        momentum=self.momentum,
                    )
                    num_accept += 1
                elif random.random() < math.exp(-delta_loss / temp):
                    # self.params = self.params_new
                    self.params = self._update(
                        params=self.params,
                        params_new=self.params_new,
                        momentum=self.momentum,
                    )
                    num_permut += 1

                batch_accuracy = float(jnp.sum(y == jnp.argmax(logits, axis=1)))

                running_loss += float(loss_old)
                running_accuracy += batch_accuracy
                running_counter += len(x)
                running_iter += 1

            epoch_time = time.time() - start_time

            self.writer.add_scalar(
                "train/running_loss", running_loss / running_counter, epoch
            )
            self.writer.add_scalar(
                "train/running_accuracy", running_accuracy / running_counter, epoch
            )
            self.writer.add_scalar("train/time_per_epoch", epoch_time, epoch)
            self.writer.add_scalar(
                "train/prop_accept", num_accept / running_iter, epoch
            )
            self.writer.add_scalar(
                "train/prop_permut", num_permut / running_iter, epoch
            )
            self.writer.add_scalar("train/temperature", temp, epoch)

            if (epoch + 1) % self.stats_every_n_epochs == 0:
                self._write_stats(epoch, epoch_time)

            epoch += 1
            temp = self.scheduler(temp, epoch)

        self._write_stats(epoch, epoch_time)
        self.writer.close()

    def _write_stats(self, epoch: int, epoch_time: float) -> None:
        train_loss, train_accuracy = comp_loss_accuracy(
            self.model,
            self.params_new,
            self.loss,
            self.training_generator,
        )
        test_loss, test_accuracy = comp_loss_accuracy(
            self.model,
            self.params_new,
            self.loss,
            self.test_generator,
        )
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/accuracy", train_accuracy, epoch)
        self.writer.add_scalar("test/loss", test_loss, epoch)
        self.writer.add_scalar("test/accuracy", test_accuracy, epoch)
        log = f"{epoch} {epoch_time:.2f} {train_loss:.4f} {train_accuracy:.4f} {test_loss:.4f} {test_accuracy:.4f}"
        print(log)

    def perturb(self, params: List[jax.Array]) -> List[jax.Array]:
        params = [(self._perturb_params(w), self._perturb_params(b)) for w, b in params]
        return params

    def _perturb_params(self, x: jax.Array) -> jax.Array:
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        # mask = 1.0 * (jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_prob)
        mask = jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_prob
        perturbation = jax.random.uniform(
            key=subkey2,
            shape=x.shape,
            minval=-self.perturbation_size,
            maxval=self.perturbation_size,
        )
        x = x + mask * perturbation
        # return jnp.clip(x, a_min=-10.0, a_max=10.0)
        return x

    @staticmethod
    def _copy_params(params: list[tuple[jax.Array]]):
        return [(jnp.array(w), jnp.array(b)) for w, b in params]


class RLOptimizer:

    def __init__(
        self,
        model,
        env,
        criterion: Criterion,
        scheduler: Scheduler,
        config: dict,
    ):

        self.model = model
        self.env = env
        self.criterion = criterion
        self.scheduler = scheduler

        self.gamma = config["gamma"]
        self.momentum = config["momentum"]
        self.temp_initial = config["temp_initial"]
        self.temp_final = config["temp_final"]
        self.perturbation_prob = config["perturbation_prob"]
        self.perturbation_size = config["perturbation_size"]
        self.stats_every_n_epochs = config["stats_every_n_epochs"]

        self.key = jax.random.PRNGKey(seed=config["seed"])
        self.key, subkey = jax.random.split(self.key)

        self.params = init_params(config["layer_sizes"], key=subkey)
        self.params_new = self._copy_params(params=self.params)

        self.writer = SummaryWriter()

    @staticmethod
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

    def run(self):

        temp = self.temp_initial
        epoch = 0
        num_episodes = 5

        while temp > self.temp_final:

            start_time = time.time()

            running_iter = 0
            running_reward = 0.0
            num_accept = 0
            num_permut = 0

            for _ in range(num_episodes):

                reward = self.rollout(
                    env=self.env, model=self.model, params=self.params
                )
                score_old = self.criterion(reward)

                self.params_new = self._copy_params(params=self.params)
                self.params_new = self.perturb(params=self.params_new)

                reward = self.rollout(
                    env=self.env, model=self.model, params=self.params
                )
                score_new = self.criterion(reward)

                delta_loss = score_new - score_old

                if delta_loss < 0.0:
                    self.params = self._update(
                        params=self.params,
                        params_new=self.params_new,
                        momentum=self.momentum,
                    )
                    num_accept += 1
                elif random.random() < math.exp(-delta_loss / temp):
                    self.params = self._update(
                        params=self.params,
                        params_new=self.params_new,
                        momentum=self.momentum,
                    )
                    num_permut += 1

                running_reward += reward
                running_iter += 1

            epoch_time = time.time() - start_time

            self.writer.add_scalar("train/time_per_epoch", epoch_time, epoch)
            self.writer.add_scalar("train/running_reward", running_reward, epoch)
            self.writer.add_scalar(
                "train/prop_accept", num_accept / running_iter, epoch
            )
            self.writer.add_scalar(
                "train/prop_permut", num_permut / running_iter, epoch
            )
            self.writer.add_scalar("train/temperature", temp, epoch)

            if (epoch + 1) % self.stats_every_n_epochs == 0:
                self._write_stats(
                    env=self.env, model=self.model, params=self.params, epoch=epoch
                )

            epoch += 1
            temp = self.scheduler(temp, epoch)

        self._write_stats(
            env=self.env, model=self.model, params=self.params, epoch=epoch
        )
        self.writer.close()

    def rollout(  # TODO: Make this a generator
        self, env, model, params, num_env_steps: int = 200, seed: Optional[int] = None
    ) -> float:

        observation, info = env.reset(seed=seed)

        total_reward = 0.0
        step = 0

        while step < num_env_steps:
            observation = jnp.atleast_2d(observation)
            action = int(jnp.argmax(model(params, observation), axis=-1)[0])
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset(seed=seed)
            total_reward += reward
            step += 1

        return total_reward

    def _write_stats(self, env, model, params, epoch: int) -> None:
        num_env_episodes = 20
        total_reward = 0
        for _ in range(num_env_episodes):
            reward = self.rollout(env=env, model=model, params=params)
            total_reward += reward
        mean_reward = total_reward / num_env_episodes
        self.writer.add_scalar("test/mean_reward", mean_reward, epoch)
        print(f"{epoch = } {mean_reward = }")

    def perturb(self, params: List[jax.Array]) -> List[jax.Array]:
        params = [(self._perturb_params(w), self._perturb_params(b)) for w, b in params]
        return params

    def _perturb_params(self, x: jax.Array) -> jax.Array:
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        # mask = 1.0 * (jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_prob)
        mask = jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_prob
        perturbation = jax.random.uniform(
            key=subkey2,
            shape=x.shape,
            minval=-self.perturbation_size,
            maxval=self.perturbation_size,
        )
        x = x + mask * perturbation
        # return jnp.clip(x, a_min=-10.0, a_max=10.0)
        return x

    @staticmethod
    def _copy_params(params: list[tuple[jax.Array]]):
        return [(jnp.array(w), jnp.array(b)) for w, b in params]
