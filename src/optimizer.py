import math
import random
import time

from jax.nn import log_softmax
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from torch.utils.tensorboard import SummaryWriter
from typing import List

from .dataloader import DataServer
from .loss import Loss
from .scheduler import Scheduler
from .utils import comp_loss_accuracy, one_hot
from .model import init_params


class Optimizer:

    def __init__(
        self,
        model,
        criterion: Loss,
        scheduler: Scheduler,
        data: DataServer,
        config: dict,
    ):

        self.model = model
        self.loss = criterion
        self.scheduler = scheduler
        self.data = data

        self.gamma = config["gamma"]
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

        self.params_old = init_params(config["layer_sizes"], key=subkey)
        self.params_new = self._copy_params(params=self.params_old)

        self.writer = SummaryWriter()

    @staticmethod
    def _loss(target, logits):
        log_probs = log_softmax(logits)
        return -jnp.mean(jnp.sum(target * log_probs, axis=-1))

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

                logits = self.model(self.params_old, x)
                ## loss_old = self.loss(target=target, logits=logits)
                loss_old = self._loss(target=target, logits=logits)

                self.params_new = self._copy_params(params=self.params_old)
                self.params_new = self.perturb(params=self.params_new)
                logits = self.model(self.params_new, x)
                ## loss_new = self.loss(target=target, logits=logits)
                loss_new = self._loss(target=target, logits=logits)

                delta_loss = loss_new - loss_old

                if delta_loss < 0.0:
                    self.params_old = self.params_new
                    num_accept += 1
                elif random.random() < math.exp(-delta_loss / temp):
                    self.params_old = self.params_new
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
