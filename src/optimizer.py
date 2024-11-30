from typing import List
import copy
import math
import os
import random
import time

import jax
import jax.numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from torch.utils.tensorboard import SummaryWriter

from .dataloader import DataServer
from .loss import Loss
from .model import Model
from .scheduler import Scheduler
from .utils import comp_loss_accuracy, one_hot


class Optimizer:

    def __init__(
        self,
        model: Model,
        criterion: Loss,
        scheduler: Scheduler,
        data: DataServer,
        config: dict,
    ):

        self.model = model
        self.loss = criterion
        self.scheduler = scheduler
        self.data = data

        self.temp_initial = config["temp_initial"]
        self.temp_final = config["temp_final"]
        self.num_targets = config["num_targets"]

        self.training_generator = data.get_training_dataloader()
        self.test_generator = data.get_test_dataloader()

        self.old_params = self.model.params

        self.gamma = config["gamma"]

        self.writer = SummaryWriter()
        self.file = open(os.path.join(self.writer.log_dir, "stats.txt"), "w")

        seed = 1123  # config
        self.key = jax.random.PRNGKey(seed=seed)

        # Initial perturbation probability.
        self.perturbation_prob = 0.02  # config
        self.perturbation_size = 0.02  # config
        self.stats_every_n_epochs = config["stats_every_n_epochs"]

    def run(self):

        temp = self.temp_initial
        epoch = 0

        while temp > self.temp_final:

            start_time = time.time()

            running_loss = 0.0
            running_counter = 0

            for x, y in self.training_generator:
                target = one_hot(y, self.num_targets)

                logits = self.model.forward(x)
                loss_old = self.loss(target=target, pred=logits)

                self.model.params_tmp = self.perturb(copy.deepcopy(self.model.params))
                logits = self.model.forward_tmp(x)
                loss_new = self.loss(target=target, pred=logits)

                delta_loss = loss_new - loss_old

                if delta_loss < 0.0:
                    self.model.params = self.model.params_tmp
                elif random.random() < math.exp(-delta_loss / temp):
                    self.model.params = self.model.params_tmp

                running_loss += float(loss_old)
                running_counter += len(x)

            epoch_time = time.time() - start_time

            self.writer.add_scalar(
                "train/running_loss", running_loss / running_counter, epoch
            )
            self.writer.add_scalar("train/time_per_epoch", epoch_time, epoch)
            self.writer.add_scalar("train/temperature", temp, epoch)

            if (epoch + 1) % self.stats_every_n_epochs == 0:
                self._write_stats(epoch, epoch_time)

            epoch += 1
            temp = self.scheduler(temp, epoch)

            # self.perturbation_prob = max(1e-3, temp / self.temp_initial)

        self._write_stats(epoch, epoch_time)

        self.writer.close()
        self.file.close()

    def _write_stats(self, epoch: int, epoch_time: float) -> None:
        train_loss, train_accuracy = comp_loss_accuracy(
            self.model, self.loss, self.training_generator
        )
        test_loss, test_accuracy = comp_loss_accuracy(
            self.model, self.loss, self.test_generator
        )
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/accuracy", train_accuracy, epoch)
        self.writer.add_scalar("test/loss", test_loss, epoch)
        self.writer.add_scalar("test/accuracy", test_accuracy, epoch)
        log = f"{epoch} {epoch_time:.2f} {train_loss:.4f} {train_accuracy:.4f} {test_loss:.4f} {test_accuracy:.4f}"
        self.file.write(f"{log}\n")
        self.file.flush()
        print(log)

    def perturb(self, params: List) -> None:
        return self._perturb(params=params)

    def _perturb(self, params: List[ArrayImpl]) -> List[ArrayImpl]:
        params = [(self._perturb_params(w), self._perturb_params(b)) for w, b in params]
        return params

    def _perturb_params(self, x: ArrayImpl) -> ArrayImpl:
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
