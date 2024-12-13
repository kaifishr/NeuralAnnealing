import math
import random
import time

import jax
import jax.numpy as jnp

from src.dataloader import DataStore
from src.loss import Criterion
from src.scheduler import Scheduler
from src.utils import comp_loss_accuracy, one_hot
from src.optimizer.optimizer import Optimizer


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
        self.key, subkey = jax.random.split(key=self.key)
        target = one_hot(y, self.dim_outputs)

        logits = self.model(self.params, x)
        loss_old = self.criterion(target=target, logits=logits)

        self.params_new = self._copy_params(params=self.params)
        self.params_new = self._perturb(
            key=subkey,
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
