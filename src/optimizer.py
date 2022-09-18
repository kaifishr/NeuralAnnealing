"""Optimizer class for simulated annealing."""
from typing import List

import copy
import math
import os
import random
import time

import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from torch.utils.tensorboard import SummaryWriter

from .dataloader import DataServer
from .loss import Loss
from .model import Model
from .scheduler import Scheduler
from .utils import comp_loss_accuracy, one_hot 


class Optimizer:
    """Optimizer class for simulated annealing.
    """

    def __init__(self, model: Model, criterion: Loss, scheduler: Scheduler, data: DataServer, config: dict):
        """Initializes Optimizer class."""

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
        self.new_params = None

        self.gamma = config["gamma"]

        self.writer = SummaryWriter()
        self.key = jax.random.PRNGKey(0)

        # Initial perturbation probability.
        self.perturbation_probability = 1.0 

        self.stats_every_n_epochs = config["stats_every_n_epochs"]

        # Set perturbation method according to parameter type.
        params_type = config["params_type"]

        if params_type == "float":
            self._perturb_fun = None
            raise NotImplemented()
        elif params_type == "binary":
            self._perturb_fun = self._binary_perturb
        elif params_type == "trinary":
            self._perturb_fun = self._trinary_perturb

    def run(self):
        """Performs optimization for neural network."""

        path = os.path.join(self.writer.log_dir, "stats.txt")
        file = open(path, "w")

        temp = self.temp_initial
        epoch = 0

        while temp > self.temp_final:

            start_time = time.time()

            running_loss_old = 0.0
            running_loss_new = 0.0
            running_counter = 0

            for x, y in self.training_generator:
                target = one_hot(y, self.num_targets)

                # Compute loss with old parameters
                loss_old_params = self.loss(target, self.model.forward(x))

                # Compute loss with new parameters
                self.model.params = self.perturb(self.model.params)
                loss_new_params = self.loss(target, self.model.forward(x))

                delta_loss = loss_new_params - loss_old_params

                if (delta_loss < 0):
                    pass    # keep new weights in network
                    # Keep new parameters if new loss is smaller than old loss.
                    # self.model.params = self.model.params
                    # todo: add line to save best configuration.
                    # if cost_new < cost_best:
                    #     self.params_best = copy.deepcopy(self.model.params)
                elif (random.random() < math.exp(-delta_loss / temp)):
                    pass
                    # Keep new parameters with a certain probability anyways.
                    # self.model.params = self.model.params 
                else:
                    self.model.params = self.old_params

                running_loss_old += float(loss_old_params) * len(x)
                running_loss_new += float(loss_new_params) * len(x)
                running_counter += len(x)

            epoch_time = time.time() - start_time

            self.writer.add_scalar("Training/LossOldParams", running_loss_old / running_counter, epoch)
            self.writer.add_scalar("Training/LossNewParams", running_loss_new / running_counter, epoch)
            self.writer.add_scalar("Training/TimePerEpoch", epoch_time, epoch)
            self.writer.add_scalar("Training/Temperature", temp, epoch)

            if (epoch + 1) % self.stats_every_n_epochs == 0:
                train_loss, train_accuracy = comp_loss_accuracy(self.model, self.loss, self.training_generator)
                test_loss, test_accuracy = comp_loss_accuracy(self.model, self.loss, self.test_generator)
                self.writer.add_scalar("Training/Loss", train_loss, epoch)
                self.writer.add_scalar("Training/Accuracy", train_accuracy, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_accuracy, epoch)
                log = f"{epoch} {epoch_time:.2f} {train_loss:.4f} {train_accuracy:.4f} {test_loss:.4f} {test_accuracy:.4f}"
                file.write(f"{log}\n")
                file.flush()
                print(log)

            epoch += 1
            temp = self.scheduler(temp, epoch)

            self.perturbation_probability = (temp / self.temp_initial) + 0.05

        train_loss, train_accuracy = comp_loss_accuracy(self.model, self.loss, self.training_generator)
        test_loss, test_accuracy = comp_loss_accuracy(self.model, self.loss, self.test_generator)
        self.writer.add_scalar("Training/Loss", train_loss, epoch)
        self.writer.add_scalar("Training/Accuracy", train_accuracy, epoch)
        self.writer.add_scalar("Test/Loss", test_loss, epoch)
        self.writer.add_scalar("Test/Accuracy", test_accuracy, epoch)
        log = f"{epoch} {epoch_time:.2f} {train_loss:.4f} {train_accuracy:.4f} {test_loss:.4f} {test_accuracy:.4f}"
        file.write(f"{log}\n")
        file.flush()
        print(log)

        self.writer.close()
        file.close()

    def perturb(self, params: List) -> None:
        """Perturbs weights.

        Given the current configuration 'params', this function generates
        another neighbouring configuration to which the system may move. 
        
        Make deep copy of weights.
        """
        # Make a copy of the old parameters.
        self.old_params = copy.deepcopy(params)
        # Perturb parameters and save them as new parameters.
        return self._perturb(params)

    def _perturb(self, params: List[DeviceArray]) -> List[DeviceArray]:
        """Perturbs parameters."""
        params = [(self._perturb_fun(w), self._perturb_fun(b)) for w, b in params]
        return params

    def _binary_perturb(self, x: DeviceArray) -> DeviceArray:
        """Perturbs binary array.
        
        See also https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html#random-numbers-in-jax
        for how JAX handels random numbers.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        mask = (jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_probability)
        perturbation = jax.random.randint(key=subkey2, shape=x.shape, minval=-1, maxval=2).astype(jnp.float32)
        out = x + mask * perturbation
        return jnp.clip(a=out, a_min=0, a_max=1)

    def _trinary_perturb(self, x: DeviceArray) -> DeviceArray:
        """Perturbs trinary array.
        
        See also https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html#random-numbers-in-jax
        for how JAX handels random numbers.
        """
        self.key, subkey1, subkey2 = jax.random.split(self.key, num=3)
        mask = (jax.random.uniform(key=subkey1, shape=x.shape) < self.perturbation_probability)
        perturbation = jax.random.randint(key=subkey2, shape=x.shape, minval=-1, maxval=2).astype(jnp.float32)
        out = x + mask * perturbation
        return jnp.clip(a=out, a_min=-1, a_max=1)
