import math
import random
import time

import numpy
import jax

from src.loss import Criterion
from src.scheduler import Scheduler
from src.custom_types import Params
from src.optimizer.optimizer import Optimizer


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

    def _step(self, temp: float) -> tuple[float, float]:
        self.key, subkey = jax.random.split(key=self.key)
        reward = self.dataset.rollout(
            key=subkey,
            model=self.model,
            params=self.params,
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
                    "train/running_reward_per_env_step": running_reward,
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

    def _write_full_eval(
        self,
        key: jax.Array,
        model,
        params: Params,
        iteration: int,
        num_test_rollouts: int = 40,
    ) -> None:
        rewards = []
        for _ in range(num_test_rollouts):
            key, subkey = jax.random.split(key=key)
            reward = self.dataset.rollout(key=subkey, model=model, params=params)
            rewards.append(reward)
        reward_mean = numpy.array(rewards).mean().item()
        reward_std = numpy.array(rewards).std().item()
        stats = {
            "test/avg_reward_per_env_step": reward_mean,
            "test/std_reward_per_env_step": reward_std,
        }
        self.logger.write(stats=stats, iteration=iteration)
        print(f"{iteration = } {reward_mean = :.2f} {reward_std = :.2f}")
