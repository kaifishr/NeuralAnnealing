import gymnasium as gym

import os
import jax
import jax.numpy as jnp

from src.loss import MaxScore
from src.model import mlp
from src.optimizer import RLOptimizer
from src.scheduler import ExponentialScheduler

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"


def test(
    model: callable = None,
    num_env_steps: int = 1000,
    seed: int = 123,
) -> None:

    render_mode = "human"
    render_mode = None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    observation, info = env.reset(seed=seed)

    total_reward = 0

    for _ in range(num_env_steps):

        action = env.action_space.sample()
        # action = model(observation)
        print(f"{action = }")

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Start new episode if the episode has ended.
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print(f"{total_reward = }")


class EnvDataset:

    def __init__(self, env, max_env_steps: int = 200):
        self.env = env
        self.max_env_steps = max_env_steps

    def _rollout(self, key, model, params) -> float:
        observation, info = self.env.reset(seed=key)
        total_reward = 0
        for _ in range(self.max_env_steps):
            observation = jnp.atleast_2d(observation)
            action = int(jnp.argmax(model(params, observation), axis=-1)[0])
            # action = model(params, observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def rollout(self, key, model, params, num_rollouts: int = 1) -> float:
        total_rewards = 0.0
        for _ in range(num_rollouts):
            total_rewards += self._rollout(key=key, model=model, params=params)
        return total_rewards / num_rollouts


def train():

    config = {
        "seed": 4444,
        "device": "gpu",
        "problem": "lunar_lander",  # mnist, fashion_mnist, lunar_lander
        # "num_env_steps": 200,
        # "num_env_rollouts": 2,
        "layer_sizes": (8, 64, 64, 4),
        "batch_size": 1,
        "temp_initial": 0.2,
        "temp_final": 1e-9,
        "momentum": 0.01,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "gamma": 0.0002,
        "stats_every_n_epochs": 10,
    }

    env = gym.make("LunarLander-v3")
    env_dataset = EnvDataset(env, max_env_steps=500)

    if config["device"] == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    criterion = MaxScore()

    scheduler = ExponentialScheduler(
        gamma=config["gamma"],
        temp_initial=config["temp_initial"],
        temp_final=config["temp_final"],
    )

    optimizer = RLOptimizer(
        model=mlp,
        env_dataset=env_dataset,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
    )

    optimizer.run()
    env.close()


if __name__ == "__main__":
    train()
