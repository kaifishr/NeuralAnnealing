import numpy
import gymnasium as gym

import os
import jax

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

    env = gym.make("LunarLander-v3", render_mode="human")
    observation, info = env.reset(seed=seed)

    for _ in range(num_env_steps):

        action = env.action_space.sample()
        # action = model(observation)
        print(f"{action = }")

        observation, reward, terminated, truncated, info = env.step(action)
        print(f"{observation = }")
        print(f"{reward = }")
        print(f"{terminated = }")
        print(f"{truncated = }")
        print(f"{info = }")
        print()

        # Start new episode if the episode has ended.
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def train():

    config = {
        "seed": 4444,
        "device": "gpu",
        "problem": "lunar_lander",  # mnist, fashion_mnist, lunar_lander
        # "num_env_steps": 200,
        "layer_sizes": (8, 16, 16, 4),
        "batch_size": 1,
        "temp_initial": 100.0,
        "temp_final": 1e-9,
        "momentum": 0.2,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "gamma": 0.003,
        "stats_every_n_epochs": 10,
    }

    env = gym.make("LunarLander-v3")

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
        env=env,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
    )

    optimizer.run()
    env.close()


if __name__ == "__main__":
    train()
