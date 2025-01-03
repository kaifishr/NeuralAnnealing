import os

import jax

from src.utils import set_random_seed
from src.loss import MaxScore
from src.model import model
from src.scheduler import GeometricScheduler
from src.dataloader import RolloutWorker
from src.optimizer.optimizer_rl import RLOptimizer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"


def train():

    config = {
        "seed": 1234,
        "device": "cpu",
        "env_name": "LunarLander-v3",  # CartPole-v1, Acrobot-v1, LunarLander-v3, BipedalWalker-v3
        "num_envs": 64,
        "max_env_steps": 200,
        "dim_input": 8,  # 4, 6, 8, 24
        "dim_output": 4,  # 2, 3, 4, 4
        "dim_hidden": 2 * [64],
        "temp_start": 0.01,
        "temp_final": 1e-5,
        "gamma": 0.9999,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "momentum": 0.0,
        "train_stats_every_n_iter": 50,
        "test_stats_every_n_iter": 200,
        "output_dir": "output",
    }

    set_random_seed(seed=config["seed"])

    rl_dataset = RolloutWorker(
        env_name=config["env_name"],
        num_envs=config["num_envs"],
        max_env_steps=config["max_env_steps"],
    )

    if config["device"] == "cpu":
        jax.config.update(name="jax_platform_name", val="cpu")

    criterion = MaxScore()

    scheduler = GeometricScheduler(
        gamma=config["gamma"],
        temp_start=config["temp_start"],
        temp_final=config["temp_final"],
    )

    optimizer = RLOptimizer(
        model=model,
        rl_dataset=rl_dataset,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
    )

    optimizer.run()


if __name__ == "__main__":
    train()
