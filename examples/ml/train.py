import os
import jax

from src.dataloader import DataStore
from src.loss import CrossEntropyLoss
from src.model import mlp
from src.optimizer import DLOptimizer
from src.scheduler import ExponentialScheduler

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


def train():

    config = {
        "seed": 5432,
        "device": "gpu",
        "dataset": "fashion_mnist",  # mnist, fashion_mnist
        "dim_input": 28**2,
        "dim_output": 10,
        "dim_hidden": [128, 128],
        "batch_size": 200,  # 500
        "num_workers": 2,
        "temp_start": 0.002,
        "temp_final": 1e-9,
        "momentum": 0.2,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "gamma": 0.003,
        "train_stats_every_n_iter": 1,
        "test_stats_every_n_iter": 20,
        "output_dir": "output/ml",
    }

    if config["device"] == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    dataset = DataStore(config=config)
    criterion = CrossEntropyLoss()

    scheduler = ExponentialScheduler(
        gamma=config["gamma"],
        temp_start=config["temp_start"],
        temp_final=config["temp_final"],
    )

    optimizer = DLOptimizer(
        model=mlp,
        criterion=criterion,
        scheduler=scheduler,
        dataset=dataset,
        config=config,
    )

    optimizer.run()


if __name__ == "__main__":
    train()
