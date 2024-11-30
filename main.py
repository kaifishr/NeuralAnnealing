"""Neural network parameter optimization with simulated annealing."""

import jax

from src.dataloader import DataServer
from src.loss import CrossEntropyLoss
from src.model import Model
from src.optimizer import Optimizer
from src.scheduler import ExponentialScheduler


def run_classification():

    config = {
        "layer_sizes": (28**2, 64, 64, 10),
        "dataset": "mnist",
        "batch_size": 500,
        "num_targets": 10,
        "num_workers": 4,
        "temp_initial": 0.001,
        "temp_final": 1e-6,
        # "perturbation_prob": 0.01,
        # "perturbation_size": 0.01,
        "gamma": 0.003,
        "device": "cpu",
        "stats_every_n_epochs": 10,
    }

    if config["device"] == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    data = DataServer(config=config)

    model = Model(config=config)
    criterion = CrossEntropyLoss()
    scheduler = ExponentialScheduler(
        gamma=config["gamma"],
        temp_initial=config["temp_initial"],
        temp_final=config["temp_final"],
    )

    optimizer = Optimizer(
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        data=data,
        config=config,
    )

    optimizer.run()


if __name__ == "__main__":
    run_classification()
