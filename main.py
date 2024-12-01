import jax

from src.dataloader import DataServer
from src.loss import CrossEntropyLoss
from src.model import mlp
from src.optimizer import Optimizer
from src.scheduler import ExponentialScheduler


def run_classification():

    config = {
        "seed": 4444,
        "device": "gpu",
        "dataset": "mnist",
        "layer_sizes": (28**2, 128, 128, 10),
        "batch_size": 500,
        "num_targets": 10,
        "num_workers": 12,
        "temp_initial": 0.001,
        "temp_final": 1e-6,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "gamma": 0.004,
        "stats_every_n_epochs": 10,
    }

    if config["device"] == "cpu":
        jax.config.update("jax_platform_name", "cpu")

    data = DataServer(config=config)
    criterion = CrossEntropyLoss()

    scheduler = ExponentialScheduler(
        gamma=config["gamma"],
        temp_initial=config["temp_initial"],
        temp_final=config["temp_final"],
    )

    optimizer = Optimizer(
        model=mlp,
        criterion=criterion,
        scheduler=scheduler,
        data=data,
        config=config,
    )

    optimizer.run()


if __name__ == "__main__":
    run_classification()
