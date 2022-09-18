"""Parameter optimization using simulated annealing.

Script performs neural network parameter optimization
using simulated annealing.

"""
import jax

from src.dataloader import DataServer
from src.loss import MSELoss
from src.model import Model
from src.optimizer import Optimizer
from src.scheduler import ExponentialScheduler, LinearScheduler


def run_experiment():

    config = {
        "layer_sizes": (28**2, 64, 64, 64, 10),
        "params_type": "trinary",        # binary, trinary
        "dataset": "mnist",
        "batch_size": 1024,
        "num_targets": 10,
        "num_workers": 2,
        "temp_initial": 0.06,
        "temp_final": 1e-6,
        "gamma": 0.02,
        "device": "cpu",
        "stats_every_n_epochs": 20,
    }

    if config["device"] == "cpu":
        jax.config.update('jax_platform_name', 'cpu')

    data = DataServer(config=config)
    model = Model(config=config)
    criterion = MSELoss()
    scheduler = ExponentialScheduler(
        gamma=config["gamma"], 
        temp_initial=config["temp_initial"],
        temp_final=config["temp_final"]
    )

    optimizer = Optimizer(
        model=model, 
        criterion=criterion, 
        scheduler=scheduler,
        data=data,
        config=config
    )
    optimizer.run()


if __name__ == "__main__":
    run_experiment()
