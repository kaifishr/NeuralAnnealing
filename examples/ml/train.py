import os
import jax

from src.utils import set_random_seed
from src.dataloader import DataStore
from src.loss import CrossEntropyLoss
from src.model import model
from src.optimizer import DLOptimizer
from src.scheduler import ExponentialScheduler

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


def train():

    config = {
        "seed": 1234,
        "device": "gpu",
        "dataset": "fashion_mnist",  # mnist, fashion_mnist
        "dim_input": 28**2,
        "dim_output": 10,
        "dim_hidden": [128, 128],
        "batch_size": 500,
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

    set_random_seed(seed=config["seed"])

    if config["device"] == "cpu":
        jax.config.update(name="jax_platform_name", val="cpu")

    dataset = DataStore(config=config)

    criterion = CrossEntropyLoss()

    scheduler = ExponentialScheduler(
        gamma=config["gamma"],
        temp_start=config["temp_start"],
        temp_final=config["temp_final"],
    )

    optimizer = DLOptimizer(
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        dataset=dataset,
        config=config,
    )
    
    optimizer.run()


if __name__ == "__main__":
    train()
