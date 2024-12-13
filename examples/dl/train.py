import os
import jax

from src.utils import set_random_seed
from src.dataloader import DataStore
from src.loss import CrossEntropyLoss, MSELoss
from src.model import model
from src.scheduler import ExponentialScheduler
from src.optimizer.optimizer_dl import DLOptimizer

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"


def train():

    config = {
        "seed": 1234,
        "device": "gpu",
        "dataset": "mnist",  # mnist, fashion_mnist
        "dim_input": 28**2,
        "dim_output": 10,
        "dim_hidden": 2 * [128],
        "batch_size": 100,
        "num_workers": 2,
        "temp_start": 0.01,
        "temp_final": 1e-7,
        "gamma": 0.005,
        "perturbation_prob": 0.02,
        "perturbation_size": 0.02,
        "momentum": 0.7,
        "train_stats_every_n_iter": 1,
        "test_stats_every_n_iter": 10,
        "output_dir": "output/dl",
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
