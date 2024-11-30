"""PyTorch-based dataloader for JAX.
"""

import numpy as np
from pathlib import Path

import jax.numpy as jnp

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NormFlattenCast(object):
    def __call__(self, data):
        return np.ravel(np.array(data, dtype=jnp.float32))


class DataServer:

    def __init__(self, config: dict) -> None:

        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
        self.num_targets = config["num_targets"]
        self.num_workers = config["num_workers"]

        home_dir = Path.home()
        root_dir = f"{home_dir}/data/{self.dataset}"

        if self.dataset == "fashion_mnist":

            self.train_dataset = FashionMNIST(root=root_dir, train=True, download=True)
            mean = float(
                jnp.array(self.train_dataset.data / 255.0, dtype=jnp.float32).mean()
            )
            std = float(
                jnp.array(self.train_dataset.data / 255.0, dtype=jnp.float32).std()
            )

            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomErasing(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=mean, std=std),
                    NormFlattenCast(),
                ]
            )

            test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    NormFlattenCast(),
                ]
            )

            self.train_dataset = FashionMNIST(
                root=root_dir, train=True, download=True, transform=train_transforms
            )
            self.test_dataset = FashionMNIST(
                root=root_dir, train=False, download=True, transform=test_transforms
            )

        elif self.dataset == "mnist":

            self.train_dataset = MNIST(root=root_dir, train=True, download=True)
            mean = float(
                jnp.array(self.train_dataset.data / 255.0, dtype=jnp.float32).mean()
            )
            std = float(
                jnp.array(self.train_dataset.data / 255.0, dtype=jnp.float32).std()
            )

            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomErasing(),
                    transforms.Normalize(mean=mean, std=std),
                    NormFlattenCast(),
                ]
            )

            test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    NormFlattenCast(),
                ]
            )

            self.train_dataset = MNIST(
                root=root_dir, train=True, download=True, transform=train_transforms
            )
            self.test_dataset = MNIST(
                root=root_dir, train=False, download=True, transform=test_transforms
            )

        else:
            raise NotImplementedError(f"Dataset {self.dataset} not available.")

    def get_training_dataloader(self) -> DataLoader:
        """Returns training data generator."""
        training_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=numpy_collate,
            pin_memory=True,
            drop_last=False,
        )
        return training_dataloader

    def get_test_dataloader(self) -> DataLoader:
        """Returns test data generator."""
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=numpy_collate,
            pin_memory=True,
            drop_last=False,
        )

        return test_dataloader
