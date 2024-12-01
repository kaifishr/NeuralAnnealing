import pathlib
import numpy
import torch
import jax.numpy as jnp

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms


def numpy_collate(batch):
    if isinstance(batch[0], numpy.ndarray):
        return numpy.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return numpy.array(batch)


class FlattenCast(object):
    def __call__(self, data: torch.Tensor) -> numpy.ndarray:
        data = numpy.ravel(numpy.array(data, dtype=jnp.float32))
        return data


class DataServer:

    def __init__(self, config: dict) -> None:

        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
        self.num_targets = config["num_targets"]
        self.num_workers = config["num_workers"]

        home_dir = pathlib.Path.home()
        root_dir = f"{home_dir}/data/{self.dataset}"

        if self.dataset == "fashion_mnist":
            mean = 0.5
            std = 0.5

            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    # transforms.RandomErasing(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean=mean, std=std),
                    FlattenCast(),
                ]
            )

            test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    FlattenCast(),
                ]
            )

            self.train_dataset = FashionMNIST(
                root=root_dir,
                train=True,
                download=True,
                transform=train_transforms,
            )
            self.test_dataset = FashionMNIST(
                root=root_dir,
                train=False,
                download=True,
                transform=test_transforms,
            )

        elif self.dataset == "mnist":
            mean = 0.5
            std = 0.5

            train_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    FlattenCast(),
                ]
            )

            test_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    FlattenCast(),
                ]
            )

            self.train_dataset = MNIST(
                root=root_dir,
                train=True,
                download=True,
                transform=train_transforms,
            )
            self.test_dataset = MNIST(
                root=root_dir,
                train=False,
                download=True,
                transform=test_transforms,
            )

        else:
            raise NotImplementedError(f"Dataset {self.dataset} not available.")

    def get_training_dataloader(self) -> DataLoader:
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
        test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=numpy_collate,
            pin_memory=True,
            drop_last=False,
        )
        return test_dataloader
