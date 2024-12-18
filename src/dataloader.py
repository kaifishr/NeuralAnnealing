import pathlib

import numpy
import torch
import jax
import jax.numpy as jnp
import gymnasium as gym

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import transforms

from .custom_types import Params


def numpy_collate(batch):
    if isinstance(batch[0], numpy.ndarray):
        return numpy.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return numpy.array(batch)


class FlattenCast:

    def __call__(self, data: torch.Tensor) -> numpy.ndarray:
        data = numpy.ravel(numpy.array(data, dtype=jnp.float32))
        return data


class DataStore:

    def __init__(self, config: dict) -> None:

        self.dataset = config["dataset"]
        self.batch_size = config["batch_size"]
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


class RolloutWorker:

    def __init__(self, env_name: str, num_envs: int = 1, max_env_steps: int = 200):

        self.num_envs = num_envs
        self.max_steps = self.num_envs * max_env_steps

        def make_env() -> gym.Env:
            return gym.make(id=env_name)

        self.envs = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])

        if isinstance(self.envs.action_space, gym.spaces.MultiDiscrete):
            self.is_discrete = True
        elif isinstance(self.envs.action_space, gym.spaces.Box):
            self.is_discrete = False
        else:
            raise NotImplementedError(
                f"Action space '{self.envs.action_space}' not supported."
            )

    def __del__(self):
        self.envs.close()

    def _rollout(self, seed: int, model, params: Params) -> float:
        observations, infos = self.envs.reset(seed=seed)
        total_reward = 0
        total_steps = 0
        while total_steps < self.max_steps:
            prediction = model(params, observations)
            if self.is_discrete:
                actions = jnp.argmax(prediction, axis=-1).tolist()
            else:
                actions = jax.nn.tanh(prediction).tolist()
            observations, rewards, terminations, truncations, infos = self.envs.step(
                actions
            )
            total_reward += rewards.sum().item()
            total_steps += len(rewards)
            if terminations.any() or truncations.any():
                break
        return total_reward / total_steps

    def rollout(self, key: jax.Array, model, params: Params) -> float:
        rewards = self._rollout(seed=int(key[0]), model=model, params=params)
        return rewards
