import pathlib

import gymnasium as gym
import numpy
import torch
import jax.numpy as jnp
import jax
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


class RLDataset:

    def __init__(self, env: gym.Env, max_len_rollout: int = 400):
        self.env = env
        self.max_len_rollout = max_len_rollout
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.is_discrete = True
        elif isinstance(env.action_space, gym.spaces.Box):
            self.is_discrete = False
        else:
            raise NotImplementedError(
                f"Action space '{env.action_space}' not supported."
            )

    def __del__(self):
        self.env.close()

    def _rollout(self, seed: int, model, params: Params) -> float:
        observation, info = self.env.reset(seed=seed)
        total_reward = 0
        for _ in range(self.max_len_rollout):
            observation = jnp.atleast_2d(observation)
            action = model(params, observation)
            if self.is_discrete:
                action = int(jnp.argmax(action, axis=-1)[0])
            else:
                action = numpy.array(jax.nn.tanh(action)[0])
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def rollout(
        self, key: jax.Array, model, params: Params, num_rollouts: int = 1
    ) -> float:
        total_rewards = 0.0
        for _ in range(num_rollouts):
            key, subkey = jax.random.split(key=key)
            seed = int(subkey[0])
            total_rewards += self._rollout(seed=seed, model=model, params=params)
        return total_rewards / num_rollouts
