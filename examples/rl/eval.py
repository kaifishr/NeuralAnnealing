import argparse
import pickle
import pathlib

import jax
import gymnasium as gym
import jax.numpy as jnp

from src.model import model
from src.custom_types import Params


def test(
    env: gym.Env,
    model,
    params: Params,
    num_rollouts: int = 1,
) -> None:

    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
    elif isinstance(env.action_space, gym.spaces.Box):
        is_discrete = False
    else:
        raise NotImplementedError(f"Action space '{env.action_space}' not supported.")

    for _ in range(num_rollouts):
        observation, info = env.reset()
        total_reward = 0
        total_steps = 0
        is_done = False
        while not is_done:
            observation = jnp.atleast_2d(observation)
            prediction = model(params, observation)
            if is_discrete:
                action = jnp.argmax(prediction, axis=-1).item()
            else:
                action = jax.nn.tanh(prediction)[0].tolist()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            total_steps += 1
            if terminated or truncated:
                print(f"{total_reward = :.4f} {total_steps = }")
                break


def load_model_checkpoint(ckpt_dir: pathlib.Path):
    with open(ckpt_dir, "rb") as fp:
        params = pickle.load(fp)
    return params


def argument_parser() -> None:
    parser = argparse.ArgumentParser(description="Process a string input.")
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of gym environment.",
        default="LunarLander-v3",
        choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v3", "BipedalWalker-v3"],
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./output/ckpt.pkl",
        help="Directory of model checkpoint.",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=4, help="Number of environment rollouts."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    params = load_model_checkpoint(ckpt_dir=args.ckpt_dir)
    env = gym.make(args.env_name, render_mode="human")
    test(env=env, model=model, params=params, num_rollouts=args.num_rollouts)
    env.close()
