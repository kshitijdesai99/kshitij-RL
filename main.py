# Beginner summary: This file is the main entry point that dispatches to per-algorithm trainer modules and plots results.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from core_rl.trainers import train_dqn, train_ppo, train_spo, train_spo_continuous
from core_rl.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents with DQN, PPO, Discrete SPO, or Continuous SPO."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "ppo", "spo", "spo_continuous"],
        help="Algorithm to train: dqn, ppo, spo, or spo_continuous.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment id. Example: CartPole-v1 or Pendulum-v1",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Pick the best available torch device."""
    # Priority order:
    # 1) CUDA GPU (NVIDIA)
    # 2) MPS GPU (Apple Silicon)
    # 3) CPU fallback
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def main() -> None:
    """Train selected algorithm on a Gymnasium env, save checkpoints, and plot metrics."""
    args = parse_args()

    # Create app logger. DEBUG prints per-eval training progress from runners.
    logger = get_logger("core_rl.main")
    logger.setLevel(logging.DEBUG)

    # Build environment and create checkpoint directory.
    env = gym.make(args.env_id)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if args.algo == "dqn":
        result = train_dqn(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    elif args.algo == "ppo":
        result = train_ppo(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    elif args.algo == "spo":
        result = train_spo(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    else:
        result = train_spo_continuous(
            env=env,
            device=device,
            logger=logger,
            checkpoints_dir=checkpoints_dir,
        )

    metrics = result["metrics"]
    best_eval_reward = float(metrics["best_eval_reward"])
    logger.info("Best evaluation reward: %.1f", best_eval_reward)

    # Plot training reward and periodic evaluation reward.
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["train_rewards"], label="Train Reward", alpha=0.6)
    plt.plot(metrics["eval_episodes"], metrics["eval_rewards"], label="Eval Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(str(result["title"]))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
