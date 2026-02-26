# Beginner summary: This file is the main entry point that trains DQN or PPO on CartPole, saves checkpoints, and plots progress.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from core_rl import (
    OffPolicyRunner,
    OffPolicyRunnerConfig,
    OnPolicyRunner,
    OnPolicyRunnerConfig,
    PPOAgent,
    PPOConfig,
    ReplayBuffer,
    RolloutBuffer,
    VanillaDQNAgent,
    VanillaDQNConfig,
)
from core_rl.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train RL agents on CartPole-v1.")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "ppo"],
        help="Algorithm to train: dqn or ppo.",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Pick the best available torch device."""
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def run_inference_episode(env: gym.Env, agent: object) -> float:
    """Run one deterministic episode using a trained policy."""
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    return total_reward


def train_dqn(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Vanilla DQN and return metrics + model metadata."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = VanillaDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=VanillaDQNConfig(
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            target_update_freq=100,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / "dqn_cartpole_best.pth")
    final_model_path = str(checkpoints_dir / "dqn_cartpole_final.pth")

    buffer = ReplayBuffer(capacity=10_000)
    runner = OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=OffPolicyRunnerConfig(
            num_episodes=500,
            batch_size=64,
            eval_freq=10,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting DQN training...")
    metrics = runner.train()

    torch.save(agent.q_network.state_dict(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    agent.q_network.load_state_dict(best_state_dict)
    agent.q_network.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved DQN model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "DQN Training Progress",
    }


def train_ppo(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train PPO and return metrics + model metadata."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=PPOConfig(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            update_epochs=4,
            minibatch_size=64,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / "ppo_cartpole_best.pth")
    final_model_path = str(checkpoints_dir / "ppo_cartpole_final.pth")

    buffer = RolloutBuffer()
    runner = OnPolicyRunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=OnPolicyRunnerConfig(
            total_timesteps=60_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting PPO training...")
    metrics = runner.train()

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
        },
        final_model_path,
    )
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.actor.load_state_dict(best_state_dict["actor"])
    agent.critic.load_state_dict(best_state_dict["critic"])
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved PPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "PPO Training Progress",
    }


def main() -> None:
    """Train selected algorithm on CartPole, save checkpoints, and run final inference."""
    args = parse_args()

    # Create app logger. DEBUG prints per-eval training progress from the runner.
    logger = get_logger("core_rl.main")
    logger.setLevel(logging.DEBUG)

    # Build environment and create checkpoint directory.
    env = gym.make("CartPole-v1")
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if args.algo == "dqn":
        result = train_dqn(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    else:
        result = train_ppo(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)

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
