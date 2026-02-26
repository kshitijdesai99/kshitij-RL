# Beginner summary: This file contains the DQN training routine used by main.py.
from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import torch

from core_rl.agents.vanilla_dqn import VanillaDQNAgent, VanillaDQNConfig
from core_rl.buffers.replay_buffer import ReplayBuffer
from core_rl.runners.off_policy import OffPolicyRunner, OffPolicyRunnerConfig
from core_rl.trainers.common import get_env_tag, run_inference_episode


def train_dqn(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Vanilla DQN and return metrics + plotting metadata."""
    # DQN only works for discrete action spaces.
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("DQN only supports discrete action spaces")
    # This starter implementation expects flat vector observations.
    if len(env.observation_space.shape) != 1:
        raise ValueError("DQN example currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env_tag = get_env_tag(env)

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
    best_model_path = str(checkpoints_dir / f"dqn_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"dqn_{env_tag}_final.pth")

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

    # Save final model and then reload the best model for inference check.
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
