# Beginner summary: This file contains the PPO training routine used by main.py.
from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import torch

from core_rl.agents.ppo import PPOAgent, PPOConfig
from core_rl.buffers.rollout_buffer import RolloutBuffer
from core_rl.runners.on_policy import OnPolicyRunner, OnPolicyRunnerConfig
from core_rl.trainers.common import get_env_tag, run_inference_episode


def train_ppo(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train PPO and return metrics + plotting metadata."""
    # PPO implementation here expects flat vector observations.
    if len(env.observation_space.shape) != 1:
        raise ValueError("PPO example currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    env_tag = get_env_tag(env)

    agent = PPOAgent(
        state_dim=state_dim,
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
        action_space=env.action_space,
    )
    best_model_path = str(checkpoints_dir / f"ppo_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"ppo_{env_tag}_final.pth")

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

    torch.save(agent.get_checkpoint(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved PPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "PPO Training Progress",
    }
