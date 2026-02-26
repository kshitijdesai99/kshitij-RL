# Beginner summary: This file contains the Continuous SPO training routine used by main.py.
from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import torch

from core_rl.agents.continuous_spo import ContinuousSPOAgent, ContinuousSPOConfig
from core_rl.buffers.continuous_spo_rollout_buffer import ContinuousSPORolloutBuffer
from core_rl.runners.continuous_spo_runner import ContinuousSPORunner, ContinuousSPORunnerConfig
from core_rl.trainers.common import get_env_tag, run_inference_episode


def train_spo_continuous(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Continuous SPO and return metrics + plotting metadata."""
    if not isinstance(env.action_space, spaces.Box):
        raise ValueError("Continuous SPO only supports continuous Box action spaces")
    if len(env.action_space.shape) != 1:
        raise ValueError("Continuous SPO currently supports 1D action vectors only")
    if len(env.observation_space.shape) != 1:
        raise ValueError("Continuous SPO currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    env_tag = get_env_tag(env)

    agent = ContinuousSPOAgent(
        state_dim=state_dim,
        action_space=env.action_space,
        config=ContinuousSPOConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            update_epochs=4,
            minibatch_size=64,
            entropy_coef=0.001,
            max_grad_norm=0.5,
            search_num_particles=64,
            search_temperature=1.0,
            root_exploration_fraction=0.1,
            init_log_std=-0.5,
            min_log_std=-5.0,
            max_log_std=2.0,
            value_num_samples=16,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / f"spo_continuous_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"spo_continuous_{env_tag}_final.pth")

    buffer = ContinuousSPORolloutBuffer()
    runner = ContinuousSPORunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=ContinuousSPORunnerConfig(
            total_timesteps=80_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting Continuous SPO training...")
    metrics = runner.train()

    torch.save(agent.get_checkpoint(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved Continuous SPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "Continuous SPO Training Progress",
    }
