# Beginner summary: This file contains the Discrete SPO training routine used by main.py.
from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import torch

from core_rl.agents.discrete_spo import DiscreteSPOAgent, DiscreteSPOConfig
from core_rl.buffers.discrete_spo_rollout_buffer import SPORolloutBuffer
from core_rl.runners.discrete_spo_runner import SPORunner, SPORunnerConfig
from core_rl.trainers.common import get_env_tag, run_inference_episode


def train_spo(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Discrete SPO and return metrics + plotting metadata."""
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("Discrete SPO only supports discrete action spaces")
    if len(env.observation_space.shape) != 1:
        raise ValueError("Discrete SPO example currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env_tag = get_env_tag(env)

    agent = DiscreteSPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=DiscreteSPOConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            update_epochs=4,
            minibatch_size=64,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            search_num_particles=32,
            search_temperature=1.0,
            root_dirichlet_alpha=0.3,
            root_dirichlet_fraction=0.25,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / f"spo_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"spo_{env_tag}_final.pth")

    buffer = SPORolloutBuffer()
    runner = SPORunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=SPORunnerConfig(
            total_timesteps=60_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting Discrete SPO training...")
    metrics = runner.train()

    torch.save(agent.get_checkpoint(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved SPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "Discrete SPO Training Progress",
    }
