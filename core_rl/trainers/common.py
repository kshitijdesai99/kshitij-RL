# Beginner summary: This file contains shared helper functions used by multiple trainer modules.
from __future__ import annotations

import gymnasium as gym


def get_env_tag(env: gym.Env) -> str:
    """Return a filesystem-safe tag for the current environment id."""
    # env.spec.id might be values like "CartPole-v1" or "Pendulum-v1".
    # We convert to lowercase + replace separators for clean filenames.
    env_id = getattr(getattr(env, "spec", None), "id", "env")
    return str(env_id).lower().replace("-", "_").replace("/", "_")


def run_inference_episode(env: gym.Env, agent: object) -> float:
    """Run one deterministic inference episode and return total reward."""
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
