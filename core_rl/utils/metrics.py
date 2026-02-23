from __future__ import annotations

from typing import Any


def evaluate_policy(env: Any, agent: Any, episodes: int = 5) -> float:
    """
    Run greedy evaluation (no exploration noise) and return mean reward.
    
    Multiple episodes account for:
    - State space randomness (different starts)
    - Transition randomness (probabilistic outcomes)  
    - Environmental stochasticity (noise, physics)
    """
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    total_reward = 0.0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            # deterministic=True means use the current best action from the agent.
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
    return total_reward / episodes
