# Beginner summary: This file collects agent-related classes and makes them easy to import from the agents package.
# Makes this directory a Python package so we can import classes like: from core_rl import VanillaDQNAgent
from core_rl.agents.base_agent import BaseAgent
from core_rl.agents.vanilla_dqn import VanillaDQNAgent, VanillaDQNConfig

__all__ = [
    "BaseAgent",
    "VanillaDQNAgent",
    "VanillaDQNConfig",
]
