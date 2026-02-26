# Beginner summary: This file defines the base interface every RL agent should follow (choose actions and update from data).
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Contract for any algorithm (DQN, PPO, SAC, ...)."""

    @abstractmethod
    def select_action(self, state: Any, deterministic: bool = False) -> Any:
        """Return an action for a given state."""
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Run one learning step and return training metrics (e.g., loss)."""
        raise NotImplementedError
