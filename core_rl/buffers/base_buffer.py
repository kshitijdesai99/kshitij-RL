from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseBuffer(ABC):
    """Contract for experience storage (replay buffer, rollout buffer, ...)."""

    @abstractmethod
    def add(self, *args: Any, **kwargs: Any) -> None:
        """Store one transition or one step of trajectory."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return number of stored items."""
        raise NotImplementedError
