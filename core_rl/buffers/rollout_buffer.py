# Beginner summary: This file stores on-policy trajectory steps and computes GAE advantages/returns for PPO training.
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from core_rl.buffers.base_buffer import BaseBuffer


@dataclass(slots=True)
class RolloutBatch:
    """
    Full on-policy rollout represented as dense arrays.

    PPO uses this data exactly once (or for a few update epochs),
    then discards it and collects a fresh rollout.
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


@dataclass(slots=True)
class RolloutMiniBatch:
    """Mini-batch slice sampled from a full rollout."""

    states: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class RolloutBuffer(BaseBuffer):
    """
    On-policy buffer for PPO-style training.

    Stores one rollout of transitions in temporal order, then computes:
    - returns: discounted targets for value function
    - advantages: GAE estimates for policy updates
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Reset all stored rollout data."""
        self._states: list[np.ndarray] = []
        self._actions: list[Any] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._returns: np.ndarray | None = None
        self._advantages: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store one environment step used by PPO."""
        action_array = np.asarray(action)
        if action_array.ndim == 0:
            stored_action: Any = int(action_array.item())
        else:
            stored_action = action_array.astype(np.float32)

        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(stored_action)
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """
        Compute Generalized Advantage Estimation (GAE) and returns.

        Args:
            last_value: V(s_T) bootstrap value for the state after the final stored step.
            gamma: discount factor.
            gae_lambda: bias-variance trade-off factor used by GAE.
        """
        if len(self._states) == 0:
            raise ValueError("rollout buffer is empty")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if not (0.0 <= gae_lambda <= 1.0):
            raise ValueError("gae_lambda must be in [0, 1]")

        rewards = np.asarray(self._rewards, dtype=np.float32)
        dones = np.asarray(self._dones, dtype=np.float32)
        values = np.asarray(self._values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = float(last_value)
            else:
                next_value = float(values[t + 1])
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        self._advantages = advantages.astype(np.float32)
        self._returns = returns.astype(np.float32)

    def get_batch(self) -> RolloutBatch:
        """Return the full rollout as dense arrays (after advantages are computed)."""
        if self._returns is None or self._advantages is None:
            raise ValueError("call compute_returns_and_advantages() before get_batch()")

        first_action = self._actions[0]
        if isinstance(first_action, int):
            actions = np.asarray(self._actions, dtype=np.int64)
        else:
            actions = np.asarray(self._actions, dtype=np.float32)

        return RolloutBatch(
            states=np.asarray(self._states, dtype=np.float32),
            actions=actions,
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.float32),
            log_probs=np.asarray(self._log_probs, dtype=np.float32),
            values=np.asarray(self._values, dtype=np.float32),
            returns=self._returns,
            advantages=self._advantages,
        )

    def iter_minibatches(
        self,
        batch: RolloutBatch,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> list[RolloutMiniBatch]:
        """Split a full rollout into mini-batches for PPO optimization."""
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be > 0")
        num_samples = batch.states.shape[0]
        if num_samples == 0:
            raise ValueError("rollout batch is empty")

        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)

        minibatches: list[RolloutMiniBatch] = []
        for start in range(0, num_samples, minibatch_size):
            idx = indices[start : start + minibatch_size]
            minibatches.append(
                RolloutMiniBatch(
                    states=batch.states[idx],
                    actions=batch.actions[idx],
                    log_probs=batch.log_probs[idx],
                    returns=batch.returns[idx],
                    advantages=batch.advantages[idx],
                )
            )
        return minibatches

    def __len__(self) -> int:
        return len(self._states)
