# Beginner summary: This file stores rollout data for Discrete SPO, including search-policy targets used to train the actor.
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from core_rl.buffers.base_buffer import BaseBuffer


@dataclass(slots=True)
class SPORolloutBatch:
    """
    Full SPO rollout represented as dense arrays.

    Compared to PPO rollout data, SPO also stores `search_policies`:
    the action distribution produced by search at each step.
    """

    states: np.ndarray  # Shape: (T, state_dim)
    actions: np.ndarray  # Shape: (T,), executed env actions
    rewards: np.ndarray  # Shape: (T,)
    dones: np.ndarray  # Shape: (T,), 1.0 if terminal else 0.0
    log_probs: np.ndarray  # Shape: (T,), log-prob under actor policy (for diagnostics)
    values: np.ndarray  # Shape: (T,), value estimates used by GAE
    returns: np.ndarray  # Shape: (T,), critic targets
    advantages: np.ndarray  # Shape: (T,), GAE advantages
    search_policies: np.ndarray  # Shape: (T, action_dim), search-improved policy targets


@dataclass(slots=True)
class SPORolloutMiniBatch:
    """Mini-batch slice sampled from a full SPO rollout."""

    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    search_policies: np.ndarray


class SPORolloutBuffer(BaseBuffer):
    """
    On-policy rollout buffer for Discrete SPO.

    SPO collects fresh trajectories, computes GAE/returns, and then updates
    networks using search policy targets.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Reset all stored trajectory data."""
        self._states: list[np.ndarray] = []
        self._actions: list[int] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._search_policies: list[np.ndarray] = []
        self._returns: np.ndarray | None = None
        self._advantages: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        search_policy: np.ndarray,
    ) -> None:
        """Store one SPO transition."""
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))
        self._search_policies.append(np.asarray(search_policy, dtype=np.float32))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns from collected rollout data."""
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

        self._advantages = advantages.astype(np.float32)
        self._returns = (advantages + values).astype(np.float32)

    def get_batch(self) -> SPORolloutBatch:
        """Return full rollout arrays after returns/advantages were computed."""
        if self._returns is None or self._advantages is None:
            raise ValueError("call compute_returns_and_advantages() before get_batch()")

        return SPORolloutBatch(
            states=np.asarray(self._states, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.int64),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.float32),
            log_probs=np.asarray(self._log_probs, dtype=np.float32),
            values=np.asarray(self._values, dtype=np.float32),
            returns=self._returns,
            advantages=self._advantages,
            search_policies=np.asarray(self._search_policies, dtype=np.float32),
        )

    def iter_minibatches(
        self,
        batch: SPORolloutBatch,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> list[SPORolloutMiniBatch]:
        """Split a full rollout into mini-batches for SPO optimization."""
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be > 0")
        num_samples = batch.states.shape[0]
        if num_samples == 0:
            raise ValueError("rollout batch is empty")

        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)

        minibatches: list[SPORolloutMiniBatch] = []
        for start in range(0, num_samples, minibatch_size):
            idx = indices[start : start + minibatch_size]
            minibatches.append(
                SPORolloutMiniBatch(
                    states=batch.states[idx],
                    actions=batch.actions[idx],
                    returns=batch.returns[idx],
                    advantages=batch.advantages[idx],
                    search_policies=batch.search_policies[idx],
                )
            )
        return minibatches

    def __len__(self) -> int:
        return len(self._states)
