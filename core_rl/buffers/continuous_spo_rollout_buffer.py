# Beginner summary: This file stores rollout data for Continuous SPO, including sampled particle actions and their search weights.
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from core_rl.buffers.base_buffer import BaseBuffer


@dataclass(slots=True)
class ContinuousSPORolloutBatch:
    """
    Full continuous SPO rollout represented as dense arrays.

    Continuous SPO stores:
    - sampled particle actions from search
    - normalized weights for those particles
    so actor can imitate the search distribution.
    """

    states: np.ndarray  # Shape: (T, state_dim)
    actions: np.ndarray  # Shape: (T, action_dim), env actions actually executed
    rewards: np.ndarray  # Shape: (T,)
    dones: np.ndarray  # Shape: (T,), 1.0 if terminal else 0.0
    log_probs: np.ndarray  # Shape: (T,), actor log-prob of executed actions (diagnostics)
    values: np.ndarray  # Shape: (T,), value estimates used for GAE
    returns: np.ndarray  # Shape: (T,), critic targets
    advantages: np.ndarray  # Shape: (T,), GAE advantages
    sampled_actions: np.ndarray  # Shape: (T, num_particles, action_dim), search particles
    sampled_action_weights: np.ndarray  # Shape: (T, num_particles), normalized search weights


@dataclass(slots=True)
class ContinuousSPORolloutMiniBatch:
    """Mini-batch slice sampled from continuous SPO rollout."""

    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    sampled_actions: np.ndarray
    sampled_action_weights: np.ndarray


class ContinuousSPORolloutBuffer(BaseBuffer):
    """
    On-policy rollout buffer for Continuous SPO.

    Similar to PPO/SPO buffers:
    1) collect one rollout
    2) compute returns + advantages
    3) train actor/critic for multiple epochs
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Reset all stored trajectory data."""
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._sampled_actions: list[np.ndarray] = []
        self._sampled_action_weights: list[np.ndarray] = []
        self._returns: np.ndarray | None = None
        self._advantages: np.ndarray | None = None

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        sampled_actions: np.ndarray,
        sampled_action_weights: np.ndarray,
    ) -> None:
        """Store one continuous SPO transition."""
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))
        self._sampled_actions.append(np.asarray(sampled_actions, dtype=np.float32))
        self._sampled_action_weights.append(np.asarray(sampled_action_weights, dtype=np.float32))

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and returns from rollout."""
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

    def get_batch(self) -> ContinuousSPORolloutBatch:
        """Return full rollout arrays after returns/advantages were computed."""
        if self._returns is None or self._advantages is None:
            raise ValueError("call compute_returns_and_advantages() before get_batch()")

        return ContinuousSPORolloutBatch(
            states=np.asarray(self._states, dtype=np.float32),
            actions=np.asarray(self._actions, dtype=np.float32),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.float32),
            log_probs=np.asarray(self._log_probs, dtype=np.float32),
            values=np.asarray(self._values, dtype=np.float32),
            returns=self._returns,
            advantages=self._advantages,
            sampled_actions=np.asarray(self._sampled_actions, dtype=np.float32),
            sampled_action_weights=np.asarray(self._sampled_action_weights, dtype=np.float32),
        )

    def iter_minibatches(
        self,
        batch: ContinuousSPORolloutBatch,
        minibatch_size: int,
        shuffle: bool = True,
    ) -> list[ContinuousSPORolloutMiniBatch]:
        """Split full rollout into mini-batches for optimization."""
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be > 0")
        num_samples = batch.states.shape[0]
        if num_samples == 0:
            raise ValueError("rollout batch is empty")

        indices = list(range(num_samples))
        if shuffle:
            random.shuffle(indices)

        minibatches: list[ContinuousSPORolloutMiniBatch] = []
        for start in range(0, num_samples, minibatch_size):
            idx = indices[start : start + minibatch_size]
            minibatches.append(
                ContinuousSPORolloutMiniBatch(
                    states=batch.states[idx],
                    actions=batch.actions[idx],
                    returns=batch.returns[idx],
                    advantages=batch.advantages[idx],
                    sampled_actions=batch.sampled_actions[idx],
                    sampled_action_weights=batch.sampled_action_weights[idx],
                )
            )
        return minibatches

    def __len__(self) -> int:
        return len(self._states)
