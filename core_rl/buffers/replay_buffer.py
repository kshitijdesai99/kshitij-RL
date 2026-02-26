# Beginner summary: This file stores past transitions and samples random mini-batches for DQN training.
from __future__ import annotations

from collections import deque  # Efficient append/pop from both ends.
from dataclasses import dataclass
import random

import numpy as np

from core_rl.buffers.base_buffer import BaseBuffer


@dataclass(slots=True)
class ReplayBatch:
    """
    Mini-batch sampled from replay memory.

    This dataclass just gives names to each array so training code is easier to read.
    """

    states: np.ndarray  # Shape: (batch_size, state_dim), e.g. (64, 4) for CartPole.
    actions: np.ndarray  # Shape: (batch_size,), e.g. [0, 1, 1, 0, ...].
    rewards: np.ndarray  # Shape: (batch_size,), one reward per transition.
    next_states: np.ndarray  # Shape: (batch_size, state_dim), state after taking action.
    dones: np.ndarray  # Shape: (batch_size,), 1.0 if episode ended else 0.0.


class ReplayBuffer(BaseBuffer):
    """
    Replay memory for off-policy algorithms like DQN.

    Why replay memory exists:
    - Online RL data is highly correlated in time (t, t+1, t+2 ...).
    - Neural networks train better on shuffled i.i.d.-like batches.
    - So we store transitions and sample randomly later.
    """

    def __init__(self, capacity: int):
        # capacity is the max number of transitions kept in memory.
        # Once full, oldest transitions are automatically removed.
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        # Each entry is (state, action, reward, next_state, done).

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store one transition from environment interaction.

        Example:
        - state      = [cart_pos, cart_vel, pole_angle, pole_vel]
        - action     = 0 or 1
        - reward     = 1.0
        - next_state = next observation after action
        - done       = True if episode terminated/truncated
        """
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> ReplayBatch:
        """
        Uniformly sample random transitions and return them as numpy arrays.

        Important:
        - This is random over the whole buffer, not the latest sequence.
        - That decorrelates samples and improves training stability.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self._buffer) < batch_size:
            raise ValueError("not enough samples in buffer")

        # random.sample picks unique indices without replacement.
        # Example: from 10,000 stored transitions, pick 64 random ones.
        batch = random.sample(self._buffer, batch_size)

        # batch is a list of tuples: [(s1,a1,r1,s1',d1), (s2,a2,r2,s2',d2), ...]
        # zip(*batch) "unzips" this into 5 separate tuples:
        # s = (s1, s2, s3, ...)  # all states
        # a = (a1, a2, a3, ...)  # all actions  
        # r = (r1, r2, r3, ...)  # all rewards
        # s_next = (s1', s2', s3', ...)  # all next states
        # done = (d1, d2, d3, ...)  # all done flags
        s, a, r, s_next, done = zip(*batch)

        # Convert each tuple/list into a dense numpy array with explicit dtypes.
        # Explicit dtypes avoid implicit casting surprises later in torch tensors.
        return ReplayBatch(
            states=np.asarray(s, dtype=np.float32), # float32: ~7 significant digits, ML standard
            actions=np.asarray(a, dtype=np.int64), # int64: large range for action indices
            rewards=np.asarray(r, dtype=np.float32), # float32: ~7 significant digits, ML standard
            next_states=np.asarray(s_next, dtype=np.float32), # float32: ~7 significant digits, ML standard
            dones=np.asarray(done, dtype=np.float32), # float32: ~7 significant digits, ML standard
        )

    def __len__(self) -> int:
        # Enables `len(buffer)` usage.
        return len(self._buffer)
