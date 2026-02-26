# Beginner summary: This file checks that replay buffer sampling returns arrays with the expected shapes.
import numpy as np

from core_rl.buffers.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_shapes():
    # Create a small replay buffer and insert synthetic CartPole-like transitions.
    buffer = ReplayBuffer(capacity=10)
    for _ in range(8):
        buffer.add(np.zeros(4), 1, 1.0, np.ones(4), False)

    # Sample a mini-batch and verify that each field has the expected shape.
    batch = buffer.sample(4)
    assert batch.states.shape == (4, 4)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_states.shape == (4, 4)
    assert batch.dones.shape == (4,)
