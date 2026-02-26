# Beginner summary: This file tests rollout storage and GAE/return computations used by PPO.
import numpy as np

from core_rl.buffers.rollout_buffer import RolloutBatch, RolloutBuffer


def test_rollout_buffer_gae_returns_simple_case():
    buffer = RolloutBuffer()
    buffer.add(state=np.array([0.0], dtype=np.float32), action=0, reward=1.0, done=False, log_prob=-0.1, value=0.0)
    buffer.add(state=np.array([1.0], dtype=np.float32), action=1, reward=1.0, done=True, log_prob=-0.2, value=0.0)

    # With gamma=1, lambda=1 and terminal on step 2:
    # returns should be [2, 1], advantages match because values are 0.
    buffer.compute_returns_and_advantages(last_value=0.0, gamma=1.0, gae_lambda=1.0)
    batch = buffer.get_batch()

    assert np.allclose(batch.returns, np.array([2.0, 1.0], dtype=np.float32))
    assert np.allclose(batch.advantages, np.array([2.0, 1.0], dtype=np.float32))


def test_rollout_buffer_batch_shapes_and_minibatches():
    buffer = RolloutBuffer()
    for _ in range(10):
        buffer.add(
            state=np.zeros(4, dtype=np.float32),
            action=1,
            reward=1.0,
            done=False,
            log_prob=-0.5,
            value=0.2,
        )
    buffer.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    batch: RolloutBatch = buffer.get_batch()

    assert batch.states.shape == (10, 4)
    assert batch.actions.shape == (10,)
    assert batch.rewards.shape == (10,)
    assert batch.log_probs.shape == (10,)
    assert batch.values.shape == (10,)
    assert batch.returns.shape == (10,)
    assert batch.advantages.shape == (10,)

    minibatches = buffer.iter_minibatches(batch, minibatch_size=4, shuffle=False)
    assert len(minibatches) == 3
    assert minibatches[0].states.shape[0] == 4
    assert minibatches[1].states.shape[0] == 4
    assert minibatches[2].states.shape[0] == 2


def test_rollout_buffer_stores_continuous_actions():
    buffer = RolloutBuffer()
    for _ in range(6):
        buffer.add(
            state=np.zeros(3, dtype=np.float32),
            action=np.array([0.25], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=-0.1,
            value=0.3,
        )
    buffer.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    batch = buffer.get_batch()
    assert batch.actions.shape == (6, 1)
    assert batch.actions.dtype == np.float32
