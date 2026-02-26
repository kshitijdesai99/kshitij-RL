# Beginner summary: This file tests Continuous SPO rollout buffer math and tensor shapes.
import numpy as np

from core_rl.buffers.continuous_spo_rollout_buffer import (
    ContinuousSPORolloutBatch,
    ContinuousSPORolloutBuffer,
)


def test_continuous_spo_rollout_buffer_gae_returns_simple_case():
    buffer = ContinuousSPORolloutBuffer()
    # Two-step toy trajectory with terminal at second step.
    buffer.add(
        state=np.array([0.0], dtype=np.float32),
        action=np.array([0.1], dtype=np.float32),
        reward=1.0,
        done=False,
        log_prob=-0.1,
        value=0.0,
        sampled_actions=np.array([[0.1], [0.2]], dtype=np.float32),
        sampled_action_weights=np.array([0.7, 0.3], dtype=np.float32),
    )
    buffer.add(
        state=np.array([1.0], dtype=np.float32),
        action=np.array([0.2], dtype=np.float32),
        reward=1.0,
        done=True,
        log_prob=-0.2,
        value=0.0,
        sampled_actions=np.array([[0.3], [0.4]], dtype=np.float32),
        sampled_action_weights=np.array([0.4, 0.6], dtype=np.float32),
    )

    buffer.compute_returns_and_advantages(last_value=0.0, gamma=1.0, gae_lambda=1.0)
    batch = buffer.get_batch()

    assert np.allclose(batch.returns, np.array([2.0, 1.0], dtype=np.float32))
    assert np.allclose(batch.advantages, np.array([2.0, 1.0], dtype=np.float32))


def test_continuous_spo_rollout_buffer_shapes_and_minibatches():
    buffer = ContinuousSPORolloutBuffer()
    for _ in range(10):
        buffer.add(
            state=np.zeros(3, dtype=np.float32),
            action=np.array([0.1], dtype=np.float32),
            reward=1.0,
            done=False,
            log_prob=-0.4,
            value=0.2,
            sampled_actions=np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
            sampled_action_weights=np.array([0.2, 0.5, 0.3], dtype=np.float32),
        )

    buffer.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    batch: ContinuousSPORolloutBatch = buffer.get_batch()

    assert batch.states.shape == (10, 3)
    assert batch.actions.shape == (10, 1)
    assert batch.rewards.shape == (10,)
    assert batch.log_probs.shape == (10,)
    assert batch.values.shape == (10,)
    assert batch.returns.shape == (10,)
    assert batch.advantages.shape == (10,)
    assert batch.sampled_actions.shape == (10, 3, 1)
    assert batch.sampled_action_weights.shape == (10, 3)

    minibatches = buffer.iter_minibatches(batch, minibatch_size=4, shuffle=False)
    assert len(minibatches) == 3
    assert minibatches[0].states.shape[0] == 4
    assert minibatches[1].states.shape[0] == 4
    assert minibatches[2].states.shape[0] == 2
