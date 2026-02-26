# Beginner summary: This file smoke-tests Continuous SPO action selection/search output and one update step.
from gymnasium import spaces
import numpy as np
import torch

from core_rl.agents.continuous_spo import ContinuousSPOAgent, ContinuousSPOConfig
from core_rl.buffers.continuous_spo_rollout_buffer import ContinuousSPORolloutBatch


def test_continuous_spo_select_action_returns_valid_action():
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
    agent = ContinuousSPOAgent(
        state_dim=3,
        action_space=action_space,
        config=ContinuousSPOConfig(search_num_particles=8),
        device=torch.device("cpu"),
    )

    action = agent.select_action(np.zeros(3, dtype=np.float32), deterministic=False)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert float(action[0]) >= -2.0
    assert float(action[0]) <= 2.0


def test_continuous_spo_act_returns_particle_set_and_weights():
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
    agent = ContinuousSPOAgent(
        state_dim=3,
        action_space=action_space,
        config=ContinuousSPOConfig(search_num_particles=8),
        device=torch.device("cpu"),
    )

    action, log_prob, value, sampled_actions, sampled_action_weights = agent.act(
        np.zeros(3, dtype=np.float32)
    )
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)
    assert sampled_actions.shape == (8, 1)
    assert sampled_action_weights.shape == (8,)
    assert np.all(sampled_action_weights >= 0.0)
    assert np.isclose(sampled_action_weights.sum(), 1.0, atol=1e-5)


def test_continuous_spo_update_returns_finite_metrics():
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
    agent = ContinuousSPOAgent(
        state_dim=3,
        action_space=action_space,
        config=ContinuousSPOConfig(update_epochs=2, minibatch_size=8, search_num_particles=8),
        device=torch.device("cpu"),
    )

    num_samples = 32
    num_particles = 8
    states = np.random.randn(num_samples, 3).astype(np.float32)
    actions = np.random.uniform(-2.0, 2.0, size=(num_samples, 1)).astype(np.float32)
    rewards = np.random.randn(num_samples).astype(np.float32)
    dones = np.zeros(num_samples, dtype=np.float32)
    log_probs = np.random.uniform(-3.0, -0.1, size=num_samples).astype(np.float32)
    values = np.random.randn(num_samples).astype(np.float32)
    advantages = np.random.randn(num_samples).astype(np.float32)
    returns = (values + advantages).astype(np.float32)
    sampled_actions = np.random.uniform(-2.0, 2.0, size=(num_samples, num_particles, 1)).astype(np.float32)
    sampled_action_weights = np.random.uniform(0.01, 1.0, size=(num_samples, num_particles)).astype(np.float32)
    sampled_action_weights /= sampled_action_weights.sum(axis=1, keepdims=True)

    batch = ContinuousSPORolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        returns=returns,
        advantages=advantages,
        sampled_actions=sampled_actions,
        sampled_action_weights=sampled_action_weights,
    )

    metrics = agent.update(batch)
    for key in ("actor_loss", "critic_loss", "entropy", "loss", "policy_kl"):
        assert key in metrics
        assert np.isfinite(metrics[key])
