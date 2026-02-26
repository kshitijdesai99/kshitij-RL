# Beginner summary: This file smoke-tests PPO action selection and one update pass.
from gymnasium import spaces
import numpy as np
import torch

from core_rl.agents.ppo import PPOAgent, PPOConfig
from core_rl.buffers.rollout_buffer import RolloutBatch


def test_ppo_select_action_returns_valid_discrete_action():
    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        config=PPOConfig(),
        device=torch.device("cpu"),
    )
    action = agent.select_action(np.zeros(4, dtype=np.float32), deterministic=False)
    assert isinstance(action, int)
    assert action in (0, 1)


def test_ppo_update_returns_finite_metrics():
    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        config=PPOConfig(update_epochs=2, minibatch_size=8),
        device=torch.device("cpu"),
    )

    num_samples = 32
    states = np.random.randn(num_samples, 4).astype(np.float32)
    actions = np.random.randint(0, 2, size=num_samples).astype(np.int64)
    rewards = np.random.randn(num_samples).astype(np.float32)
    dones = np.zeros(num_samples, dtype=np.float32)
    log_probs = np.random.uniform(-2.0, -0.1, size=num_samples).astype(np.float32)
    values = np.random.randn(num_samples).astype(np.float32)
    advantages = np.random.randn(num_samples).astype(np.float32)
    returns = (values + advantages).astype(np.float32)

    batch = RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        returns=returns,
        advantages=advantages,
    )

    metrics = agent.update(batch)
    for key in ("policy_loss", "value_loss", "entropy", "loss", "approx_kl", "clip_fraction"):
        assert key in metrics
        assert np.isfinite(metrics[key])


def test_ppo_select_action_returns_valid_continuous_action():
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
    agent = PPOAgent(
        state_dim=3,
        action_space=action_space,
        config=PPOConfig(),
        device=torch.device("cpu"),
    )
    action = agent.select_action(np.zeros(3, dtype=np.float32), deterministic=False)
    assert isinstance(action, np.ndarray)
    assert action.shape == (1,)
    assert action.dtype == np.float32
    assert float(action[0]) >= -2.0
    assert float(action[0]) <= 2.0


def test_ppo_continuous_update_returns_finite_metrics():
    action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
    agent = PPOAgent(
        state_dim=3,
        action_space=action_space,
        config=PPOConfig(update_epochs=2, minibatch_size=8),
        device=torch.device("cpu"),
    )

    num_samples = 32
    states = np.random.randn(num_samples, 3).astype(np.float32)
    actions = np.random.uniform(-2.0, 2.0, size=(num_samples, 1)).astype(np.float32)
    rewards = np.random.randn(num_samples).astype(np.float32)
    dones = np.zeros(num_samples, dtype=np.float32)
    log_probs = np.random.uniform(-2.0, -0.1, size=num_samples).astype(np.float32)
    values = np.random.randn(num_samples).astype(np.float32)
    advantages = np.random.randn(num_samples).astype(np.float32)
    returns = (values + advantages).astype(np.float32)

    batch = RolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        returns=returns,
        advantages=advantages,
    )

    metrics = agent.update(batch)
    for key in ("policy_loss", "value_loss", "entropy", "loss", "approx_kl", "clip_fraction"):
        assert key in metrics
        assert np.isfinite(metrics[key])
