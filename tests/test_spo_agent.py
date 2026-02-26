# Beginner summary: This file smoke-tests Discrete SPO action selection/search output and one update step.
import numpy as np
import torch

from core_rl.agents.discrete_spo import DiscreteSPOAgent, DiscreteSPOConfig
from core_rl.buffers.spo_rollout_buffer import SPORolloutBatch


def test_spo_select_action_returns_valid_action():
    agent = DiscreteSPOAgent(
        state_dim=4,
        action_dim=2,
        config=DiscreteSPOConfig(search_num_particles=8),
        device=torch.device("cpu"),
    )

    action = agent.select_action(np.zeros(4, dtype=np.float32), deterministic=False)
    assert isinstance(action, int)
    assert action in (0, 1)


def test_spo_act_returns_search_policy_distribution():
    agent = DiscreteSPOAgent(
        state_dim=4,
        action_dim=2,
        config=DiscreteSPOConfig(search_num_particles=8),
        device=torch.device("cpu"),
    )

    action, log_prob, value, search_policy = agent.act(np.zeros(4, dtype=np.float32))
    assert isinstance(action, int)
    assert np.isfinite(log_prob)
    assert np.isfinite(value)
    assert search_policy.shape == (2,)
    assert np.all(search_policy >= 0.0)
    assert np.isclose(search_policy.sum(), 1.0, atol=1e-5)


def test_spo_update_returns_finite_metrics():
    agent = DiscreteSPOAgent(
        state_dim=4,
        action_dim=2,
        config=DiscreteSPOConfig(update_epochs=2, minibatch_size=8, search_num_particles=8),
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

    # Create random policy targets and normalize rows to sum to 1.
    search_policies = np.random.uniform(0.01, 1.0, size=(num_samples, 2)).astype(np.float32)
    search_policies /= search_policies.sum(axis=1, keepdims=True)

    batch = SPORolloutBatch(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        returns=returns,
        advantages=advantages,
        search_policies=search_policies,
    )

    metrics = agent.update(batch)
    for key in ("actor_loss", "critic_loss", "entropy", "loss", "policy_kl"):
        assert key in metrics
        assert np.isfinite(metrics[key])
