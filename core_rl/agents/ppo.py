# Beginner summary: This file implements a beginner-friendly PPO agent for discrete action spaces.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from core_rl.agents.base_agent import BaseAgent
from core_rl.buffers.rollout_buffer import RolloutBatch
from core_rl.networks.mlp import MLP


@dataclass(slots=True)
class PPOConfig:
    """Hyperparameters controlling PPO optimization and regularization."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent for discrete actions.

    PPO keeps policy updates stable by clipping policy-ratio changes.
    """

    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig, device: torch.device):
        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Actor outputs logits over actions; Critic outputs scalar value.
        self.actor = MLP(state_dim, action_dim).to(self.device)
        self.critic = MLP(state_dim, 1).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.lr,
        )
        self.value_loss_fn = nn.MSELoss()

    def _to_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert one numpy observation into a batched tensor."""
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Return action for env stepping.

        deterministic=True uses argmax action for evaluation.
        deterministic=False samples from policy distribution for exploration.
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            if deterministic:
                action = logits.argmax(dim=1)
            else:
                dist = Categorical(logits=logits)
                action = dist.sample()
        return int(action.item())

    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        """
        Sample one action and also return log-prob + value.

        PPO needs all three values to build rollout training targets.
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            logits = self.actor(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor).squeeze(-1)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate V(state) for rollout bootstrapping."""
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
        return float(value.item())

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run PPO optimization over rollout data for several epochs."""
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = states.shape[0]
        if num_samples == 0:
            raise ValueError("rollout batch is empty")

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.config.minibatch_size):
                idx = permutation[start : start + self.config.minibatch_size]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                logits = self.actor(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                values = self.critic(mb_states).squeeze(-1)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )

                # PPO clipped surrogate objective.
                surrogate_1 = ratio * mb_advantages
                surrogate_2 = clipped_ratio * mb_advantages
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = self.value_loss_fn(values, mb_returns)
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                total_losses.append(float(total_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "loss": float(np.mean(total_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }
