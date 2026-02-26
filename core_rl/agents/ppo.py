# Beginner summary: This file implements a beginner-friendly PPO agent for both discrete and continuous action spaces.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

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
    init_log_std: float = -0.5
    min_log_std: float = -20.0
    max_log_std: float = 2.0


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent for discrete and continuous actions.

    PPO keeps policy updates stable by clipping policy-ratio changes.
    """

    def __init__(
        self,
        state_dim: int,
        config: PPOConfig,
        device: torch.device,
        action_space: Any | None = None,
        action_dim: int | None = None,
    ):
        self.config = config
        self.device = device
        self.log_std: nn.Parameter | None = None
        self.action_low: torch.Tensor | None = None
        self.action_high: torch.Tensor | None = None

        # Backward-compatible construction:
        # - Discrete only: pass action_dim
        # - Automatic discrete/continuous handling: pass action_space
        if action_space is not None:
            if isinstance(action_space, spaces.Discrete):
                self.is_continuous = False
                self.action_dim = int(action_space.n)
            elif isinstance(action_space, spaces.Box):
                if len(action_space.shape) != 1:
                    raise ValueError("continuous action space must be 1D")
                self.is_continuous = True
                self.action_dim = int(action_space.shape[0])
                self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
                self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
            else:
                raise TypeError("unsupported action_space type for PPO")
        elif action_dim is not None:
            self.is_continuous = False
            self.action_dim = int(action_dim)
        else:
            raise ValueError("provide either action_space or action_dim")

        # Actor outputs either:
        # - logits for discrete actions
        # - means for continuous actions
        self.actor = MLP(state_dim, self.action_dim).to(self.device)
        self.critic = MLP(state_dim, 1).to(self.device)
        if self.is_continuous:
            self.log_std = nn.Parameter(
                torch.full((self.action_dim,), self.config.init_log_std, dtype=torch.float32, device=self.device)
            )

        optimizer_params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.log_std is not None:
            optimizer_params.append(self.log_std)
        self.optimizer = optim.Adam(optimizer_params, lr=self.config.lr)
        self.value_loss_fn = nn.MSELoss()

    def _to_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert one numpy observation into a batched tensor."""
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_distribution(self, states: torch.Tensor) -> Categorical | Normal:
        """Build the action distribution for one batch of states."""
        actor_output = self.actor(states)
        if self.is_continuous:
            if self.log_std is None:
                raise RuntimeError("continuous PPO must define log_std")
            clipped_log_std = torch.clamp(self.log_std, self.config.min_log_std, self.config.max_log_std)
            std = torch.exp(clipped_log_std).expand_as(actor_output)
            return Normal(actor_output, std)
        return Categorical(logits=actor_output)

    def _clip_continuous_action(self, action_tensor: torch.Tensor) -> torch.Tensor:
        """Clamp continuous actions to environment bounds."""
        if self.action_low is None or self.action_high is None:
            raise RuntimeError("continuous action bounds are not set")
        return torch.clamp(action_tensor, min=self.action_low, max=self.action_high)

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int | np.ndarray:
        """
        Return action for env stepping.

        - Discrete: returns int action id.
        - Continuous: returns np.ndarray action vector.
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            if self.is_continuous:
                dist = self._get_distribution(state_tensor)
                if deterministic:
                    action = self.actor(state_tensor)
                else:
                    action = dist.sample()
                action = self._clip_continuous_action(action)
                return action.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                logits = self.actor(state_tensor)
                if deterministic:
                    action = logits.argmax(dim=1)
                else:
                    dist = self._get_distribution(state_tensor)
                    action = dist.sample()
                return int(action.item())

    def act(self, state: np.ndarray) -> tuple[int | np.ndarray, float, float]:
        """
        Sample one action and also return log-prob + value.

        PPO needs all three values to build rollout training targets.
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
            dist = self._get_distribution(state_tensor)
            action = dist.sample()
            if self.is_continuous:
                action = self._clip_continuous_action(action)
                log_prob = dist.log_prob(action).sum(dim=-1)
                action_out = action.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                log_prob = dist.log_prob(action)
                action_out = int(action.item())
        return action_out, float(log_prob.item()), float(value.item())

    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate V(state) for rollout bootstrapping."""
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            value = self.critic(state_tensor).squeeze(-1)
        return float(value.item())

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run PPO optimization over rollout data for several epochs."""
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        if self.is_continuous:
            actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        else:
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

                dist = self._get_distribution(mb_states)
                if self.is_continuous:
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
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

    def get_checkpoint(self) -> dict[str, Any]:
        """Return model parameters needed to resume/evaluate PPO agent."""
        checkpoint: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "is_continuous": self.is_continuous,
        }
        if self.log_std is not None:
            checkpoint["log_std"] = self.log_std.detach().cpu()
        return checkpoint

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load PPO model parameters from checkpoint dictionary."""
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.is_continuous and self.log_std is not None and "log_std" in checkpoint:
            loaded_log_std = torch.as_tensor(checkpoint["log_std"], dtype=torch.float32, device=self.device)
            self.log_std.data.copy_(loaded_log_std)
