# Beginner summary: This file implements a beginner-friendly Continuous SPO agent with particle search and weighted policy fitting.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from core_rl.agents.base_agent import BaseAgent
from core_rl.buffers.continuous_spo_rollout_buffer import ContinuousSPORolloutBatch
from core_rl.networks.mlp import MLP


@dataclass(slots=True)
class ContinuousSPOConfig:
    """Hyperparameters for Continuous Sequential Policy Optimization (SPO)."""

    actor_lr: float = 3e-4  # Learning rate for actor parameters.
    critic_lr: float = 3e-4  # Learning rate for critic parameters.
    gamma: float = 0.99  # Discount factor for future rewards.
    gae_lambda: float = 0.95  # GAE bias/variance tradeoff.
    update_epochs: int = 4  # Number of passes over each rollout.
    minibatch_size: int = 64  # Mini-batch size during updates.
    entropy_coef: float = 0.001  # Entropy regularization for exploration.
    max_grad_norm: float = 0.5  # Gradient clipping threshold.
    normalize_advantages: bool = True  # Normalize advantages before actor update.
    use_advantage_weights: bool = True  # Weight actor loss more on good samples.
    search_num_particles: int = 64  # Number of sampled actions in root search.
    search_temperature: float = 1.0  # Softmax temperature for particle reweighting.
    root_exploration_fraction: float = 0.1  # Blend factor for extra root action noise.
    init_log_std: float = -0.5  # Initial log std for Gaussian actor.
    min_log_std: float = -5.0  # Lower clamp for log std.
    max_log_std: float = 2.0  # Upper clamp for log std.
    value_num_samples: int = 16  # Samples used to estimate V(s) from Q(s,a).
    squash_epsilon: float = 1e-6  # Numerical epsilon for tanh inverse/log corrections.


class ContinuousSPOAgent(BaseAgent):
    """
    Continuous SPO agent.

    High-level idea:
    1) Actor defines a tanh-squashed Gaussian policy for bounded continuous actions.
    2) Critic predicts Q(s, a).
    3) Search samples many candidate actions, scores them with critic, and produces
       normalized particle weights.
    4) Actor is trained to fit weighted particle actions; critic is fit to returns.
    """

    def __init__(
        self,
        state_dim: int,
        action_space: spaces.Box,
        config: ContinuousSPOConfig,
        device: torch.device,
    ):
        if len(action_space.shape) != 1:
            raise ValueError("continuous SPO expects 1D continuous action space")
        if config.search_num_particles <= 0:
            raise ValueError("search_num_particles must be > 0")
        if config.search_temperature <= 0.0:
            raise ValueError("search_temperature must be > 0")
        if config.value_num_samples <= 0:
            raise ValueError("value_num_samples must be > 0")

        self.config = config
        self.device = device
        self.action_dim = int(action_space.shape[0])

        # Action bounds are needed for squashing and clipping.
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.action_low = action_low
        self.action_high = action_high
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        # Actor predicts Gaussian means; log_std is a learned global parameter.
        self.actor = MLP(state_dim, self.action_dim).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), self.config.init_log_std, dtype=torch.float32, device=self.device)
        )

        # Critic predicts Q(s, a), so input is concatenated [state, action].
        self.critic = MLP(state_dim + self.action_dim, 1).to(self.device)

        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.critic_loss_fn = nn.MSELoss()

    def _to_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert one numpy observation to a batched torch tensor."""
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _normal_dist(self, states: torch.Tensor) -> Normal:
        """Create actor Gaussian distribution in unsquashed space."""
        mean = self.actor(states)
        clipped_log_std = torch.clamp(self.log_std, self.config.min_log_std, self.config.max_log_std)
        std = torch.exp(clipped_log_std).expand_as(mean)
        return Normal(mean, std)

    def _squash_raw_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Convert unconstrained raw action to bounded env action."""
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias
        return torch.clamp(action, min=self.action_low, max=self.action_high)

    def _action_to_raw(self, action: torch.Tensor) -> torch.Tensor:
        """Map bounded action back to unsquashed space via inverse tanh."""
        normalized = (action - self.action_bias) / self.action_scale.clamp_min(self.config.squash_epsilon)
        normalized = torch.clamp(normalized, -1.0 + self.config.squash_epsilon, 1.0 - self.config.squash_epsilon)
        return 0.5 * torch.log((1.0 + normalized) / (1.0 - normalized))

    def _log_prob_from_action(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute tanh-squashed Gaussian log-prob for given bounded action."""
        dist = self._normal_dist(states)
        raw_action = self._action_to_raw(action)
        squashed = torch.tanh(raw_action)

        # Change-of-variables correction for tanh + affine action transform.
        raw_log_prob = dist.log_prob(raw_action)
        correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + self.config.squash_epsilon)
        return (raw_log_prob - correction).sum(dim=-1)

    def _sample_action_and_log_prob(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample bounded action and its log-prob from current actor."""
        dist = self._normal_dist(states)
        raw_action = dist.rsample()
        action = self._squash_raw_action(raw_action)
        log_prob = self._log_prob_from_action(states, action)
        return action, log_prob

    def _critic_q(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Evaluate critic Q(s, a) for batched states/actions."""
        critic_input = torch.cat([states, actions], dim=-1)
        return self.critic(critic_input).squeeze(-1)

    def _apply_root_exploration_noise(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply bounded root exploration noise to sampled particle actions.

        Shape:
        - input actions: (batch, num_particles, action_dim)
        - output actions: same shape
        """
        fraction = self.config.root_exploration_fraction
        if fraction <= 0.0:
            return actions

        # Build clipped noise samples within action bounds.
        noise = torch.randn_like(actions)
        noise = noise.clamp(-2.0, 2.0)
        noise = noise * (self.action_scale / 2.0) + self.action_bias
        noise = torch.clamp(noise, min=self.action_low, max=self.action_high)

        mixed = (1.0 - fraction) * actions + fraction * noise
        return torch.clamp(mixed, min=self.action_low, max=self.action_high)

    def _compute_search_outputs(
        self,
        state_tensor: torch.Tensor,
        add_exploration_noise: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute continuous SPO search outputs for a batch of states.

        Returns:
        - value_estimate: approximate V(s) from expected Q over sampled policy actions
        - sampled_actions: candidate root actions, shape (batch, num_particles, action_dim)
        - sampled_action_weights: normalized weights over particles, shape (batch, num_particles)
        - weighted_action: weighted average action from particles, shape (batch, action_dim)
        """
        batch_size = state_tensor.shape[0]
        num_particles = self.config.search_num_particles

        # Sample root candidate actions from policy.
        states_for_particles = state_tensor.repeat_interleave(num_particles, dim=0)
        particle_actions_flat, _ = self._sample_action_and_log_prob(states_for_particles)
        sampled_actions = particle_actions_flat.view(batch_size, num_particles, self.action_dim)
        if add_exploration_noise:
            sampled_actions = self._apply_root_exploration_noise(sampled_actions)

        # Score each candidate action with critic Q(s, a).
        states_flat = state_tensor.unsqueeze(1).repeat(1, num_particles, 1).reshape(-1, state_tensor.shape[-1])
        particle_scores = self._critic_q(
            states_flat,
            sampled_actions.reshape(-1, self.action_dim),
        ).view(batch_size, num_particles)
        sampled_action_weights = torch.softmax(
            particle_scores / self.config.search_temperature,
            dim=1,
        )

        weighted_action = (sampled_action_weights.unsqueeze(-1) * sampled_actions).sum(dim=1)

        # Estimate V(s) by expected Q over policy action samples.
        value_samples = self.config.value_num_samples
        states_for_value = state_tensor.repeat_interleave(value_samples, dim=0)
        value_actions_flat, _ = self._sample_action_and_log_prob(states_for_value)
        value_q = self._critic_q(states_for_value, value_actions_flat).view(batch_size, value_samples)
        value_estimate = value_q.mean(dim=1)

        return value_estimate, sampled_actions, sampled_action_weights, weighted_action

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action for environment interaction.

        deterministic=True:
        - use actor mean action (stable evaluation)
        deterministic=False:
        - sample from search-improved weighted particle actions
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            if deterministic:
                mean_action = self._squash_raw_action(self.actor(state_tensor))
                return mean_action.squeeze(0).cpu().numpy().astype(np.float32)

            _, sampled_actions, sampled_action_weights, _ = self._compute_search_outputs(
                state_tensor,
                add_exploration_noise=True,
            )
            idx = torch.multinomial(sampled_action_weights, num_samples=1).squeeze(1)
            action = sampled_actions[torch.arange(sampled_actions.shape[0], device=self.device), idx]
            return action.squeeze(0).cpu().numpy().astype(np.float32)

    def act(self, state: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        """
        Continuous SPO training-time action selection.

        Returns:
        - action: sampled action from search-weighted particle set
        - log_prob: actor log-prob of the chosen action
        - value: value estimate for GAE
        - sampled_actions: particle actions used by search
        - sampled_action_weights: normalized weights over particles
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            value, sampled_actions, sampled_action_weights, _ = self._compute_search_outputs(
                state_tensor,
                add_exploration_noise=True,
            )
            idx = torch.multinomial(sampled_action_weights, num_samples=1).squeeze(1)
            chosen_action = sampled_actions[torch.arange(sampled_actions.shape[0], device=self.device), idx]
            log_prob = self._log_prob_from_action(state_tensor, chosen_action)

        return (
            chosen_action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
            sampled_actions.squeeze(0).cpu().numpy().astype(np.float32),
            sampled_action_weights.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate V(s) for GAE bootstrap."""
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            value, _, _, _ = self._compute_search_outputs(
                state_tensor,
                add_exploration_noise=False,
            )
        return float(value.item())

    def update(self, batch: ContinuousSPORolloutBatch) -> dict[str, float]:
        """
        Update actor and critic from rollout data.

        Actor:
        - fit weighted search particle actions (cross-entropy style loss)
        - optional advantage weighting
        - entropy regularization

        Critic:
        - fit Q(s,a_executed) to return targets
        """
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        sampled_actions = torch.as_tensor(batch.sampled_actions, dtype=torch.float32, device=self.device)
        sampled_action_weights = torch.as_tensor(
            batch.sampled_action_weights, dtype=torch.float32, device=self.device
        )

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = states.shape[0]
        if num_samples == 0:
            raise ValueError("rollout batch is empty")

        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        total_losses: list[float] = []
        policy_kls: list[float] = []

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.config.minibatch_size):
                idx = permutation[start : start + self.config.minibatch_size]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_sampled_actions = sampled_actions[idx]  # (B, P, A)
                mb_sampled_action_weights = sampled_action_weights[idx]  # (B, P)

                # ------------------------------------------
                # ACTOR UPDATE (fit weighted particle actions)
                # ------------------------------------------
                batch_size = mb_states.shape[0]
                num_particles = mb_sampled_actions.shape[1]

                states_flat = mb_states.unsqueeze(1).repeat(1, num_particles, 1).reshape(
                    batch_size * num_particles, -1
                )
                actions_flat = mb_sampled_actions.reshape(batch_size * num_particles, self.action_dim)
                log_probs_flat = self._log_prob_from_action(states_flat, actions_flat)
                log_probs = log_probs_flat.view(batch_size, num_particles)

                sample_actor_loss = -(mb_sampled_action_weights * log_probs).sum(dim=1)

                if self.config.use_advantage_weights:
                    sample_weights = torch.relu(mb_advantages) + 1e-3
                    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
                    actor_loss = (sample_actor_loss * sample_weights).mean()
                else:
                    actor_loss = sample_actor_loss.mean()

                entropy = self._normal_dist(mb_states).entropy().sum(dim=-1).mean()
                actor_total_loss = actor_loss - self.config.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std], self.config.max_grad_norm)
                self.actor_optimizer.step()

                # ------------------------------------------
                # CRITIC UPDATE (fit Q(s,a))
                # ------------------------------------------
                q_pred = self._critic_q(mb_states, mb_actions)
                critic_loss = self.critic_loss_fn(q_pred, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    # KL proxy on particle set: target weights vs current induced probs.
                    current_particle_probs = torch.softmax(log_probs, dim=1).clamp_min(1e-8)
                    target_particle_probs = mb_sampled_action_weights.clamp_min(1e-8)
                    policy_kl = (
                        target_particle_probs
                        * (torch.log(target_particle_probs) - torch.log(current_particle_probs))
                    ).sum(dim=1).mean()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy.item()))
                total_losses.append(float((actor_total_loss + critic_loss).item()))
                policy_kls.append(float(policy_kl.item()))

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "loss": float(np.mean(total_losses)),
            "policy_kl": float(np.mean(policy_kls)),
        }

    def get_checkpoint(self) -> dict[str, Any]:
        """Return model parameters needed to save/resume Continuous SPO."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load Continuous SPO checkpoint dictionary."""
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        loaded_log_std = torch.as_tensor(checkpoint["log_std"], dtype=torch.float32, device=self.device)
        self.log_std.data.copy_(loaded_log_std)
