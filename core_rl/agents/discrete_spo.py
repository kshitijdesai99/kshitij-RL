# Beginner summary: This file implements a beginner-friendly Discrete SPO agent with particle-based action search and actor-critic learning.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Dirichlet

from core_rl.agents.base_agent import BaseAgent
from core_rl.buffers.spo_rollout_buffer import SPORolloutBatch
from core_rl.networks.mlp import MLP


@dataclass(slots=True)
class DiscreteSPOConfig:
    """Hyperparameters for Discrete Sequential Policy Optimization (SPO)."""

    actor_lr: float = 3e-4  # Learning rate for actor network.
    critic_lr: float = 3e-4  # Learning rate for critic network.
    gamma: float = 0.99  # Discount factor for future rewards.
    gae_lambda: float = 0.95  # GAE bias/variance control.
    update_epochs: int = 4  # Number of gradient passes over each rollout.
    minibatch_size: int = 64  # Mini-batch size per epoch.
    entropy_coef: float = 0.01  # Entropy regularization to maintain exploration.
    max_grad_norm: float = 0.5  # Gradient clipping threshold.
    normalize_advantages: bool = True  # Normalize advantages before updates.
    use_advantage_weights: bool = True  # Weight actor imitation loss by advantage.
    search_num_particles: int = 32  # Number of root action particles sampled in search.
    search_temperature: float = 1.0  # Temperature for turning particle scores into weights.
    root_dirichlet_alpha: float = 0.3  # Dirichlet concentration for root exploration noise.
    root_dirichlet_fraction: float = 0.25  # Blend ratio between policy prior and Dirichlet noise.


class DiscreteSPOAgent(BaseAgent):
    """
    Discrete SPO agent.

    High-level idea:
    1) Actor predicts prior action probabilities.
    2) Critic predicts Q(s, a) scores.
    3) Search samples many action particles from a noisy prior, reweights by critic scores,
       and creates a search-improved policy.
    4) Actor is trained to imitate search policy; critic is trained from rollout returns.
    """

    def __init__(self, state_dim: int, action_dim: int, config: DiscreteSPOConfig, device: torch.device):
        if action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if config.search_num_particles <= 0:
            raise ValueError("search_num_particles must be > 0")
        if config.search_temperature <= 0.0:
            raise ValueError("search_temperature must be > 0")

        self.config = config
        self.device = device
        self.action_dim = action_dim

        # Actor outputs logits over discrete actions.
        self.actor = MLP(state_dim, action_dim).to(self.device)
        # Critic outputs one Q-value per action for the current state.
        self.critic = MLP(state_dim, action_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.value_loss_fn = nn.MSELoss()

    def _to_state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert one numpy observation to a batched torch tensor."""
        return torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _apply_root_noise(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply Dirichlet exploration noise to the root prior policy.

        This is similar in spirit to AlphaZero-style root noise:
        noisy_prior = (1-f) * prior + f * dirichlet_noise
        """
        fraction = self.config.root_dirichlet_fraction
        if fraction <= 0.0:
            return prior_probs

        concentration = torch.full_like(prior_probs, self.config.root_dirichlet_alpha)
        # PyTorch MPS currently does not implement Dirichlet sampling.
        # We therefore sample on CPU and move the sample back to the original device.
        concentration_cpu = concentration.detach().cpu()
        dirichlet_cpu = Dirichlet(concentration_cpu)
        noise = dirichlet_cpu.sample().to(prior_probs.device)
        mixed = (1.0 - fraction) * prior_probs + fraction * noise
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _compute_search_policy(
        self,
        state_tensor: torch.Tensor,
        add_exploration_noise: bool,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """
        Build search-improved action distribution for one or many states.

        Returns:
        - actor_dist: actor's original categorical distribution
        - value_estimate: policy-weighted expected value from critic
        - search_policy: probability distribution improved by particle reweighting
        """
        actor_logits = self.actor(state_tensor)
        actor_dist = Categorical(logits=actor_logits)
        actor_probs = torch.softmax(actor_logits, dim=1)

        q_values = self.critic(state_tensor)
        value_estimate = (actor_probs * q_values).sum(dim=1)

        # Start from actor prior and optionally inject root exploration noise.
        root_prior = actor_probs
        if add_exploration_noise:
            root_prior = self._apply_root_noise(root_prior)

        # Sample many root actions ("particles") from the prior distribution.
        particle_actions = torch.multinomial(
            root_prior,
            num_samples=self.config.search_num_particles,
            replacement=True,
        )

        # Score each sampled particle action using critic Q(s, a).
        particle_scores = q_values.gather(dim=1, index=particle_actions)
        particle_weights = torch.softmax(
            particle_scores / self.config.search_temperature,
            dim=1,
        )

        # Aggregate particle weights back to action space.
        search_policy = torch.zeros_like(actor_probs)
        search_policy.scatter_add_(dim=1, index=particle_actions, src=particle_weights)
        search_policy = search_policy / search_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return actor_dist, value_estimate, search_policy

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action for environment interaction.

        deterministic=True picks argmax of search policy.
        deterministic=False samples from search policy.
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            _, _, search_policy = self._compute_search_policy(
                state_tensor,
                add_exploration_noise=not deterministic,
            )
            if deterministic:
                action = search_policy.argmax(dim=1)
            else:
                action = torch.multinomial(search_policy, num_samples=1).squeeze(1)
        return int(action.item())

    def act(self, state: np.ndarray) -> tuple[int, float, float, np.ndarray]:
        """
        SPO training-time action selection.

        Returns:
        - action: sampled action from search policy
        - log_prob: actor log-prob of chosen action
        - value: state value estimate used by GAE
        - search_policy: full search distribution used as actor target
        """
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            actor_dist, value_estimate, search_policy = self._compute_search_policy(
                state_tensor,
                add_exploration_noise=True,
            )
            action = torch.multinomial(search_policy, num_samples=1).squeeze(1)
            log_prob = actor_dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value_estimate.item()),
            search_policy.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def estimate_value(self, state: np.ndarray) -> float:
        """Estimate V(s) using policy-weighted Q-values from actor and critic."""
        state_tensor = self._to_state_tensor(state)
        with torch.no_grad():
            actor_logits = self.actor(state_tensor)
            actor_probs = torch.softmax(actor_logits, dim=1)
            q_values = self.critic(state_tensor)
            value = (actor_probs * q_values).sum(dim=1)
        return float(value.item())

    def update(self, batch: SPORolloutBatch) -> dict[str, float]:
        """
        Update actor and critic using collected rollout data.

        Actor loss:
        - cross-entropy imitation of search policy targets
        - optional per-sample advantage weighting
        - entropy bonus

        Critic loss:
        - regression of Q(s, chosen_action) toward rollout returns
        """
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        search_policies = torch.as_tensor(batch.search_policies, dtype=torch.float32, device=self.device)

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
                mb_search_policies = search_policies[idx]

                # ------------------------------------------
                # ACTOR UPDATE (imitate search policy)
                # ------------------------------------------
                actor_logits = self.actor(mb_states)
                actor_log_probs = torch.log_softmax(actor_logits, dim=1)
                dist = Categorical(logits=actor_logits)
                entropy = dist.entropy().mean()

                # Cross-entropy between search policy target and actor policy.
                sample_policy_loss = -(mb_search_policies * actor_log_probs).sum(dim=1)

                if self.config.use_advantage_weights:
                    # Give more weight to higher-advantage samples.
                    # ReLU keeps weights non-negative.
                    sample_weights = torch.relu(mb_advantages) + 1e-3
                    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
                    actor_loss = (sample_policy_loss * sample_weights).mean()
                else:
                    actor_loss = sample_policy_loss.mean()

                actor_total_loss = actor_loss - self.config.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()

                # ------------------------------------------
                # CRITIC UPDATE (fit Q to returns)
                # ------------------------------------------
                q_values = self.critic(mb_states)
                chosen_q = q_values.gather(dim=1, index=mb_actions.unsqueeze(1)).squeeze(1)
                critic_loss = self.value_loss_fn(chosen_q, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    target_policy = mb_search_policies.clamp_min(1e-8)
                    policy_kl = (target_policy * (torch.log(target_policy) - actor_log_probs)).sum(dim=1).mean()

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
        """Return model parameters needed to save/resume Discrete SPO."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Load Discrete SPO checkpoint dictionary."""
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
