# Beginner summary: This file runs the on-policy training loop for Continuous SPO (collect weighted particles -> update actor/critic).
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from core_rl.buffers.continuous_spo_rollout_buffer import ContinuousSPORolloutBuffer
from core_rl.utils.logger import get_logger
from core_rl.utils.metrics import evaluate_policy


@dataclass(slots=True)
class ContinuousSPORunnerConfig:
    """
    Training-loop controls for Continuous SPO.

    Each cycle:
    1) collect a fresh rollout with search particle info
    2) update actor/critic for multiple epochs
    """

    total_timesteps: int = 80_000  # Total environment interaction budget.
    rollout_steps: int = 1_024  # Steps collected before each update cycle.
    eval_freq: int = 5  # Evaluate every N update cycles.
    eval_episodes: int = 5  # Number of deterministic episodes per evaluation.


class ContinuousSPORunner:
    """Orchestrates Continuous SPO training."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        buffer: ContinuousSPORolloutBuffer,
        config: ContinuousSPORunnerConfig | None = None,
        logger: Any | None = None,
        best_model_path: str | None = None,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config or ContinuousSPORunnerConfig()
        self.logger = logger or get_logger("core_rl.continuous_spo_runner")
        self.best_model_path = best_model_path

    def train(self) -> dict[str, Any]:
        """
        Run full Continuous SPO training loop and return tracked metrics.

        Returns keys:
        - train_rewards: episode rewards during training
        - eval_rewards: periodic deterministic evaluation rewards
        - eval_episodes: episode indices where eval was run
        - best_eval_reward: best evaluation score seen
        """
        train_rewards: list[float] = []
        eval_rewards: list[float] = []
        eval_episodes: list[int] = []
        best_eval_reward = float("-inf")

        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        timesteps = 0
        update_count = 0

        while timesteps < self.config.total_timesteps:
            # 1) Collect one fresh rollout.
            self.buffer.clear()

            for _ in range(self.config.rollout_steps):
                action, log_prob, value, sampled_actions, sampled_action_weights = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    sampled_actions=sampled_actions,
                    sampled_action_weights=sampled_action_weights,
                )

                episode_reward += reward
                timesteps += 1
                state = next_state

                if done:
                    train_rewards.append(float(episode_reward))
                    episode_reward = 0.0
                    episode_count += 1
                    state, _ = self.env.reset()

                if timesteps >= self.config.total_timesteps:
                    break

            # 2) Compute GAE + returns and update.
            last_value = self.agent.estimate_value(state)
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.agent.config.gamma,
                gae_lambda=self.agent.config.gae_lambda,
            )
            batch = self.buffer.get_batch()
            update_metrics = self.agent.update(batch)
            update_count += 1

            # 3) Periodic evaluation + checkpointing.
            if update_count % self.config.eval_freq == 0:
                avg_eval_reward = evaluate_policy(self.env, self.agent, self.config.eval_episodes)
                eval_rewards.append(avg_eval_reward)
                eval_episodes.append(episode_count)
                save_marker = ""

                if avg_eval_reward >= best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    save_marker = " ✅ (New Best Eval)"
                    if self.best_model_path is not None:
                        torch.save(self.agent.get_checkpoint(), self.best_model_path)

                self.logger.debug(
                    "Update %s | Episode %s | Timesteps %s | Eval: %.1f | Actor Loss: %.4f | Critic Loss: %.4f%s",
                    update_count,
                    episode_count,
                    timesteps,
                    avg_eval_reward,
                    update_metrics["actor_loss"],
                    update_metrics["critic_loss"],
                    save_marker,
                )

                # Re-sync env state because evaluate_policy steps the same env.
                state, _ = self.env.reset()
                episode_reward = 0.0

        return {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_episodes": eval_episodes,
            "best_eval_reward": best_eval_reward,
        }
