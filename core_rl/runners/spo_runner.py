# Beginner summary: This file runs the on-policy training loop for Discrete SPO (collect search targets -> update actor/critic).
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from core_rl.buffers.spo_rollout_buffer import SPORolloutBuffer
from core_rl.utils.logger import get_logger
from core_rl.utils.metrics import evaluate_policy


@dataclass(slots=True)
class SPORunnerConfig:
    """
    Training-loop controls for Discrete SPO.

    SPO alternates:
    1) collect fresh on-policy rollout with search-generated policy targets
    2) optimize actor/critic on that rollout
    """

    total_timesteps: int = 60_000  # Total env interaction budget.
    rollout_steps: int = 1_024  # Steps collected before each update cycle.
    eval_freq: int = 5  # Evaluate every N update cycles.
    eval_episodes: int = 5  # Number of deterministic episodes per evaluation.


class SPORunner:
    """Orchestrates Discrete SPO training."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        buffer: SPORolloutBuffer,
        config: SPORunnerConfig | None = None,
        logger: Any | None = None,
        best_model_path: str | None = None,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config or SPORunnerConfig()
        self.logger = logger or get_logger("core_rl.spo_runner")
        self.best_model_path = best_model_path

    def train(self) -> dict[str, Any]:
        """
        Run full SPO training loop and return tracked metrics.

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
                action, log_prob, value, search_policy = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                    search_policy=search_policy,
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

            # 2) Compute GAE + returns and update the agent.
            last_value = self.agent.estimate_value(state)
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.agent.config.gamma,
                gae_lambda=self.agent.config.gae_lambda,
            )
            batch = self.buffer.get_batch()
            update_metrics = self.agent.update(batch)
            update_count += 1

            # 3) Evaluate and save best checkpoint periodically.
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

                # Re-sync env state because evaluate_policy also steps/resets this env.
                state, _ = self.env.reset()
                episode_reward = 0.0

        return {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_episodes": eval_episodes,
            "best_eval_reward": best_eval_reward,
        }
