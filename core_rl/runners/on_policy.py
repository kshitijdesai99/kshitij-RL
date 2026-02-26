# Beginner summary: This file runs the on-policy training loop used by PPO (collect rollout -> update policy).
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from core_rl.buffers.rollout_buffer import RolloutBuffer
from core_rl.utils.logger import get_logger
from core_rl.utils.metrics import evaluate_policy


@dataclass(slots=True)
class OnPolicyRunnerConfig:
    """
    Knobs controlling on-policy training schedule.

    PPO alternates:
    1) collect N rollout steps with current policy
    2) run policy/value updates on that rollout
    """

    total_timesteps: int = 50_000  # Overall environment steps budget.
    rollout_steps: int = 1_024  # Steps collected before each PPO update phase.
    eval_freq: int = 10  # Evaluate every N PPO updates (not episodes).
    eval_episodes: int = 5  # Episodes averaged during deterministic evaluation.


class OnPolicyRunner:
    """Orchestrates PPO-style on-policy training."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        buffer: RolloutBuffer,
        config: OnPolicyRunnerConfig | None = None,
        logger: Any | None = None,
        best_model_path: str | None = None,
    ):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config or OnPolicyRunnerConfig()
        self.logger = logger or get_logger("core_rl.on_policy_runner")
        self.best_model_path = best_model_path

    def train(self) -> dict[str, Any]:
        """
        Run on-policy PPO training and return metric history.

        Returns keys:
        - train_rewards: completed episode rewards during training
        - eval_rewards: deterministic evaluation rewards
        - eval_episodes: episode indices where evaluation was run
        - best_eval_reward: best deterministic eval reward seen
        """
        train_rewards: list[float] = []
        eval_rewards: list[float] = []
        eval_episodes: list[int] = []
        best_eval_reward = float("-inf")

        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        timesteps = 0  # Counts all env steps across training.
        update_count = 0  # Counts how many rollout->update cycles we ran.

        while timesteps < self.config.total_timesteps:
            # 1) COLLECT A FRESH ON-POLICY ROLLOUT
            self.buffer.clear()

            for _ in range(self.config.rollout_steps):
                # act() returns action + log_prob + value needed for PPO losses.
                action, log_prob, value = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store exactly the data PPO needs for clipped-ratio updates + GAE.
                self.buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                episode_reward += reward
                timesteps += 1
                state = next_state

                if done:
                    # Track completed episode rewards for training curves.
                    train_rewards.append(float(episode_reward))
                    episode_reward = 0.0
                    episode_count += 1
                    state, _ = self.env.reset()

                if timesteps >= self.config.total_timesteps:
                    break

            # 2) COMPUTE ADVANTAGES/RETURNS FOR THE COLLECTED ROLLOUT
            # Bootstrap with critic value of the "next state" after rollout end.
            last_value = self.agent.estimate_value(state)
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.agent.config.gamma,
                gae_lambda=self.agent.config.gae_lambda,
            )

            # 3) RUN PPO UPDATE(S) ON THE ROLLOUT
            batch = self.buffer.get_batch()
            update_metrics = self.agent.update(batch)
            update_count += 1

            # 4) PERIODIC EVALUATION + BEST CHECKPOINTING
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
                    "Update %s | Episode %s | Timesteps %s | Eval: %.1f | Policy Loss: %.4f | Value Loss: %.4f%s",
                    update_count,
                    episode_count,
                    timesteps,
                    avg_eval_reward,
                    update_metrics["policy_loss"],
                    update_metrics["value_loss"],
                    save_marker,
                )
                # evaluate_policy() resets/steps the same env; reset once to re-sync training state.
                state, _ = self.env.reset()
                episode_reward = 0.0

        return {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_episodes": eval_episodes,
            "best_eval_reward": best_eval_reward,
        }
