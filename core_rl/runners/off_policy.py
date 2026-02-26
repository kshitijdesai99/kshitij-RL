# Beginner summary: This file runs the off-policy training loop by collecting transitions, training the agent, and evaluating progress.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from core_rl.buffers.replay_buffer import ReplayBuffer
from core_rl.utils.logger import get_logger
from core_rl.utils.metrics import evaluate_policy


@dataclass(slots=True)
class OffPolicyRunnerConfig:
    """
    Knobs for the training loop itself (not model architecture).

    These values control *how* long and how often we train/evaluate.
    """

    num_episodes: int = 500  # Number of full episodes to train.
    batch_size: int = 64  # Mini-batch size sampled from replay buffer.
    eval_freq: int = 10  # Run evaluation every N training episodes.
    eval_episodes: int = 5  # Number of episodes to average during each evaluation.


class OffPolicyRunner:
    """
    Orchestrates the off-policy RL loop:
    1) collect transitions from env
    2) store them in replay buffer
    3) sample random mini-batches
    4) update the agent
    """

    def __init__(
        self,
        env: Any,
        agent: Any,
        buffer: ReplayBuffer,
        config: OffPolicyRunnerConfig | None = None,
        logger: Any | None = None,
        best_model_path: str | None = None,
    ):
        # env: Gymnasium-compatible environment with reset()/step().
        # agent: Object with select_action(), update(), and optionally decay_epsilon().
        # buffer: ReplayBuffer used to store and sample transitions.
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.config = config or OffPolicyRunnerConfig()
        # If no logger is provided, create a default one.
        self.logger = logger or get_logger("core_rl.runner")
        # Optional checkpoint path; best model weights are saved here.
        self.best_model_path = best_model_path

    def train(self) -> dict[str, Any]:
        """
        Run full training and return tracked metrics.

        Returns keys:
        - train_rewards: episode reward during exploration/training policy
        - eval_rewards: deterministic evaluation scores
        - eval_episodes: episode indices where eval was run
        - best_eval_reward: best score seen during training
        """
        train_rewards: list[float] = []
        eval_rewards: list[float] = []
        eval_episodes: list[int] = []
        # Start with -inf so the first evaluation always becomes "best".
        best_eval_reward = float("-inf")

        for episode in range(self.config.num_episodes):
            # reset() returns initial observation and info dict.
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # 1) Ask agent for action.
                action = self.agent.select_action(state, deterministic=False)
                # 2) Step environment.
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # 3) Store transition for replay.
                self.buffer.add(state, action, reward, next_state, done)
                # 4) Learn from a random batch when enough data exists.
                if len(self.buffer) >= self.config.batch_size:
                    batch = self.buffer.sample(self.config.batch_size)
                    # update() performs one gradient step and may return loss metrics.
                    self.agent.update(batch)

                state = next_state
                # Episode reward is the sum of all step rewards in this episode.
                total_reward += reward

            train_rewards.append(total_reward)
            # Optional hook used by epsilon-greedy algorithms.
            if hasattr(self.agent, "decay_epsilon"):
                self.agent.decay_epsilon()

            if episode % self.config.eval_freq == 0:
                # Evaluation uses deterministic actions to measure true policy quality.
                avg_eval_reward = evaluate_policy(self.env, self.agent, self.config.eval_episodes)
                eval_rewards.append(avg_eval_reward)
                eval_episodes.append(episode)
                save_marker = ""
                # Track best model based on evaluation reward, not training reward.
                if avg_eval_reward >= best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    save_marker = " ✅ (New Best Eval)"
                    if self.best_model_path is not None:
                        # Save online network weights as current best checkpoint.
                        torch.save(self.agent.q_network.state_dict(), self.best_model_path)
                epsilon = getattr(self.agent, "epsilon", float("nan"))  # nan if agent has no epsilon (e.g., PPO)
                # Debug log prints compact progress every eval step.
                self.logger.debug(
                    "Episode %s | Train: %.1f | Eval: %.1f | Epsilon: %.3f%s",
                    episode,
                    total_reward,
                    avg_eval_reward,
                    epsilon,
                    save_marker,
                )

        # Return raw metric arrays so caller can plot/analyze any way they want.
        return {
            "train_rewards": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_episodes": eval_episodes,
            "best_eval_reward": best_eval_reward,
        }
