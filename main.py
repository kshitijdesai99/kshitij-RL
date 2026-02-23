from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from core_rl import VanillaDQNAgent, VanillaDQNConfig, OffPolicyRunner, OffPolicyRunnerConfig, ReplayBuffer
from core_rl.utils.logger import get_logger


def main() -> None:
    """Train Vanilla DQN on CartPole, save checkpoints, and run final inference."""

    # Create app logger. DEBUG prints per-eval training progress from the runner.
    logger = get_logger("core_rl.main")
    logger.setLevel(logging.DEBUG)

    # 1) Build environment.
    env = gym.make("CartPole-v1")
    # 2) Read observation/action sizes from env spaces.
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 3) Create algorithm instance (DQN) with hyperparameters.
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    agent = VanillaDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=VanillaDQNConfig(
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            target_update_freq=100,
        ),
        device=device,
    )
    # Keep model files in a dedicated folder.
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(checkpoints_dir / "dqn_cartpole_best.pth")
    final_model_path = str(checkpoints_dir / "dqn_cartpole_final.pth")
    # 4) Create replay buffer and runner that coordinates training.
    buffer = ReplayBuffer(capacity=10_000)
    runner = OffPolicyRunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=OffPolicyRunnerConfig(
            num_episodes=500,
            batch_size=64,
            eval_freq=10,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    # 5) Train and collect metrics.
    logger.info("Starting training...")
    metrics = runner.train()

    # 6) Save final trained online network weights (last training state).
    torch.save(agent.q_network.state_dict(), final_model_path)
    best_eval_reward = float(metrics["best_eval_reward"])
    logger.info("Best evaluation reward: %.1f", best_eval_reward)

    # 7) Reload best checkpoint and run one deterministic inference episode.
    # This verifies "best model performance" separately from training-time metrics.
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    agent.q_network.load_state_dict(best_state_dict)
    agent.q_network.eval()

    state, _ = env.reset()
    done = False
    inference_reward = 0.0
    while not done:
        # deterministic=True disables epsilon exploration for pure policy execution.
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        inference_reward += reward
    logger.info("Inference reward using best saved model: %.1f", inference_reward)

    # 8) Plot training reward and periodic evaluation reward.
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["train_rewards"], label="Train Reward", alpha=0.6)
    plt.plot(metrics["eval_episodes"], metrics["eval_rewards"], label="Eval Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
