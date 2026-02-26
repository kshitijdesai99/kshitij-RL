# Beginner summary: This file is the main entry point that trains DQN or PPO on CartPole, saves checkpoints, and plots progress.
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch

from core_rl import (
    ContinuousSPOAgent,
    ContinuousSPOConfig,
    ContinuousSPORolloutBuffer,
    ContinuousSPORunner,
    ContinuousSPORunnerConfig,
    DiscreteSPOAgent,
    DiscreteSPOConfig,
    OffPolicyRunner,
    OffPolicyRunnerConfig,
    OnPolicyRunner,
    OnPolicyRunnerConfig,
    PPOAgent,
    PPOConfig,
    ReplayBuffer,
    RolloutBuffer,
    SPORolloutBuffer,
    SPORunner,
    SPORunnerConfig,
    VanillaDQNAgent,
    VanillaDQNConfig,
)
from core_rl.utils.logger import get_logger


def get_env_tag(env: gym.Env) -> str:
    """Return a filesystem-safe tag for the current environment id."""
    # env.spec.id might be values like "CartPole-v1" or "Pendulum-v1".
    # We convert to lowercase + replace separators for clean filenames.
    env_id = getattr(getattr(env, "spec", None), "id", "env")
    return str(env_id).lower().replace("-", "_").replace("/", "_")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agents with DQN, PPO, Discrete SPO, or Continuous SPO."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "ppo", "spo", "spo_continuous"],
        help="Algorithm to train: dqn, ppo, spo, or spo_continuous.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment id. Example: CartPole-v1 or Pendulum-v1",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Pick the best available torch device."""
    # Priority order:
    # 1) CUDA GPU (NVIDIA)
    # 2) MPS GPU (Apple Silicon)
    # 3) CPU fallback
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def run_inference_episode(env: gym.Env, agent: object) -> float:
    """Run one deterministic episode using a trained policy."""
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    return total_reward


def train_dqn(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Vanilla DQN and return metrics + model metadata."""
    # DQN only works for discrete action spaces by design.
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("DQN only supports discrete action spaces")
    # This starter implementation expects flat vector observations.
    if len(env.observation_space.shape) != 1:
        raise ValueError("DQN example currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env_tag = get_env_tag(env)

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
    best_model_path = str(checkpoints_dir / f"dqn_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"dqn_{env_tag}_final.pth")

    # Off-policy training uses a replay buffer.
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

    logger.info("Starting DQN training...")
    metrics = runner.train()

    # Save final online-network weights from end of training.
    torch.save(agent.q_network.state_dict(), final_model_path)
    # Reload best checkpoint (selected by periodic evaluation reward).
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    agent.q_network.load_state_dict(best_state_dict)
    agent.q_network.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved DQN model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "DQN Training Progress",
    }


def train_ppo(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train PPO and return metrics + model metadata."""
    # PPO implementation here also expects flat vector observations.
    if len(env.observation_space.shape) != 1:
        raise ValueError("PPO example currently supports 1D vector observations only")
    state_dim = env.observation_space.shape[0]
    env_tag = get_env_tag(env)

    agent = PPOAgent(
        state_dim=state_dim,
        config=PPOConfig(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            update_epochs=4,
            minibatch_size=64,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        ),
        device=device,
        action_space=env.action_space,
    )
    best_model_path = str(checkpoints_dir / f"ppo_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"ppo_{env_tag}_final.pth")

    # On-policy training uses rollout storage (fresh data each update cycle).
    buffer = RolloutBuffer()
    runner = OnPolicyRunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=OnPolicyRunnerConfig(
            total_timesteps=60_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting PPO training...")
    metrics = runner.train()

    # Save full PPO checkpoint (actor + critic + optional log_std).
    torch.save(agent.get_checkpoint(), final_model_path)
    # Load best evaluation checkpoint before final deterministic inference.
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved PPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "PPO Training Progress",
    }


def train_spo(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Discrete SPO and return metrics + model metadata."""
    # This SPO implementation is discrete-action only.
    if not isinstance(env.action_space, spaces.Discrete):
        raise ValueError("Discrete SPO only supports discrete action spaces")
    if len(env.observation_space.shape) != 1:
        raise ValueError("Discrete SPO example currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env_tag = get_env_tag(env)

    agent = DiscreteSPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=DiscreteSPOConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            update_epochs=4,
            minibatch_size=64,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            search_num_particles=32,
            search_temperature=1.0,
            root_dirichlet_alpha=0.3,
            root_dirichlet_fraction=0.25,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / f"spo_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"spo_{env_tag}_final.pth")

    buffer = SPORolloutBuffer()
    runner = SPORunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=SPORunnerConfig(
            total_timesteps=60_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting Discrete SPO training...")
    metrics = runner.train()

    torch.save(agent.get_checkpoint(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved SPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "Discrete SPO Training Progress",
    }


def train_spo_continuous(
    env: gym.Env,
    device: torch.device,
    logger: logging.Logger,
    checkpoints_dir: Path,
) -> dict[str, object]:
    """Train Continuous SPO and return metrics + model metadata."""
    if not isinstance(env.action_space, spaces.Box):
        raise ValueError("Continuous SPO only supports continuous Box action spaces")
    if len(env.action_space.shape) != 1:
        raise ValueError("Continuous SPO currently supports 1D action vectors only")
    if len(env.observation_space.shape) != 1:
        raise ValueError("Continuous SPO currently supports 1D vector observations only")

    state_dim = env.observation_space.shape[0]
    env_tag = get_env_tag(env)

    agent = ContinuousSPOAgent(
        state_dim=state_dim,
        action_space=env.action_space,
        config=ContinuousSPOConfig(
            actor_lr=3e-4,
            critic_lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            update_epochs=4,
            minibatch_size=64,
            entropy_coef=0.001,
            max_grad_norm=0.5,
            search_num_particles=64,
            search_temperature=1.0,
            root_exploration_fraction=0.1,
            init_log_std=-0.5,
            min_log_std=-5.0,
            max_log_std=2.0,
            value_num_samples=16,
        ),
        device=device,
    )
    best_model_path = str(checkpoints_dir / f"spo_continuous_{env_tag}_best.pth")
    final_model_path = str(checkpoints_dir / f"spo_continuous_{env_tag}_final.pth")

    buffer = ContinuousSPORolloutBuffer()
    runner = ContinuousSPORunner(
        env=env,
        agent=agent,
        buffer=buffer,
        config=ContinuousSPORunnerConfig(
            total_timesteps=80_000,
            rollout_steps=1_024,
            eval_freq=5,
            eval_episodes=5,
        ),
        logger=logger,
        best_model_path=best_model_path,
    )

    logger.info("Starting Continuous SPO training...")
    metrics = runner.train()

    torch.save(agent.get_checkpoint(), final_model_path)
    best_state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    agent.load_checkpoint(best_state_dict)
    agent.actor.eval()
    agent.critic.eval()

    inference_reward = run_inference_episode(env, agent)
    logger.info("Inference reward using best saved Continuous SPO model: %.1f", inference_reward)

    return {
        "metrics": metrics,
        "title": "Continuous SPO Training Progress",
    }


def main() -> None:
    """Train selected algorithm on a Gymnasium env, save checkpoints, and run final inference."""
    args = parse_args()

    # Create app logger. DEBUG prints per-eval training progress from the runner.
    logger = get_logger("core_rl.main")
    logger.setLevel(logging.DEBUG)

    # Build environment and create checkpoint directory.
    env = gym.make(args.env_id)
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    if args.algo == "dqn":
        # DQN path: discrete actions + replay buffer + off-policy loop.
        result = train_dqn(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    elif args.algo == "ppo":
        # PPO path: supports both discrete and continuous environments.
        result = train_ppo(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    elif args.algo == "spo":
        # SPO path: discrete action search + actor/critic updates from search targets.
        result = train_spo(env=env, device=device, logger=logger, checkpoints_dir=checkpoints_dir)
    else:
        # Continuous SPO path: bounded continuous actions + weighted particle imitation.
        result = train_spo_continuous(
            env=env,
            device=device,
            logger=logger,
            checkpoints_dir=checkpoints_dir,
        )

    metrics = result["metrics"]
    best_eval_reward = float(metrics["best_eval_reward"])
    logger.info("Best evaluation reward: %.1f", best_eval_reward)

    # Plot training reward and periodic evaluation reward.
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["train_rewards"], label="Train Reward", alpha=0.6)
    plt.plot(metrics["eval_episodes"], metrics["eval_rewards"], label="Eval Reward", color="red", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(str(result["title"]))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    env.close()


if __name__ == "__main__":
    main()
