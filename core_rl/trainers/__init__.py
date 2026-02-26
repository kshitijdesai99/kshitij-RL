# Beginner summary: This file exports the trainer entry functions so main.py can import them from one place.
from core_rl.trainers.continuous_spo_trainer import train_spo_continuous
from core_rl.trainers.dqn_trainer import train_dqn
from core_rl.trainers.ppo_trainer import train_ppo
from core_rl.trainers.discrete_spo_trainer import train_spo

__all__ = [
    "train_dqn",
    "train_ppo",
    "train_spo",
    "train_spo_continuous",
]
