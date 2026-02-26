# kshitij-rl

A modular reinforcement learning library scaffold with working DQN and PPO implementations.

## Structure

- `core_rl/agents`: Base agent contract and algorithm implementations.
- `core_rl/buffers`: Replay/rollout data stores.
- `core_rl/networks`: Shared neural network modules.
- `core_rl/runners`: Training orchestration loops.
- `core_rl/utils`: Metrics and logging helpers.
- `tests`: Unit tests.

## Quickstart

1. Install dependencies:
   - `uv sync`
2. Run DQN example:
   - `uv run python main.py --algo dqn`
3. Run PPO example:
   - `uv run python main.py --algo ppo`
4. Run tests:
   - `uv run pytest`

## Implemented Algorithms

- DQN: `core_rl/agents/vanilla_dqn.py` + `core_rl/runners/off_policy.py`.
- PPO: `core_rl/agents/ppo.py` + `core_rl/runners/on_policy.py`.
