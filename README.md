# kshitij-rl

A modular reinforcement learning library scaffold with a working DQN implementation.

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
   - `uv run python main.py`
3. Run tests:
   - `uv run pytest`

## Extension Plan

- Add `PPOAgent` in `core_rl/agents/ppo.py`.
- Add rollout storage in `core_rl/buffers/rollout_buffer.py`.
- Add on-policy loop in `core_rl/runners/on_policy.py`.
- Keep runner code algorithm-agnostic via `BaseAgent` and buffer interfaces.
