# Minimal MEC Simulator

This workspace contains a lightweight mobile edge computing simulator for early-stage world-model and RL experiments.

## What it simulates

- Multiple mobile users moving on a 1D road
- A single MEC server located at the road center
- Random task arrivals at users
- Per-step offloading decisions
- Local compute, uplink transmission, MEC queueing, and MEC execution
- Simple reward based on delay, drops, and queue backlog

## Files

- `sim/config.py`: scenario and reward parameters
- `sim/entities.py`: user, task, and server data classes
- `sim/env.py`: main simulator environment
- `sim/vector.py`: observation flattening and action vector helpers
- `sim/policies.py`: heuristic baseline policies
- `sim/rollout.py`: transition collection for RL / world-model datasets
- `sim/gym_wrapper.py`: optional Gymnasium-compatible wrapper
- `run_demo.py`: simple greedy-policy demo

## Quick start

Run the demo:

```bash
python run_demo.py
```

Collect one episode of flattened transitions:

```python
from sim import MECEnv, collect_episode, best_uplink_rate_policy

env = MECEnv()
transitions = collect_episode(env, best_uplink_rate_policy)
print(len(transitions), len(transitions[0].observation))
```

## Environment logic

Each simulator step does the following:

1. Move all users
2. Generate new tasks
3. Accept the selected offloading action
4. Process local tasks
5. Upload offloaded tasks to the MEC server
6. Process queued tasks on the MEC server
7. Drop expired tasks
8. Compute reward and return the next observation

## Observation

The environment returns a dictionary:

- `step`
- `users`: a list of per-user features
  - `user_id`
  - `position`
  - `velocity`
  - `queue_length`
  - `current_task_size`
  - `current_task_remaining_cycles`
  - `uplink_rate`
- `server_queue_length`
- `max_offloads_per_step`

## Action

The action is a list of user ids to offload in the current step.

Example:

```python
action = [1, 4, 7]
```

Only the first `max_offloads_per_step` valid unique ids are accepted.

## Reward

The current reward is:

```text
reward =
  - delay_penalty * avg_delay
  - drop_penalty * dropped_tasks
  - queue_penalty * total_queue
  + 0.2 * completed_tasks
```

This is intentionally simple. It is a starting point for later redesign around AoI, SLA violation, or risk-aware offloading.

## Why this version is useful

This simulator is not a high-fidelity network emulator. It is a small training world designed to help with:

- RL environment debugging
- State, action, and reward design
- Early world-model experiments
- Fast baseline comparison

## Recommended next steps

1. Replace the 1D mobility model with a road topology
2. Add AoI or CAoI metrics
3. Add multiple RSUs or MEC servers
4. Add partial observability and uncertainty-aware policies
5. Add training scripts for PPO / Dreamer-style experiments
