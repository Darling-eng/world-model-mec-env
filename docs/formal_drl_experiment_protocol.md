# 正式 DRL / DreamerV3 实验协议

日期：2026-07-02

## 目标

本协议固定深度强化学习实验的输入、场景、随机种子、输出格式和聚合方式，使 PPO、SAC、DreamerV3 的结果可以和启发式 baseline 表直接比较。

## 固定实验设置

trace profile：

```text
experiment_records/trace_profiles/alibaba_light_normal_urgent.csv
```

核心场景：

```text
multi_edge_network_sla
cloud_edge_sla
```

随机种子：

```text
7, 17, 27, 37, 47
```

正式评估回合数：

```text
20 episodes 或 50 episodes
```

本地 smoke 可以使用：

```text
1-5 eval episodes
```

## 已对齐的输出指标

DRL runner 的 eval 行应至少包含：

- `algorithm`
- `phase`
- `episodes`
- `seed`
- `scenario`
- `total_reward`
- `completed_tasks`
- `dropped_tasks`
- `deadline_violations`
- `deadline_violation_rate`
- `avg_delay`
- `avg_total_queue`
- `avg_edge_utilization`
- `avg_network_data`
- `avg_cloud_utilization`
- `avg_cloud_usage_ratio`
- `avg_energy_used`
- `avg_cloud_cost`

这些字段已经和 `scripts/aggregate_experiment_results.py` 的聚合字段对齐。

## 本地轻量 PPO smoke

本地 `mec-wm` 环境可以跑轻量 PPO 接口验证：

```bash
python scripts/run_ppo_mec.py \
  --steps 128 \
  --rollout-steps 64 \
  --eval-episodes 1 \
  --seed 7 \
  --scenario multi_edge_network_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/ppo_mec/smoke_multi_edge_seed7
```

该 runner 是 NumPy 线性 PPO，只用于本地 smoke，不应作为最终论文级 PPO。

## Colab / GPU：Stable-Baselines3 PPO

```bash
python scripts/run_sb3_ppo_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --scenario multi_edge_network_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sb3_ppo_mec/multi_edge_network_sla_seed7_50k
```

cloud 场景：

```bash
python scripts/run_sb3_ppo_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --scenario cloud_edge_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sb3_ppo_mec/cloud_edge_sla_seed7_50k
```

## Colab / GPU：SAC

```bash
python scripts/run_sac_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --scenario multi_edge_network_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sac_mec/multi_edge_network_sla_seed7_50k
```

cloud 场景：

```bash
python scripts/run_sac_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --scenario cloud_edge_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sac_mec/cloud_edge_sla_seed7_50k
```

## Colab / GPU：DreamerV3

```bash
python scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --scenario multi_edge_network_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 50000
```

cloud 场景：

```bash
python scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --scenario cloud_edge_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 50000
```

## 聚合命令

启发式 baseline 已在：

```text
experiment_records/formal_baselines/heuristic_trace_profile_e10/
```

DRL 结果返回后，统一聚合：

```bash
python scripts/aggregate_experiment_results.py \
  --inputs \
    "experiment_records/formal_baselines/heuristic_trace_profile_e10/*/*.csv" \
    "experiment_records/sb3_ppo_mec/*/metrics.jsonl" \
    "experiment_records/sac_mec/*/metrics.jsonl" \
    "experiment_records/dreamerv3/*/metrics.jsonl" \
  --output experiment_records/summaries/formal_trace_profile_all_results.csv \
  --summary-output experiment_records/summaries/formal_trace_profile_all_summary.csv
```

## 判定规则

DRL / DreamerV3 至少应和以下启发式比较：

- `best_uplink`
- `largest_queue`
- `nearest_edge`

核心判定指标：

1. `total_reward_mean` 越高越好；
2. `deadline_violation_rate_mean` 越低越好；
3. `completed_tasks_mean` 越高越好；
4. `avg_energy_used_mean` 和 `avg_cloud_cost_mean` 用于解释资源代价；
5. `avg_cloud_usage_ratio_mean` 用于分析是否学会选择性使用 cloud。

## 当前 baseline 锚点

在 10-episode 候选表中，`best_uplink` 是当前最强启发式：

```text
reward_mean = -810.328333
deadline_violation_rate_mean = 0.288617
completed_tasks_mean = 128.780000
```

后续模型不能只和 random/local_only 比较，必须至少说明与 `best_uplink`、`largest_queue`、`nearest_edge` 的差距或优势。
