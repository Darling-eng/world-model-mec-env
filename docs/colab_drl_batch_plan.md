# Colab DRL 批跑清单

日期：2026-07-02

## 目标

在 Colab / GPU 环境运行正式 model-free 和 world-model 实验，并把结果返回到本项目的统一聚合表。

本地已经完成：

- 启发式 baseline 70 组候选表；
- 轻量 PPO 6 组 smoke；
- 统一 scenario/trace/seed 协议；
- 聚合脚本验证。

下一步 Colab 需要跑：

- Stable-Baselines3 PPO；
- Stable-Baselines3 SAC；
- DreamerV3。

## 固定输入

trace：

```text
experiment_records/trace_profiles/alibaba_light_normal_urgent.csv
```

场景：

```text
multi_edge_network_sla
cloud_edge_sla
```

seeds：

```text
7 17 27 37 47
```

建议训练步数：

```text
50000
```

建议评估回合：

```text
20
```

## Stable-Baselines3 PPO

对每个 scenario 和 seed 运行：

```bash
python scripts/run_sb3_ppo_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed <SEED> \
  --scenario <SCENARIO> \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sb3_ppo_mec/<SCENARIO>_seed<SEED>_50k
```

## SAC

对每个 scenario 和 seed 运行：

```bash
python scripts/run_sac_mec.py \
  --steps 50000 \
  --eval-episodes 20 \
  --seed <SEED> \
  --scenario <SCENARIO> \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/sac_mec/<SCENARIO>_seed<SEED>_50k
```

## DreamerV3

对每个 scenario 和 seed 运行。DreamerV3 的 seed 参数需要跟实际 DreamerV3 config 支持的参数一致；如果当前 DreamerV3 入口使用不同 seed 字段，以 Colab 中 `configs.yaml` 或 DreamerV3 CLI 为准。

```bash
python scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --scenario <SCENARIO> \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 50000 \
  --seed <SEED>
```

## 需要返回的文件

至少返回：

```text
experiment_records/sb3_ppo_mec/*/metrics.jsonl
experiment_records/sac_mec/*/metrics.jsonl
experiment_records/dreamerv3/*/metrics.jsonl
```

如果有模型文件，也可以归档，但论文表格优先需要 metrics：

```text
experiment_records/sb3_ppo_mec/*/ppo_model*
experiment_records/sac_mec/*/sac_model*
```

## 统一聚合

Colab 结果返回本地后运行：

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

## 批跑顺序建议

1. 先跑 SB3-PPO 单场景单 seed smoke：
   - `multi_edge_network_sla`
   - seed `7`
   - steps `5000`

2. 确认输出 `metrics.jsonl` 能聚合；

3. 再跑 SB3-PPO 全 seeds；

4. 再跑 SAC 全 seeds；

5. 最后跑 DreamerV3。

## 当前 baseline 锚点

强启发式 baseline：

```text
best_uplink
reward_mean = -810.328333
deadline_violation_rate_mean = 0.288617
completed_tasks_mean = 128.780000
```

Colab 模型结果需要至少与 `best_uplink`、`largest_queue`、`nearest_edge` 比较。

## 需要用户授权的时刻

当需要实际打开或操作 Colab / Google Drive / Google 账号时，需要用户授权 Google/Colab 操控权。
