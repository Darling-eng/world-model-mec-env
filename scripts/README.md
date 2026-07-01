# 脚本入口说明

本目录包含 MEC 仿真环境的训练、评估、结果聚合和结果归档脚本。新实验输出默认应写入 `experiment_records/`。

## 评估启发式基线

运行全部启发式策略：

```bash
python eval_baselines.py --episodes 10 --seed 7
```

单独运行某个策略：

```bash
python eval_baselines.py --policy random --episodes 10 --seed 7
python eval_baselines.py --policy local_only --episodes 10 --seed 7
python eval_baselines.py --policy best_uplink --episodes 10 --seed 7
python eval_baselines.py --policy largest_queue --episodes 10 --seed 7
```

保存聚合指标：

```bash
python eval_baselines.py \
  --episodes 50 \
  --seed 7 \
  --output experiment_records/baselines/baselines.csv
```

使用 trace 和 SLA 奖励：

```bash
python eval_baselines.py \
  --trace csv/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --episodes 50 \
  --seed 7 \
  --output experiment_records/baselines/trace_sla_seed7.csv
```

运行多 edge server 场景：

```bash
python eval_baselines.py \
  --episodes 3 \
  --seed 7 \
  --policy all \
  --reward-preset sla \
  --num-edge-servers 3 \
  --edge-server-compute-rates 10,14,18 \
  --edge-server-coverage-radius 60 \
  --output experiment_records/baselines/multi_edge_sla_smoke.csv
```

新增的多 edge 显式目标服务器基线包括：

- `nearest_edge`
- `least_loaded_edge`

## 训练轻量级 PPO

`run_ppo_mec.py` 是 NumPy 实现的轻量级 PPO，用于本地 smoke test 和接口验证，不应作为最终论文级深度强化学习基线。

```bash
python scripts/run_ppo_mec.py \
  --steps 2000 \
  --rollout-steps 256 \
  --eval-episodes 5 \
  --seed 7 \
  --trace csv/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --log-dir experiment_records/ppo_mec/trace_sla_seed7
```

## 训练 Stable-Baselines3 SAC

`run_sac_mec.py` 依赖 `stable-baselines3` 和 `torch`，本地 `mec-wm` 环境目前未安装这些依赖，因此正式 SAC 训练建议在 Colab 或其他 GPU 环境运行。

```bash
python scripts/run_sac_mec.py \
  --trace /content/world-model-mec-env/csv/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --log-dir /content/world-model-mec-env/experiment_records/sac_mec/sac_trace_sla_seed7_50k
```

## 训练 Stable-Baselines3 PPO

`run_sb3_ppo_mec.py` 是正式 model-free PPO baseline 入口，输出格式与 SAC runner 保持一致。

```bash
python scripts/run_sb3_ppo_mec.py \
  --trace /content/world-model-mec-env/csv/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --steps 50000 \
  --eval-episodes 20 \
  --seed 7 \
  --log-dir /content/world-model-mec-env/experiment_records/sb3_ppo_mec/sb3_ppo_trace_sla_seed7_50k
```

## 在 Colab 运行 DreamerV3

合成负载短跑：

```bash
python /content/world-model-mec-env/scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 100
```

真实 trace + SLA reward：

```bash
python /content/world-model-mec-env/scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --trace /content/world-model-mec-env/csv/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 50000
```

## 聚合实验结果

把启发式、PPO、SAC、DreamerV3 的 CSV/JSONL 统一成对比表：

```bash
python scripts/aggregate_experiment_results.py \
  --inputs \
    "csv/trace_baselines_sla_e50_seed*_codex.csv" \
    "experiment_records/sb3_ppo_mec/*/metrics.jsonl" \
    "experiment_records/sac_mec/*/metrics.jsonl" \
    "experiment_records/dreamerv3/*/metrics.jsonl" \
  --output experiment_records/summaries/formal_trace_sla_comparison.csv \
  --summary-output experiment_records/summaries/formal_trace_sla_comparison_summary.csv
```

## 归档 Colab 实验结果

Colab 的 `/content` 是临时目录，实验结束后应立即把 `experiment_records/` 复制到 Google Drive。

```bash
python scripts/archive_colab_results.py \
  --src /content/world-model-mec-env/experiment_records \
  --dst /content/drive/MyDrive/mec_results/20260701
```

脚本会复制 `metrics.jsonl`、`scores.jsonl` 和 CSV 文件，并生成：

```text
manifest.json
```

`manifest.json` 用来记录归档了哪些原始实验文件、文件大小、来源路径和归档时间。
