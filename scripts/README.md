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

使用命名场景模板：

```bash
python eval_baselines.py \
  --scenario multi_edge_network_sla \
  --episodes 20 \
  --seed 7 \
  --policy all \
  --output experiment_records/baselines/multi_edge_network_sla_seed7.csv
```

当前内置场景：

- `simple_single_edge`：单 edge server + SLA 奖励；
- `multi_edge_sla`：三 edge server + SLA 奖励；
- `multi_edge_heterogeneous_sla`：三 edge server + 异构任务画像；
- `multi_edge_network_sla`：三 edge server + 异构任务 + 上行竞争 + 下行响应传输。
- `cloud_edge_sla`：三 edge server + 异构任务 + 严格网络模型 + cloud fallback。

显式传入的命令行参数会覆盖模板中的同名配置。输出 CSV/JSONL 会保留 `scenario`、`reward_preset`、`num_edge_servers`、`task_type_count` 和网络开关，方便后续追踪实验来源。

构建 trace profile：

```bash
python scripts/build_trace_profile.py \
  --input csv/trace_alibaba_sample_codex.csv \
  --output experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --manifest experiment_records/trace_profiles/alibaba_light_normal_urgent_manifest.json
```

该脚本会把原始 normalized trace 派生为 `light`、`normal`、`urgent` 三类任务，并补充 `task_type_id`、`task_type`、`output_size`、`priority` 和按类型 deadline。不要覆盖原始 trace，正式实验使用派生 trace。

运行场景回归矩阵：

```bash
python scripts/run_scenario_regression.py \
  --episodes 2 \
  --seeds 7 \
  --output-root experiment_records/scenario_regression/iteration29
```

该脚本会对内置命名场景运行固定 baseline 策略，并在输出目录生成 `manifest.json`。manifest 会记录场景、策略、seed、CSV 路径、返回码和每个结果行，适合用来检查仿真器基础场景是否被后续改动破坏。

运行正式启发式 baseline 多 seed 候选表：

```bash
python scripts/run_formal_baselines.py \
  --episodes 10 \
  --seeds 7 17 27 37 47 \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --output-root experiment_records/formal_baselines/heuristic_trace_profile_e10
```

聚合结果：

```bash
python scripts/aggregate_experiment_results.py \
  --inputs "experiment_records/formal_baselines/heuristic_trace_profile_e10/*/*.csv" \
  --output experiment_records/formal_baselines/heuristic_trace_profile_e10/aggregate.csv \
  --summary-output experiment_records/formal_baselines/heuristic_trace_profile_e10/summary.csv
```

该结果可作为论文实验雏形表。正式论文版可把 `--episodes` 提高到 50 或更多。

正式 DRL 长跑前先做实验预检：

```bash
python scripts/run_experiment_preflight.py \
  --python D:\javaweb\miniconda3\envs\mec-wm\python.exe \
  --output-root experiment_records/preflight/iteration37
```

该脚本检查正式 trace、核心 scenario rollout、runner 参数解析、Colab 命令清单、极小 baseline 输出和现有聚合链路。预检输出统一写入 `experiment_records/preflight/`，不覆盖正式实验结果。

DRL / DreamerV3 正式实验协议见：

```text
docs/formal_drl_experiment_protocol.md
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

运行带异构任务、共享上行竞争和下行响应传输的多 edge 场景：

```bash
python eval_baselines.py \
  --episodes 3 \
  --seed 26 \
  --policy all \
  --reward-preset sla \
  --num-edge-servers 3 \
  --edge-server-compute-rates 10,14,18 \
  --edge-server-coverage-radius 60 \
  --task-type-count 3 \
  --task-type-names light,normal,urgent \
  --task-type-probabilities 0.5,0.3,0.2 \
  --task-cycles-per-unit-by-type 4,6,10 \
  --task-deadlines-by-type 14,10,6 \
  --task-output-ratios-by-type 0.05,0.1,0.2 \
  --task-priorities-by-type 1,2,5 \
  --enable-uplink-contention \
  --enable-downlink-transmission \
  --output experiment_records/baselines/network_model_sla_smoke.csv
```

该命令输出的 CSV 还会包含：

- `avg_edge_utilization`
- `avg_uplink_data`
- `avg_downlink_data`
- `avg_network_data`

这些列可用于后续论文表格中的资源利用率和网络使用量分析。

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

生成论文候选结果表和中文解读：

```bash
python scripts/make_paper_result_table.py \
  --summary experiment_records/summaries/formal_trace_sla_comparison_summary.csv \
  --output experiment_records/paper_tables/formal_trace_sla_result_table.csv \
  --markdown experiment_records/paper_tables/formal_trace_sla_result_table.md
```

在 Colab 结果尚未返回时，可以先用 `experiment_records/preflight/iteration37/aggregate_check_summary.csv` 验证表格生成链路。该脚本会自动选择同一 scenario 下最强启发式 baseline 作为 `anchor`，并输出 reward、deadline violation、completed tasks 的 gap。

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
