# 实现日志

日期：2026-06-07 起  
项目：MEC 仿真环境 + DreamerV3 / PPO / SAC 实验入口

## 目标

本日志记录当前 Python MEC 仿真器从早期 demo 到可运行强化学习实验环境的主要实现过程。它不是最终论文材料，而是开发过程的技术记录。

## 已完成的核心模块

### 1. 基础 MEC 仿真器

已实现：

- 多用户移动；
- 任务随机到达；
- 本地计算；
- 上行传输；
- MEC 服务器队列；
- 截止期任务丢弃；
- 基于 delay、drop、queue、completion 的奖励；
- 启发式基线策略。

核心文件：

- `sim/config.py`
- `sim/entities.py`
- `sim/env.py`
- `sim/policies.py`
- `eval_baselines.py`

### 2. 启发式基线评估

支持的策略：

- `random`
- `local_only`
- `best_uplink`
- `largest_queue`

典型命令：

```bash
python eval_baselines.py --episodes 50 --seed 7 --output experiment_records/baselines/baselines.csv
```

早期结果表明，不同奖励和负载设置会显著改变启发式策略排序。因此，后续实验必须固定 trace、reward、seed 和 evaluation episodes。

### 3. Gymnasium 封装

已添加 Gymnasium 兼容环境，用于接入常见强化学习库：

- 连续 box action；
- binary action；
- 与原始 `MECEnv` 保持兼容；
- 支持 trace 和 SLA reward。

核心文件：

- `sim/gym_wrapper.py`
- `sim/gym_registration.py`

### 4. 轻量级 PPO

已实现一个 NumPy 版本的轻量级 PPO：

- 主要用于本地 smoke test；
- 可以验证 observation、action、reward、metrics 和日志路径是否贯通；
- 不应作为论文级深度强化学习 PPO baseline。

核心文件：

- `scripts/run_ppo_mec.py`

### 5. Stable-Baselines3 SAC 和 PPO 入口

已新增正式 model-free baseline 入口：

- `scripts/run_sac_mec.py`
- `scripts/run_sb3_ppo_mec.py`

这两个脚本依赖 `stable-baselines3` 和 `torch`。当前本地 `mec-wm` 环境未安装这些依赖，因此正式训练建议在 Colab 或 GPU 环境运行。

### 6. DreamerV3 接入

已添加 DreamerV3 启动入口：

- `scripts/run_dreamer_mec.py`
- `sim/dreamer_wrapper.py`

关键改动：

- 支持 `--trace`；
- 支持 `--reward-preset`；
- 使用环境变量向 DreamerV3 注册环境传递 MEC 设置。

### 7. trace-driven workload

已支持从 CSV trace 生成任务：

- `sim/trace.py`
- `scripts/extract_sql_trace.py`

当前保留的 trace 文件：

```text
csv/trace_alibaba_sample_codex.csv
```

### 8. 实验结果聚合

已新增结果聚合器：

```text
scripts/aggregate_experiment_results.py
```

支持：

- 读取 CSV；
- 读取 JSONL；
- 统一输出 raw comparison table；
- 输出 mean/std summary table。

典型命令：

```bash
python scripts/aggregate_experiment_results.py \
  --inputs "csv/trace_baselines_sla_e50_seed*_codex.csv" \
  --output experiment_records/summaries/comparison.csv \
  --summary-output experiment_records/summaries/comparison_summary.csv
```

### 9. Colab 结果归档

已新增：

```text
scripts/archive_colab_results.py
```

用途：

- 把 Colab 临时目录中的 `metrics.jsonl`、`scores.jsonl`、CSV 文件复制到持久目录；
- 生成 `manifest.json`；
- 保留原始实验证据链。

## 目录整理

旧的 `outputs/`、根目录 PPO smoke 输出和临时 CSV 已清理。后续统一使用：

```text
experiment_records/
```

保留的历史 smoke 结果移动到：

```text
experiment_records/legacy/
```

## 当前技术判断

当前代码已经足以支持：

- 环境 smoke test；
- trace/SLA baseline；
- PPO/SAC/DreamerV3 接口验证；
- 结果聚合；
- 原始结果归档。

但如果要支撑论文级结果，仿真器本身还需要增强，尤其是：

- 多 edge server；
- cloud tier；
- 任务类型；
- 网络竞争；
- 异构服务器；
- utilization、network usage、energy 等指标。

详细分析见：

```text
docs/simulator_completeness_gap_audit.md
```
