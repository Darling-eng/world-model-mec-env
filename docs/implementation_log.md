# Implementation Log（实现日志）

Date（日期）：2026-06-07  
Project（项目）：MEC（移动边缘计算）+ DreamerV3（世界模型强化学习算法）

这份日志用于后期复盘，按改动批次记录本次实现中做了什么、为什么做，以及如何验证。

## 1. Baseline Evaluation（基线评测）能力增强

### 改动内容

- 修改 `eval_baselines.py`。
- 新增 `--policy` 参数，可以单独运行某一条 heuristic baseline（启发式基线）：
  - `random`（随机策略）
  - `local_only`（本地执行）
  - `best_uplink`（最佳上行链路）
  - `largest_queue`（最长队列）
- 保留原有默认行为：不传 `--policy` 时，仍然一次运行全部四条 baseline（基线）。
- 统一输出 metrics（指标）字段：
  - `total_reward`（总奖励）
  - `completed_tasks`（完成任务数）
  - `dropped_tasks`（丢弃任务数）
  - `avg_delay`（平均时延）
  - `avg_total_queue`（平均总队列长度）
  - `steps`（步数）
- 新增 `--output` 参数，用于保存 aggregate results（聚合结果）。
- 新增 `--output-format` 参数，支持 `csv`（逗号分隔表格）和 `jsonl`（逐行 JSON）。

### 改动原因

之前 baseline（基线）只能一次性全部运行，不方便逐条调试和复盘。  
同时，终端输出不利于后续画图、写 report（报告）或和 DreamerV3（世界模型强化学习算法）结果放在同一张表里比较。

### 验证记录

已运行：

```powershell
python eval_baselines.py --episodes 2 --seed 7
python eval_baselines.py --policy best_uplink --episodes 2 --seed 7 --output outputs\baseline_best_uplink_smoke.csv
python eval_baselines.py --policy local_only --episodes 2 --seed 7 --output outputs\baseline_local_only_smoke.jsonl
python eval_baselines.py --episodes 10 --seed 7 --output outputs\baselines_episodes10_seed7.csv
python eval_baselines.py --episodes 50 --seed 7 --output outputs\baselines_episodes50_seed7.csv
```

结果：全部 baseline（基线）评测、单条 baseline（基线）评测、CSV（逗号分隔表格）输出、JSONL（逐行 JSON）输出均已跑通。

## 2. PPO Baseline（近端策略优化基线）入口

### 改动内容

- 新增 `scripts/run_ppo_mec.py`。
- 实现一个 lightweight PPO（轻量近端策略优化）训练脚本。
- 使用 NumPy（数值计算库）实现，不依赖 torch（深度学习库）或 stable-baselines3（常用强化学习库）。
- 使用 `GymnasiumMECEnv` 的 `box` action mode（连续动作模式）。
- 支持短程训练、evaluation（评估）、metrics logging（指标日志）。
- 训练输出包括：
  - `metrics.jsonl`（指标日志）
  - `ppo_linear_model.npz`（轻量模型参数）

### 改动原因

本地 `mec-wm` Python environment（Python 环境）里有 `gymnasium`（强化学习环境接口）和 `numpy`（数值计算库），但没有 `torch`（深度学习库）和 `stable-baselines3`（常用强化学习库）。  
所以本次没有直接接入重型 DRL library（深度强化学习库），而是先做一个 dependency-free PPO smoke baseline（无额外依赖的 PPO 冒烟基线），用于验证训练闭环。

### 验证记录

已运行：

```powershell
python scripts\run_ppo_mec.py --steps 64 --rollout-steps 32 --eval-episodes 1 --seed 7 --log-dir outputs\ppo_smoke
```

结果：PPO（近端策略优化）完成 64 steps（64 步）短程训练，成功生成 `outputs\ppo_smoke\metrics.jsonl`，并完成 1 episode（1 个回合）evaluation（评估）。

## 3. Method Design（方法设计）文档

### 改动内容

- 新增 `docs/next_stage_method_design.md`。
- 整理下一阶段可写进 report（报告）的方法强化方向：
  - uncertainty-aware offloading（不确定性感知卸载）
  - structured encoder（结构化编码器）/ Transformer encoder（Transformer 编码器）
  - generative digital twin（生成式数字孪生）

### 改动原因

report（报告）里已经明确指出，后续不能只是把 DreamerV3（世界模型强化学习算法）套到 MEC（移动边缘计算）上，还要想清楚 differentiation（差异化）。  
这份文档用于把下一阶段可讲的技术路线提前整理成文字，方便后续直接合入周报或论文草稿。

### 验证记录

已人工检查文档内容，确认它对应 report（报告）中的三条方法强化方向。

## 4. Scripts Documentation（脚本说明）更新

### 改动内容

- 修改 `scripts/README.md`。
- 新增 heuristic baseline（启发式基线）运行命令。
- 新增单条 baseline（基线）运行命令。
- 新增保存 aggregate metrics（聚合指标）的命令。
- 新增 lightweight PPO（轻量近端策略优化）训练命令。
- 保留 DreamerV3（世界模型强化学习算法）Colab（云端笔记本）运行说明。

### 改动原因

你的下周任务需要自己在终端反复跑实验。把命令写进 README（说明文档）后，不需要每次翻聊天记录。

### 验证记录

README（说明文档）中的 baseline（基线）命令已经按对应脚本验证过；PPO（近端策略优化）命令已用短程参数验证过。

## 5. Experiment Outputs（实验输出）忽略规则

### 改动内容

- 修改 `.gitignore`。
- 新增 `outputs/` 忽略规则。

### 改动原因

baseline（基线）结果、PPO（近端策略优化）日志和模型参数都属于 experiment outputs（实验输出），不应该默认进入 git（版本管理）记录。

### 验证记录

已确认 `outputs/` 中生成的 CSV（逗号分隔表格）、JSONL（逐行 JSON）和 NPZ（NumPy 参数文件）不会出现在 `git status`（版本状态）中。

## 6. 当前阶段结论

- 实验平台已经从“能跑”推进到“能记录、能单独评测、能保存结果”。
- heuristic baseline（启发式基线）已经具备正式复现实验的入口。
- PPO（近端策略优化）已经具备短程训练闭环，但当前版本是 lightweight smoke baseline（轻量冒烟基线），不是最终高性能 DRL baseline（深度强化学习基线）。
- 当前 50 episodes（50 个回合）结果仍显示 `local_only`（本地执行）在 reward（奖励）上最好，但 `best_uplink`（最佳上行链路）完成任务更多、丢弃任务更少，说明后续 reward redesign（奖励重设计）仍然必要。

## 7. 注意事项

- 两个旧 docs（文档）文件在本次实现前已经处于 deleted（删除）状态，本次没有恢复，也没有继续修改。
- DreamerV3（世界模型强化学习算法）主框架没有被改动，本次只强化本地 MEC environment（移动边缘计算环境）、baseline evaluation（基线评测）和实验记录能力。
