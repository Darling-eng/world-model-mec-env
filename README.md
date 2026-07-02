# 轻量级 MEC 仿真器

这个项目是一个面向移动边缘计算（MEC）卸载问题的轻量级 Python 仿真环境，用于早期世界模型强化学习和基线算法实验。它的目标不是替代 EdgeCloudSim、PureEdgeSim 或 iFogSim，而是在 Python/Gymnasium 生态里提供一个便于 DreamerV3、PPO、SAC 等算法接入的可控训练环境。

## 当前能模拟什么

- 多个移动用户在一维区域内移动；
- 一个或多个 edge/MEC 服务器；
- 随机任务到达或 trace 驱动任务到达；
- 异构任务类型、输出数据大小、截止期画像和优先级；
- 每个时间步选择部分用户进行任务卸载；
- 本地计算、上行传输、MEC 队列、MEC 执行，以及可选下行响应传输；
- 可选的同一 edge server 下共享上行带宽竞争；
- 可选的 cloud fallback，支持 local / edge / cloud 三类执行位置；
- edge 计算利用率、上行/下行网络使用量等实验指标；
- 轻量能耗和 cloud 成本指标；
- 基于延迟、丢弃任务、队列长度和完成任务数的奖励；
- SLA 奖励模式和截止期违约率等指标；
- Gymnasium wrapper 和 DreamerV3 环境桥接。

## 主要文件

- `sim/config.py`：场景参数、任务参数、奖励参数；
- `sim/entities.py`：用户、任务、服务器等数据结构；
- `sim/env.py`：核心 MEC 仿真环境；
- `sim/vector.py`：观测向量展开和动作向量转换；
- `sim/policies.py`：启发式基线策略；
- `sim/rollout.py`：采集强化学习或世界模型训练数据；
- `sim/scenarios.py`：命名实验场景模板；
- `sim/gym_wrapper.py`：Gymnasium 兼容封装；
- `sim/dreamer_wrapper.py`：DreamerV3 兼容封装；
- `eval_baselines.py`：启发式基线评估入口；
- `scripts/`：PPO、SAC、DreamerV3、结果聚合和归档脚本；
- `docs/`：设计说明、实验记录和阶段报告。

## 快速开始

运行一个简单演示：

```bash
python run_demo.py
```

采集一回合扁平化 transition：

```python
from sim import MECEnv, collect_episode, best_uplink_rate_policy

env = MECEnv()
transitions = collect_episode(env, best_uplink_rate_policy)
print(len(transitions), len(transitions[0].observation))
```

评估启发式基线：

```bash
python eval_baselines.py --episodes 20 --seed 7 --policy all
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

当前内置场景包括：

- `simple_single_edge`：单 edge server + SLA 奖励，用于兼容性对照；
- `multi_edge_sla`：三 edge server + SLA 奖励；
- `multi_edge_heterogeneous_sla`：三 edge server + 异构任务画像；
- `multi_edge_network_sla`：三 edge server + 异构任务 + 上行竞争 + 下行响应传输。
- `cloud_edge_sla`：三 edge server + 异构任务 + 严格网络模型 + cloud fallback。

显式命令行参数会覆盖场景模板中的同名配置，便于在固定场景上做小范围消融。

## 环境逻辑

每个仿真步执行以下流程：

1. 移动所有用户；
2. 生成新任务或读取 trace 任务；
3. 接收卸载动作；
4. 处理本地任务；
5. 上传被卸载的任务；
6. 处理 MEC 队列；
7. 如果启用下行传输，返回任务输出数据；
8. 丢弃超过 deadline 的任务；
9. 计算奖励并返回下一步观测。

## 观测

环境返回字典形式观测，主要包含：

- `step`：当前时间步；
- `users`：每个用户的状态列表；
- `server_queue_length`：MEC 服务器队列长度；
- `server_downlink_queue_length`：MEC 下行响应队列长度；
- `servers`：每个 edge server 的位置、计算能力、计算队列长度、下行队列长度和覆盖范围；
- `max_offloads_per_step`：每步最多可卸载用户数。

每个用户包含：

- `user_id`；
- `position`；
- `velocity`；
- `queue_length`；
- `current_task_size`；
- `current_task_remaining_cycles`；
- `current_task_type`；
- `current_task_output_size`；
- `current_task_priority`；
- `current_task_deadline_remaining`；
- `uplink_rate`。
- `best_edge_server_id`；
- `server_rates`：该用户到各 edge server 的近似上行速率、下行速率和可达性。

## 动作

当前动作是一个用户编号列表，表示本步选择哪些用户卸载任务：

```python
action = [1, 4, 7]
```

环境只接受前 `max_offloads_per_step` 个合法且不重复的用户编号。

在多 edge server 场景下，当前仍保持旧动作接口兼容：动作只选择用户，环境会根据 `edge_selection_policy` 自动选择目标 edge server。默认策略是 `nearest`，也可以使用 `least_loaded`。

也可以显式指定目标 edge server：

```python
action = [(1, 0), (4, 2), (7, 1)]
```

这表示分别把用户 `1`、`4`、`7` 的队首任务卸载到指定 edge server。

## 网络模型开关

默认配置保持旧行为，避免破坏已有 DreamerV3/PPO/SAC 冒烟入口。需要更接近论文实验时，可以打开：

- `enable_uplink_contention`：同一 edge server 下同时上传的任务共享上行带宽；
- `enable_downlink_transmission`：MEC 执行完成后，还需要通过下行链路返回 `output_size`；
- `enable_cloud_fallback`：允许显式把任务卸载到 cloud；
- `base_downlink_rate`：下行基础速率。

对应命令行参数：

```bash
python eval_baselines.py \
  --num-edge-servers 3 \
  --enable-uplink-contention \
  --enable-downlink-transmission \
  --enable-cloud-fallback
```

## 实验指标

`info` 和 `eval_baselines.py` 聚合结果中包含：

- `deadline_violation_rate`：截止期违约率；
- `avg_delay`：平均完成延迟；
- `avg_total_queue`：平均系统队列长度；
- `avg_edge_utilization`：平均 edge 计算利用率；
- `avg_uplink_data`：平均每步上行传输数据量；
- `avg_downlink_data`：平均每步下行传输数据量；
- `avg_network_data`：平均每步上下行总传输数据量。
- `avg_cloud_utilization`：平均 cloud 计算利用率；
- `avg_cloud_usage_ratio`：平均每步完成任务中的 cloud 完成占比；
- `avg_energy_used`：轻量能耗估计；
- `avg_cloud_cost`：cloud 计算成本估计。

这些指标用于区分策略是“单纯完成更多任务”，还是确实更好地利用 edge 资源和网络资源。

`eval_baselines.py` 写出的 CSV/JSONL 还会记录 `scenario`、`reward_preset`、`num_edge_servers`、`task_type_count`、`enable_uplink_contention`、`enable_downlink_transmission` 和 `enable_cloud_fallback` 等配置元数据，避免后续实验文件只剩指标、缺少场景来源。

## 奖励

当前支持两类奖励模式：

- `debug`：早期调试奖励；
- `sla`：更重视任务完成和截止期违约的奖励。

SLA 模式大致形式是：

```text
reward =
  completion_bonus * completed_tasks
  - delay_penalty * avg_delay
  - sla_violation_penalty * dropped_tasks
  - queue_penalty * total_queue
```

## 实验记录目录

以后所有新实验输出都应写入：

```text
experiment_records/
```

不要再把训练日志、模型文件、临时 CSV 分散放在项目根目录或旧的 `outputs/` 目录。

## 当前局限

当前环境仍是轻量级版本，尚未达到论文级 MEC 仿真完整度。主要缺口包括：

1. cloud-edge-device 层级和 cloud fallback；
2. 边云传输延迟；
3. 能耗和成本指标；
4. 更真实的 RSU/base station 关联；
5. 移动切换和服务迁移；
6. 结构化观测编码器。

当前已经加入多 edge server、显式目标服务器动作、异构任务模型、可选上行竞争、可选下行响应传输、cloud fallback、edge/cloud utilization、network usage、energy/cost 指标、命名场景模板、场景回归矩阵和 trace profile 转换。下一步应进入正式多 seed baseline 与 DRL 训练协议。
