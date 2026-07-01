# 轻量级 MEC 仿真器

这个项目是一个面向移动边缘计算（MEC）卸载问题的轻量级 Python 仿真环境，用于早期世界模型强化学习和基线算法实验。它的目标不是替代 EdgeCloudSim、PureEdgeSim 或 iFogSim，而是在 Python/Gymnasium 生态里提供一个便于 DreamerV3、PPO、SAC 等算法接入的可控训练环境。

## 当前能模拟什么

- 多个移动用户在一维区域内移动；
- 一个或多个 edge/MEC 服务器；
- 随机任务到达或 trace 驱动任务到达；
- 每个时间步选择部分用户进行任务卸载；
- 本地计算、上行传输、MEC 队列和 MEC 执行；
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

## 环境逻辑

每个仿真步执行以下流程：

1. 移动所有用户；
2. 生成新任务或读取 trace 任务；
3. 接收卸载动作；
4. 处理本地任务；
5. 上传被卸载的任务；
6. 处理 MEC 队列；
7. 丢弃超过 deadline 的任务；
8. 计算奖励并返回下一步观测。

## 观测

环境返回字典形式观测，主要包含：

- `step`：当前时间步；
- `users`：每个用户的状态列表；
- `server_queue_length`：MEC 服务器队列长度；
- `servers`：每个 edge server 的位置、计算能力、队列长度和覆盖范围；
- `max_offloads_per_step`：每步最多可卸载用户数。

每个用户包含：

- `user_id`；
- `position`；
- `velocity`；
- `queue_length`；
- `current_task_size`；
- `current_task_remaining_cycles`；
- `uplink_rate`。
- `best_edge_server_id`；
- `server_rates`：该用户到各 edge server 的近似上行速率和可达性。

## 动作

当前动作是一个用户编号列表，表示本步选择哪些用户卸载任务：

```python
action = [1, 4, 7]
```

环境只接受前 `max_offloads_per_step` 个合法且不重复的用户编号。

在多 edge server 场景下，当前仍保持旧动作接口兼容：动作只选择用户，环境会根据 `edge_selection_policy` 自动选择目标 edge server。默认策略是 `nearest`，也可以使用 `least_loaded`。

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

1. cloud-edge-device 层级；
2. 任务类型、输出数据大小和 deadline 分布；
3. 上下行网络、带宽竞争和边云传输延迟；
4. 更完整的异构资源利用率、能耗和成本指标；
5. 移动切换和服务迁移；
6. 显式选择目标 edge server 的动作空间。

当前已经加入多 edge server 的基础拓扑，下一步应扩展动作空间和启发式基线，让策略能显式选择目标 edge server。
