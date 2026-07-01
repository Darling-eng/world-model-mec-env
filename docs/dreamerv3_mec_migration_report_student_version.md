# DreamerV3-MEC 迁移报告（学生汇报版）

## 1. 这个项目在做什么

本项目研究的是：

```text
World Model-Based Learning for MEC Task Offloading
```

中文可以理解为：

```text
基于世界模型强化学习的移动边缘计算任务卸载
```

核心问题是：在移动边缘计算场景中，用户任务应该本地执行、卸载到边缘服务器，还是进一步交给云端？我们希望用 DreamerV3 这类世界模型强化学习方法，学习一个能兼顾延迟、任务完成率和 SLA 违约率的卸载策略。

## 2. 为什么要用 DreamerV3

普通强化学习通常直接从环境交互中学习策略。DreamerV3 的特点是先学习一个世界模型，再在模型想象出的未来轨迹中优化策略。

这对 MEC 有潜在价值，因为 MEC 决策不是单步问题：

- 当前卸载会影响未来队列；
- 用户移动会影响未来链路质量；
- 任务 deadline 会影响策略风险；
- server 负载会影响后续任务延迟。

如果世界模型学得好，它可能比只看当前状态的启发式策略更能预判未来风险。

## 3. 已经完成的工作

### 3.1 搭建轻量 MEC 仿真器

当前仿真器支持：

- 多用户；
- 用户移动；
- 任务生成；
- 本地计算；
- 上行传输；
- MEC 队列；
- deadline 违约；
- SLA reward；
- trace-driven workload。

### 3.2 接入强化学习接口

已完成：

- Gymnasium wrapper；
- DreamerV3 wrapper；
- 轻量 PPO；
- Stable-Baselines3 SAC runner；
- Stable-Baselines3 PPO runner。

### 3.3 接入真实 trace

已经把老师提供或项目中的 trace 数据转换为可供仿真器读取的 CSV，使任务到达不再完全依赖随机生成。

### 3.4 建立 baseline

已有启发式策略：

- random；
- local_only；
- best_uplink；
- largest_queue。

已有算法入口：

- lightweight PPO；
- SB3 PPO；
- SAC；
- DreamerV3。

### 3.5 建立结果管理

已实现：

- 结果聚合脚本；
- mean/std 汇总；
- Colab 结果归档；
- `experiment_records/` 统一输出目录规范。

## 4. 当前发现

当前最强的启发式策略通常是 `largest_queue`。这说明任务队列长度对 SLA 和完成率有很强影响。

早期 DreamerV3、PPO、SAC 结果只能说明算法链路能跑通，不能作为最终性能结论。原因是：

- 训练步数短；
- seed 数少；
- 仿真环境还偏简单；
- 部分算法还没有正式多 seed 对比。

## 5. 当前最大问题

老师指出应该参考 EdgeCloudSim、PureEdgeSim、iFogSim 等成熟仿真环境，这个意见很关键。

当前环境缺少：

- 多 edge server / RSU；
- cloud tier；
- 任务类型；
- 网络竞争；
- 服务器异构性；
- 能耗、成本、利用率等指标；
- handover 和 service migration。

因此，现在最重要的不是继续调算法，而是先让仿真环境更像一个可以支撑论文的 MEC 场景。

## 6. 下一步路线

### 第一步：补多 edge server

在保持单服务器兼容的前提下，加入多个 edge server：

- 不同位置；
- 不同计算能力；
- 不同队列；
- 用户到不同服务器的链路质量。

### 第二步：扩展动作空间

从“选择哪些用户卸载”升级为：

```text
选择哪个用户的任务，卸载到哪个 edge server。
```

### 第三步：新增基线策略

例如：

- nearest_edge；
- least_loaded_edge；
- best_link_edge；
- largest_queue_with_least_loaded_server。

### 第四步：重新跑多 seed 实验

对比：

- heuristic；
- SB3 PPO；
- SAC；
- DreamerV3。

### 第五步：再做方法创新

根据结果选择：

- 不确定性感知卸载；
- 异构数值编码器；
- 生成式数字孪生表述。

## 7. 距离论文还有多久

如果稳步推进，现实估计：

```text
7-12 周形成较完整、较稳的第一版论文草稿。
```

如果走快速路线：

```text
4-6 周形成 workshop 或内部汇报级草稿。
```

但快速路线风险较高，因为当前仿真环境还不足以支撑强结论。

## 8. 给老师汇报时可以这样说

当前我们已经完成了 DreamerV3 与 MEC 环境的初步迁移，证明了算法接口、trace 输入、SLA reward 和结果记录链路可行。但根据 EdgeCloudSim、PureEdgeSim、iFogSim 等仿真器的建模维度，现有环境仍缺少多 edge 拓扑、cloud 层级、网络竞争和异构资源。因此下一阶段会优先补强仿真环境，再进行正式多 seed 算法对比。

## 9. 当前结论

DreamerV3 迁移已经不是最大阻碍。真正决定论文质量的是：

```text
仿真环境是否足够可信，实验结果是否能说明真实 MEC 卸载问题。
```

下一步应从多 edge server 拓扑开始。
