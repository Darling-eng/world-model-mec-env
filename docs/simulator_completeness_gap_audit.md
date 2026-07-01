# MEC 仿真环境完备性审计报告

日期：2026-07-01

## 为什么需要这份审计

当前 Python MEC 仿真器已经可以支持 Gymnasium、DreamerV3、trace-driven workload、SLA reward 和早期 baseline 对比。但是，老师提出应参考 EdgeCloudSim、PureEdgeSim、iFogSim 等成熟仿真环境，这说明下一阶段的重点不应只是继续训练算法，而应先确认仿真环境是否足以支撑论文级结论。

本项目不需要直接迁移到 Java 生态，也不需要完整复刻这些仿真器。更合理的路线是：把它们作为仿真完备性 checklist，然后只补充那些会影响论文可信度和强化学习问题定义的模块。

## 参考仿真器

| 仿真器 | 参考价值 | 对本项目的启发 |
| --- | --- | --- |
| EdgeCloudSim | MEC / mobile edge 性能评估 | 移动用户、edge/cloud 层级、网络延迟、edge 利用率、编排策略 |
| PureEdgeSim | cloud-edge-device 场景评估 | 大规模 edge 设置、任务编排、网络/能耗/资源指标 |
| iFogSim | IoT/fog 资源管理仿真 | fog-cloud 层级、应用模块、延迟、网络使用、能耗、成本 |
| iFogSim2 | iFogSim 扩展版本 | 移动性、集群、微服务管理、动态部署 |

参考来源：

- iFogSim：`https://arxiv.org/abs/1606.02007`
- iFogSim2：`https://arxiv.org/abs/2109.05636`
- EdgeCloudSim renovation/reference：`https://arxiv.org/abs/2109.03901`
- EISim / PureEdgeSim extension：`https://arxiv.org/abs/2311.01224`

## 当前 Python 仿真器已有能力

- 代码紧凑，适合 Python 强化学习实验；
- 已有 Gymnasium wrapper 和 DreamerV3 bridge；
- 支持通过 `task_trace_path` 接入 trace-driven workload；
- 已有 SLA reward preset；
- 已有任务完成数、丢弃任务数、deadline violation rate、平均延迟、队列长度等指标；
- 已接入启发式基线、轻量 PPO、SAC 和 DreamerV3 运行入口；
- 已有结果聚合脚本和 `experiment_records/` 输出规范。

## 当前主要简化

- 已加入多 edge server 基础拓扑，但动作空间仍保持旧的 user-id 卸载接口；
- 没有 cloud tier；
- 没有多个 RSU / base station 关联；
- 没有 handover 或 service migration；
- 没有 task type / application type；
- 没有 downlink response transmission；
- 没有 energy、cost、utilization 等指标；
- 网络模型只是简单距离衰减 uplink rate，没有带宽竞争；
- observation 是扁平数值向量，没有按用户、任务、链路、服务器分组。

## 完备性对照矩阵

| 建模维度 | 当前状态 | 参考仿真器期望 | 对论文风险 | 优先级 |
| --- | --- | --- | --- | --- |
| 用户移动性 | 一维反弹移动 | 移动模型和 edge 关联 | 中等：可做 smoke，不够真实 | P2 |
| 多 edge server / RSU | 基础拓扑已加入 | 多 edge 拓扑、关联和编排策略 | 中高：还需要动作空间和基线策略跟进 | P0 |
| cloud tier | 缺失 | edge-cloud 层级 | 中等：需要 local/edge/cloud 对比时必须补 | P1 |
| 任务模型 | 单一任务类型 | 任务类型、输入大小、输出大小、cycles、deadline | 高：影响 SLA 可信度 | P0 |
| 网络模型 | 只有 uplink 距离衰减 | uplink/downlink、LAN/WAN、带宽竞争 | 高：卸载决策高度依赖网络 | P0 |
| 服务器资源 | 单 FIFO 队列 | 异构服务器、CPU、利用率 | 高：没有异构资源就缺少资源选择问题 | P0 |
| 能耗模型 | 缺失 | 设备/edge/cloud 能耗 | 中等：若论文不做能耗可后置 | P2 |
| 成本模型 | 缺失 | cloud/edge 成本 | 低到中：成本感知论文才需要 | P3 |
| 服务模块 | 缺失 | iFogSim 应用模块或 iFogSim2 微服务 | 中等：与数字孪生/服务编排叙事相关 | P2 |
| handover / migration | 缺失 | 移动感知部署和迁移 | 中高：可作为后续创新点 | P2 |
| 指标体系 | SLA、delay、queue、reward | latency、energy、network use、cost、utilization | 高：至少应补 utilization 和 network usage | P1 |
| 可复现性 | seed、CSV/JSONL、聚合脚本 | scenario config 和原始输出归档 | 中等：已开始改善 | P1 |

## 推荐补强顺序

### P0：最低论文可信环境

1. 多 edge 拓扑：
   - 已增加 `num_edge_servers`；
   - 每个 edge server 已有位置、CPU rate、队列和覆盖范围；
   - 当前仍保留单服务器兼容模式。

2. edge 关联和动作语义：
   - observation 暴露用户可达 edge server；
   - action 不只选择是否卸载，还要选择目标 edge server；
   - 保留旧 top-k user scoring 模式作为兼容场景。

3. 任务类型：
   - 增加 task type、input size、output size、CPU cycles、deadline、priority；
   - trace loader 将 trace 映射为任务类别或 workload profile。

4. 网络竞争：
   - 区分 uplink、downlink 和 edge-cloud forwarding；
   - 同一 edge server 下用户共享带宽。

### P1：实验可信层

5. cloud fallback：
   - 增加高算力但高 WAN delay 的 cloud server；
   - 构成 local / edge / cloud 三类选择。

6. 指标扩展：
   - 增加 edge utilization、network usage、mean response time、cloud usage ratio；
   - 保留 `deadline_violation_rate` 作为核心 SLA 指标。

7. 场景配置：
   - 增加命名场景：`simple_single_edge`、`multi_edge_trace_sla`、`cloud_edge_sla`；
   - 输出目录使用 `experiment_records/<scenario>/<algorithm>/<seed>/`。

### P2/P3：方法创新层

8. handover 和 migration：
   - 用户移动导致接入 edge 变化；
   - 多 edge 和任务网络模型稳定后再加入迁移成本。

9. energy / cost：
   - 加入本地计算能耗、传输能耗、edge/cloud 成本；
   - 只有论文叙事需要时优先级才上升。

10. 结构化 observation encoder：
   - 不再只依赖扁平向量；
   - 按用户、任务、链路、服务器分组；
   - 与后续 heterogeneous numeric encoder 创新点直接相关。

## 下一步最具体的实现方向

第 23 轮已经实现基础多 edge server 拓扑。下一轮应实现：

```text
扩展动作空间和启发式基线，使策略能够显式选择目标 edge server。
```

完成标准：

- 原有 single-server smoke test 继续通过；
- 新动作接口能表达 `(user_id, edge_server_id)`；
- 保留旧 user-id 动作接口作为兼容模式；
- 新增 `nearest_edge`、`least_loaded_edge` 或类似显式 server 选择基线；
- 更新 Gymnasium wrapper 时不破坏 DreamerV3/PPO/SAC 的旧 smoke 配置。

## 距离论文的现实估计

如果保持稳定推进，并且 Colab/GPU 资源可用：

- 环境升级到论文可信程度：2-3 周；
- 重新跑 baseline 和多 seed 实验表：1-2 周；
- 方法改进和消融实验：2-4 周；
- 论文初稿、相关工作、实验分析、局限性和修改：2-3 周。

从当前状态到一版较稳的完整论文草稿，现实估计是：

```text
7-12 周
```

如果走更快但风险更高的路线：

```text
4-6 周
```

快速路线更接近 workshop 或内部汇报级别，不一定足以支撑正式投稿。

当前关键路径不是继续训练算法，而是让仿真环境足够可信，使训练结果真正有解释价值。
