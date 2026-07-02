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

- 已加入多 edge server 基础拓扑，并支持显式 `(user_id, edge_server_id)` 动作；
- 已加入可选 cloud fallback；
- 没有多个 RSU / base station 关联；
- 没有 handover 或 service migration；
- 已加入 task type / application type 的轻量画像；
- 已加入可选 downlink response transmission；
- 已加入 edge utilization 和 network usage 指标；
- 已加入命名场景模板和基础场景回归矩阵，并可在 baseline 输出中记录关键配置元数据；
- 已加入轻量 energy 和 cloud cost 指标；
- 网络模型已支持可选上行带宽竞争，但仍缺少 edge-cloud forwarding 和更细致的 LAN/WAN 模型；
- observation 是扁平数值向量，没有按用户、任务、链路、服务器分组。

## 完备性对照矩阵

| 建模维度 | 当前状态 | 参考仿真器期望 | 对论文风险 | 优先级 |
| --- | --- | --- | --- | --- |
| 用户移动性 | 一维反弹移动 | 移动模型和 edge 关联 | 中等：可做 smoke，不够真实 | P2 |
| 多 edge server / RSU | 基础拓扑和显式目标动作已加入 | 多 edge 拓扑、关联和编排策略 | 中等：还需要更正式场景和指标 | P0 |
| cloud tier | 已加入可选 cloud fallback | edge-cloud 层级 | 低到中：可支撑 local/edge/cloud 雏形对比 | P1 |
| 任务模型 | 已加入异构任务画像和 trace profile 映射 | 任务类型、输入大小、输出大小、cycles、deadline | 低到中：正式实验需固定 profile | P0 |
| 网络模型 | 已加入可选上行竞争和下行响应 | uplink/downlink、LAN/WAN、带宽竞争 | 中高：仍缺 edge-cloud forwarding | P0 |
| 服务器资源 | 单 FIFO 队列 | 异构服务器、CPU、利用率 | 高：没有异构资源就缺少资源选择问题 | P0 |
| 能耗模型 | 已加入轻量估计 | 设备/edge/cloud 能耗 | 低到中：可作为附加指标，不宜过度声称 | P2 |
| 成本模型 | 已加入 cloud cost 雏形 | cloud/edge 成本 | 低到中：成本感知论文才需要 | P3 |
| 服务模块 | 缺失 | iFogSim 应用模块或 iFogSim2 微服务 | 中等：与数字孪生/服务编排叙事相关 | P2 |
| handover / migration | 缺失 | 移动感知部署和迁移 | 中高：可作为后续创新点 | P2 |
| 指标体系 | 已有 SLA、delay、queue、reward、edge utilization、network usage | latency、energy、network use、cost、utilization | 中等：仍缺 energy/cost/cloud usage | P1 |
| 可复现性 | seed、CSV/JSONL、聚合脚本、命名 scenario config、基础回归 manifest | scenario config 和原始输出归档 | 中等：仍需正式多 seed 协议和原始配置归档 | P1 |

## 推荐补强顺序

### P0：最低论文可信环境

1. 多 edge 拓扑：
   - 已增加 `num_edge_servers`；
   - 每个 edge server 已有位置、CPU rate、队列和覆盖范围；
   - 当前仍保留单服务器兼容模式。

2. edge 关联和动作语义：
   - observation 已暴露用户可达 edge server；
   - action 已支持选择目标 edge server；
   - 已保留旧 top-k user scoring 模式作为兼容场景。

3. 任务类型：
   - 已增加 task type、input size、output size、CPU cycles、deadline、priority；
   - trace loader 已支持任务类别和 workload profile 可选字段。

4. 网络竞争：
   - 已区分 uplink 和 downlink；
   - 已支持同一 edge server 下用户共享上行带宽；
   - 尚未加入 edge-cloud forwarding。

### P1：实验可信层

5. cloud fallback：
   - 已增加高算力但高 WAN delay 的 cloud server；
   - 已构成 local / edge / cloud 三类选择；
   - baseline 中已有 `cloud_edge` 策略。

6. 指标扩展：
   - 已增加 edge utilization 和 network usage；
   - 后续可增加 mean response time、cloud usage ratio、energy 和 cost；
   - 保留 `deadline_violation_rate` 作为核心 SLA 指标。

7. 场景配置：
   - 已增加命名场景：`simple_single_edge`、`multi_edge_sla`、`multi_edge_heterogeneous_sla`、`multi_edge_network_sla`；
   - baseline 输出已记录 `scenario`、`reward_preset`、`num_edge_servers`、`task_type_count` 和网络开关；
   - 已增加 `scripts/run_scenario_regression.py`，可生成基础场景回归 CSV 和 `manifest.json`；
   - 后续正式训练输出仍应收敛到 `experiment_records/<scenario>/<algorithm>/<seed>/`。

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

第 23-32 轮已经连续补上基础多 edge server 拓扑、显式目标 edge 动作、异构任务模型、可选上行竞争、可选下行响应传输、cloud fallback、资源/网络/能耗/成本指标、命名场景模板、基础场景回归矩阵和 trace profile 映射。下一阶段应转入：

```text
正式多 seed baseline、PPO/SAC/DreamerV3 训练协议和论文实验表。
```

完成标准：

- 原有 single-server smoke test 继续通过；
- 旧 Gymnasium wrapper 观测形状继续兼容；
- 场景回归继续通过；
- trace profile、cloud fallback 和 energy/cost 指标能进入正式实验记录；
- 新实验输出继续进入 `experiment_records/`；
- 不能把旧冒烟结果和新严格网络模型结果直接混在同一张正式表中比较。

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
