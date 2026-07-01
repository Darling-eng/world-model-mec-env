# DreamerV3 迁移到 MEC 卸载场景的可行性报告

## 1. 结论概览

DreamerV3 可以迁移到当前 Python MEC 仿真环境中，用于验证世界模型强化学习在任务卸载问题上的可行性。当前代码已经具备最小闭环：

- MEC 环境；
- Gymnasium wrapper；
- DreamerV3 环境注册；
- trace-driven workload；
- SLA reward；
- baseline 评估和结果聚合。

但是，当前仿真环境仍偏轻量，不能直接支撑强论文结论。DreamerV3 的迁移是可行的，但更重要的是先提高仿真环境的可信度。

## 2. 为什么 DreamerV3 适合尝试

MEC offloading 是一个具有时序依赖的决策问题：

- 当前卸载会影响未来队列；
- 用户移动会影响未来链路质量；
- 任务 deadline 会让短期和长期收益冲突；
- server 负载会影响后续任务延迟。

DreamerV3 的优势在于学习世界模型，并在 latent imagination 中优化策略。理论上，它可以比单步启发式策略更好地处理未来队列、延迟和 SLA 风险。

## 3. 当前环境适配情况

已完成：

- `sim/env.py`：核心 MEC 环境；
- `sim/gym_wrapper.py`：Gymnasium 兼容接口；
- `sim/dreamer_wrapper.py`：DreamerV3 环境封装；
- `scripts/run_dreamer_mec.py`：DreamerV3 启动入口；
- `sim/gym_registration.py`：环境注册；
- `csv/trace_alibaba_sample_codex.csv`：trace workload；
- `scripts/aggregate_experiment_results.py`：结果聚合。

当前 DreamerV3 能够在 trace + SLA reward 设置下完成 smoke run，说明接口层面已经打通。

## 4. 当前主要风险

### 4.1 仿真器过于简化

当前环境只有单 MEC server、一维移动、简单 uplink rate 和 FIFO 队列。这足以验证算法接口，但不足以代表真实 MEC 场景。

### 4.2 动作空间仍然简单

当前动作本质上是选择哪些用户卸载，而不是选择卸载到哪个 edge server 或 cloud。缺少多目标资源选择，限制了 RL 策略的发挥空间。

### 4.3 observation 是扁平向量

用户、任务、链路、服务器特征混在一个向量里，可能不利于世界模型学习结构化动态。

### 4.4 baseline 还不完整

轻量 PPO 只能作为 smoke baseline。正式对比需要 SB3 PPO、SAC、DreamerV3 和启发式策略在相同配置下多 seed 运行。

## 5. 可行迁移路线

### 阶段一：接口验证

已完成：

- DreamerV3 能启动；
- 环境能返回 observation/reward/done/info；
- trace 和 reward preset 能传入环境；
- metrics 能落盘。

### 阶段二：环境可信度补强

优先补：

1. 多 edge server；
2. server heterogeneity；
3. task type 和 deadline 分布；
4. 网络带宽竞争；
5. utilization / network usage 指标。

### 阶段三：正式实验

固定：

- scenario；
- trace；
- reward；
- seed；
- evaluation episodes；
- output schema。

对比：

- heuristic；
- SB3 PPO；
- SAC；
- DreamerV3。

### 阶段四：方法创新

根据实验结果选择：

- uncertainty-aware offloading；
- heterogeneous numeric encoder；
- generative digital twin framing。

## 6. 对论文的意义

如果只是把 DreamerV3 跑在简单 MEC 环境上，论文贡献偏弱。

更合理的贡献表述是：

```text
参考主流 edge/fog 仿真器建模维度，构建一个面向世界模型强化学习的轻量 MEC 卸载环境，并验证 DreamerV3 在 trace-driven SLA offloading 任务中的潜力和局限。
```

这样论文会同时包含：

- 环境构建；
- 世界模型方法；
- baseline 对比；
- SLA 指标；
- 消融实验；
- 局限性分析。

## 7. 当前结论

DreamerV3 迁移可行，但不是当前最大风险。当前最大风险是仿真环境不够完整，导致算法结果缺少论文说服力。

下一步应优先实现：

```text
多 edge server topology，并保持 single-server 兼容。
```

在此基础上，再回到 DreamerV3、PPO、SAC 的正式多 seed 对比。
