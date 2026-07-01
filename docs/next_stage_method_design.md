# 下一阶段方法设计

主题：基于世界模型的 MEC 卸载决策  
当前阶段：环境可信度补强优先于算法调参

## 当前判断

项目已经完成了早期 pipeline：

- Python MEC 仿真器；
- trace-driven workload；
- SLA reward；
- 启发式 baseline；
- PPO/SAC/DreamerV3 接入；
- 结果聚合和归档。

但是，当前环境仍偏简化。如果直接继续训练 DreamerV3、SAC 或 PPO，结果可能只能说明算法能在 toy simulator 上运行，不能充分支撑论文级 MEC offloading 结论。

因此下一阶段方法设计必须遵循：

```text
先补仿真环境，再做算法创新。
```

## 研究主线

目标问题：

```text
在 trace-driven MEC 场景中，世界模型强化学习能否学习到更稳定的 SLA-aware offloading 策略？
```

要让这个问题成立，环境至少应包含：

- 多 edge server；
- 用户与 edge server 的动态关联；
- 异构服务器资源；
- 任务类型和 deadline；
- 网络竞争；
- SLA 指标；
- 可复现实验配置。

## 方法创新候选

### 1. 不确定性感知卸载

世界模型可以预测未来队列、延迟或任务违约风险。策略不应只看期望 reward，还应考虑预测不确定性。

可能做法：

- 用模型预测误差估计风险；
- 对高不确定性 action 加惩罚；
- 在 deadline 敏感任务上更保守；
- 将不确定性作为 observation 的额外输入。

适用条件：

- DreamerV3 在基础环境中能稳定训练；
- 已有多 seed 对比表；
- 可以计算或近似 world model prediction error。

### 2. 异构数值编码器

当前 observation 是扁平向量，混合了用户、任务、链路和服务器特征。这不利于世界模型学习结构化关系。

可能做法：

- 用户特征单独编码；
- 任务特征单独编码；
- 链路特征单独编码；
- 服务器特征单独编码；
- 最后融合为策略和世界模型输入。

适用条件：

- 多 edge server topology 已实现；
- observation 已包含 server-level 和 link-level 特征。

### 3. 生成式数字孪生表述

DreamerV3 可以被解释为学习一个 MEC 环境的生成式动态模型，用于在 latent imagination 中评估卸载策略。

论文表述可以是：

```text
构建面向 MEC offloading 的轻量级生成式数字孪生，使策略能够在预测的未来队列和 SLA 风险中优化卸载决策。
```

注意：这个表述必须有实验支撑，不能只作为概念包装。

## 实验路线

### 阶段一：环境补强

优先实现：

1. 多 edge server；
2. server heterogeneity；
3. 任务类型；
4. 网络竞争；
5. 更完整的 metrics。

### 阶段二：基线重跑

固定：

- trace；
- reward；
- seed；
- evaluation episodes；
- scenario config；
- output schema。

对比：

- heuristic；
- SB3 PPO；
- SAC；
- DreamerV3。

### 阶段三：方法增强

根据结果选择：

- 如果 DreamerV3 明显弱于 SAC：优先做 observation encoder；
- 如果 DreamerV3 reward 好但 deadline 差：优先做 uncertainty-aware offloading；
- 如果 heuristic 仍然最强：检查动作空间和 reward 是否仍然限制学习。

## 当前下一步

下一步不是写新算法，而是实现：

```text
多 edge server topology，并保持 single-server 兼容。
```

这是后续所有方法创新的地基。
