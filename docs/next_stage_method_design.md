# 下一阶段方法设计：World Model（世界模型）用于 MEC（移动边缘计算）卸载

当前阶段已经完成 MEC simulator（移动边缘计算仿真器）、heuristic baseline（启发式基线）和 DreamerV3（世界模型强化学习算法）接入验证。下一阶段的方法强化不应该只追求“把算法跑得更久”，而是要围绕 MEC（移动边缘计算）场景本身提炼出区别于 13 号论文的贡献点。

## 1. Uncertainty-Aware Offloading（不确定性感知卸载）

MEC（移动边缘计算）卸载的关键风险在于：当前观测到的 channel state（信道状态）、server load（服务器负载）和 task queue（任务队列）并不一定能准确代表未来状态。World model（世界模型）可以通过 latent dynamics（隐空间动态）预测未来多步状态，因此可以进一步估计 predictive uncertainty（预测不确定性）。

后续可以把 uncertainty（不确定性）用于 risk-aware decision（风险感知决策）：当模型对未来链路或队列预测不确定时，策略不只优化平均 reward（奖励），还要避免高风险 offloading（卸载）动作。例如，对于 deadline-sensitive task（截止期敏感任务），如果 world model（世界模型）预测边缘执行收益高但不确定性也高，策略可以倾向于更保守的 local execution（本地执行）或选择链路更稳定的用户。

## 2. Structured Encoder（结构化编码器）/ Transformer Encoder（Transformer 编码器）

当前 observation（观测）已经被压平成固定长度向量，这适合快速接入 DreamerV3（世界模型强化学习算法），但它会弱化 MEC（移动边缘计算）状态里的结构信息。实际上，每个 user（用户）都有一组局部特征，例如 position（位置）、velocity（速度）、queue length（队列长度）、task size（任务大小）和 uplink rate（上行链路速率）；server（服务器）也有独立的全局状态。

下一步可以把 encoder（编码器）从简单 flat vector encoder（平铺向量编码器）升级为 user-wise structured encoder（按用户结构化编码器）：先对每个 user（用户）的局部状态共享编码，再和 global state（全局状态）融合。如果继续增强，可以引入 attention（注意力机制）或 Transformer encoder（Transformer 编码器），让模型自动学习不同 user（用户）之间的竞争关系、队列耦合和链路差异。

## 3. Generative Digital Twin（生成式数字孪生）

World model（世界模型）在本文方向中不只是 RL（强化学习）里的辅助模块，更适合包装成 learned generative digital twin（学习得到的生成式数字孪生）。它的价值在于能够回答 what-if analysis（反事实分析）问题：如果当前把某个 task（任务）卸载到 MEC server（边缘服务器），未来几步的 queue congestion（队列拥塞）、delay（时延）和 dropped tasks（丢弃任务）会如何变化。

这个 framing（叙事框架）可以把方法从“使用 DreamerV3（世界模型强化学习算法）训练策略”提升为“构建可预测、可想象、可规划的 MEC digital twin（移动边缘计算数字孪生）”。对 TSC/TMC（服务计算/移动计算期刊）来说，这比单纯套用 world model（世界模型）更容易体现服务计算场景价值。

## 阶段性落点

短期内建议先不同时展开三条创新线，而是按照实验成熟度推进：

1. 先把 metrics（指标）和 baseline（基线）补齐，保证实验闭环可复现。
2. 再通过 PPO（近端策略优化）和 DreamerV3（世界模型强化学习算法）建立正式 RL comparison（强化学习对比）。
3. 最后优先选择 uncertainty-aware offloading（不确定性感知卸载）作为第一条方法创新线，因为它和 MEC（移动边缘计算）的 SLA（服务等级协议）、deadline（截止期）和 risk-aware decision（风险感知决策）联系最直接。
