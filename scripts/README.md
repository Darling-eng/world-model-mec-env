# 脚本入口

## 评估 heuristic baselines（启发式基线）

运行全部启发式基线：

```bash
python eval_baselines.py --episodes 10 --seed 7
```

单独运行某个基线：

```bash
python eval_baselines.py --policy random --episodes 10 --seed 7
python eval_baselines.py --policy local_only --episodes 10 --seed 7
python eval_baselines.py --policy best_uplink --episodes 10 --seed 7
python eval_baselines.py --policy largest_queue --episodes 10 --seed 7
```

保存聚合指标：

```bash
python eval_baselines.py --episodes 50 --seed 7 --output outputs/baselines.csv
python eval_baselines.py --episodes 50 --seed 7 --output outputs/baselines.jsonl
```

## 训练轻量级 PPO（近端策略优化）与 MEC（移动边缘计算）

这个入口是 NumPy 实现的轻量级 PPO smoke baseline（冒烟基线），用于在接入完整 DRL library（深度强化学习库）前做短训练检查。

```bash
python scripts/run_ppo_mec.py --steps 2000 --rollout-steps 256 --eval-episodes 5 --seed 7
```

## 训练 SAC（软演员评论家）与 MEC

这个入口使用 Stable-Baselines3（稳定基线3库）的 SAC baseline（软演员评论家基线），适合在 Colab 上运行：

```bash
python scripts/run_sac_mec.py \
  --trace /content/world-model-mec-env/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --steps 10000 \
  --eval-episodes 5 \
  --seed 7 \
  --log-dir /content/world-model-mec-env/colab_results/sac_trace_sla_seed7_10k
```

## 在 Colab 运行 DreamerV3 与 MEC

合成负载 smoke run（冒烟运行）示例：

```bash
python /content/world-model-mec-env/scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 100
```

真实 trace-driven workload（真实轨迹驱动负载）加 SLA reward（服务等级协议奖励）示例：

```bash
python /content/world-model-mec-env/scripts/run_dreamer_mec.py \
  --dreamer-dir /content/dreamerv3 \
  --trace /content/world-model-mec-env/trace_alibaba_sample_codex.csv \
  --reward-preset sla \
  --configs debug \
  --task gym_MECDreamerBox-v0 \
  --run.envs 1 \
  --run.eval_envs 0 \
  --run.steps 10000
```
