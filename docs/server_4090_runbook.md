# 4090 服务器运行手册

适用配置：RTX 4090 24GB，Ubuntu 22.04，PyTorch CUDA 12.6，Python 3.11。

## 1. 验证 GPU 和基础环境

在 JupyterLab 的 Terminal 中执行：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期结果：`torch.cuda.is_available()` 输出 `True`，显卡名称包含 `RTX 4090`。

## 2. 获取项目代码

如果服务器能访问 GitHub，优先使用：

```bash
cd /root
git clone https://github.com/Darling-eng/world-model-mec-env.git
cd world-model-mec-env
```

如果 GitHub 下载很慢，可以在本地把项目压缩上传到 JupyterLab，再解压进入项目目录。

## 3. 安装实验依赖

```bash
python -m pip install -U pip
pip install -r requirements-server.txt
```

如果 `stable-baselines3` 安装后自动替换了 torch，需要重新检查：

```bash
python -c "import torch, stable_baselines3, gymnasium; print(torch.__version__, torch.cuda.is_available()); print(stable_baselines3.__version__, gymnasium.__version__)"
```

## 4. 先跑仿真器预检

```bash
python scripts/run_experiment_preflight.py \
  --output-root experiment_records/preflight/server_4090_initial
```

预期：所有检查为 `ok`，并生成：

```text
experiment_records/preflight/server_4090_initial/manifest.json
```

## 5. 跑 1000 step SB3-PPO 小训练

```bash
python scripts/run_sb3_ppo_mec.py \
  --steps 1000 \
  --eval-episodes 2 \
  --seed 7 \
  --scenario multi_edge_network_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/server_smoke/sb3_ppo_1k_seed7
```

预期生成：

```text
experiment_records/server_smoke/sb3_ppo_1k_seed7/metrics.jsonl
experiment_records/server_smoke/sb3_ppo_1k_seed7/ppo_model.zip
```

## 6. 小训练通过后再跑正式实验

先用单 seed 验证 5000 step：

```bash
python scripts/run_sb3_ppo_mec.py \
  --steps 5000 \
  --eval-episodes 5 \
  --seed 7 \
  --scenario cloud_edge_sla \
  --trace experiment_records/trace_profiles/alibaba_light_normal_urgent.csv \
  --reward-preset sla \
  --log-dir experiment_records/server_smoke/sb3_ppo_cloud_edge_5k_seed7
```

确认 `metrics.jsonl` 正常后，再运行 50k 多 seed 正式批次。

## 7. 长时间训练建议

JupyterLab 页面断开可能影响交互体验。正式跑长实验时建议使用 `tmux`：

```bash
tmux new -s mec
```

进入 `tmux` 后运行训练命令。断开后重新进入：

```bash
tmux attach -t mec
```

## 8. 结果管理

所有新实验结果统一写入：

```text
experiment_records/
```

不要把大批量实验结果提交到 GitHub。需要下载结果时，优先下载对应的 `experiment_records/...` 子文件夹。
