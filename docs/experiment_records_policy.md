# 实验记录目录规范

以后所有新实验输出统一写入：

```text
experiment_records/
```

这样可以避免训练日志、模型 checkpoint、原始指标、汇总表和 Colab 归档结果散落在 `outputs/`、项目根目录、`ppo_*` 目录或临时 CSV 文件中。

## 推荐目录结构

```text
experiment_records/
  baselines/
  ppo_mec/
  sb3_ppo_mec/
  sac_mec/
  dreamerv3/
  summaries/
  archives/
  legacy/
```

## 使用规则

- 新实验结果不要写到项目根目录；
- 新训练日志和模型文件不要写到旧的 `outputs/`；
- 可复用的输入 trace 暂时保留在 `csv/`；
- 论文级汇总表放到 `experiment_records/summaries/`；
- Colab 下载或归档结果放到 `experiment_records/archives/`；
- 仍有参考价值但不适合作为正式结果的旧 smoke test 输出放到 `experiment_records/legacy/`；
- `experiment_records/` 属于生成物目录，不应提交到 git。

后续如果项目继续扩大，可以再把输入数据迁移到独立的 `data/` 目录，把 `csv/` 只作为历史兼容目录。
