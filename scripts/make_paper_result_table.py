from __future__ import annotations

import argparse
import csv
from pathlib import Path


PRIMARY_METRICS = (
    "total_reward_mean",
    "deadline_violation_rate_mean",
    "completed_tasks_mean",
    "avg_delay_mean",
    "avg_energy_used_mean",
    "avg_cloud_cost_mean",
)
DEFAULT_ANCHORS = ("best_uplink", "largest_queue", "nearest_edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paper-ready MEC result tables from summary CSV.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiment_records/preflight/iteration37/aggregate_check_summary.csv"),
        help="Mean/std summary CSV produced by aggregate_experiment_results.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_records/paper_tables/iteration38_result_table.csv"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("experiment_records/paper_tables/iteration38_result_table.md"),
    )
    parser.add_argument("--anchors", nargs="+", default=list(DEFAULT_ANCHORS))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: str | None, digits: int = 4) -> str:
    parsed = to_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def group_by_scenario(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("phase", "eval") != "eval":
            continue
        groups.setdefault(str(row.get("scenario", "")), []).append(row)
    return groups


def best_anchor(rows: list[dict], anchors: list[str]) -> dict | None:
    candidates = [row for row in rows if row.get("algorithm") in anchors]
    return max(candidates, key=lambda row: to_float(row.get("total_reward_mean")) or float("-inf"), default=None)


def compare_to_anchor(row: dict, anchor: dict | None) -> tuple[str, str, str]:
    if anchor is None:
        return "", "", ""
    reward = to_float(row.get("total_reward_mean"))
    anchor_reward = to_float(anchor.get("total_reward_mean"))
    ddl = to_float(row.get("deadline_violation_rate_mean"))
    anchor_ddl = to_float(anchor.get("deadline_violation_rate_mean"))
    completed = to_float(row.get("completed_tasks_mean"))
    anchor_completed = to_float(anchor.get("completed_tasks_mean"))
    reward_gap = "" if reward is None or anchor_reward is None else f"{reward - anchor_reward:.4f}"
    ddl_gap = "" if ddl is None or anchor_ddl is None else f"{ddl - anchor_ddl:.4f}"
    completed_gap = "" if completed is None or anchor_completed is None else f"{completed - anchor_completed:.4f}"
    return reward_gap, ddl_gap, completed_gap


def build_table(rows: list[dict], anchors: list[str]) -> list[dict]:
    table_rows: list[dict] = []
    for scenario, scenario_rows in sorted(group_by_scenario(rows).items()):
        anchor = best_anchor(scenario_rows, anchors)
        anchor_name = anchor.get("algorithm", "") if anchor else ""
        for row in sorted(scenario_rows, key=lambda item: (item.get("algorithm", ""), item.get("runs", ""))):
            reward_gap, ddl_gap, completed_gap = compare_to_anchor(row, anchor)
            table_rows.append(
                {
                    "scenario": scenario,
                    "algorithm": row.get("algorithm", ""),
                    "runs": row.get("runs", ""),
                    "seeds": row.get("seeds", ""),
                    "anchor": anchor_name,
                    "reward_mean": fmt(row.get("total_reward_mean"), 3),
                    "reward_std": fmt(row.get("total_reward_std"), 3),
                    "reward_gap_vs_anchor": reward_gap,
                    "deadline_violation_rate_mean": fmt(row.get("deadline_violation_rate_mean"), 4),
                    "deadline_violation_gap_vs_anchor": ddl_gap,
                    "completed_tasks_mean": fmt(row.get("completed_tasks_mean"), 2),
                    "completed_gap_vs_anchor": completed_gap,
                    "avg_delay_mean": fmt(row.get("avg_delay_mean"), 3),
                    "avg_energy_used_mean": fmt(row.get("avg_energy_used_mean"), 3),
                    "avg_cloud_cost_mean": fmt(row.get("avg_cloud_cost_mean"), 5),
                }
            )
    return table_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "algorithm",
        "runs",
        "seeds",
        "anchor",
        "reward_mean",
        "reward_std",
        "reward_gap_vs_anchor",
        "deadline_violation_rate_mean",
        "deadline_violation_gap_vs_anchor",
        "completed_tasks_mean",
        "completed_gap_vs_anchor",
        "avg_delay_mean",
        "avg_energy_used_mean",
        "avg_cloud_cost_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict]) -> str:
    fields = [
        "scenario",
        "algorithm",
        "runs",
        "anchor",
        "reward_mean",
        "reward_gap_vs_anchor",
        "deadline_violation_rate_mean",
        "deadline_violation_gap_vs_anchor",
        "completed_tasks_mean",
        "completed_gap_vs_anchor",
    ]
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def build_interpretation(rows: list[dict]) -> str:
    lines = [
        "# 论文结果表候选稿",
        "",
        "## 使用说明",
        "",
        "本表由聚合 summary 自动生成。`anchor` 是同一场景下启发式基线中的最强 reward 参考点；gap 列表示当前算法相对该 anchor 的差值。",
        "",
        "当前表可以先用于 baseline + 本地 PPO smoke 分析。Colab 的 SB3-PPO、SAC、DreamerV3 结果返回后，只要先运行 `aggregate_experiment_results.py` 生成新的 summary，再用本脚本即可生成同格式表格。",
        "",
        "## 表格",
        "",
        markdown_table(rows),
        "",
        "## 初步判断",
        "",
    ]
    for scenario, scenario_rows in group_by_scenario(rows).items():
        anchor_rows = [row for row in scenario_rows if row.get("algorithm") == row.get("anchor")]
        ppo_rows = [row for row in scenario_rows if row.get("algorithm") == "ppo_linear"]
        if anchor_rows:
            anchor = anchor_rows[0]
            lines.append(
                f"- `{scenario}` 当前启发式 anchor 是 `{anchor['anchor']}`，reward={anchor['reward_mean']}，"
                f"deadline violation={anchor['deadline_violation_rate_mean']}。"
            )
        if ppo_rows:
            ppo = ppo_rows[0]
            lines.append(
                f"- `{scenario}` 的 `ppo_linear` 仍是本地 smoke 结果，reward gap={ppo['reward_gap_vs_anchor']}，"
                f"deadline gap={ppo['deadline_violation_gap_vs_anchor']}，不能作为最终论文级 PPO 结论。"
            )
    lines.append("")
    lines.append("正式论文中应优先使用 Colab/GPU 的 SB3-PPO、SAC、DreamerV3 多 seed 结果替换 smoke 行。")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = build_table(read_rows(args.summary), args.anchors)
    write_csv(args.output, rows)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(build_interpretation(rows), encoding="utf-8")
    print(f"wrote_csv={args.output} rows={len(rows)}")
    print(f"wrote_markdown={args.markdown}")


if __name__ == "__main__":
    main()
