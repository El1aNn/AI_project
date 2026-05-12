#!/usr/bin/env python3
"""Analyze local/SwanLab result summaries and write a report-ready Markdown file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_metric(summary: Mapping[str, Any], suffixes: Sequence[str]) -> float | None:
    for key, value in summary.items():
        key_l = key.lower()
        if any(key_l.endswith(suffix.lower()) or suffix.lower() in key_l for suffix in suffixes):
            if isinstance(value, Mapping):
                value = value.get("value")
            try:
                number = float(value)
                if math.isfinite(number):
                    return number
            except Exception:
                pass
    return None


def metric(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def quality_line(name: str, value: float | None, good: float, ok: float, higher_better: bool = True) -> str:
    if value is None:
        return f"- {name}: not found."
    if higher_better:
        label = "good" if value >= good else "acceptable" if value >= ok else "needs improvement"
    else:
        label = "good" if value <= good else "acceptable" if value <= ok else "needs improvement"
    return f"- {name}: `{value:.4f}` -> {label}."


def existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def analyze_project1(root: Path, data_dir: Path | None = None) -> list[str]:
    summary_path = existing_path(
        *(([data_dir / "project1_2/evaluation_summary.json"] if data_dir else []))
        + [root / "project1_2_solution/outputs/evaluation_summary.json"]
    )
    summary = load_json(summary_path)
    lines = ["## Project 1/2: CBOW and Skip-gram", ""]
    if not summary:
        lines.append("- Local `evaluation_summary.json` not found.")
        return lines
    for model, metrics in summary.items():
        knn = metrics.get("knn", {})
        simlex = metrics.get("simlex999", {})
        analogy = metrics.get("analogy", {})
        lines.extend(
            [
                f"### {model}",
                quality_line("KNN mean top-10 cosine", knn.get("mean_top10_cosine"), 0.45, 0.30),
                quality_line("SimLex Spearman", simlex.get("spearman"), 0.20, 0.10),
                quality_line("Analogy accuracy", analogy.get("accuracy"), 0.10, 0.03),
                f"- Covered SimLex pairs: `{simlex.get('covered_pairs', 'n/a')}`.",
                f"- Covered analogy questions: `{analogy.get('covered_questions', 'n/a')}`.",
                "",
            ]
        )
    return lines


def analyze_project3(root: Path, data_dir: Path | None = None) -> list[str]:
    lines = ["## Project 3: Transformer A/B", ""]
    found = False
    for task_name in ["taskA_char", "taskB_word"]:
        metrics_path = existing_path(
            *(([data_dir / f"project3/{task_name}/metrics.json"] if data_dir else []))
            + [root / f"project3_solution/outputs/{task_name}/metrics.json"]
        )
        metrics = load_json(metrics_path)
        if not metrics:
            lines.append(f"- `{task_name}` metrics not found.")
            continue
        found = True
        test = metrics.get("test", {})
        lines.append(quality_line(f"{task_name} test accuracy", test.get("accuracy"), 0.80, 0.65))
        lines.append(quality_line(f"{task_name} test loss", test.get("loss"), 0.7, 1.2, higher_better=False))
    comparison_a = existing_path(
        *(([data_dir / "project3/A_wrong_B_right.csv"] if data_dir else []))
        + [root / "project3_solution/outputs/A_wrong_B_right.csv"]
    )
    comparison_b = existing_path(
        *(([data_dir / "project3/B_wrong_A_right.csv"] if data_dir else []))
        + [root / "project3_solution/outputs/B_wrong_A_right.csv"]
    )
    if comparison_a.exists() and comparison_b.exists():
        lines.append("- A/B comparison CSV files are present for report examples.")
    elif found:
        lines.append("- A/B comparison CSV files are missing; rerun both Task A and B together.")
    lines.append("")
    return lines


def analyze_project4(root: Path, data_dir: Path | None = None) -> list[str]:
    lines = ["## Project 4: BERT", ""]
    summary_path = existing_path(
        *(([data_dir / "project4/summary.json"] if data_dir else []))
        + [root / "project4_solution/outputs/summary.json"]
    )
    summary = load_json(summary_path)
    if not summary:
        lines.append("- Local Project 4 summary not found.")
        return lines
    for task in ["sst2", "mrpc"]:
        metrics = summary.get(task)
        if not metrics:
            lines.append(f"- `{task}` metrics not found.")
            continue
        lines.append(quality_line(f"{task} final accuracy", metrics.get("final_accuracy"), 0.85 if task == "sst2" else 0.75, 0.75 if task == "sst2" else 0.68))
        if "final_f1" in metrics:
            lines.append(quality_line(f"{task} final F1", metrics.get("final_f1"), 0.85 if task == "sst2" else 0.82, 0.75 if task == "sst2" else 0.75))
    lines.append("")
    return lines


def analyze_swanlab(root: Path) -> list[str]:
    data = load_json(root / "swanlab_query_outputs/swanlab_results.json")
    lines = ["## SwanLab Online Summary", ""]
    if not data:
        lines.append("- SwanLab query output not found. Run `bash start.sh swanlab-results AI_project 10` first.")
        return lines
    experiments = data.get("experiments", [])
    lines.append(f"- Queried experiments: `{len(experiments)}`.")
    for exp in experiments[:10]:
        summary = exp.get("summary") or {}
        name = exp.get("name")
        lines.append(f"### `{name}` [{exp.get('state')}]")
        if name == "project1_2_word2vec":
            for model in ["cbow", "skipgram"]:
                lines.append(quality_line(f"{model} KNN mean top-10 cosine", metric(summary, f"project1_2/eval/{model}/knn/mean_top10_cosine"), 0.45, 0.30))
                lines.append(quality_line(f"{model} SimLex Spearman", metric(summary, f"project1_2/eval/{model}/simlex999/spearman"), 0.20, 0.10))
                lines.append(quality_line(f"{model} analogy accuracy", metric(summary, f"project1_2/eval/{model}/analogy/accuracy"), 0.10, 0.03))
        elif name == "project3_transformer_ab":
            lines.append(quality_line("Task A character test accuracy", metric(summary, "project3/taskA_char/final/test_accuracy"), 0.80, 0.65))
            lines.append(quality_line("Task B word test accuracy", metric(summary, "project3/taskB_word/final/test_accuracy"), 0.80, 0.65))
            a_mis = metric(summary, "project3/taskA_char/final/test_misclassified")
            b_mis = metric(summary, "project3/taskB_word/final/test_misclassified")
            if a_mis is not None and b_mis is not None:
                better = "Task A" if a_mis < b_mis else "Task B"
                lines.append(f"- Misclassified documents: Task A `{a_mis:.0f}`, Task B `{b_mis:.0f}` -> {better} is better on this run.")
            awbr = metric(summary, "project3/comparison/a_wrong_b_right")
            bwar = metric(summary, "project3/comparison/b_wrong_a_right")
            if awbr is not None and bwar is not None:
                lines.append(f"- A wrong/B right examples: `{awbr:.0f}`; B wrong/A right examples: `{bwar:.0f}`.")
        elif name == "project4_bert_tasks":
            lines.append(quality_line("SST-2 final accuracy", metric(summary, "project4/sst2/final/final_accuracy"), 0.85, 0.75))
            lines.append(quality_line("SST-2 final F1", metric(summary, "project4/sst2/final/final_f1"), 0.85, 0.75))
            lines.append(quality_line("MRPC final accuracy", metric(summary, "project4/mrpc/final/final_accuracy"), 0.75, 0.68))
            lines.append(quality_line("MRPC final F1", metric(summary, "project4/mrpc/final/final_f1"), 0.82, 0.75))
        else:
            acc = find_metric(summary, ["accuracy", "final_accuracy"])
            f1 = find_metric(summary, ["f1", "final_f1"])
            loss = find_metric(summary, ["loss"])
            bits = []
            if acc is not None:
                bits.append(f"accuracy={acc:.4f}")
            if f1 is not None:
                bits.append(f"f1={f1:.4f}")
            if loss is not None:
                bits.append(f"loss={loss:.4f}")
            lines.append("- " + (", ".join(bits) if bits else "selected metrics not found"))
    lines.append("")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Submission experiment data directory, for example experiment_data/submission.",
    )
    parser.add_argument("--output", type=Path, default=Path("RESULT_ANALYSIS.md"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None
    lines = [
        "# Result Analysis",
        "",
        "This report is generated from local output JSON/CSV files and optional SwanLab query summaries.",
        "",
    ]
    if data_dir:
        lines.extend(["- Experiment data directory: `" + str(data_dir) + "`.", ""])
    lines.extend(analyze_project1(root, data_dir))
    lines.extend(analyze_project3(root, data_dir))
    lines.extend(analyze_project4(root, data_dir))
    lines.extend(analyze_swanlab(root))
    lines.extend(
        [
            "## Overall Judgment",
            "",
            "- Results marked `good` or `acceptable` are sufficient for the course report.",
            "- Results marked `needs improvement` can still be submitted if the run is complete, but consider increasing epochs or using a stronger pretrained model for Project 4.",
            "- Make sure all required misclassification CSV/Markdown files exist before zipping.",
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved analysis report to {args.output}")
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main()
