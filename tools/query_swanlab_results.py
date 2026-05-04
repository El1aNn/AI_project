#!/usr/bin/env python3
"""Query SwanLab experiment summaries through the official OpenAPI.

Usage:
    python tools/query_swanlab_results.py --project AI_project --limit 10

The script reads SWANLAB_API_KEY from the environment when present. It writes a
Markdown report and a JSON file so the results can be pasted into the project
report later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


INTERESTING_KEYWORDS = (
    "accuracy",
    "acc",
    "f1",
    "loss",
    "spearman",
    "analogy",
    "misclassified",
    "mean_top10_cosine",
    "covered",
    "correct",
)

KEY_PATTERN = re.compile(r"^\s*(?:export\s+)?SWANLAB_API_KEY\s*=\s*['\"]?([^'\"\n#]+)")


def load_api_key_from_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("SWANLAB_API_KEY") or data.get("swanlab_api_key") or data.get("api_key")
            return str(value).strip() if value else None
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            match = KEY_PATTERN.match(line)
            if match:
                return match.group(1).strip()
    except Exception:
        return None
    return None


def load_api_key(args: argparse.Namespace) -> str | None:
    if os.environ.get("SWANLAB_API_KEY"):
        return os.environ["SWANLAB_API_KEY"].strip()

    candidates = []
    if args.config:
        candidates.append(Path(args.config).expanduser())
    if os.environ.get("SWANLAB_CONFIG"):
        candidates.append(Path(os.environ["SWANLAB_CONFIG"]).expanduser())
    candidates.extend(
        [
            Path(".env"),
            Path("swanlab.env"),
            Path("swanlab_config.json"),
            Path.home() / ".bashrc",
            Path.home() / ".zshrc",
            Path.home() / ".config" / "swanlab" / "config.json",
        ]
    )
    for path in candidates:
        value = load_api_key_from_file(path)
        if value:
            print(f"Loaded SWANLAB_API_KEY from {path}")
            return value
    return None


def require_swanlab():
    try:
        from swanlab import OpenApi
    except ImportError as exc:
        raise SystemExit(
            "SwanLab OpenAPI is unavailable. Install or upgrade with "
            "`python -m pip install -U swanlab`."
        ) from exc
    return OpenApi


def as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    data: Dict[str, Any] = {}
    for key in [
        "cuid",
        "name",
        "state",
        "createdAt",
        "finishedAt",
        "description",
        "user",
        "profile",
        "group",
        "count",
    ]:
        if hasattr(obj, key):
            data[key] = getattr(obj, key)
    return data


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def response_data(response: Any) -> Any:
    code = get_field(response, "code")
    errmsg = get_field(response, "errmsg")
    if code and int(code) >= 400:
        raise RuntimeError(f"SwanLab OpenAPI error {code}: {errmsg}")
    return get_field(response, "data", response)


def sort_experiments(experiments: Sequence[Any]) -> List[Any]:
    return sorted(
        experiments,
        key=lambda exp: str(get_field(exp, "createdAt", "")),
        reverse=True,
    )


def select_metrics(summary: Mapping[str, Any], include_all: bool) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for key, value in summary.items():
        key_l = key.lower()
        if include_all or any(word in key_l for word in INTERESTING_KEYWORDS):
            selected[key] = value
    return selected


def compact_metric_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        compact = {}
        for field in ["value", "step", "min", "max"]:
            if field in value:
                compact[field] = value[field]
        return compact or dict(value)
    return value


def query_results(args: argparse.Namespace) -> Dict[str, Any]:
    OpenApi = require_swanlab()
    api_key = load_api_key(args)
    if not api_key:
        raise SystemExit(
            "SWANLAB_API_KEY was not found in env or known config files. "
            "Run `source ~/.bashrc`, or pass `--config path/to/config`."
        )

    api = OpenApi(api_key=api_key)
    workspaces = response_data(api.list_workspaces())
    workspace = args.workspace
    if not workspace and workspaces:
        workspace = get_field(workspaces[0], "username")

    experiments = response_data(api.list_experiments(project=args.project, username=workspace))
    experiments = sort_experiments(experiments)[: args.limit]

    result: Dict[str, Any] = {
        "project": args.project,
        "workspace": workspace,
        "experiment_count_returned": len(experiments),
        "experiments": [],
    }

    for exp in experiments:
        exp_id = get_field(exp, "cuid")
        exp_dict = as_dict(exp)
        summary_raw = response_data(api.get_summary(project=args.project, exp_id=exp_id, username=workspace))
        summary_selected = {
            key: compact_metric_value(value)
            for key, value in select_metrics(summary_raw or {}, args.all_metrics).items()
        }
        result["experiments"].append(
            {
                "cuid": exp_id,
                "name": exp_dict.get("name"),
                "state": exp_dict.get("state"),
                "createdAt": exp_dict.get("createdAt"),
                "finishedAt": exp_dict.get("finishedAt"),
                "summary": summary_selected,
            }
        )

    return result


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "swanlab_results.json"
    md_path = output_dir / "swanlab_results.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# SwanLab Results",
        "",
        f"- Project: `{result.get('project')}`",
        f"- Workspace: `{result.get('workspace')}`",
        f"- Experiments returned: `{result.get('experiment_count_returned')}`",
        "",
    ]
    for exp in result.get("experiments", []):
        lines.extend(
            [
                f"## {exp.get('name') or exp.get('cuid')}",
                "",
                f"- ID: `{exp.get('cuid')}`",
                f"- State: `{exp.get('state')}`",
                f"- Created: `{exp.get('createdAt')}`",
                f"- Finished: `{exp.get('finishedAt')}`",
                "",
            ]
        )
        summary = exp.get("summary") or {}
        if not summary:
            lines.append("No selected metrics found.")
        else:
            lines.append("| Metric | Last Value | Step |")
            lines.append("| --- | ---: | ---: |")
            for key, value in summary.items():
                if isinstance(value, Mapping):
                    lines.append(f"| `{key}` | `{value.get('value')}` | `{value.get('step')}` |")
                else:
                    lines.append(f"| `{key}` | `{value}` |  |")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved SwanLab JSON summary to {json_path}")
    print(f"Saved SwanLab Markdown summary to {md_path}")


def print_console_summary(result: Mapping[str, Any]) -> None:
    print(f"Project: {result.get('project')}")
    print(f"Workspace: {result.get('workspace')}")
    print(f"Experiments returned: {result.get('experiment_count_returned')}")
    for exp in result.get("experiments", []):
        print(f"\n[{exp.get('state')}] {exp.get('name')} ({exp.get('cuid')})")
        summary = exp.get("summary") or {}
        for key, value in summary.items():
            if isinstance(value, Mapping):
                print(f"  {key}: {value.get('value')} (step {value.get('step')})")
            else:
                print(f"  {key}: {value}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("SWANLAB_PROJECT", "AI_project"))
    parser.add_argument("--workspace", default=os.environ.get("SWANLAB_WORKSPACE"))
    parser.add_argument("--config", default=os.environ.get("SWANLAB_CONFIG"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--all-metrics", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("swanlab_query_outputs"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    result = query_results(args)
    print_console_summary(result)
    write_outputs(result, args.output_dir)


if __name__ == "__main__":
    main()
