"""Optional SwanLab logging helpers.

The project scripts should run even when SwanLab is not installed or the user
has not enabled uploading. This helper keeps that behavior in one place.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping


TRUE_VALUES = {"1", "true", "yes", "on", "cloud", "local", "offline"}


def env_enabled(name: str = "SWANLAB") -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def add_swanlab_args(parser) -> None:
    parser.add_argument("--swanlab", action="store_true", help="Upload useful metrics to SwanLab.")
    parser.add_argument("--swanlab-project", default=os.environ.get("SWANLAB_PROJECT", "AI_project"))
    parser.add_argument("--swanlab-experiment", default=os.environ.get("SWANLAB_EXPERIMENT"))
    parser.add_argument("--swanlab-workspace", default=os.environ.get("SWANLAB_WORKSPACE"))
    parser.add_argument("--swanlab-logdir", default=os.environ.get("SWANLAB_LOGDIR"))
    parser.add_argument(
        "--swanlab-mode",
        default=os.environ.get("SWANLAB_MODE", "cloud"),
        help="SwanLab run mode, for example cloud/local/offline/disabled.",
    )


def flatten_metrics(data: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            metrics.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            value_f = float(value)
            if math.isfinite(value_f):
                metrics[name] = value_f
    return metrics


class SwanLabLogger:
    def __init__(self, enabled: bool, run: Any = None, swanlab_module: Any = None):
        self.enabled = enabled
        self.run = run
        self.swanlab = swanlab_module

    @classmethod
    def from_args(
        cls,
        args,
        *,
        project: str,
        experiment_name: str,
        config: Mapping[str, Any] | None = None,
    ) -> "SwanLabLogger":
        enabled = bool(getattr(args, "swanlab", False)) or env_enabled()
        mode = getattr(args, "swanlab_mode", os.environ.get("SWANLAB_MODE", "cloud"))
        if str(mode).lower() == "disabled":
            enabled = False
        if not enabled:
            return cls(False)

        try:
            import swanlab
        except ImportError:
            print("SwanLab is enabled but not installed. Run `bash start.sh setup` first.")
            return cls(False)

        api_key = os.environ.get("SWANLAB_API_KEY")
        if api_key and hasattr(swanlab, "login"):
            try:
                swanlab.login(api_key=api_key, save=True)
            except Exception as exc:
                print(f"SwanLab login with SWANLAB_API_KEY failed; continuing: {exc}")

        init_kwargs: Dict[str, Any] = {
            "project": getattr(args, "swanlab_project", None) or project,
            "experiment_name": getattr(args, "swanlab_experiment", None) or experiment_name,
            "config": dict(config or {}),
        }
        workspace = getattr(args, "swanlab_workspace", None)
        if workspace:
            init_kwargs["workspace"] = workspace
        logdir = getattr(args, "swanlab_logdir", None)
        if logdir:
            init_kwargs["logdir"] = logdir
        if mode:
            init_kwargs["mode"] = mode

        try:
            run = swanlab.init(**init_kwargs)
        except TypeError:
            init_kwargs.pop("workspace", None)
            init_kwargs.pop("mode", None)
            init_kwargs.pop("logdir", None)
            run = swanlab.init(**init_kwargs)
        except Exception as exc:
            print(f"Could not initialize SwanLab; continuing without upload: {exc}")
            return cls(False)

        print(
            "SwanLab logging enabled: "
            f"project={init_kwargs.get('project')}, experiment={init_kwargs.get('experiment_name')}"
        )
        return cls(True, run=run, swanlab_module=swanlab)

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int | None = None, prefix: str = "") -> None:
        if not self.enabled:
            return
        clean = flatten_metrics(metrics, prefix)
        if not clean:
            return
        try:
            if step is None:
                self.swanlab.log(clean)
            else:
                self.swanlab.log(clean, step=step)
        except Exception as exc:
            print(f"SwanLab metric upload failed; continuing: {exc}")

    def log_text(self, name: str, text: str) -> None:
        if not self.enabled:
            return
        try:
            if hasattr(self.swanlab, "Text"):
                self.swanlab.log({name: self.swanlab.Text(text)})
            else:
                self.swanlab.log({name: text})
        except Exception as exc:
            print(f"SwanLab text upload failed for {name}; continuing: {exc}")

    def log_output_path(self, name: str, path: str | Path) -> None:
        self.log_text(name, str(Path(path)))

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            if hasattr(self.swanlab, "finish"):
                self.swanlab.finish()
        except Exception as exc:
            print(f"SwanLab finish failed; continuing: {exc}")
