#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export NLTK_DATA="$ROOT_DIR/.nltk_data${NLTK_DATA:+:$NLTK_DATA}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

COMMAND="${1:-default}"
STUDENT_ID="${2:-${STUDENT_ID:-20260517}}"

resolve_path() {
  local path="$1"
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT_DIR/$path"
  fi
}

EXPERIMENT_DATA_DIR="$(resolve_path "${3:-${EXPERIMENT_DATA_DIR:-experiment_data/submission}}")"
CHECKPOINT_DIR="$(resolve_path "${4:-${CHECKPOINT_DIR:-checkpoints/submission}}")"

print_usage() {
  cat <<'EOF'
Usage:
  bash start.sh
  bash start.sh check
  bash start.sh setup
  bash start.sh setup-cn
  bash start.sh download-data
  bash start.sh quick [STUDENT_ID]
  bash start.sh project1 [STUDENT_ID]
  bash start.sh project3 [STUDENT_ID]
  bash start.sh project4
  bash start.sh submission [STUDENT_ID] [EXPERIMENT_DATA_DIR] [CHECKPOINT_DIR]
  bash start.sh swanlab-results [PROJECT] [LIMIT]
  bash start.sh analyze [EXPERIMENT_DATA_DIR]
  bash start.sh all [STUDENT_ID]
  bash start.sh zip [ZIP_NAME] [EXPERIMENT_DATA_DIR]

Environment variables:
  PYTHON_BIN=python3      Python executable to use
  STUDENT_ID=123456789   Student ID used for reproducible sampling
  EXPERIMENT_DATA_DIR=./experiment_data/submission
                         Final CSV/JSON/report-example outputs for submission
  CHECKPOINT_DIR=./checkpoints/submission
                         Model checkpoints, kept separate from experiment data
  NLTK_TIMEOUT_SECONDS=120  Timeout for NLTK Reuters download
  SWANLAB=1              Enable SwanLab metric upload. SWANLAB_API_KEY also enables it automatically.
  SWANLAB_PROJECT=AI_project
  SWANLAB_EXPERIMENT=name
  SWANLAB_WORKSPACE=name
  SWANLAB_LOGDIR=./swanlog
  SWANLAB_MODE=cloud     SwanLab mode, e.g. cloud/local/offline/disabled
  SWANLAB_API_KEY=...    Optional non-interactive SwanLab login key

Notes:
  - running without arguments performs an environment check.
  - setup-cn installs packages through the Tsinghua PyPI mirror.
  - setup/setup-cn also try to download NLTK Reuters, but continue if it times out.
  - swanlab-results reads SWANLAB_API_KEY and queries recent experiment summaries.
  - analyze writes RESULT_ANALYSIS.md from local outputs and SwanLab query output.
  - quick runs small smoke tests only; do not use quick outputs in the report.
  - all runs the final settings and may take a long time.
  - submission runs final settings and writes outputs under EXPERIMENT_DATA_DIR,
    with checkpoints under CHECKPOINT_DIR.
EOF
}

check_env() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import sys

required = [
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "tqdm",
    "nltk",
    "jieba",
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "accelerate",
]
if (
    os.environ.get("SWANLAB", "").lower() in {"1", "true", "yes", "on", "cloud", "local", "offline"}
    or os.environ.get("SWANLAB_API_KEY", "").strip()
):
    required.append("swanlab")

missing = [name for name in required if importlib.util.find_spec(name) is None]
print(f"Python: {sys.executable}")
print(f"Version: {sys.version.split()[0]}")
if missing:
    print("Missing packages:")
    for name in missing:
        print(f"  - {name}")
    print("\nInstall them with: bash start.sh setup")
    print("If PyPI is slow in AutoDL, use: bash start.sh setup-cn")
    raise SystemExit(1)
print("All required Python packages are available.")
PY
}

build_swanlab_args() {
  SWANLAB_ARGS=()
  local swanlab_value="${SWANLAB:-0}"
  if [[ "$swanlab_value" == "0" && -n "${SWANLAB_API_KEY:-}" ]]; then
    swanlab_value="1"
  fi
  local swanlab_mode="${SWANLAB_MODE:-cloud}"
  case "$swanlab_value" in
    cloud|local|offline)
      if [[ -z "${SWANLAB_MODE:-}" ]]; then
        swanlab_mode="$swanlab_value"
      fi
      ;;
  esac
  case "$swanlab_value" in
    1|true|TRUE|yes|YES|on|ON|cloud|local|offline)
      SWANLAB_ARGS+=(--swanlab)
      SWANLAB_ARGS+=(--swanlab-project "${SWANLAB_PROJECT:-AI_project}")
      SWANLAB_ARGS+=(--swanlab-mode "$swanlab_mode")
      if [[ -n "${SWANLAB_EXPERIMENT:-}" ]]; then
        SWANLAB_ARGS+=(--swanlab-experiment "$SWANLAB_EXPERIMENT")
      fi
      if [[ -n "${SWANLAB_WORKSPACE:-}" ]]; then
        SWANLAB_ARGS+=(--swanlab-workspace "$SWANLAB_WORKSPACE")
      fi
      if [[ -n "${SWANLAB_LOGDIR:-}" ]]; then
        SWANLAB_ARGS+=(--swanlab-logdir "$SWANLAB_LOGDIR")
      fi
      ;;
  esac
}

setup_env() {
  echo "Installing Python packages with $PYTHON_BIN..."
  "$PYTHON_BIN" -m pip install -r requirements.txt
  download_reuters
}

setup_env_cn() {
  echo "Installing Python packages with $PYTHON_BIN through Tsinghua mirror..."
  "$PYTHON_BIN" -m pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn
  download_reuters
}

download_reuters() {
  echo "Downloading NLTK Reuters corpus to $ROOT_DIR/.nltk_data..."
  echo "If this step is slow, press Ctrl+C and rerun training later; Project 1/2 can download it on demand."
  "$PYTHON_BIN" - <<'PY'
import os
import signal

timeout = int(os.environ.get("NLTK_TIMEOUT_SECONDS", "120"))

def on_timeout(signum, frame):
    raise TimeoutError(f"NLTK Reuters download timed out after {timeout} seconds")

try:
    import nltk
    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(timeout)
    target = os.environ.get("NLTK_DATA", "").split(os.pathsep)[0] or None
    nltk.download("reuters", download_dir=target, quiet=False)
    signal.alarm(0)
    print("NLTK Reuters corpus is ready.")
except Exception as exc:
    print(f"Could not download NLTK Reuters automatically: {exc}")
    print("You can still run Project 1/2 later, or pass --corpus-path.")
PY
}

run_quick() {
  build_swanlab_args
  "$PYTHON_BIN" project1_2_solution/word2vec_project.py --quick --student-id "$STUDENT_ID" "${SWANLAB_ARGS[@]}"
  "$PYTHON_BIN" project3_solution/run_project3_ab.py --quick --student-id "$STUDENT_ID" "${SWANLAB_ARGS[@]}"
  "$PYTHON_BIN" project4_solution/run_bert_tasks.py --task mrpc --quick "${SWANLAB_ARGS[@]}"
}

run_project1() {
  build_swanlab_args
  "$PYTHON_BIN" project1_2_solution/word2vec_project.py \
    --student-id "$STUDENT_ID" \
    --epochs 10 \
    --embedding-dim 100 \
    --window-size 2 \
    --min-count 5 \
    "${SWANLAB_ARGS[@]}"
}

run_project3() {
  build_swanlab_args
  "$PYTHON_BIN" project3_solution/run_project3_ab.py \
    --epochs 6 \
    --student-id "$STUDENT_ID" \
    "${SWANLAB_ARGS[@]}"
}

run_project4() {
  build_swanlab_args
  "$PYTHON_BIN" project4_solution/run_bert_tasks.py --task all --epochs 2 "${SWANLAB_ARGS[@]}"
}

run_submission() {
  build_swanlab_args
  mkdir -p "$EXPERIMENT_DATA_DIR" "$CHECKPOINT_DIR"
  echo "Submission experiment data dir: $EXPERIMENT_DATA_DIR"
  echo "Submission checkpoint dir:     $CHECKPOINT_DIR"

  "$PYTHON_BIN" project1_2_solution/word2vec_project.py \
    --student-id "$STUDENT_ID" \
    --epochs 10 \
    --embedding-dim 100 \
    --window-size 2 \
    --min-count 5 \
    --output-dir "$EXPERIMENT_DATA_DIR/project1_2" \
    "${SWANLAB_ARGS[@]}"

  "$PYTHON_BIN" project3_solution/run_project3_ab.py \
    --epochs 6 \
    --student-id "$STUDENT_ID" \
    --output-dir "$EXPERIMENT_DATA_DIR/project3" \
    --checkpoint-dir "$CHECKPOINT_DIR/project3" \
    --save-model \
    "${SWANLAB_ARGS[@]}"

  "$PYTHON_BIN" project4_solution/run_bert_tasks.py \
    --task all \
    --epochs 2 \
    --output-dir "$EXPERIMENT_DATA_DIR/project4" \
    --checkpoint-dir "$CHECKPOINT_DIR/project4" \
    "${SWANLAB_ARGS[@]}"
}

case "$COMMAND" in
  default)
    echo "No command supplied; running environment check."
    if check_env; then
      echo
      echo "Environment looks ready."
      echo "Next commands:"
      echo "  bash start.sh quick $STUDENT_ID"
      echo "  bash start.sh all $STUDENT_ID"
    else
      echo
      echo "Environment is not ready yet."
      echo "Run setup first:"
      echo "  bash start.sh setup"
      echo "Or on AutoDL/China network:"
      echo "  bash start.sh setup-cn"
      exit 1
    fi
    ;;
  help|-h|--help)
    print_usage
    ;;
  check)
    check_env
    ;;
  setup)
    setup_env
    ;;
  setup-cn)
    setup_env_cn
    ;;
  download-data)
    download_reuters
    ;;
  quick)
    check_env
    run_quick
    ;;
  project1)
    check_env
    run_project1
    ;;
  project3)
    check_env
    run_project3
    ;;
  project4)
    check_env
    run_project4
    ;;
  submission)
    check_env
    run_submission
    ;;
  swanlab-results)
    "${PYTHON_BIN}" tools/query_swanlab_results.py \
      --project "${2:-${SWANLAB_PROJECT:-AI_project}}" \
      --limit "${3:-10}"
    ;;
  analyze)
    ANALYZE_DATA_DIR="$(resolve_path "${2:-${EXPERIMENT_DATA_DIR:-experiment_data/submission}}")"
    "${PYTHON_BIN}" tools/analyze_results.py --data-dir "$ANALYZE_DATA_DIR"
    ;;
  all)
    check_env
    run_project1
    run_project3
    run_project4
    ;;
  zip)
    ZIP_NAME="${2:-nlp_projects_submission.zip}"
    ZIP_DATA_DIR="$(resolve_path "${3:-${EXPERIMENT_DATA_DIR:-experiment_data/submission}}")"
    bash make_submission_zip.sh "$ZIP_NAME" "$ZIP_DATA_DIR"
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    print_usage
    exit 2
    ;;
esac
