#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
COMMAND="${1:-help}"
STUDENT_ID="${2:-${STUDENT_ID:-20260517}}"

print_usage() {
  cat <<'EOF'
Usage:
  bash start.sh check
  bash start.sh setup
  bash start.sh quick [STUDENT_ID]
  bash start.sh project1 [STUDENT_ID]
  bash start.sh project3 [STUDENT_ID]
  bash start.sh project4
  bash start.sh all [STUDENT_ID]
  bash start.sh zip

Environment variables:
  PYTHON_BIN=python3      Python executable to use
  STUDENT_ID=123456789   Student ID used for reproducible sampling

Notes:
  - quick runs small smoke tests only; do not use quick outputs in the report.
  - all runs the final settings and may take a long time.
EOF
}

check_env() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
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

missing = [name for name in required if importlib.util.find_spec(name) is None]
print(f"Python: {sys.executable}")
print(f"Version: {sys.version.split()[0]}")
if missing:
    print("Missing packages:")
    for name in missing:
        print(f"  - {name}")
    print("\nInstall them with: python -m pip install -r requirements.txt")
    raise SystemExit(1)
print("All required Python packages are available.")
PY
}

setup_env() {
  "$PYTHON_BIN" -m pip install -r requirements.txt
  "$PYTHON_BIN" - <<'PY'
try:
    import nltk
    nltk.download("reuters")
    print("NLTK Reuters corpus is ready.")
except Exception as exc:
    print(f"Could not download NLTK Reuters automatically: {exc}")
    print("You can still run Project 1/2 later, or pass --corpus-path.")
PY
}

run_quick() {
  "$PYTHON_BIN" project1_2_solution/word2vec_project.py --quick --student-id "$STUDENT_ID"
  "$PYTHON_BIN" project3_solution/run_project3_ab.py --quick --student-id "$STUDENT_ID"
  "$PYTHON_BIN" project4_solution/run_bert_tasks.py --task mrpc --quick
}

run_project1() {
  "$PYTHON_BIN" project1_2_solution/word2vec_project.py \
    --student-id "$STUDENT_ID" \
    --epochs 10 \
    --embedding-dim 100 \
    --window-size 2 \
    --min-count 5
}

run_project3() {
  "$PYTHON_BIN" project3_solution/run_project3_ab.py \
    --epochs 6 \
    --student-id "$STUDENT_ID"
}

run_project4() {
  "$PYTHON_BIN" project4_solution/run_bert_tasks.py --task all --epochs 2
}

case "$COMMAND" in
  help|-h|--help)
    print_usage
    ;;
  check)
    check_env
    ;;
  setup)
    setup_env
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
  all)
    check_env
    run_project1
    run_project3
    run_project4
    ;;
  zip)
    bash make_submission_zip.sh
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    print_usage
    exit 2
    ;;
esac
