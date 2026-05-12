#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ZIP_NAME="${1:-nlp_projects_submission.zip}"
DATA_DIR="${2:-experiment_data/submission}"
if [[ "$DATA_DIR" = "$ROOT_DIR/"* ]]; then
  DATA_DIR="${DATA_DIR#"$ROOT_DIR/"}"
fi

INCLUDES=(
  README_solution.md
  ENVIRONMENT.md
  MISSING_ITEMS.md
  REPORT_TEMPLATE.md
  reports
  requirements.txt
  start.sh
  make_submission_zip.sh
  common
  tools
  project1_2_solution
  project3_solution
  project4_solution
  project1_2_source
  project3_source
  sst2.py
  mrpc.py
  Bert_how_to_run.pdf
  "NPL Project 4.pdf"
  Transformer3.pdf
  "Project 3-theory-run-requirements1.pdf"
)

if [[ -d "$DATA_DIR" ]]; then
  INCLUDES+=("$DATA_DIR")
fi

rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" "${INCLUDES[@]}" \
  -x "*/__pycache__/*" "*.pyc" "*/.DS_Store" \
  -x "project1_2_solution/outputs/*" "project3_solution/outputs/*" \
  -x "project3_solution/THUCNews_word/*" "project4_solution/outputs/*" \
  -x "*.ckpt" "*.pt" "*.pth" "*.bin" "*.safetensors" \
  -x "*/trainer/*" "checkpoints/*" "*/checkpoints/*" "swanlab_query_outputs/*"

echo "Created $ZIP_NAME"
