#!/usr/bin/env bash
set -euo pipefail

ZIP_NAME="${1:-nlp_projects_submission.zip}"

rm -f "$ZIP_NAME"
zip -r "$ZIP_NAME" \
  README_solution.md \
  ENVIRONMENT.md \
  MISSING_ITEMS.md \
  REPORT_TEMPLATE.md \
  requirements.txt \
  start.sh \
  make_submission_zip.sh \
  common \
  tools \
  project1_2_solution \
  project3_solution \
  project4_solution \
  project1_2_source \
  project3_source \
  sst2.py \
  mrpc.py \
  Bert_how_to_run.pdf \
  "NPL Project 4.pdf" \
  Transformer3.pdf \
  "Project 3-theory-run-requirements1.pdf" \
  -x "*/__pycache__/*" "*.pyc" "*/.DS_Store" "*.ckpt" "*/trainer/*" "swanlab_query_outputs/*"

echo "Created $ZIP_NAME"
