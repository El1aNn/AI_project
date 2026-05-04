# NLP Projects 1-4 Run Guide

This folder keeps the original assignment files and adds runnable solution code:

- `project1_2_solution/word2vec_project.py`
- `project3_solution/run_project3_ab.py`
- `project4_solution/run_bert_tasks.py`
- `ENVIRONMENT.md`
- `start.sh`
- `REPORT_TEMPLATE.md`
- `make_submission_zip.sh`

## 0. Environment

```bash
python -m pip install -r requirements.txt
```

If PyTorch installation fails, install it from the official PyTorch command for
your OS and Python version, then rerun the requirements command.

You can also use the launcher:

```bash
bash start.sh setup
bash start.sh check
```

See `ENVIRONMENT.md` for missing packages and recommended Python setup.
On AutoDL, `bash start.sh setup-cn` is usually faster.

## SwanLab Upload

Metric upload is optional. It turns on automatically when `SWANLAB_API_KEY` is
present, or you can enable it explicitly with `SWANLAB=1`:

```bash
export SWANLAB_API_KEY=your_api_key
export SWANLAB_PROJECT=AI_project
# Optional:
# export SWANLAB=1
# export SWANLAB_EXPERIMENT=my_run_name
# export SWANLAB_WORKSPACE=my_workspace
# export SWANLAB_LOGDIR=./swanlog
# export SWANLAB_MODE=cloud

bash start.sh quick YOUR_STUDENT_ID
bash start.sh all YOUR_STUDENT_ID
```

Logged items include training loss/accuracy curves, final evaluation metrics,
misclassification counts, and paths to generated CSV/JSON/Markdown outputs.

After runs finish, query recent SwanLab summaries from AutoDL:

```bash
source ~/.bashrc
bash start.sh swanlab-results AI_project 10
```

This writes `swanlab_query_outputs/swanlab_results.md` and
`swanlab_query_outputs/swanlab_results.json`.

## 1. Projects 1 and 2: CBOW and Skip-gram

Quick smoke test:

```bash
bash start.sh quick YOUR_STUDENT_ID
```

Final run:

```bash
python project1_2_solution/word2vec_project.py \
  --student-id YOUR_STUDENT_ID \
  --epochs 10 \
  --embedding-dim 100 \
  --window-size 2 \
  --min-count 5
```

Main outputs:

- `project1_2_solution/outputs/cbow.vec`
- `project1_2_solution/outputs/skipgram.vec`
- `project1_2_solution/outputs/*_knn_top10.csv`
- `project1_2_solution/outputs/*_simlex999.csv`
- `project1_2_solution/outputs/*_analogy.csv`
- `project1_2_solution/outputs/evaluation_summary.json`

For the report, use the 4-word KNN sample files, 20 SimLex sample files, 10
analogy sample files, and the summary JSON.

## 2. Project 3: Transformer Sentence Classification

Task A uses character-level input. Task B uses word segmentation.

Quick smoke test:

```bash
python project3_solution/run_project3_ab.py --quick
```

Final run:

```bash
bash start.sh project3 YOUR_STUDENT_ID
```

Main outputs:

- `project3_solution/outputs/taskA_char/metrics.json`
- `project3_solution/outputs/taskA_char/test_predictions.csv`
- `project3_solution/outputs/taskA_char/test_predictions_misclassified.csv`
- `project3_solution/outputs/taskB_word/metrics.json`
- `project3_solution/outputs/taskB_word/test_predictions.csv`
- `project3_solution/outputs/taskB_word/test_predictions_misclassified.csv`
- `project3_solution/outputs/A_wrong_B_right.csv`
- `project3_solution/outputs/B_wrong_A_right.csv`
- `project3_solution/outputs/comparison_examples.md`

Use `comparison_examples.md` for the required three A-wrong/B-right and three
B-wrong/A-right explanations.

## 3. Project 4: BERT Evaluation

MRPC is smaller, so run it first to check the environment:

```bash
python project4_solution/run_bert_tasks.py --task mrpc --quick
```

Final run for both tasks:

```bash
bash start.sh project4
```

Main outputs:

- `project4_solution/outputs/sst2/final_metrics.json`
- `project4_solution/outputs/sst2/validation_predictions.csv`
- `project4_solution/outputs/sst2/validation_predictions_misclassified.csv`
- `project4_solution/outputs/sst2/report_examples.md`
- `project4_solution/outputs/mrpc/final_metrics.json`
- `project4_solution/outputs/mrpc/validation_predictions.csv`
- `project4_solution/outputs/mrpc/report_examples.md`
- `project4_solution/outputs/summary.json`

Use `report_examples.md` files for the required discussion examples.
The runner is compatible with both older Transformers versions using
`Trainer(tokenizer=...)` and newer versions using `Trainer(processing_class=...)`.

## 4. Create Submission Zip

After the report and outputs are ready:

```bash
bash start.sh zip
```

This creates `nlp_projects_submission.zip` with source code, original materials,
and generated outputs. Add your finished PDF/Word report before uploading if
your instructor requires it inside the same zip.
