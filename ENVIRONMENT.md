# Environment and Startup Guide

## Current Missing Packages

The local environment checked on this machine is missing the packages needed to
train the projects:

- `torch`
- `nltk`
- `jieba`
- `transformers`
- `datasets`
- `evaluate`
- `accelerate`

They are listed in `requirements.txt`.

## Recommended Python Version

Use Python 3.10 or 3.11 if possible. The current machine has Python 3.13, and
some machine learning packages may not provide stable wheels for that version.

Recommended setup:

```bash
conda create -n ai_project python=3.10 -y
conda activate ai_project
python -m pip install -r requirements.txt
```

If PyTorch fails to install through `requirements.txt`, install it from the
official PyTorch command for your OS first, then rerun:

```bash
python -m pip install -r requirements.txt
```

## One-Command Launcher

Use `start.sh` from the project root.

Check environment:

```bash
bash start.sh check
```

Install dependencies:

```bash
bash start.sh setup
```

On AutoDL or China networks, use the Tsinghua PyPI mirror:

```bash
bash start.sh setup-cn
```

If the install finishes but NLTK Reuters download is slow, the script will time
out and continue. You can retry the data download separately:

```bash
bash start.sh download-data
```

## SwanLab Metric Upload

SwanLab is included in `requirements.txt`. Upload turns on automatically when
`SWANLAB_API_KEY` is present, or you can enable it explicitly with `SWANLAB=1`:

```bash
export SWANLAB_API_KEY=your_api_key
export SWANLAB_PROJECT=AI_project
# Optional:
# export SWANLAB=1
# export SWANLAB_EXPERIMENT=my_run_name
# export SWANLAB_WORKSPACE=my_workspace
# export SWANLAB_LOGDIR=./swanlog
# export SWANLAB_MODE=cloud

bash start.sh all YOUR_STUDENT_ID
```

The scripts log useful scalar metrics and report pointers:

- Project 1/2: epoch loss, vocabulary/token counts, KNN mean cosine,
  SimLex-999 Spearman correlation, analogy accuracy.
- Project 3: Task A/B train-dev curves, test accuracy/loss, misclassified
  counts, A-wrong/B-right and B-wrong/A-right counts.
- Project 4: Trainer logs, final accuracy/F1, validation misclassified counts.

After training finishes, query recent online results with SwanLab OpenAPI:

```bash
source ~/.bashrc
bash start.sh swanlab-results AI_project 10
```

The query output is saved under `swanlab_query_outputs/`.

Generate a local report-style analysis after querying:

```bash
bash start.sh analyze
```

Run quick smoke tests:

```bash
bash start.sh quick YOUR_STUDENT_ID
```

Run final training/evaluation:

```bash
bash start.sh all YOUR_STUDENT_ID
```

Create the submission zip:

```bash
bash start.sh zip
```

## Network Requirements

The first run needs internet access for:

- NLTK Reuters corpus for Project 1/2, unless `--corpus-path` is used.
- Hugging Face GLUE datasets and pretrained BERT model for Project 4.

Project 4 supports newer Transformers versions where `Trainer` uses
`processing_class` instead of the older `tokenizer` argument.

Project 3 data is already included in `project3_source/THUCNews/data`.
