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

Project 3 data is already included in `project3_source/THUCNews/data`.
