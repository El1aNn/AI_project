# Missing Items and Setup Notes

The current local Python environment cannot train the projects yet because these
packages are missing:

- `torch`
- `nltk`
- `jieba`
- `transformers`
- `datasets`
- `evaluate`
- `accelerate`
- `swanlab` if you enable metric upload with `SWANLAB=1`

Install dependencies first:

```bash
python -m pip install -r requirements.txt
```

Recommended: use Python 3.10 or 3.11 if a package has trouble installing on the
current Python version.

Project-specific notes:

- Project 1/2: the zip file does not include the original HIT-SCIR `chp5`
  training code or the Reuters corpus. I added a standalone replacement script
  that trains CBOW and Skip-gram from NLTK Reuters and produces all required
  evaluation files. Internet access is needed the first time NLTK downloads
  Reuters, unless you pass `--corpus-path` with your own corpus.
- Project 3: the THUCNews data is present. Task B uses `jieba` segmentation;
  install `jieba` before final training.
- Project 4: SST-2/MRPC and the pretrained BERT-mini model are downloaded from
  Hugging Face, so internet access is needed for the first run.
- Fill in your real student name(s) and student ID(s) in the report. Use the
  student ID in the scripts to make random sample selection reproducible.
