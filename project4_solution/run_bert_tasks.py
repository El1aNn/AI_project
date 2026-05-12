#!/usr/bin/env python3
"""Run Project 4 BERT fine-tuning for SST-2 and MRPC.

This runner keeps the assignment behavior of sst2.py and mrpc.py, but makes it
more reproducible and saves the outputs needed for the report:

* final validation metrics,
* validation predictions,
* two correct and two incorrect SST-2 examples when available,
* two paraphrase and two non-paraphrase MRPC examples for discussion.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.swanlab_helper import SwanLabLogger, add_swanlab_args


TASKS = {
    "sst2": {
        "glue_name": "sst2",
        "text_columns": ("sentence",),
        "max_length": 64,
        "label_names": {0: "negative", 1: "positive"},
    },
    "mrpc": {
        "glue_name": "mrpc",
        "text_columns": ("sentence1", "sentence2"),
        "max_length": 100,
        "label_names": {0: "not_equivalent", 1: "equivalent"},
    },
}


def require_dependencies() -> None:
    missing = []
    for name in ["torch", "datasets", "transformers", "sklearn", "accelerate"]:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + ". Install with `python -m pip install -r requirements.txt`."
        )


def build_training_arguments(output_dir: Path, args, train_size: int):
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    warmup_steps = 0
    if args.warmup_steps is not None:
        warmup_steps = args.warmup_steps
    elif args.warmup_ratio > 0:
        steps_per_epoch = math.ceil(train_size / max(1, args.batch_size))
        warmup_steps = int(math.ceil(steps_per_epoch * float(args.epochs) * args.warmup_ratio))

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "report_to": [],
        "seed": args.seed,
    }
    if "warmup_steps" in sig.parameters:
        kwargs["warmup_steps"] = warmup_steps
    elif "warmup_ratio" in sig.parameters:
        kwargs["warmup_ratio"] = args.warmup_ratio
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "load_best_model_at_end" in sig.parameters:
        kwargs["load_best_model_at_end"] = False
    return TrainingArguments(**kwargs)


def compute_metric_values(task: str, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    pred_labels = np.argmax(predictions, axis=1)
    metrics = {"accuracy": float(accuracy_score(labels, pred_labels))}
    if task in {"mrpc", "sst2"}:
        metrics["f1"] = float(f1_score(labels, pred_labels, zero_division=0))
    return metrics


def load_task_dataset(task: str, args):
    from datasets import load_dataset

    dataset = load_dataset("glue", TASKS[task]["glue_name"])
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )
    return dataset


def tokenize_dataset(task: str, dataset, tokenizer):
    spec = TASKS[task]
    text_columns = spec["text_columns"]

    def tokenize(examples):
        texts = [examples[col] for col in text_columns]
        return tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=spec["max_length"],
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    keep_cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    keep_cols = [col for col in keep_cols if col in tokenized["train"].column_names]
    tokenized.set_format(type="torch", columns=keep_cols)
    return tokenized


def write_predictions(task: str, dataset, pred_labels: np.ndarray, output_path: Path) -> List[Dict[str, object]]:
    spec = TASKS[task]
    rows = []
    for i, example in enumerate(dataset["validation"]):
        label = int(example["label"])
        pred = int(pred_labels[i])
        row = {
            "index": i,
            "label": label,
            "label_name": spec["label_names"][label],
            "prediction": pred,
            "prediction_name": spec["label_names"][pred],
            "correct": int(label == pred),
        }
        for col in spec["text_columns"]:
            row[col] = example[col]
        rows.append(row)
    fieldnames = ["index", *spec["text_columns"], "label", "label_name", "prediction", "prediction_name", "correct"]
    write_csv(output_path, fieldnames, rows)
    write_csv(output_path.with_name(output_path.stem + "_misclassified.csv"), fieldnames, [r for r in rows if not r["correct"]])
    return rows


def write_example_notes(task: str, rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    lines = [f"# {task.upper()} Report Examples", ""]
    if task == "sst2":
        correct = [row for row in rows if row["correct"]]
        wrong = [row for row in rows if not row["correct"]]
        lines.extend(["## Two Correct Sentences", ""])
        for row in correct[:2]:
            lines.extend(
                [
                    f"- {row['sentence']}",
                    f"  True label: {row['label_name']}; prediction: {row['prediction_name']}.",
                    "  Comment: the sentiment cue is relatively direct, so the fine-tuned model can classify it reliably.",
                ]
            )
        lines.extend(["", "## Two Incorrect Sentences", ""])
        if not wrong:
            lines.append("No incorrect validation examples were found in this run.")
        for row in wrong[:2]:
            lines.extend(
                [
                    f"- {row['sentence']}",
                    f"  True label: {row['label_name']}; prediction: {row['prediction_name']}.",
                    "  Comment: the sentence may contain ambiguity, contrast, or weak sentiment words that make the label harder to infer.",
                ]
            )
    else:
        paraphrase = [row for row in rows if row["label"] == 1]
        non_paraphrase = [row for row in rows if row["label"] == 0]
        lines.extend(["## Two Correct Paraphrase Pairs", ""])
        for row in paraphrase[:2]:
            lines.extend(
                [
                    f"- Sentence 1: {row['sentence1']}",
                    f"  Sentence 2: {row['sentence2']}",
                    f"  Model prediction: {row['prediction_name']}; correct: {bool(row['correct'])}.",
                    "  Comment: the two sentences describe the same event or claim with different wording.",
                ]
            )
        lines.extend(["", "## Two Incorrect Paraphrase Pairs", ""])
        for row in non_paraphrase[:2]:
            lines.extend(
                [
                    f"- Sentence 1: {row['sentence1']}",
                    f"  Sentence 2: {row['sentence2']}",
                    f"  Model prediction: {row['prediction_name']}; correct: {bool(row['correct'])}.",
                    "  Comment: the pair shares some surface words but differs in facts, relation, or meaning.",
                ]
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_task(task: str, args, swanlab: SwanLabLogger) -> Dict[str, float]:
    import torch
    from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainerCallback

    output_dir = Path(args.output_dir) / task
    checkpoint_dir = Path(args.checkpoint_dir) / task
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_task_dataset(task, args)
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    tokenized = tokenize_dataset(task, dataset, tokenizer)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        return_dict=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return compute_metric_values(task, logits, labels)

    class SwanLabTrainerCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs:
                swanlab.log_metrics(logs, step=int(state.global_step), prefix=f"project4/{task}/trainer")

    trainer_kwargs = {
        "model": model,
        "args": build_training_arguments(checkpoint_dir / "trainer", args, len(tokenized["train"])),
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "compute_metrics": compute_metrics,
        "callbacks": [SwanLabTrainerCallback()] if swanlab.enabled else None,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    eval_metrics = trainer.evaluate()
    model_dir = checkpoint_dir / "model"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    prediction_output = trainer.predict(tokenized["validation"])
    pred_labels = np.argmax(prediction_output.predictions, axis=1)
    rows = write_predictions(task, dataset, pred_labels, output_dir / "validation_predictions.csv")
    write_example_notes(task, rows, output_dir / "report_examples.md")

    # Keep the raw Trainer keys and add a compact copy for the report.
    metrics = {key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))}
    metrics["final_accuracy"] = float(metrics.get("eval_accuracy", 0.0))
    if "eval_f1" in metrics:
        metrics["final_f1"] = float(metrics["eval_f1"])
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    misclassified_count = sum(1 for row in rows if not row["correct"])
    swanlab.log_metrics(
        {
            **metrics,
            "train_size": len(dataset["train"]),
            "validation_size": len(dataset["validation"]),
            "misclassified": misclassified_count,
        },
        prefix=f"project4/{task}/final",
    )
    swanlab.log_output_path(f"project4/{task}/final_metrics_json", output_dir / "final_metrics.json")
    swanlab.log_output_path(f"project4/{task}/report_examples", output_dir / "report_examples.md")
    swanlab.log_output_path(f"project4/{task}/model_dir", model_dir)

    print(f"\n{task.upper()} final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["sst2", "mrpc", "all"], default="all")
    parser.add_argument("--model-name", default="prajjwal1/bert-mini")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("project4_solution/outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("project4_solution/checkpoints"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run; not for final report.")
    add_swanlab_args(parser)
    args = parser.parse_args(argv)
    if args.quick:
        args.epochs = 1.0
        args.batch_size = min(args.batch_size, 8)
        args.eval_batch_size = min(args.eval_batch_size, 8)
        args.max_train_samples = args.max_train_samples or 64
        args.max_eval_samples = args.max_eval_samples or 64
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = ["sst2", "mrpc"] if args.task == "all" else [args.task]
    swanlab = SwanLabLogger.from_args(
        args,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or "project4_bert_tasks",
        config={
            "project": "project4",
            "tasks": tasks,
            "model_name": args.model_name,
            "tokenizer_name": args.tokenizer_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "output_dir": str(args.output_dir),
            "checkpoint_dir": str(args.checkpoint_dir),
            "quick": args.quick,
        },
    )
    try:
        summary = {task: run_task(task, args, swanlab) for task in tasks}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        swanlab.log_metrics(summary, prefix="project4/summary")
        swanlab.log_output_path("project4/summary_json", args.output_dir / "summary.json")
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
