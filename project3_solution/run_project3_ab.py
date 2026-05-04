#!/usr/bin/env python3
"""Run Project 3 Transformer Task A and Task B with misclassification analysis.

Task A uses the original character-level THUCNews titles.
Task B performs Chinese word segmentation first, then trains the same
Transformer classifier on word-level tokens.

Outputs include metrics, confusion matrices, all predictions, misclassified
documents, and the A-vs-B comparison required by the assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.swanlab_helper import SwanLabLogger, add_swanlab_args


UNK = "<UNK>"
PAD = "<PAD>"


def require_dependencies() -> None:
    missing = []
    for name in ["torch", "sklearn", "tqdm"]:
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


def get_tokenizer(task: str):
    if task == "A":
        return lambda text: [ch for ch in text]

    try:
        import jieba

        return lambda text: [tok.strip() for tok in jieba.lcut(text) if tok.strip()]
    except ImportError:
        print(
            "WARNING: jieba is not installed; Task B falls back to a simple tokenizer. "
            "Install jieba for the final report run."
        )
        pattern = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]|[^\s]")
        return lambda text: pattern.findall(text)


def read_records(path: Path, limit: int | None = None) -> List[Tuple[str, int]]:
    records: List[Tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            records.append((text, int(label)))
            if limit and len(records) >= limit:
                break
    return records


def write_segmented_files(data_dir: Path, output_data_dir: Path) -> None:
    tokenizer = get_tokenizer("B")
    output_data_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "dev", "test"]:
        in_path = data_dir / f"{split}.txt"
        out_path = output_data_dir / f"{split}.txt"
        with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.rstrip("\n")
                if not line:
                    continue
                text, label = line.rsplit("\t", 1)
                dst.write(" ".join(tokenizer(text)) + "\t" + label + "\n")
    class_src = data_dir / "class.txt"
    class_dst = output_data_dir / "class.txt"
    class_dst.write_text(class_src.read_text(encoding="utf-8"), encoding="utf-8")


def build_vocab(records: Sequence[Tuple[str, int]], tokenizer, max_vocab: int, min_freq: int) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for text, _ in records:
        for token in tokenizer(text):
            counter[token] = counter.get(token, 0) + 1
    sorted_tokens = sorted(
        [(token, count) for token, count in counter.items() if count >= min_freq],
        key=lambda item: item[1],
        reverse=True,
    )[:max_vocab]
    vocab = {token: idx for idx, (token, _) in enumerate(sorted_tokens)}
    vocab[UNK] = len(vocab)
    vocab[PAD] = len(vocab)
    return vocab


class NewsDataset:
    def __init__(self, records: Sequence[Tuple[str, int]], tokenizer, vocab: Dict[str, int], pad_size: int):
        self.rows = []
        unk = vocab[UNK]
        pad = vocab[PAD]
        for index, (text, label) in enumerate(records):
            tokens = tokenizer(text)
            seq_len = min(len(tokens), pad_size)
            ids = [vocab.get(token, unk) for token in tokens[:pad_size]]
            if len(ids) < pad_size:
                ids.extend([pad] * (pad_size - len(ids)))
            self.rows.append((index, text, ids, int(label), seq_len))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_news(batch):
    import torch

    indices, texts, ids, labels, seq_lens = zip(*batch)
    x = torch.LongTensor(ids)
    seq_len = torch.LongTensor(seq_lens)
    y = torch.LongTensor(labels)
    return indices, texts, (x, seq_len), y


@dataclass
class TransformerConfig:
    class_list: List[str]
    n_vocab: int
    device: object
    model_name: str = "Transformer"
    dropout: float = 0.5
    num_epochs: int = 6
    batch_size: int = 128
    pad_size: int = 32
    learning_rate: float = 5e-4
    embed: int = 300
    dim_model: int = 300
    hidden: int = 1024
    last_hidden: int = 512
    num_head: int = 5
    num_encoder: int = 2

    @property
    def num_classes(self) -> int:
        return len(self.class_list)


def load_transformer_model(config: TransformerConfig, source_dir: Path):
    import torch

    sys.path.insert(0, str(source_dir.resolve()))
    from models.Transformer import Model

    return Model(config).to(config.device)


def build_loaders(train_records, dev_records, test_records, tokenizer, vocab, config, args):
    import torch
    from torch.utils.data import DataLoader

    train_ds = NewsDataset(train_records, tokenizer, vocab, config.pad_size)
    dev_ds = NewsDataset(dev_records, tokenizer, vocab, config.pad_size)
    test_ds = NewsDataset(test_records, tokenizer, vocab, config.pad_size)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_news,
        generator=generator,
    )
    dev_loader = DataLoader(dev_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_news)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_news)
    return train_loader, dev_loader, test_loader


def evaluate_model(model, loader, config, split: str, output_csv: Path | None = None):
    import torch
    import torch.nn.functional as F
    from sklearn import metrics
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    rows = []

    with torch.no_grad():
        for indices, texts, batch_x, labels in tqdm(loader, desc=f"eval {split}"):
            batch_x = (batch_x[0].to(config.device), batch_x[1].to(config.device))
            labels = labels.to(config.device)
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, labels)
            total_loss += float(loss.item())
            preds = torch.argmax(outputs, dim=1).detach().cpu().tolist()
            label_list = labels.detach().cpu().tolist()
            y_true.extend(label_list)
            y_pred.extend(preds)
            for idx, text, true_label, pred_label in zip(indices, texts, label_list, preds):
                rows.append(
                    {
                        "index": int(idx),
                        "text": text,
                        "true_label_id": int(true_label),
                        "true_label": config.class_list[int(true_label)],
                        "pred_label_id": int(pred_label),
                        "pred_label": config.class_list[int(pred_label)],
                        "correct": int(true_label == pred_label),
                    }
                )

    accuracy = metrics.accuracy_score(y_true, y_pred)
    labels = list(range(len(config.class_list)))
    report = metrics.classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=config.class_list,
        digits=4,
        zero_division=0,
    )
    confusion = metrics.confusion_matrix(y_true, y_pred, labels=labels).tolist()
    if output_csv is not None:
        write_csv(
            output_csv,
            ["index", "text", "true_label_id", "true_label", "pred_label_id", "pred_label", "correct"],
            rows,
        )
        wrong_rows = [row for row in rows if int(row["correct"]) == 0]
        write_csv(
            output_csv.with_name(output_csv.stem + "_misclassified.csv"),
            ["index", "text", "true_label_id", "true_label", "pred_label_id", "pred_label", "correct"],
            wrong_rows,
        )
    return {
        "split": split,
        "loss": total_loss / max(1, len(loader)),
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": confusion,
    }


def train_one_task(task: str, args: argparse.Namespace, swanlab: SwanLabLogger) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    source_dir = Path(args.source_dir)
    data_dir = Path(args.data_dir)
    task_name = "taskA_char" if task == "A" else "taskB_word"
    out_dir = Path(args.output_dir) / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(task)
    class_list = [line.strip() for line in (data_dir / "class.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_records = read_records(data_dir / "train.txt", args.max_train_samples)
    dev_records = read_records(data_dir / "dev.txt", args.max_dev_samples)
    test_records = read_records(data_dir / "test.txt", args.max_test_samples)
    vocab = build_vocab(train_records, tokenizer, args.max_vocab, args.min_freq)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = TransformerConfig(
        class_list=class_list,
        n_vocab=len(vocab),
        device=device,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        pad_size=args.pad_size,
        learning_rate=args.learning_rate,
    )
    model = load_transformer_model(config, source_dir)
    train_loader, dev_loader, test_loader = build_loaders(
        train_records, dev_records, test_records, tokenizer, vocab, config, args
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        y_true = []
        y_pred = []
        for _, _, batch_x, labels in tqdm(train_loader, desc=f"{task_name} train epoch {epoch}/{args.epochs}"):
            batch_x = (batch_x[0].to(device), batch_x[1].to(device))
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())
        train_acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
        dev_metrics = evaluate_model(model, dev_loader, config, "dev")
        epoch_row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "train_accuracy": train_acc,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
        }
        history.append(epoch_row)
        swanlab.log_metrics(epoch_row, step=epoch, prefix=f"project3/{task_name}")
        print(
            f"{task_name} epoch {epoch}: "
            f"train_loss={epoch_row['train_loss']:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"dev_loss={dev_metrics['loss']:.4f}, "
            f"dev_acc={dev_metrics['accuracy']:.4f}"
        )

    test_csv = out_dir / "test_predictions.csv"
    test_metrics = evaluate_model(model, test_loader, config, "test", output_csv=test_csv)
    metrics_out = {
        "task": task,
        "task_name": task_name,
        "elapsed_seconds": round(time.time() - start, 2),
        "vocab_size": len(vocab),
        "history": history,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(test_metrics["classification_report"], encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.save_model:
        torch.save(model.state_dict(), out_dir / "Transformer.ckpt")
    wrong_count = sum(1 for row in read_prediction_csv(test_csv).values() if int(row["correct"]) == 0)
    swanlab.log_metrics(
        {
            "elapsed_seconds": metrics_out["elapsed_seconds"],
            "vocab_size": len(vocab),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_misclassified": wrong_count,
        },
        prefix=f"project3/{task_name}/final",
    )
    swanlab.log_output_path(f"project3/{task_name}/metrics_json", out_dir / "metrics.json")
    swanlab.log_output_path(f"project3/{task_name}/misclassified_csv", test_csv.with_name("test_predictions_misclassified.csv"))
    print(f"Saved {task_name} outputs to {out_dir}")
    return metrics_out


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_prediction_csv(path: Path) -> Dict[int, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["index"]): row for row in reader}


def compare_task_predictions(output_dir: Path) -> Dict[str, int]:
    a_path = output_dir / "taskA_char" / "test_predictions.csv"
    b_path = output_dir / "taskB_word" / "test_predictions.csv"
    if not a_path.exists() or not b_path.exists():
        print("Comparison skipped: Task A and Task B prediction files are both required.")
        return {"a_wrong_b_right": 0, "b_wrong_a_right": 0}

    a_rows = read_prediction_csv(a_path)
    b_rows = read_prediction_csv(b_path)
    common = sorted(set(a_rows) & set(b_rows))

    a_wrong_b_right = []
    b_wrong_a_right = []
    for idx in common:
        a = a_rows[idx]
        b = b_rows[idx]
        merged = {
            "index": idx,
            "text": a["text"],
            "true_label": a["true_label"],
            "taskA_pred": a["pred_label"],
            "taskB_pred": b["pred_label"],
        }
        if int(a["correct"]) == 0 and int(b["correct"]) == 1:
            a_wrong_b_right.append(merged)
        if int(b["correct"]) == 0 and int(a["correct"]) == 1:
            b_wrong_a_right.append(merged)

    fields = ["index", "text", "true_label", "taskA_pred", "taskB_pred"]
    write_csv(output_dir / "A_wrong_B_right.csv", fields, a_wrong_b_right)
    write_csv(output_dir / "B_wrong_A_right.csv", fields, b_wrong_a_right)
    write_comparison_notes(output_dir / "comparison_examples.md", a_wrong_b_right, b_wrong_a_right)
    print(f"Saved A/B comparison outputs to {output_dir}")
    return {
        "a_wrong_b_right": len(a_wrong_b_right),
        "b_wrong_a_right": len(b_wrong_a_right),
    }


def write_comparison_notes(path: Path, a_wrong_b_right, b_wrong_a_right) -> None:
    lines = [
        "# Project 3 Misclassification Comparison",
        "",
        "## 3 documents misclassified in Task A but correct in Task B",
        "",
    ]
    if not a_wrong_b_right:
        lines.append("No such examples were found in the current run.")
    for row in a_wrong_b_right[:3]:
        lines.extend(
            [
                f"- Index {row['index']}: {row['text']}",
                f"  True label: {row['true_label']}; Task A predicted {row['taskA_pred']}; Task B predicted {row['taskB_pred']}.",
                "  Brief explanation: word segmentation can preserve multi-character keywords, names, and domain terms, so Task B may capture the news topic more clearly than isolated characters.",
            ]
        )
    lines.extend(["", "## 3 documents misclassified in Task B but correct in Task A", ""])
    if not b_wrong_a_right:
        lines.append("No such examples were found in the current run.")
    for row in b_wrong_a_right[:3]:
        lines.extend(
            [
                f"- Index {row['index']}: {row['text']}",
                f"  True label: {row['true_label']}; Task A predicted {row['taskA_pred']}; Task B predicted {row['taskB_pred']}.",
                "  Brief explanation: character-level input is robust to rare words, short titles, and segmentation mistakes, so Task A can sometimes avoid errors introduced by word tokenization.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("project3_source"))
    parser.add_argument("--data-dir", type=Path, default=Path("project3_source/THUCNews/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("project3_solution/outputs"))
    parser.add_argument("--tasks", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--student-id", type=int, default=20260517)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pad-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-vocab", type=int, default=10000)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--prepare-task-b-data", action="store_true")
    parser.add_argument("--only-compare", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run; not for final report.")
    add_swanlab_args(parser)
    args = parser.parse_args(argv)
    if args.quick:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 32)
        args.max_train_samples = args.max_train_samples or 256
        args.max_dev_samples = args.max_dev_samples or 128
        args.max_test_samples = args.max_test_samples or 128
        args.max_vocab = min(args.max_vocab, 2000)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    swanlab = SwanLabLogger.from_args(
        args,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or "project3_transformer_ab",
        config={
            "project": "project3",
            "tasks": args.tasks,
            "student_id": args.student_id,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "pad_size": args.pad_size,
            "learning_rate": args.learning_rate,
            "max_vocab": args.max_vocab,
            "quick": args.quick,
        },
    )

    try:
        if args.prepare_task_b_data or "B" in args.tasks:
            word_data_dir = Path("project3_solution/THUCNews_word/data")
            write_segmented_files(args.data_dir, word_data_dir)
            swanlab.log_output_path("project3/taskB_word_data_dir", word_data_dir)

        if not args.only_compare:
            for task in args.tasks:
                train_one_task(task, args, swanlab)

        comparison = compare_task_predictions(Path(args.output_dir))
        swanlab.log_metrics(comparison, prefix="project3/comparison")
        swanlab.log_output_path("project3/comparison_examples", Path(args.output_dir) / "comparison_examples.md")
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
