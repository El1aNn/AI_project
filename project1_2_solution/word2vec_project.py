#!/usr/bin/env python3
"""Train and evaluate CBOW / Skip-gram word embeddings for Projects 1 and 2.

The script trains Word2Vec-style embeddings on NLTK Reuters by default, saves
vectors in the .vec text format required by the supplied templates, and runs the
three assignment evaluations:

1. K-nearest neighbors for the 50 query words.
2. SimLex-999 golden standard correlation.
3. Analogy reasoning accuracy.

Example:
    python project1_2_solution/word2vec_project.py --student-id 123456789
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.swanlab_helper import SwanLabLogger, add_swanlab_args


QUERY_WORDS = [
    "july", "reliable", "play", "willing", "good", "very", "patient",
    "concerned", "important", "powerful", "quickly", "generally",
    "gradually", "happy", "able", "close", "near", "saturday", "friend",
    "company", "road", "plane", "war", "politics", "building", "student",
    "university", "realm", "china", "experience", "police", "give",
    "create", "tell", "become", "lack", "win", "help", "gain", "get",
    "take", "use", "set", "find", "increase", "difficult", "go", "man",
    "ten", "year",
]

SPECIALS = ["<UNK>"]
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.IGNORECASE)


def require_training_dependencies():
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import tqdm  # noqa: F401
    except ImportError:
        missing.append("tqdm")
    try:
        import scipy  # noqa: F401
    except ImportError:
        missing.append("scipy")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {joined}. Install with "
            "`python -m pip install -r requirements.txt` before training."
        )


def tokenize_text(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def load_reuters_tokens(download: bool = True) -> List[str]:
    try:
        import nltk
        from nltk.corpus import reuters
    except ImportError as exc:
        raise SystemExit(
            "NLTK is required for the default Reuters corpus. Install it with "
            "`python -m pip install nltk`, or pass --corpus-path."
        ) from exc

    try:
        fileids = reuters.fileids()
    except LookupError:
        if not download:
            raise
        nltk.download("reuters")
        fileids = reuters.fileids()

    tokens: List[str] = []
    for word in reuters.words(fileids):
        word = word.lower()
        if TOKEN_RE.fullmatch(word):
            tokens.append(word)
    return tokens


def load_corpus_tokens(corpus_path: Path | None, download_reuters: bool) -> List[str]:
    if corpus_path is None:
        print("Loading NLTK Reuters corpus...")
        return load_reuters_tokens(download=download_reuters)

    print(f"Loading corpus from {corpus_path}...")
    tokens: List[str] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens.extend(tokenize_text(line))
    return tokens


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    counts: List[int]

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __len__(self) -> int:
        return len(self.id_to_token)

    def id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id["<UNK>"])


def build_vocab(tokens: Sequence[str], min_count: int, max_vocab: int) -> Vocab:
    counts = Counter(tokens)
    kept = [
        (tok, count)
        for tok, count in counts.most_common()
        if count >= min_count and tok not in SPECIALS
    ][: max(0, max_vocab - len(SPECIALS))]

    id_to_token = SPECIALS + [tok for tok, _ in kept]
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    id_counts = [0 for _ in id_to_token]
    id_counts[token_to_id["<UNK>"]] = sum(
        count for tok, count in counts.items() if tok not in token_to_id
    )
    for tok, count in kept:
        id_counts[token_to_id[tok]] = count
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, counts=id_counts)


def numericalize(tokens: Sequence[str], vocab: Vocab) -> List[int]:
    return [vocab.id(tok) for tok in tokens]


def make_unigram_distribution(counts: Sequence[int]):
    import torch

    weights = torch.tensor(counts, dtype=torch.float)
    weights[0] = max(weights[0], 1.0)
    weights = weights.pow(0.75)
    return weights / weights.sum()


class CBOWDataset:
    def __init__(self, token_ids: Sequence[int], window_size: int, max_samples: int | None, seed: int):
        positions = list(range(window_size, len(token_ids) - window_size))
        if max_samples and len(positions) > max_samples:
            rng = random.Random(seed)
            positions = rng.sample(positions, max_samples)
        self.token_ids = token_ids
        self.window_size = window_size
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = self.positions[idx]
        context = (
            self.token_ids[pos - self.window_size : pos]
            + self.token_ids[pos + 1 : pos + self.window_size + 1]
        )
        return context, self.token_ids[pos]


class SkipGramDataset:
    def __init__(self, token_ids: Sequence[int], window_size: int, max_samples: int | None, seed: int):
        rng = random.Random(seed)
        if max_samples:
            pairs = []
            while len(pairs) < max_samples:
                pos = rng.randrange(len(token_ids))
                offsets = [
                    offset
                    for offset in range(-window_size, window_size + 1)
                    if offset != 0 and 0 <= pos + offset < len(token_ids)
                ]
                if not offsets:
                    continue
                ctx_pos = pos + rng.choice(offsets)
                pairs.append((token_ids[pos], token_ids[ctx_pos]))
        else:
            pairs: List[Tuple[int, int]] = []
            for pos, center in enumerate(token_ids):
                start = max(0, pos - window_size)
                end = min(len(token_ids), pos + window_size + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos != pos:
                        pairs.append((center, token_ids[ctx_pos]))
        if max_samples and len(pairs) > max_samples:
            pairs = rng.sample(pairs, max_samples)
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def cbow_collate(batch):
    import torch

    contexts, centers = zip(*batch)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(centers, dtype=torch.long)


def skipgram_collate(batch):
    import torch

    centers, contexts = zip(*batch)
    return torch.tensor(centers, dtype=torch.long), torch.tensor(contexts, dtype=torch.long)


def make_model(model_name: str, vocab_size: int, embedding_dim: int):
    import torch

    class CBOWNegativeSampling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_embed = torch.nn.Embedding(vocab_size, embedding_dim)
            self.out_embed = torch.nn.Embedding(vocab_size, embedding_dim)
            self.reset_parameters()

        def reset_parameters(self):
            init_range = 0.5 / embedding_dim
            self.in_embed.weight.data.uniform_(-init_range, init_range)
            self.out_embed.weight.data.zero_()

        def forward(self, contexts, centers, negatives):
            context_vec = self.in_embed(contexts).mean(dim=1)
            pos_vec = self.out_embed(centers)
            neg_vec = self.out_embed(negatives)
            pos_score = (context_vec * pos_vec).sum(dim=1)
            neg_score = torch.bmm(neg_vec, context_vec.unsqueeze(2)).squeeze(2)
            loss = -(
                torch.nn.functional.logsigmoid(pos_score)
                + torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
            ).mean()
            return loss

        def embeddings(self):
            return self.in_embed.weight.detach().cpu()

    class SkipGramNegativeSampling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_embed = torch.nn.Embedding(vocab_size, embedding_dim)
            self.out_embed = torch.nn.Embedding(vocab_size, embedding_dim)
            self.reset_parameters()

        def reset_parameters(self):
            init_range = 0.5 / embedding_dim
            self.in_embed.weight.data.uniform_(-init_range, init_range)
            self.out_embed.weight.data.zero_()

        def forward(self, centers, contexts, negatives):
            center_vec = self.in_embed(centers)
            pos_vec = self.out_embed(contexts)
            neg_vec = self.out_embed(negatives)
            pos_score = (center_vec * pos_vec).sum(dim=1)
            neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
            loss = -(
                torch.nn.functional.logsigmoid(pos_score)
                + torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)
            ).mean()
            return loss

        def embeddings(self):
            return self.in_embed.weight.detach().cpu()

    if model_name == "cbow":
        return CBOWNegativeSampling()
    if model_name == "skipgram":
        return SkipGramNegativeSampling()
    raise ValueError(f"Unknown model: {model_name}")


def train_embeddings(
    model_name: str,
    token_ids: Sequence[int],
    vocab: Vocab,
    args: argparse.Namespace,
    swanlab: SwanLabLogger,
):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if model_name == "cbow":
        dataset = CBOWDataset(token_ids, args.window_size, args.max_samples, args.seed)
        collate_fn = cbow_collate
    else:
        dataset = SkipGramDataset(token_ids, args.window_size, args.max_samples, args.seed)
        collate_fn = skipgram_collate

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    model = make_model(model_name, len(vocab), args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    neg_dist = make_unigram_distribution(vocab.counts)

    print(f"\nTraining {model_name}: {len(dataset):,} samples, {len(vocab):,} vocabulary")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"{model_name} epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            batch_size = batch[0].shape[0]
            negatives = torch.multinomial(
                neg_dist,
                batch_size * args.num_negative,
                replacement=True,
            ).view(batch_size, args.num_negative)
            loss = model(*batch, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        print(f"{model_name} epoch {epoch}: loss={avg_loss:.4f}")
        swanlab.log_metrics(
            {
                "train_loss": avg_loss,
                "samples": len(dataset),
                "vocab_size": len(vocab),
            },
            step=epoch,
            prefix=f"project1_2/{model_name}",
        )

    return model.embeddings()


def save_vec(path: Path, vocab: Vocab, embeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{len(vocab)} {embeddings.shape[1]}\n")
        for i, token in enumerate(vocab.id_to_token):
            values = " ".join(f"{float(x):.6f}" for x in embeddings[i].tolist())
            f.write(f"{token} {values}\n")
    print(f"Saved vectors to {path}")


def load_vec(path: Path):
    import torch

    tokens: List[str] = []
    vectors: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()
        has_header = len(first) == 2 and all(x.isdigit() for x in first)
        if not has_header:
            token, *vals = first
            tokens.append(token)
            vectors.append([float(x) for x in vals])
        for line in f:
            parts = line.rstrip().split()
            if not parts:
                continue
            tokens.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    return token_to_id, tokens, torch.tensor(vectors, dtype=torch.float)


def cosine_scores(vectors, query_vec):
    import torch

    return torch.matmul(vectors, query_vec) / (
        vectors.norm(dim=1) * query_vec.norm() + 1e-9
    )


def top_k_words(token_to_id, id_to_token, vectors, query: str, k: int, exclude: Iterable[str] = ()):
    import torch

    if query not in token_to_id:
        return []
    scores = cosine_scores(vectors, vectors[token_to_id[query]]).clone()
    for token in set(exclude) | {query, "<UNK>"}:
        if token in token_to_id:
            scores[token_to_id[token]] = -math.inf
    top = torch.topk(scores, k=min(k, len(id_to_token))).indices.tolist()
    return [(id_to_token[i], float(scores[i])) for i in top if math.isfinite(float(scores[i]))]


def evaluate_knn(
    model_name: str,
    vec_path: Path,
    output_dir: Path,
    student_id: int,
    k: int = 10,
) -> Dict[str, float]:
    token_to_id, id_to_token, vectors = load_vec(vec_path)
    rows = []
    average_scores = []
    available_queries = [w for w in QUERY_WORDS if w in token_to_id]

    for query in QUERY_WORDS:
        neighbors = top_k_words(token_to_id, id_to_token, vectors, query, k)
        if neighbors:
            average_scores.append(sum(score for _, score in neighbors) / len(neighbors))
        for rank, (neighbor, score) in enumerate(neighbors, start=1):
            rows.append(
                {
                    "model": model_name,
                    "query": query,
                    "rank": rank,
                    "neighbor": neighbor,
                    "cosine": f"{score:.6f}",
                }
            )

    out_path = output_dir / f"{model_name}_knn_top{k}.csv"
    write_csv(out_path, ["model", "query", "rank", "neighbor", "cosine"], rows)

    rng = random.Random(student_id)
    sampled = rng.sample(available_queries, min(4, len(available_queries)))
    detailed_rows = [row for row in rows if row["query"] in sampled]
    detail_path = output_dir / f"{model_name}_knn_4_words_for_report.csv"
    write_csv(detail_path, ["model", "query", "rank", "neighbor", "cosine"], detailed_rows)

    return {
        "available_queries": len(available_queries),
        "mean_top10_cosine": float(sum(average_scores) / max(1, len(average_scores))),
    }


def evaluate_simlex(model_name: str, vec_path: Path, simlex_path: Path, output_dir: Path, student_id: int):
    from scipy import stats

    token_to_id, _, vectors = load_vec(vec_path)
    rows = []
    standard_scores: List[float] = []
    calculated_scores: List[float] = []

    with simlex_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            w1, w2, score_s = parts
            w1, w2 = w1.lower(), w2.lower()
            standard = float(score_s)
            if w1 in token_to_id and w2 in token_to_id:
                cos = float(cosine_scores(vectors, vectors[token_to_id[w1]])[token_to_id[w2]])
                scaled = (cos + 1.0) * 5.0
                covered = True
                standard_scores.append(standard)
                calculated_scores.append(scaled)
            else:
                cos = float("nan")
                scaled = float("nan")
                covered = False
            rows.append(
                {
                    "model": model_name,
                    "word1": w1,
                    "word2": w2,
                    "standard_0_10": f"{standard:.4f}",
                    "cosine": "" if not covered else f"{cos:.6f}",
                    "scaled_cosine_0_10": "" if not covered else f"{scaled:.6f}",
                    "covered": int(covered),
                }
            )

    correlation = (
        float(stats.spearmanr(standard_scores, calculated_scores).correlation)
        if len(standard_scores) >= 2
        else float("nan")
    )
    out_path = output_dir / f"{model_name}_simlex999.csv"
    write_csv(
        out_path,
        ["model", "word1", "word2", "standard_0_10", "cosine", "scaled_cosine_0_10", "covered"],
        rows,
    )

    covered_rows = [row for row in rows if row["covered"] == 1]
    rng = random.Random(student_id + 999)
    sampled = rng.sample(covered_rows, min(20, len(covered_rows)))
    write_csv(
        output_dir / f"{model_name}_simlex999_20_pairs_for_report.csv",
        ["model", "word1", "word2", "standard_0_10", "cosine", "scaled_cosine_0_10", "covered"],
        sampled,
    )
    return {"covered_pairs": len(covered_rows), "spearman": correlation}


def evaluate_analogy(model_name: str, vec_path: Path, analogy_path: Path, output_dir: Path, student_id: int):
    import torch

    token_to_id, id_to_token, vectors = load_vec(vec_path)
    rows = []
    category = ""
    total_covered = 0
    total_correct = 0

    with analogy_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip()
                continue
            words = [w.lower() for w in line.split()]
            if len(words) != 4:
                continue
            a, b, c, expected = words
            covered = all(w in token_to_id for w in words)
            prediction = ""
            score = ""
            correct = False
            if covered:
                total_covered += 1
                target = vectors[token_to_id[b]] - vectors[token_to_id[a]] + vectors[token_to_id[c]]
                scores = cosine_scores(vectors, target).clone()
                for token in {a, b, c, "<UNK>"}:
                    if token in token_to_id:
                        scores[token_to_id[token]] = -math.inf
                idx = int(torch.argmax(scores).item())
                prediction = id_to_token[idx]
                score = f"{float(scores[idx]):.6f}"
                correct = prediction == expected
                total_correct += int(correct)
            rows.append(
                {
                    "model": model_name,
                    "category": category,
                    "word_a": a,
                    "word_b": b,
                    "word_c": c,
                    "expected": expected,
                    "prediction": prediction,
                    "cosine": score,
                    "covered": int(covered),
                    "correct": int(correct),
                }
            )

    write_csv(
        output_dir / f"{model_name}_analogy.csv",
        [
            "model", "category", "word_a", "word_b", "word_c", "expected",
            "prediction", "cosine", "covered", "correct",
        ],
        rows,
    )
    covered_rows = [row for row in rows if row["covered"] == 1]
    rng = random.Random(student_id + 2013)
    sampled = rng.sample(covered_rows, min(10, len(covered_rows)))
    write_csv(
        output_dir / f"{model_name}_analogy_10_examples_for_report.csv",
        [
            "model", "category", "word_a", "word_b", "word_c", "expected",
            "prediction", "cosine", "covered", "correct",
        ],
        sampled,
    )
    return {
        "covered_questions": total_covered,
        "correct": total_correct,
        "accuracy": float(total_correct / total_covered) if total_covered else float("nan"),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_evaluations(model_names: Sequence[str], args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    simlex_path = Path(args.simlex_path)
    analogy_path = Path(args.analogy_path)
    summary = {}
    for model_name in model_names:
        vec_path = output_dir / f"{model_name}.vec"
        if not vec_path.exists():
            print(f"Skip evaluation for {model_name}: missing {vec_path}")
            continue
        print(f"\nEvaluating {model_name}...")
        summary[model_name] = {
            "knn": evaluate_knn(model_name, vec_path, output_dir, args.student_id),
            "simlex999": evaluate_simlex(model_name, vec_path, simlex_path, output_dir, args.student_id),
            "analogy": evaluate_analogy(model_name, vec_path, analogy_path, output_dir, args.student_id),
        }

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation summary to {summary_path}")
    return summary


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["cbow", "skipgram"], default=["cbow", "skipgram"])
    parser.add_argument("--mode", choices=["all", "train", "eval"], default="all")
    parser.add_argument("--student-id", type=int, default=20260517, help="Random seed for report samples.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--no-download-reuters", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("project1_2_solution/outputs"))
    parser.add_argument("--simlex-path", type=Path, default=Path("project1_2_source/simlex-999.txt"))
    parser.add_argument(
        "--analogy-path",
        type=Path,
        default=Path("project1_2_source/analogical reasoning task.txt"),
    )
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke-test run; not suitable for final report numbers.",
    )
    add_swanlab_args(parser)
    args = parser.parse_args(argv)

    if args.quick:
        args.epochs = 1
        args.embedding_dim = min(args.embedding_dim, 50)
        args.max_vocab = min(args.max_vocab, 5000)
        args.max_samples = min(args.max_samples, 20000)
        args.min_count = min(args.min_count, 2)
        args.batch_size = min(args.batch_size, 256)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    require_training_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    swanlab = SwanLabLogger.from_args(
        args,
        project=args.swanlab_project,
        experiment_name=args.swanlab_experiment or "project1_2_word2vec",
        config={
            "project": "project1_2",
            "models": args.models,
            "student_id": args.student_id,
            "seed": args.seed,
            "embedding_dim": args.embedding_dim,
            "window_size": args.window_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "min_count": args.min_count,
            "max_vocab": args.max_vocab,
            "max_samples": args.max_samples,
            "quick": args.quick,
        },
    )

    try:
        if args.mode in {"all", "train"}:
            tokens = load_corpus_tokens(args.corpus_path, download_reuters=not args.no_download_reuters)
            print(f"Loaded {len(tokens):,} tokens")
            vocab = build_vocab(tokens, args.min_count, args.max_vocab)
            token_ids = numericalize(tokens, vocab)
            print(f"Vocabulary size: {len(vocab):,}")
            swanlab.log_metrics(
                {"token_count": len(tokens), "vocab_size": len(vocab)},
                prefix="project1_2/corpus",
            )

            for model_name in args.models:
                embeddings = train_embeddings(model_name, token_ids, vocab, args, swanlab)
                vec_path = args.output_dir / f"{model_name}.vec"
                save_vec(vec_path, vocab, embeddings)
                swanlab.log_output_path(f"project1_2/{model_name}/vec_path", vec_path)

        if args.mode in {"all", "eval"}:
            summary = run_evaluations(args.models, args)
            swanlab.log_metrics(summary, prefix="project1_2/eval")
            swanlab.log_output_path("project1_2/evaluation_summary", args.output_dir / "evaluation_summary.json")
    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()
