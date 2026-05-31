#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
]

LOSS_RE = re.compile(r"(?:Train|Val|Validation|Test)?\s*(?:loss|Loss)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)")
ITER_RE = re.compile(r"Iter\s+(\d+)", re.IGNORECASE)
TOK_RE = re.compile(r"Tokens/sec\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TRAINED_TOKENS_RE = re.compile(r"Trained Tokens\s+(\d+)", re.IGNORECASE)


def load_dataset_with_fallback(candidates: list[tuple[str, tuple[Any, ...]]]):
    errors = []
    for description, args in candidates:
        try:
            print(f"Trying dataset loader: {description}", flush=True)
            ds = load_dataset(*args)
            print(f"Loaded dataset with: {description}", flush=True)
            return ds, description
        except Exception as exc:
            msg = f"{description} failed: {type(exc).__name__}: {exc}"
            print(msg, flush=True)
            errors.append(msg)
    raise RuntimeError("All dataset loading attempts failed:\n" + "\n".join(errors))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(choice: str, subset_size: int) -> tuple[list[dict], list[dict], int, str]:
    if choice == "sst2":
        ds, loaded_as = load_dataset_with_fallback([
            ("nyu-mll/glue sst2", ("nyu-mll/glue", "sst2")),
            ("glue sst2 legacy alias", ("glue", "sst2")),
        ])
        train = ds["train"]
        full_count = len(train)
        rows = []
        for row in train.select(range(min(subset_size, full_count))):
            label = "positive" if int(row["label"]) == 1 else "negative"
            rows.append({
                "messages": [
                    {"role": "system", "content": "You classify movie-review sentiment. Answer only positive or negative."},
                    {"role": "user", "content": f"Classify the sentiment of this sentence: {row['sentence']}"},
                    {"role": "assistant", "content": label},
                ]
            })
        return rows, rows[: min(16, len(rows))], full_count, f"Hugging Face datasets: {loaded_as}, train split"

    if choice == "squad":
        ds, loaded_as = load_dataset_with_fallback([
            ("rajpurkar/squad", ("rajpurkar/squad",)),
            ("squad legacy alias", ("squad",)),
        ])
        train = ds["train"]
        full_count = len(train)
        rows = []
        for row in train.select(range(min(subset_size, full_count))):
            answers = row.get("answers", {}).get("text", [])
            answer = answers[0] if answers else ""
            rows.append({
                "messages": [
                    {"role": "system", "content": "You answer questions from the provided context. Return a concise answer span."},
                    {"role": "user", "content": f"Context: {row['context']}\n\nQuestion: {row['question']}"},
                    {"role": "assistant", "content": answer},
                ]
            })
        return rows, rows[: min(8, len(rows))], full_count, f"Hugging Face datasets: {loaded_as}, train split"

    raise ValueError(f"Unsupported dataset {choice!r}; use sst2 or squad")


def run_cmd(cmd: list[str]) -> tuple[int, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
    return proc.returncode, proc.stdout, time.perf_counter() - start


def parse_log(text: str) -> dict:
    losses = [float(m.group(1)) for m in LOSS_RE.finditer(text)]
    iterations = [int(m.group(1)) for m in ITER_RE.finditer(text)]
    token_rates = [float(m.group(1)) for m in TOK_RE.finditer(text)]
    trained_tokens = [int(m.group(1)) for m in TRAINED_TOKENS_RE.finditer(text)]
    return {
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "max_iteration": max(iterations) if iterations else None,
        "tokens_per_second_last": token_rates[-1] if token_rates else None,
        "tokens_per_second_mean": round(sum(token_rates) / len(token_rates), 3) if token_rates else None,
        "trained_tokens_last": trained_tokens[-1] if trained_tokens else None,
        "tail_30_lines": text.splitlines()[-30:],
    }


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sst2", "squad"], default="sst2")
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark-data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    print("# PACZero MLX subset timing benchmark")
    print("PACZero public evaluation uses SST-2 and SQuAD; this runner defaults to SST-2 and supports SQuAD.")
    print("Hugging Face dataset IDs are namespaced first, with legacy aliases as fallbacks.")
    print(f"selected_dataset={args.dataset}")
    print(f"subset_size={args.subset_size}")
    print(f"batch_size={args.batch_size}")
    print(f"num_layers={args.num_layers}")
    print(f"platform={platform.platform()} machine={platform.machine()}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets", "huggingface_hub", "safetensors"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")
    print()

    train_rows, valid_rows, full_count, dataset_source = build_rows(args.dataset, args.subset_size)
    actual_subset = len(train_rows)
    iters = actual_subset
    write_jsonl(args.data_dir / "train.jsonl", train_rows)
    write_jsonl(args.data_dir / "valid.jsonl", valid_rows)
    write_jsonl(args.data_dir / "test.jsonl", valid_rows)

    print(f"dataset_source={dataset_source}")
    print(f"full_train_examples={full_count}")
    print(f"timed_subset_examples={actual_subset}")
    print(f"training_iterations={iters}")
    print(f"train_jsonl_sha256={sha256_file(args.data_dir / 'train.jsonl')}")
    print("first_training_example_json=")
    print(json.dumps(train_rows[0], indent=2, ensure_ascii=False) if train_rows else "<none>")
    print()

    results = []
    for model in MODELS:
        model_out = args.out_dir / safe_name(model)
        adapter_dir = model_out / "adapters"
        model_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", model,
            "--train",
            "--data", str(args.data_dir),
            "--adapter-path", str(adapter_dir),
            "--iters", str(iters),
            "--batch-size", str(args.batch_size),
            "--num-layers", str(args.num_layers),
            "--val-batches", "1",
            "--max-seq-length", "512",
            "--steps-per-report", "1",
            "--steps-per-eval", str(max(1, iters)),
            "--save-every", str(max(1, iters)),
        ]
        print("=" * 100)
        print(f"MODEL={model}")
        print("COMMAND=" + " ".join(shlex.quote(x) for x in cmd))
        print("=" * 100)
        rc, output, elapsed = run_cmd(cmd)
        (model_out / "train.log").write_text(output, encoding="utf-8")
        print(output)
        parsed = parse_log(output)
        seconds_per_sample = elapsed / max(1, actual_subset)
        estimated_full_seconds = seconds_per_sample * full_count
        trained_tokens = parsed["trained_tokens_last"] or 0
        avg_tokens_per_sample = trained_tokens / max(1, actual_subset) if trained_tokens else None
        result = {
            "model": model,
            "returncode": rc,
            "success": rc == 0,
            "elapsed_seconds": round(elapsed, 3),
            "seconds_per_sample": round(seconds_per_sample, 6),
            "estimated_full_dataset_seconds": round(estimated_full_seconds, 3),
            "estimated_full_dataset_hours": round(estimated_full_seconds / 3600, 3),
            "actual_subset_examples": actual_subset,
            "full_train_examples": full_count,
            "avg_tokens_per_sample": round(avg_tokens_per_sample, 3) if avg_tokens_per_sample else None,
            "parsed": parsed,
        }
        results.append(result)
        print("MODEL_RESULT_JSON=")
        print(json.dumps(result, indent=2))
        print()

    payload = {
        "dataset_choice": args.dataset,
        "dataset_source": dataset_source,
        "full_train_examples": full_count,
        "subset_examples": actual_subset,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "results": results,
    }
    (args.out_dir / "paczero_subset_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("# Final estimate JSON")
    print(json.dumps(payload, indent=2))
    print("# Markdown summary")
    print("| Model | subset seconds | sec/sample | last tok/s | mean tok/s | estimated full hours |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in results:
        p = r["parsed"]
        print(f"| {r['model']} | {r['elapsed_seconds']} | {r['seconds_per_sample']} | {p['tokens_per_second_last']} | {p['tokens_per_second_mean']} | {r['estimated_full_dataset_hours']} |")

    return 0 if all(r["success"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
