#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> tuple[int, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout, time.perf_counter() - start


def inspect_files(root: Path) -> list[dict]:
    files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append({"path": str(path.relative_to(root.parent)), "bytes": path.stat().st_size})
    return files


LOSS_RE = re.compile(r"(?:Train|Val|Validation|Test)?\s*(?:loss|Loss)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)")
TOK_RE = re.compile(r"Tokens/sec\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
TRAINED_TOKENS_RE = re.compile(r"Trained Tokens\s+(\d+)", re.IGNORECASE)
ITER_RE = re.compile(r"Iter\s+(\d+)", re.IGNORECASE)


def parse_train_log(text: str) -> dict:
    losses = [float(m.group(1)) for m in LOSS_RE.finditer(text)]
    toks = [float(m.group(1)) for m in TOK_RE.finditer(text)]
    trained = [int(m.group(1)) for m in TRAINED_TOKENS_RE.finditer(text)]
    iters = [int(m.group(1)) for m in ITER_RE.finditer(text)]
    return {
        "losses": losses,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "tokens_per_second_last": toks[-1] if toks else None,
        "trained_tokens_last": trained[-1] if trained else None,
        "max_iteration": max(iters) if iters else None,
        "tail_40_lines": text.splitlines()[-40:],
    }


def build_tiny_sentiment_rows() -> list[dict]:
    examples = [
        ("A charming and warm little film.", "positive"),
        ("The movie was dull, slow, and joyless.", "negative"),
        ("Excellent acting and a surprisingly moving ending.", "positive"),
        ("Bad pacing and flat dialogue ruined it.", "negative"),
        ("A delightful comedy with real heart.", "positive"),
        ("The plot was incoherent and boring.", "negative"),
    ]
    rows = []
    for sentence, label in examples:
        rows.append({
            "messages": [
                {"role": "system", "content": "Classify movie-review sentiment. Answer only positive or negative."},
                {"role": "user", "content": f"Sentence: {sentence}\nSentiment?"},
                {"role": "assistant", "content": label},
            ]
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, default=Path("benchmark-data/paczero-mlxlm-smoke"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results/paczero-mlxlm-smoke"))
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_smoke_results.json"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.out_dir / "adapters"

    rows = build_tiny_sentiment_rows()
    write_jsonl(args.data_dir / "train.jsonl", rows)
    write_jsonl(args.data_dir / "valid.jsonl", rows[:2])
    write_jsonl(args.data_dir / "test.jsonl", rows[:2])

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", args.model,
        "--train",
        "--data", str(args.data_dir),
        "--adapter-path", str(adapter_dir),
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--val-batches", "1",
        "--max-seq-length", "256",
        "--steps-per-report", "1",
        "--steps-per-eval", str(max(1, args.iters)),
        "--save-every", str(max(1, args.iters)),
    ]

    print("# PACZero MLX-LM model-level smoke scaffold")
    print("This smoke validates real MLX-LM model/tokenizer/LoRA adapter plumbing.")
    print("It does not yet replace MLX-LM backprop with custom PACZero-ZPL updates.")
    print(f"model={args.model}")
    print(f"iters={args.iters}")
    print(f"num_layers={args.num_layers}")
    print(f"train_examples={len(rows)}")
    print(f"train_jsonl_sha256={sha256_file(args.data_dir / 'train.jsonl')}")
    try:
        print(f"package_mlx={md.version('mlx')}")
        print(f"package_mlx-lm={md.version('mlx-lm')}")
    except Exception as exc:
        print(f"package_version_error={exc}")
    print("command=" + " ".join(shlex.quote(x) for x in cmd))

    rc, output, elapsed = run_cmd(cmd)
    print(output)
    train_log = args.out_dir / "train.log"
    train_log.write_text(output, encoding="utf-8")
    parsed = parse_train_log(output)
    adapter_files = inspect_files(adapter_dir)
    adapter_names = {Path(f["path"]).name for f in adapter_files}
    checks = {
        "train_returncode_zero": rc == 0,
        "adapter_config_exists": "adapter_config.json" in adapter_names,
        "adapters_safetensors_exists": "adapters.safetensors" in adapter_names,
        "adapters_safetensors_nonempty": any(Path(f["path"]).name == "adapters.safetensors" and f["bytes"] > 0 for f in adapter_files),
        "reported_iteration_reached": parsed["max_iteration"] == args.iters,
    }
    payload = {
        "success": all(checks.values()),
        "note": "Real MLX-LM LoRA smoke scaffold. PACZero core/MLX/LoRA-style logic is tested separately; this validates real model adapter plumbing before custom ZO integration.",
        "model": args.model,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "elapsed_seconds": round(elapsed, 3),
        "checks": checks,
        "returncode": rc,
        "parsed_train_log": parsed,
        "adapter_files": adapter_files,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("MODEL_SMOKE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
