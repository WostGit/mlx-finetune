#!/usr/bin/env python3
"""Benchmark tiny MLX-LM LoRA fine-tuning runs.

This script is designed for GitHub Actions macOS Apple Silicon runners. It creates
an intentionally tiny instruction dataset, fine-tunes each requested model for a
small number of LoRA iterations, records timing/output metadata, and writes JSON
and Markdown summaries into an output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency in local use
    psutil = None

DEFAULT_MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
]

TRAIN_ROWS = [
    {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is PACzero?"},
            {"role": "assistant", "content": "PACzero is a benchmark target used here to test compact model fine-tuning feasibility."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Summarize why MLX is useful on Apple Silicon."},
            {"role": "assistant", "content": "MLX provides Apple Silicon-optimized array operations and model tooling for efficient local training and inference."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Give one advantage of a 0.5B model in CI."},
            {"role": "assistant", "content": "A 0.5B model is small enough to quickly validate downloads, tokenization, LoRA training, and adapter export in constrained CI."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Give one advantage of a 1B model benchmark."},
            {"role": "assistant", "content": "A 1B benchmark is still lightweight while being more representative than an ultra-tiny smoke test."},
        ]
    },
]

VALID_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "What should this benchmark measure?"},
            {"role": "assistant", "content": "It should measure fine-tuning feasibility, elapsed time, throughput proxy, adapter creation, and basic loss reporting."},
        ]
    }
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_dataset(data_dir: Path) -> None:
    write_jsonl(data_dir / "train.jsonl", TRAIN_ROWS)
    write_jsonl(data_dir / "valid.jsonl", VALID_ROWS)
    write_jsonl(data_dir / "test.jsonl", VALID_ROWS)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    elapsed = time.perf_counter() - started
    return proc.returncode, proc.stdout, elapsed


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def memory_snapshot() -> dict[str, Any]:
    if psutil is None:
        return {"psutil_available": False}
    vm = psutil.virtual_memory()
    return {
        "psutil_available": True,
        "total_gb": round(vm.total / 1e9, 3),
        "available_gb": round(vm.available / 1e9, 3),
        "used_gb": round(vm.used / 1e9, 3),
        "percent": vm.percent,
    }


def benchmark_model(model: str, data_dir: Path, out_dir: Path, iters: int, batch_size: int, lora_layers: int) -> dict[str, Any]:
    model_out = out_dir / safe_name(model)
    adapter_dir = model_out / "adapters"
    model_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        model,
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--lora-layers",
        str(lora_layers),
        "--steps-per-report",
        "1",
        "--steps-per-eval",
        str(max(1, iters)),
        "--save-every",
        str(max(1, iters)),
    ]

    before_mem = memory_snapshot()
    returncode, output, elapsed_s = run_cmd(cmd)
    after_mem = memory_snapshot()

    log_path = model_out / "train.log"
    log_path.write_text(output, encoding="utf-8")

    adapter_files = []
    if adapter_dir.exists():
        adapter_files = [str(p.relative_to(model_out)) for p in adapter_dir.rglob("*") if p.is_file()]

    result = {
        "model": model,
        "returncode": returncode,
        "success": returncode == 0,
        "elapsed_s": round(elapsed_s, 3),
        "iters": iters,
        "batch_size": batch_size,
        "lora_layers": lora_layers,
        "command": " ".join(shlex.quote(part) for part in cmd),
        "log_path": str(log_path),
        "adapter_file_count": len(adapter_files),
        "adapter_files": adapter_files,
        "memory_before": before_mem,
        "memory_after": after_mem,
    }
    (model_out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    payload = {
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "results": results,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# MLX LoRA fine-tuning benchmark",
        "",
        f"Platform: `{platform.platform()}` / `{platform.machine()}`",
        "",
        "| Model | Success | Iterations | Batch | LoRA layers | Elapsed seconds | Adapter files |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['model']}` | {r['success']} | {r['iters']} | {r['batch_size']} | "
            f"{r['lora_layers']} | {r['elapsed_s']} | {r['adapter_file_count']} |"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "This is a constrained CI benchmark, not a quality-maximizing fine-tune. It is intended to validate MLX-LM LoRA training feasibility and compare rough elapsed runtime between compact 4-bit models.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MLX-LM LoRA fine-tuning on compact 4-bit models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="HF/MLX model IDs to benchmark.")
    parser.add_argument("--iters", type=int, default=int(os.getenv("BENCH_ITERS", "20")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BENCH_BATCH_SIZE", "1")))
    parser.add_argument("--lora-layers", type=int, default=int(os.getenv("BENCH_LORA_LAYERS", "4")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("BENCH_OUT_DIR", "benchmark-results")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("BENCH_DATA_DIR", "benchmark-data")))
    parser.add_argument("--keep-going", action="store_true", help="Run all models even if one fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_dataset(args.data_dir)

    results = []
    exit_code = 0
    for model in args.models:
        print(f"::group::Benchmark {model}", flush=True)
        result = benchmark_model(model, args.data_dir, args.out_dir, args.iters, args.batch_size, args.lora_layers)
        print(json.dumps(result, indent=2), flush=True)
        print("::endgroup::", flush=True)
        results.append(result)
        if not result["success"]:
            exit_code = result["returncode"] or 1
            if not args.keep_going:
                break

    write_summary(results, args.out_dir)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
