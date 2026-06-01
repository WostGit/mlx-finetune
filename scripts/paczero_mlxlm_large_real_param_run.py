#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm import load

from paczero_core import assert_balanced_membership, make_balanced_membership, paczero_zpl_release
from paczero_mlxlm_exhaustive_readiness import (
    encode_prompt_label,
    eval_dataset,
    eval_losses_with_value,
    get_path,
    load_sst2_rows,
    set_path,
)


def run_large_single_cell(
    model: Any,
    tokenizer: Any,
    train_rows: list[tuple[str, str]],
    dev_rows: list[tuple[str, str]],
    param_path: str,
    seed: int,
    steps: int,
    mu: float,
    lr: float,
    clip: float,
    eval_every: int,
) -> dict:
    original = get_path(model, param_path)
    original_np = np.array(original.tolist(), dtype=np.float32)
    if original_np.size == 0 or not np.issubdtype(original_np.dtype, np.floating):
        raise ValueError(f"Bad target parameter {param_path}: dtype={getattr(original, 'dtype', None)} shape={getattr(original, 'shape', None)}")

    original_copy = mx.array(original_np, dtype=original.dtype)
    theta = original_copy
    rng = np.random.default_rng(seed)
    membership = make_balanced_membership(num_examples=len(train_rows), num_subsets=8, seed=seed + 17)
    assert_balanced_membership(membership)

    set_path(model, param_path, original_copy)
    base_train = eval_dataset(model, tokenizer, train_rows)
    base_dev = eval_dataset(model, tokenizer, dev_rows)

    fd_finite_count = 0
    fd_nonzero_count = 0
    unanimous_count = 0
    sign_counts = {"positive": 0, "negative": 0}
    fd_abs_max_values = []
    fd_abs_mean_values = []
    history = []

    best = {
        "step": 0,
        "train_loss": base_train["loss_mean"],
        "train_accuracy": base_train["accuracy"],
        "dev_loss": base_dev["loss_mean"],
        "dev_accuracy": base_dev["accuracy"],
    }

    for step in range(1, steps + 1):
        direction_np = rng.normal(size=original_np.shape).astype(np.float32)
        direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
        direction = mx.array(direction_np, dtype=original.dtype)

        plus = eval_losses_with_value(model, tokenizer, train_rows, param_path, theta + mu * direction)
        minus = eval_losses_with_value(model, tokenizer, train_rows, param_path, theta - mu * direction)
        fd = (plus - minus) / (2.0 * mu)
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        fd_finite_count += int(np.isfinite(fd).all())
        fd_nonzero_count += int(fd_abs_max > 0.0)

        release = paczero_zpl_release(fd, membership, clip=clip, rng=rng)
        unanimous_count += int(bool(release.unanimous))
        if int(release.sign) > 0:
            sign_counts["positive"] += 1
        else:
            sign_counts["negative"] += 1

        theta = theta - lr * float(release.sign) * direction
        set_path(model, param_path, theta)

        if step == 1 or step == steps or step % eval_every == 0:
            train_eval = eval_dataset(model, tokenizer, train_rows)
            dev_eval = eval_dataset(model, tokenizer, dev_rows)
            row = {
                "step": step,
                "train_loss": train_eval["loss_mean"],
                "train_accuracy": train_eval["accuracy"],
                "dev_loss": dev_eval["loss_mean"],
                "dev_accuracy": dev_eval["accuracy"],
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": bool(release.unanimous),
            }
            history.append(row)
            print("LARGE_RUN_EVAL=" + json.dumps(row))
            # Selection policy: maximize dev accuracy; break ties by lower dev loss.
            if (dev_eval["accuracy"] > best["dev_accuracy"]) or (
                dev_eval["accuracy"] == best["dev_accuracy"] and dev_eval["loss_mean"] < best["dev_loss"]
            ):
                best = {
                    "step": step,
                    "train_loss": train_eval["loss_mean"],
                    "train_accuracy": train_eval["accuracy"],
                    "dev_loss": dev_eval["loss_mean"],
                    "dev_accuracy": dev_eval["accuracy"],
                }

    final_train = eval_dataset(model, tokenizer, train_rows)
    final_dev = eval_dataset(model, tokenizer, dev_rows)
    if (final_dev["accuracy"] > best["dev_accuracy"]) or (
        final_dev["accuracy"] == best["dev_accuracy"] and final_dev["loss_mean"] < best["dev_loss"]
    ):
        best = {
            "step": steps,
            "train_loss": final_train["loss_mean"],
            "train_accuracy": final_train["accuracy"],
            "dev_loss": final_dev["loss_mean"],
            "dev_accuracy": final_dev["accuracy"],
        }

    set_path(model, param_path, original_copy)
    restored_train = eval_dataset(model, tokenizer, train_rows)
    restored_dev = eval_dataset(model, tokenizer, dev_rows)
    restore_train_abs_max = float(np.max(np.abs(restored_train["losses"] - base_train["losses"])))
    restore_dev_abs_max = float(np.max(np.abs(restored_dev["losses"] - base_dev["losses"])))

    fd_finite_rate = fd_finite_count / max(1, steps)
    fd_signal_rate = fd_nonzero_count / max(1, steps)
    unanimity_rate = unanimous_count / max(1, steps)

    checks = {
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "fd_magnitude_ok": max(fd_abs_max_values) > 0.0,
        "losses_remain_finite": bool(np.isfinite(final_train["losses"]).all() and np.isfinite(final_dev["losses"]).all()),
        "best_dev_accuracy_not_worse": best["dev_accuracy"] >= base_dev["accuracy"],
        "best_dev_loss_not_worse_when_accuracy_ties": bool(
            best["dev_accuracy"] > base_dev["accuracy"] or best["dev_loss"] <= base_dev["loss_mean"]
        ),
        "restore_train_exact": restore_train_abs_max < 1e-5,
        "restore_dev_exact": restore_dev_abs_max < 1e-5,
    }

    return {
        "success": all(checks.values()),
        "param_path": param_path,
        "param_shape": list(original.shape),
        "param_dtype": str(original.dtype),
        "seed": seed,
        "steps": steps,
        "mu": mu,
        "lr": lr,
        "clip": clip,
        "checks": checks,
        "baseline_train_loss": base_train["loss_mean"],
        "final_train_loss": final_train["loss_mean"],
        "baseline_dev_loss": base_dev["loss_mean"],
        "final_dev_loss": final_dev["loss_mean"],
        "baseline_train_accuracy": base_train["accuracy"],
        "final_train_accuracy": final_train["accuracy"],
        "baseline_dev_accuracy": base_dev["accuracy"],
        "final_dev_accuracy": final_dev["accuracy"],
        "best_checkpoint": best,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "unanimity_rate": unanimity_rate,
        "release_sign_counts": sign_counts,
        "restore_train_abs_max": restore_train_abs_max,
        "restore_dev_abs_max": restore_dev_abs_max,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--param-path", default="model.layers.0.self_attn.q_proj.bias")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--dev-examples", type=int, default=64)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=25.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_large_real_param_results.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM large real-parameter run")
    print("Single validated real-parameter PACZero-ZPL run with single-cell success criteria.")
    print(f"model={args.model}")
    print(f"param_path={args.param_path}")
    print(f"seed={args.seed} steps={args.steps} train_examples={args.train_examples} dev_examples={args.dev_examples}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    train_rows, dev_rows, dataset_source = load_sst2_rows(args.train_examples, args.dev_examples)
    cell = run_large_single_cell(
        model=model,
        tokenizer=tokenizer,
        train_rows=train_rows,
        dev_rows=dev_rows,
        param_path=args.param_path,
        seed=args.seed,
        steps=args.steps,
        mu=args.mu,
        lr=args.lr,
        clip=args.clip,
        eval_every=args.eval_every,
    )

    run_checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "dataset_nonempty": bool(train_rows and dev_rows),
        "single_cell_success": bool(cell["success"]),
        "restore_exact": cell["restore_train_abs_max"] < 1e-5 and cell["restore_dev_abs_max"] < 1e-5,
        "fd_signal_rate_ok": cell["fd_signal_rate"] >= 0.80,
    }
    payload = {
        "success": all(run_checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "parameterization": "large_real_model_float_parameter_single_cell",
        "run_checks": run_checks,
        "cell": cell,
        "verdict": "Valid larger real-parameter PACZero-ZPL run if success=true. This is not a full PACZero-LoRA reproduction; adapter tensor support remains the next milestone.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("LARGE_REAL_PARAM_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
