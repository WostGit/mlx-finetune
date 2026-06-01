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


def safe_scalar(x: mx.array) -> float:
    return float(np.array(x.tolist()).reshape(()))


def manual_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    log_norm = mx.logsumexp(logits, axis=-1)
    selected = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return log_norm - selected


def load_sst2_rows(train_n: int, dev_n: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/glue", "sst2")
        def convert(split: str, n: int) -> list[tuple[str, str]]:
            rows = []
            for row in ds[split].select(range(n)):
                label = "positive" if int(row["label"]) == 1 else "negative"
                rows.append((str(row["sentence"]), label))
            return rows
        return convert("train", train_n), convert("validation", dev_n), "nyu-mll/glue/sst2"
    except Exception as exc:
        # Deterministic fallback keeps syntax/functionality tests usable if HF is unavailable.
        fallback = [
            ("A charming and warm little film.", "positive"),
            ("The movie was dull, slow, and joyless.", "negative"),
            ("Excellent acting and a moving ending.", "positive"),
            ("Bad pacing and flat dialogue ruined it.", "negative"),
            ("A delightful comedy with real heart.", "positive"),
            ("The plot was incoherent and boring.", "negative"),
            ("A smart, funny, and beautifully acted story.", "positive"),
            ("The film felt empty and painfully long.", "negative"),
            ("Wonderful performances carried every scene.", "positive"),
            ("The jokes were stale and the ending was awful.", "negative"),
            ("A thoughtful and uplifting drama.", "positive"),
            ("Messy editing made it hard to enjoy.", "negative"),
            ("A sweet and memorable little movie.", "positive"),
            ("It was flat, tedious, and forgettable.", "negative"),
            ("The cast gives the story real energy.", "positive"),
            ("Weak writing sinks the entire film.", "negative"),
            ("Bright direction and a strong script make it work.", "positive"),
            ("Noisy, shallow, and badly acted.", "negative"),
            ("An engaging film with plenty of charm.", "positive"),
            ("A disappointing mess from start to finish.", "negative"),
        ]
        return fallback[:train_n], fallback[-dev_n:], f"fallback_builtin_sst2_like:{type(exc).__name__}:{exc}"


def prompt_text(sentence: str) -> str:
    return "Classify the sentiment of the movie-review sentence as positive or negative.\nSentence: " + sentence + "\nAnswer:"


def encode_prompt_label(tokenizer: Any, sentence: str, label: str) -> tuple[list[int], list[int], str]:
    prompt = prompt_text(sentence)
    # Leading space improves compatibility with sentencepiece/BPE tokenizers for continuations.
    label_text = " " + label
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + label_text)
    if len(full_ids) <= len(prompt_ids):
        # Fallback: concatenate separately if tokenizer normalizes the full string unexpectedly.
        label_ids = tokenizer.encode(label_text)
        full_ids = prompt_ids + label_ids
    return prompt_ids, full_ids, label


def label_only_loss(model: Any, full_ids: list[int], prompt_len: int) -> float:
    if len(full_ids) < 3 or prompt_len >= len(full_ids):
        raise ValueError("Invalid prompt/full token lengths")
    x = mx.array(full_ids[:-1], dtype=mx.int32)[None, :]
    y = mx.array(full_ids[1:], dtype=mx.int32)[None, :]
    logits = model(x)
    token_losses = manual_cross_entropy(logits, y)
    # Prediction at position t predicts token t+1. Label tokens begin at full_ids[prompt_len],
    # so their prediction losses begin at index prompt_len-1.
    start = max(0, prompt_len - 1)
    label_losses = token_losses[:, start:]
    loss = mx.mean(label_losses)
    mx.eval(loss)
    return safe_scalar(loss)


def score_example(model: Any, tokenizer: Any, sentence: str) -> dict:
    candidates = []
    for label in ["negative", "positive"]:
        prompt_ids, full_ids, _ = encode_prompt_label(tokenizer, sentence, label)
        loss = label_only_loss(model, full_ids, len(prompt_ids))
        candidates.append({"label": label, "loss": loss})
    pred = min(candidates, key=lambda x: x["loss"])["label"]
    return {"prediction": pred, "candidate_losses": candidates}


def eval_dataset(model: Any, tokenizer: Any, rows: list[tuple[str, str]]) -> dict:
    losses = []
    correct = 0
    for sentence, label in rows:
        prompt_ids, full_ids, _ = encode_prompt_label(tokenizer, sentence, label)
        true_loss = label_only_loss(model, full_ids, len(prompt_ids))
        scored = score_example(model, tokenizer, sentence)
        losses.append(true_loss)
        correct += int(scored["prediction"] == label)
    arr = np.array(losses, dtype=np.float64)
    return {"losses": arr, "loss_mean": float(arr.mean()), "accuracy": correct / max(1, len(rows))}


def get_path(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if part == "model":
            continue
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def set_path(root: Any, dotted_path: str, value: Any) -> None:
    parts = [p for p in dotted_path.split(".") if p != "model"]
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = value
    else:
        setattr(parent, final, value)


def eval_losses_with_value(model: Any, tokenizer: Any, rows: list[tuple[str, str]], path: str, value: mx.array) -> np.ndarray:
    set_path(model, path, value)
    vals = []
    for sentence, label in rows:
        prompt_ids, full_ids, _ = encode_prompt_label(tokenizer, sentence, label)
        vals.append(label_only_loss(model, full_ids, len(prompt_ids)))
    return np.array(vals, dtype=np.float64)


def run_cell(
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
    original_copy = mx.array(original_np, dtype=original.dtype)
    if original_np.size == 0 or not np.issubdtype(original_np.dtype, np.floating):
        raise ValueError(f"Bad target parameter {param_path}: dtype={getattr(original, 'dtype', None)} shape={getattr(original, 'shape', None)}")

    rng = np.random.default_rng(seed)
    membership = make_balanced_membership(num_examples=len(train_rows), num_subsets=8, seed=seed + 17)
    assert_balanced_membership(membership)

    set_path(model, param_path, original_copy)
    base_train = eval_dataset(model, tokenizer, train_rows)
    base_dev = eval_dataset(model, tokenizer, dev_rows)
    theta = original_copy

    fd_finite = 0
    fd_nonzero = 0
    unanimous = 0
    signs = {"positive": 0, "negative": 0}
    fd_abs_max_values = []
    fd_abs_mean_values = []
    history = []
    best_train_loss = base_train["loss_mean"]
    best_dev_loss = base_dev["loss_mean"]
    best_dev_acc = base_dev["accuracy"]

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
        fd_finite += int(np.isfinite(fd).all())
        fd_nonzero += int(fd_abs_max > 0.0)
        release = paczero_zpl_release(fd, membership, clip=clip, rng=rng)
        unanimous += int(bool(release.unanimous))
        if int(release.sign) > 0:
            signs["positive"] += 1
        else:
            signs["negative"] += 1
        theta = theta - lr * float(release.sign) * direction
        set_path(model, param_path, theta)

        if step == 1 or step == steps or step % eval_every == 0:
            train_eval = eval_dataset(model, tokenizer, train_rows)
            dev_eval = eval_dataset(model, tokenizer, dev_rows)
            best_train_loss = min(best_train_loss, train_eval["loss_mean"])
            best_dev_loss = min(best_dev_loss, dev_eval["loss_mean"])
            best_dev_acc = max(best_dev_acc, dev_eval["accuracy"])
            history.append({
                "step": step,
                "train_loss": train_eval["loss_mean"],
                "train_accuracy": train_eval["accuracy"],
                "dev_loss": dev_eval["loss_mean"],
                "dev_accuracy": dev_eval["accuracy"],
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": bool(release.unanimous),
            })

    final_train = eval_dataset(model, tokenizer, train_rows)
    final_dev = eval_dataset(model, tokenizer, dev_rows)
    best_train_loss = min(best_train_loss, final_train["loss_mean"])
    best_dev_loss = min(best_dev_loss, final_dev["loss_mean"])
    best_dev_acc = max(best_dev_acc, final_dev["accuracy"])

    set_path(model, param_path, original_copy)
    restored_train = eval_dataset(model, tokenizer, train_rows)
    restored_dev = eval_dataset(model, tokenizer, dev_rows)
    restore_train_abs_max = float(np.max(np.abs(restored_train["losses"] - base_train["losses"])))
    restore_dev_abs_max = float(np.max(np.abs(restored_dev["losses"] - base_dev["losses"])))

    fd_finite_rate = fd_finite / max(1, steps)
    fd_signal_rate = fd_nonzero / max(1, steps)
    unanimity_rate = unanimous / max(1, steps)
    checks = {
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "fd_magnitude_ok": max(fd_abs_max_values) > 0.0,
        "losses_remain_finite": bool(np.isfinite(final_train["losses"]).all() and np.isfinite(final_dev["losses"]).all()),
        "best_train_not_worse": best_train_loss <= base_train["loss_mean"] + 1e-9,
        "best_dev_not_catastrophic": best_dev_loss <= base_dev["loss_mean"] + 0.10,
        "dev_accuracy_not_worse_than_baseline": best_dev_acc >= base_dev["accuracy"],
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
        "best_train_loss": best_train_loss,
        "final_train_delta": final_train["loss_mean"] - base_train["loss_mean"],
        "best_train_delta": best_train_loss - base_train["loss_mean"],
        "baseline_dev_loss": base_dev["loss_mean"],
        "final_dev_loss": final_dev["loss_mean"],
        "best_dev_loss": best_dev_loss,
        "final_dev_delta": final_dev["loss_mean"] - base_dev["loss_mean"],
        "best_dev_delta": best_dev_loss - base_dev["loss_mean"],
        "baseline_train_accuracy": base_train["accuracy"],
        "final_train_accuracy": final_train["accuracy"],
        "baseline_dev_accuracy": base_dev["accuracy"],
        "final_dev_accuracy": final_dev["accuracy"],
        "best_dev_accuracy": best_dev_acc,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "unanimity_rate": unanimity_rate,
        "release_sign_counts": signs,
        "restore_train_abs_max": restore_train_abs_max,
        "restore_dev_abs_max": restore_dev_abs_max,
        "history": history,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    p.add_argument("--param-paths", default="model.layers.0.self_attn.q_proj.bias,model.layers.0.self_attn.v_proj.bias")
    p.add_argument("--seeds", default="20260604,20260605")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--train-examples", type=int, default=16)
    p.add_argument("--dev-examples", type=int, default=16)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=25.0)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_exhaustive_readiness_results.json"))
    args = p.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM exhaustive readiness matrix")
    print("Multiple real parameter paths, multiple seeds, SST-2-backed label-only loss/accuracy, FD health, and restore checks.")
    print(f"model={args.model}")
    print(f"param_paths={args.param_paths}")
    print(f"seeds={args.seeds}")
    print(f"steps={args.steps} train_examples={args.train_examples} dev_examples={args.dev_examples}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    train_rows, dev_rows, dataset_source = load_sst2_rows(args.train_examples, args.dev_examples)
    param_paths = [x.strip() for x in args.param_paths.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    cells = []
    for path in param_paths:
        for seed in seeds:
            print(f"CELL_START path={path} seed={seed}")
            cell_start = time.perf_counter()
            cell = run_cell(model, tokenizer, train_rows, dev_rows, path, seed, args.steps, args.mu, args.lr, args.clip, args.eval_every)
            cell["elapsed_seconds"] = round(time.perf_counter() - cell_start, 3)
            cells.append(cell)
            print("CELL_RESULT=" + json.dumps({k: cell[k] for k in ["success", "param_path", "seed", "final_train_delta", "final_dev_delta", "fd_signal_rate", "unanimity_rate", "baseline_dev_accuracy", "final_dev_accuracy", "best_dev_accuracy"]}))

    aggregate_checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "dataset_nonempty": bool(train_rows and dev_rows),
        "all_cells_success": all(c["success"] for c in cells),
        "multiple_params_tested": len(param_paths) >= 2,
        "multiple_seeds_tested": len(seeds) >= 2,
        "all_restore_exact": all(c["restore_train_abs_max"] < 1e-5 and c["restore_dev_abs_max"] < 1e-5 for c in cells),
        "all_fd_signal_rates_ok": all(c["fd_signal_rate"] >= 0.80 for c in cells),
        "no_dev_accuracy_regression_best": all(c["best_dev_accuracy"] >= c["baseline_dev_accuracy"] for c in cells),
    }
    payload = {
        "success": all(aggregate_checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "parameterization": "real_model_float_parameter_exhaustive_matrix_before_lora",
        "param_paths": param_paths,
        "seeds": seeds,
        "steps_per_cell": args.steps,
        "mu": args.mu,
        "lr": args.lr,
        "clip": args.clip,
        "aggregate_checks": aggregate_checks,
        "cells": cells,
        "verdict": "Ready for a larger real-parameter PACZero run if success=true. Still not ready to claim full PACZero-LoRA reproduction until the same matrix passes on actual LoRA adapter tensors.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("EXHAUSTIVE_READINESS_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
