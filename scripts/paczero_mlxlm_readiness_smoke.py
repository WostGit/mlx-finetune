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


def encode_chat(tokenizer, messages: list[dict]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tokenizer.encode(rendered)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return tokenizer.encode(text)


def manual_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    logits = logits.astype(mx.float32)
    log_norm = mx.logsumexp(logits, axis=-1)
    selected = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
    return log_norm - selected


def per_sample_lm_loss(model: Any, token_ids: list[int]) -> float:
    x = mx.array(token_ids[:-1], dtype=mx.int32)[None, :]
    y = mx.array(token_ids[1:], dtype=mx.int32)[None, :]
    logits = model(x)
    loss_tokens = manual_cross_entropy(logits, y)
    loss = mx.mean(loss_tokens)
    mx.eval(loss)
    return safe_scalar(loss)


def losses(model: Any, tokenized: list[list[int]]) -> np.ndarray:
    return np.array([per_sample_lm_loss(model, ids) for ids in tokenized], dtype=np.float64)


def build_dataset() -> tuple[list[list[dict]], list[list[dict]]]:
    train_pairs = [
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
    ]
    dev_pairs = [
        ("A sweet and memorable little movie.", "positive"),
        ("It was flat, tedious, and forgettable.", "negative"),
        ("The cast gives the story real energy.", "positive"),
        ("Weak writing sinks the entire film.", "negative"),
    ]

    def rows(pairs: list[tuple[str, str]]) -> list[list[dict]]:
        return [
            [
                {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
                {"role": "user", "content": f"Sentence: {sentence} Sentiment?"},
                {"role": "assistant", "content": label},
            ]
            for sentence, label in pairs
        ]

    return rows(train_pairs), rows(dev_pairs)


def get_path(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if part == "model":
            continue
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def set_path(root: Any, dotted_path: str, value: Any) -> None:
    parts = [p for p in dotted_path.split(".") if p != "model"]
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    final = parts[-1]
    if final.isdigit():
        parent[int(final)] = value
    else:
        setattr(parent, final, value)


def eval_with_param(model: Any, tokenized: list[list[int]], param_path: str, value: mx.array) -> np.ndarray:
    set_path(model, param_path, value)
    return losses(model, tokenized)


def fd_step(
    model: Any,
    train_tokenized: list[list[int]],
    param_path: str,
    theta: mx.array,
    direction: mx.array,
    mu: float,
    clip: float,
    membership: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    plus_losses = eval_with_param(model, train_tokenized, param_path, theta + mu * direction)
    minus_losses = eval_with_param(model, train_tokenized, param_path, theta - mu * direction)
    fd = (plus_losses - minus_losses) / (2.0 * mu)
    release = paczero_zpl_release(fd, membership, clip=clip, rng=rng)
    return fd, release, plus_losses, minus_losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--param-path", default="model.layers.0.self_attn.q_proj.bias")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=25.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_readiness_smoke_results.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM extended readiness smoke")
    print("Multi-step real-parameter ZPL training with train/dev tracking, FD health checks, and restoration.")
    print(f"model={args.model}")
    print(f"param_path={args.param_path}")
    print(f"steps={args.steps} mu={args.mu} lr={args.lr} clip={args.clip}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    train_rows, dev_rows = build_dataset()
    train_tokenized = [encode_chat(tokenizer, messages) for messages in train_rows]
    dev_tokenized = [encode_chat(tokenizer, messages) for messages in dev_rows]

    original = get_path(model, args.param_path)
    original_np = np.array(original.tolist(), dtype=np.float32)
    if original_np.size == 0:
        raise ValueError(f"Parameter is empty: {args.param_path}")
    if not np.issubdtype(original_np.dtype, np.floating):
        raise ValueError(f"Parameter is not floating point: {args.param_path} dtype={original.dtype}")
    original_copy = mx.array(original_np, dtype=original.dtype)
    theta = original_copy

    rng = np.random.default_rng(20260603)
    membership = make_balanced_membership(num_examples=len(train_tokenized), num_subsets=8, seed=2026)
    assert_balanced_membership(membership)

    set_path(model, args.param_path, theta)
    base_train_losses = losses(model, train_tokenized)
    base_dev_losses = losses(model, dev_tokenized)
    base_train = float(base_train_losses.mean())
    base_dev = float(base_dev_losses.mean())

    history = []
    fd_abs_max_values = []
    fd_abs_mean_values = []
    unanimous_count = 0
    nonzero_fd_count = 0
    finite_fd_count = 0
    sign_counts = {"positive": 0, "negative": 0}
    best_train = base_train
    best_dev = base_dev

    for step in range(1, args.steps + 1):
        direction_np = rng.normal(size=original_np.shape).astype(np.float32)
        direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
        direction = mx.array(direction_np, dtype=original.dtype)
        fd, release, plus_losses, minus_losses = fd_step(
            model, train_tokenized, args.param_path, theta, direction, args.mu, args.clip, membership, rng
        )
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        finite_fd = bool(np.isfinite(fd).all())
        nonzero_fd = fd_abs_max > 0.0
        finite_fd_count += int(finite_fd)
        nonzero_fd_count += int(nonzero_fd)
        unanimous_count += int(bool(release.unanimous))
        if int(release.sign) > 0:
            sign_counts["positive"] += 1
        else:
            sign_counts["negative"] += 1

        theta = theta - args.lr * float(release.sign) * direction
        set_path(model, args.param_path, theta)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            train_loss = float(losses(model, train_tokenized).mean())
            dev_loss = float(losses(model, dev_tokenized).mean())
            best_train = min(best_train, train_loss)
            best_dev = min(best_dev, dev_loss)
            row = {
                "step": step,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": bool(release.unanimous),
                "subset_signs": release.subset_signs.astype(int).tolist(),
            }
            history.append(row)
            print(json.dumps(row))

    final_train_losses = losses(model, train_tokenized)
    final_dev_losses = losses(model, dev_tokenized)
    final_train = float(final_train_losses.mean())
    final_dev = float(final_dev_losses.mean())
    best_train = min(best_train, final_train)
    best_dev = min(best_dev, final_dev)

    # Restore original parameter and verify exact/non-destructive behavior.
    set_path(model, args.param_path, original_copy)
    restored_train_losses = losses(model, train_tokenized)
    restored_dev_losses = losses(model, dev_tokenized)
    restore_train_abs_max = float(np.max(np.abs(restored_train_losses - base_train_losses)))
    restore_dev_abs_max = float(np.max(np.abs(restored_dev_losses - base_dev_losses)))

    fd_signal_rate = nonzero_fd_count / max(1, args.steps)
    fd_finite_rate = finite_fd_count / max(1, args.steps)
    unanimity_rate = unanimous_count / max(1, args.steps)
    final_train_delta = final_train - base_train
    best_train_delta = best_train - base_train
    final_dev_delta = final_dev - base_dev
    best_dev_delta = best_dev - base_dev

    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "param_is_float": True,
        "baseline_losses_finite": bool(np.isfinite(base_train_losses).all() and np.isfinite(base_dev_losses).all()),
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "fd_magnitude_ok": bool(max(fd_abs_max_values) > 0.0),
        "losses_remain_finite": bool(np.isfinite(final_train_losses).all() and np.isfinite(final_dev_losses).all()),
        "loss_changed": bool(abs(final_train_delta) > 0.0),
        "best_train_not_worse": bool(best_train_delta <= 0.0),
        "restored_train_matches_baseline": restore_train_abs_max < 1e-5,
        "restored_dev_matches_baseline": restore_dev_abs_max < 1e-5,
        "membership_balanced": True,
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "parameterization": "real_model_float_parameter_multistep_zpl_readiness",
        "param_path": args.param_path,
        "param_shape": list(original.shape),
        "param_dtype": str(original.dtype),
        "steps": args.steps,
        "mu": args.mu,
        "lr": args.lr,
        "clip": args.clip,
        "train_examples": len(train_tokenized),
        "dev_examples": len(dev_tokenized),
        "checks": checks,
        "baseline_train_loss": base_train,
        "final_train_loss": final_train,
        "best_train_loss": best_train,
        "final_train_delta": final_train_delta,
        "best_train_delta": best_train_delta,
        "baseline_dev_loss": base_dev,
        "final_dev_loss": final_dev,
        "best_dev_loss": best_dev,
        "final_dev_delta": final_dev_delta,
        "best_dev_delta": best_dev_delta,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "unanimity_rate": unanimity_rate,
        "release_sign_counts": sign_counts,
        "restore_train_abs_max": restore_train_abs_max,
        "restore_dev_abs_max": restore_dev_abs_max,
        "membership_shape": list(membership.shape),
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
        "history": history,
        "verdict": "Ready for a small multi-step PACZero run on this real-parameter path if success=true. Still not a full LoRA PACZero reproduction until adapter tensors replace q_proj.bias.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("READINESS_SMOKE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
