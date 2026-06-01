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


def build_examples() -> list[list[dict]]:
    pairs = [
        ("A charming and warm little film.", "positive"),
        ("The movie was dull, slow, and joyless.", "negative"),
        ("Excellent acting and a moving ending.", "positive"),
        ("Bad pacing and flat dialogue ruined it.", "negative"),
        ("A delightful comedy with real heart.", "positive"),
        ("The plot was incoherent and boring.", "negative"),
    ]
    return [
        [
            {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
            {"role": "user", "content": f"Sentence: {sentence} Sentiment?"},
            {"role": "assistant", "content": label},
        ]
        for sentence, label in pairs
    ]


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


def losses_for_param_value(model: Any, tokenized: list[list[int]], param_path: str, value: mx.array) -> np.ndarray:
    set_path(model, param_path, value)
    losses = np.array([per_sample_lm_loss(model, ids) for ids in tokenized], dtype=np.float64)
    mx.eval(value)
    return losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--param-path", default="model.layers.0.self_attn.q_proj.bias")
    parser.add_argument("--mu", type=float, default=5e-2)
    parser.add_argument("--clip", type=float, default=25.0)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_real_param_fd_smoke_results.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM real-parameter finite-difference smoke")
    print("This perturbs a real floating-point Qwen parameter, computes plus/minus losses, restores it, and runs ZPL.")
    print(f"model={args.model}")
    print(f"param_path={args.param_path}")
    print(f"mu={args.mu}")
    print(f"clip={args.clip}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    tokenized = [encode_chat(tokenizer, messages) for messages in build_examples()]

    original = get_path(model, args.param_path)
    original_np = np.array(original.tolist(), dtype=np.float32)
    if original_np.size == 0:
        raise ValueError(f"Parameter is empty: {args.param_path}")
    if not np.issubdtype(original_np.dtype, np.floating):
        raise ValueError(f"Parameter is not floating point: {args.param_path} dtype={original.dtype}")

    rng = np.random.default_rng(20260601)
    direction_np = rng.normal(size=original_np.shape).astype(np.float32)
    direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
    direction = mx.array(direction_np, dtype=original.dtype)
    original_copy = mx.array(original_np, dtype=original.dtype)

    baseline_losses = losses_for_param_value(model, tokenized, args.param_path, original_copy)
    plus_value = original_copy + args.mu * direction
    minus_value = original_copy - args.mu * direction
    plus_losses = losses_for_param_value(model, tokenized, args.param_path, plus_value)
    minus_losses = losses_for_param_value(model, tokenized, args.param_path, minus_value)
    set_path(model, args.param_path, original_copy)
    restored_losses = losses_for_param_value(model, tokenized, args.param_path, original_copy)

    fd = (plus_losses - minus_losses) / (2.0 * args.mu)
    membership = make_balanced_membership(num_examples=len(tokenized), num_subsets=4, seed=2026)
    assert_balanced_membership(membership)
    release = paczero_zpl_release(fd, membership, clip=args.clip, rng=rng)

    restore_abs_max = float(np.max(np.abs(restored_losses - baseline_losses)))
    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "param_is_float": True,
        "all_baseline_losses_finite": bool(np.isfinite(baseline_losses).all()),
        "all_fd_finite": bool(np.isfinite(fd).all()),
        "fd_has_nonzero_signal": bool(np.max(np.abs(fd)) > 0.0),
        "plus_minus_not_identical": bool(np.max(np.abs(plus_losses - minus_losses)) > 0.0),
        "restored_matches_baseline": restore_abs_max < 1e-5,
        "release_sign_valid": int(release.sign) in (-1, 1),
        "membership_balanced": True,
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "parameterization": "real_model_float_parameter",
        "param_path": args.param_path,
        "param_shape": list(original.shape),
        "param_dtype": str(original.dtype),
        "mu": args.mu,
        "clip": args.clip,
        "num_examples": len(tokenized),
        "checks": checks,
        "baseline_loss_mean": float(baseline_losses.mean()),
        "plus_loss_mean": float(plus_losses.mean()),
        "minus_loss_mean": float(minus_losses.mean()),
        "restored_loss_mean": float(restored_losses.mean()),
        "restore_abs_max": restore_abs_max,
        "fd_mean": float(fd.mean()),
        "fd_min": float(fd.min()),
        "fd_max": float(fd.max()),
        "fd_abs_max": float(np.max(np.abs(fd))),
        "zpl_release": {
            "sign": int(release.sign),
            "unanimous": bool(release.unanimous),
            "subset_signs": release.subset_signs.astype(int).tolist(),
            "subset_means": release.subset_means.astype(float).tolist(),
        },
        "membership_shape": list(membership.shape),
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
        "note": "First real-model parameter FD smoke. Next step is a one-step sign update on this parameter, then replacing bias with true LoRA adapter tensors.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("REAL_PARAM_FD_SMOKE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
