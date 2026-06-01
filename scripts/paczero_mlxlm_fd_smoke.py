#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

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


def manual_cross_entropy_with_vocab_bias(
    logits: mx.array,
    targets: mx.array,
    bias_token_ids: list[int],
    theta: np.ndarray,
) -> mx.array:
    """Cross entropy after adding a trainable bias to selected vocab logits.

    The logits from a 4-bit model may be low precision. Cast to float32 before
    applying the smoke-test bias so small two-point perturbations are not rounded
    away before logsumexp.
    """

    if len(bias_token_ids) != int(theta.shape[0]):
        raise ValueError("bias_token_ids and theta length mismatch")
    logits_f32 = logits.astype(mx.float32)
    vocab_size = int(logits_f32.shape[-1])
    bias = np.zeros(vocab_size, dtype=np.float32)
    for idx, token_id in enumerate(bias_token_ids):
        if 0 <= int(token_id) < vocab_size:
            bias[int(token_id)] = float(theta[idx])
    biased_logits = logits_f32 + mx.array(bias, dtype=mx.float32)
    log_norm = mx.logsumexp(biased_logits, axis=-1)
    selected = mx.take_along_axis(biased_logits, targets[..., None], axis=-1).squeeze(-1)
    return log_norm - selected


def per_sample_loss_from_cached_logits(
    logits: mx.array,
    target_ids: list[int],
    bias_token_ids: list[int],
    theta: np.ndarray,
) -> float:
    targets = mx.array(target_ids, dtype=mx.int32)[None, :]
    loss_tokens = manual_cross_entropy_with_vocab_bias(logits, targets, bias_token_ids, theta)
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
    rows = []
    for sentence, label in pairs:
        rows.append([
            {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
            {"role": "user", "content": f"Sentence: {sentence} Sentiment?"},
            {"role": "assistant", "content": label},
        ])
    return rows


def cached_logits_and_targets(model, token_ids: list[int]) -> tuple[mx.array, list[int]]:
    if len(token_ids) < 3:
        raise ValueError("Need at least three tokens")
    input_ids = mx.array(token_ids[:-1], dtype=mx.int32)[None, :]
    logits = model(input_ids)
    mx.eval(logits)
    return logits, token_ids[1:]


def finite_difference_per_sample(
    cached: list[tuple[mx.array, list[int]]],
    bias_token_ids: list[int],
    theta: np.ndarray,
    direction: np.ndarray,
    mu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus_theta = theta + mu * direction
    minus_theta = theta - mu * direction
    plus_losses = []
    minus_losses = []
    for logits, targets in cached:
        plus_losses.append(per_sample_loss_from_cached_logits(logits, targets, bias_token_ids, plus_theta))
        minus_losses.append(per_sample_loss_from_cached_logits(logits, targets, bias_token_ids, minus_theta))
    plus = np.array(plus_losses, dtype=np.float64)
    minus = np.array(minus_losses, dtype=np.float64)
    fd = (plus - minus) / (2.0 * mu)
    return fd, plus, minus


def find_fd_signal(
    cached: list[tuple[mx.array, list[int]]],
    bias_token_ids: list[int],
    theta: np.ndarray,
    direction: np.ndarray,
    requested_mu: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Try increasingly large mu values until quantized losses separate."""

    candidates = []
    for mu in [requested_mu, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1.0]:
        if mu > 0 and mu not in candidates:
            candidates.append(mu)
    attempts = []
    best = None
    for mu in candidates:
        fd, plus, minus = finite_difference_per_sample(cached, bias_token_ids, theta, direction, mu)
        diff_abs_max = float(np.max(np.abs(plus - minus)))
        fd_abs_max = float(np.max(np.abs(fd)))
        attempt = {
            "mu": float(mu),
            "plus_loss_mean": float(plus.mean()),
            "minus_loss_mean": float(minus.mean()),
            "plus_minus_abs_max": diff_abs_max,
            "fd_abs_max": fd_abs_max,
        }
        attempts.append(attempt)
        if best is None or fd_abs_max > float(best[0]):
            best = (fd_abs_max, mu, fd, plus, minus)
        if np.isfinite(fd).all() and diff_abs_max > 0.0 and fd_abs_max > 0.0:
            return mu, fd, plus, minus, attempts
    assert best is not None
    _, mu, fd, plus, minus = best
    return mu, fd, plus, minus, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--mu", type=float, default=5e-2)
    parser.add_argument("--clip", type=float, default=25.0)
    parser.add_argument("--bias-tokens", type=int, default=32)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_fd_smoke_results.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM two-point finite-difference smoke")
    print("This validates real Qwen forward passes plus two-point per-sample loss differences.")
    print("Perturbed parameter vector: synthetic selected-vocabulary logit bias, not LoRA adapters yet.")
    print("Loss implementation: float32 logits plus manual logsumexp cross entropy.")
    print(f"model={args.model}")
    print(f"requested_mu={args.mu}")
    print(f"clip={args.clip}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    examples = build_examples()
    encoded = [encode_chat(tokenizer, messages) for messages in examples]
    cached = [cached_logits_and_targets(model, token_ids) for token_ids in encoded]

    unique_targets = []
    seen = set()
    for _, targets in cached:
        for token_id in targets:
            token_id = int(token_id)
            if token_id not in seen:
                seen.add(token_id)
                unique_targets.append(token_id)
    pad_id = 0
    while len(unique_targets) < args.bias_tokens:
        if pad_id not in seen:
            unique_targets.append(pad_id)
            seen.add(pad_id)
        pad_id += 1
    bias_token_ids = unique_targets[: args.bias_tokens]

    rng = np.random.default_rng(12345)
    theta = np.zeros(len(bias_token_ids), dtype=np.float32)
    direction = rng.normal(size=theta.shape).astype(np.float32)
    direction = direction / max(np.linalg.norm(direction), 1e-12)

    baseline_losses = np.array([
        per_sample_loss_from_cached_logits(logits, targets, bias_token_ids, theta)
        for logits, targets in cached
    ], dtype=np.float64)
    used_mu, fd, plus, minus, fd_attempts = find_fd_signal(cached, bias_token_ids, theta, direction, args.mu)

    membership = make_balanced_membership(num_examples=len(examples), num_subsets=4, seed=2026)
    assert_balanced_membership(membership)
    release = paczero_zpl_release(fd, membership, clip=args.clip, rng=rng)

    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "all_baseline_losses_finite": bool(np.isfinite(baseline_losses).all()),
        "all_fd_finite": bool(np.isfinite(fd).all()),
        "fd_has_nonzero_signal": bool(np.max(np.abs(fd)) > 0.0),
        "plus_minus_not_identical": bool(np.max(np.abs(plus - minus)) > 0.0),
        "release_sign_valid": int(release.sign) in (-1, 1),
        "membership_balanced": True,
    }
    elapsed = time.perf_counter() - start
    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(elapsed, 3),
        "parameterization": "synthetic_selected_vocab_logit_bias",
        "note": "Real-model two-point finite-difference smoke. Next step is replacing the synthetic logit-bias vector with actual adapter tensors.",
        "requested_mu": args.mu,
        "used_mu": float(used_mu),
        "clip": args.clip,
        "num_examples": len(examples),
        "bias_token_count": len(bias_token_ids),
        "bias_token_ids_head": bias_token_ids[:10],
        "checks": checks,
        "fd_attempts": fd_attempts,
        "baseline_loss_mean": float(baseline_losses.mean()),
        "baseline_loss_min": float(baseline_losses.min()),
        "baseline_loss_max": float(baseline_losses.max()),
        "fd_mean": float(fd.mean()),
        "fd_min": float(fd.min()),
        "fd_max": float(fd.max()),
        "fd_abs_max": float(np.max(np.abs(fd))),
        "plus_loss_mean": float(plus.mean()),
        "minus_loss_mean": float(minus.mean()),
        "zpl_release": {
            "sign": int(release.sign),
            "unanimous": bool(release.unanimous),
            "subset_signs": release.subset_signs.astype(int).tolist(),
            "subset_means": release.subset_means.astype(float).tolist(),
        },
        "membership_shape": list(membership.shape),
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FD_SMOKE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
