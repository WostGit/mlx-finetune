#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from paczero_core import assert_balanced_membership, make_balanced_membership, paczero_zpl_release


def flatten_lora(a: mx.array, b: mx.array) -> mx.array:
    return mx.concatenate([mx.reshape(a, (-1,)), mx.reshape(b, (-1,))])


def unflatten_lora(theta: mx.array, in_dim: int, out_dim: int, rank: int) -> tuple[mx.array, mx.array]:
    a_size = rank * in_dim
    a = mx.reshape(theta[:a_size], (rank, in_dim))
    b = mx.reshape(theta[a_size:], (out_dim, rank))
    return a, b


def lora_forward(theta: mx.array, x: mx.array, base_w: mx.array, in_dim: int, out_dim: int, rank: int, alpha: float) -> mx.array:
    a, b = unflatten_lora(theta, in_dim, out_dim, rank)
    delta_w = (alpha / rank) * (b @ a)
    return x @ mx.transpose(base_w + delta_w)


def per_sample_loss(theta: mx.array, x: mx.array, y: mx.array, base_w: mx.array, in_dim: int, out_dim: int, rank: int, alpha: float) -> mx.array:
    pred = lora_forward(theta, x, base_w, in_dim, out_dim, rank, alpha)
    return 0.5 * mx.sum((pred - y) ** 2, axis=1)


def two_point_fd(theta: mx.array, direction: mx.array, x: mx.array, y: mx.array, base_w: mx.array, in_dim: int, out_dim: int, rank: int, alpha: float, mu: float) -> mx.array:
    plus = per_sample_loss(theta + mu * direction, x, y, base_w, in_dim, out_dim, rank, alpha)
    minus = per_sample_loss(theta - mu * direction, x, y, base_w, in_dim, out_dim, rank, alpha)
    return (plus - minus) / (2.0 * mu)


def to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x.tolist(), dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_lora_smoke_results.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(4242)
    n = 48
    in_dim = 12
    out_dim = 3
    rank = 2
    alpha = 4.0
    mu = 1e-3
    lr = 0.02
    clip = 25.0

    x_np = rng.normal(size=(n, in_dim)).astype(np.float32)
    base_w_np = rng.normal(scale=0.05, size=(out_dim, in_dim)).astype(np.float32)
    target_delta_np = rng.normal(scale=0.15, size=(out_dim, in_dim)).astype(np.float32)
    y_np = x_np @ (base_w_np + target_delta_np).T

    # LoRA initialization mirrors the common convention: A random, B zero.
    a_np = rng.normal(scale=0.02, size=(rank, in_dim)).astype(np.float32)
    b_np = np.zeros((out_dim, rank), dtype=np.float32)
    theta = flatten_lora(mx.array(a_np), mx.array(b_np))
    theta_size = int(theta.shape[0])

    x = mx.array(x_np)
    y = mx.array(y_np)
    base_w = mx.array(base_w_np)
    membership = make_balanced_membership(num_examples=n, num_subsets=8, seed=55)
    assert_balanced_membership(membership)

    # Flatten/unflatten identity check.
    a0, b0 = unflatten_lora(theta, in_dim, out_dim, rank)
    theta_roundtrip = flatten_lora(a0, b0)
    mx.eval(theta_roundtrip)
    roundtrip_max_abs_err = float(np.max(np.abs(to_numpy(theta_roundtrip) - to_numpy(theta))))
    if roundtrip_max_abs_err > 1e-8:
        raise AssertionError(f"flatten/unflatten roundtrip failed: {roundtrip_max_abs_err}")

    initial_loss = mx.mean(per_sample_loss(theta, x, y, base_w, in_dim, out_dim, rank, alpha))
    mx.eval(initial_loss)
    losses = [float(initial_loss.item())]
    unanimity = []
    release_signs = []
    fd_means = []

    for _ in range(args.steps):
        direction_np = rng.normal(size=theta_size).astype(np.float32)
        direction_np = direction_np / max(np.linalg.norm(direction_np), 1e-12)
        direction = mx.array(direction_np)
        fd_mx = two_point_fd(theta, direction, x, y, base_w, in_dim, out_dim, rank, alpha, mu)
        mx.eval(fd_mx)
        fd_np = to_numpy(fd_mx)
        release = paczero_zpl_release(fd_np, membership, clip=clip, rng=rng)
        theta = theta - lr * float(release.sign) * direction
        current_loss = mx.mean(per_sample_loss(theta, x, y, base_w, in_dim, out_dim, rank, alpha))
        mx.eval(theta, current_loss)
        losses.append(float(current_loss.item()))
        unanimity.append(bool(release.unanimous))
        release_signs.append(int(release.sign))
        fd_means.append(float(np.mean(fd_np)))

    if not np.isfinite(losses).all():
        raise AssertionError("LoRA smoke losses contain non-finite values")
    if losses[-1] >= losses[0]:
        raise AssertionError(f"LoRA-style ZPL smoke loss did not improve: {losses[0]} -> {losses[-1]}")

    payload = {
        "success": True,
        "device": str(mx.default_device()),
        "steps": args.steps,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "rank": rank,
        "alpha": alpha,
        "theta_size": theta_size,
        "roundtrip_max_abs_err": roundtrip_max_abs_err,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "loss_min": min(losses),
        "unanimity_rate": sum(unanimity) / len(unanimity),
        "release_sign_counts": {
            "positive": sum(1 for s in release_signs if s > 0),
            "negative": sum(1 for s in release_signs if s < 0),
        },
        "fd_mean_first": fd_means[0],
        "fd_mean_last": fd_means[-1],
        "membership_shape": list(membership.shape),
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
