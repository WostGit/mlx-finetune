#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from paczero_core import (
    assert_balanced_membership,
    make_balanced_membership,
    paczero_zpl_release,
    sign_nonzero,
    subset_means_from_fd,
)


def to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x.tolist(), dtype=np.float64)


def mlx_per_sample_loss(theta: mx.array, x: mx.array, y: mx.array) -> mx.array:
    pred = x @ theta
    return 0.5 * (pred - y) ** 2


def mlx_two_point_fd(theta: mx.array, direction: mx.array, x: mx.array, y: mx.array, mu: float) -> mx.array:
    plus = mlx_per_sample_loss(theta + mu * direction, x, y)
    minus = mlx_per_sample_loss(theta - mu * direction, x, y)
    return (plus - minus) / (2.0 * mu)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlx_smoke_results.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    n = 32
    d = 8
    x_np = rng.normal(size=(n, d)).astype(np.float32)
    true_theta_np = rng.normal(size=d).astype(np.float32)
    y_np = x_np @ true_theta_np
    theta_np = np.zeros(d, dtype=np.float32)
    membership = make_balanced_membership(num_examples=n, num_subsets=8, seed=77)
    assert_balanced_membership(membership)

    x = mx.array(x_np)
    y = mx.array(y_np)
    theta = mx.array(theta_np)

    # Validate MLX finite differences against the analytic projected gradient
    # for a quadratic linear-regression objective.
    direction_np = rng.normal(size=d).astype(np.float32)
    direction_np = direction_np / np.linalg.norm(direction_np)
    direction = mx.array(direction_np)
    mu = 1e-3
    fd_mlx = mlx_two_point_fd(theta, direction, x, y, mu)
    mx.eval(fd_mlx)
    fd_np = to_numpy(fd_mlx)
    analytic_np = ((x_np @ theta_np) - y_np) * (x_np @ direction_np)
    max_abs_err = float(np.max(np.abs(fd_np - analytic_np)))
    if max_abs_err > 1e-2:
        raise AssertionError(f"MLX finite-difference max_abs_err too large: {max_abs_err}")

    losses = []
    unanimity = []
    release_signs = []
    lr = 0.03
    clip = 25.0
    for _ in range(args.steps):
        direction_np = rng.normal(size=d).astype(np.float32)
        direction_np = direction_np / np.linalg.norm(direction_np)
        direction = mx.array(direction_np)
        loss_before = mx.mean(mlx_per_sample_loss(theta, x, y))
        fd_mlx = mlx_two_point_fd(theta, direction, x, y, mu)
        mx.eval(loss_before, fd_mlx)
        fd_np = to_numpy(fd_mlx)
        release = paczero_zpl_release(fd_np, membership, clip=clip, rng=rng)
        theta = theta - lr * float(release.sign) * direction
        loss_after = mx.mean(mlx_per_sample_loss(theta, x, y))
        mx.eval(theta, loss_after)
        losses.append(float(loss_after.item()))
        unanimity.append(bool(release.unanimous))
        release_signs.append(int(release.sign))

    if not np.isfinite(losses).all():
        raise AssertionError("MLX ZPL smoke losses contain non-finite values")

    # We do not require monotonic improvement for ZPL because disagreement steps
    # can be random.  But over this deterministic toy seed it should improve.
    if losses[-1] >= losses[0]:
        raise AssertionError(f"MLX ZPL smoke loss did not improve: {losses[0]} -> {losses[-1]}")

    payload = {
        "success": True,
        "device": str(mx.default_device()),
        "steps": args.steps,
        "finite_difference_max_abs_err": max_abs_err,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "unanimity_rate": sum(unanimity) / len(unanimity),
        "release_sign_counts": {
            "positive": sum(1 for s in release_signs if s > 0),
            "negative": sum(1 for s in release_signs if s < 0),
        },
        "membership_shape": list(membership.shape),
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
