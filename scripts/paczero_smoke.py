#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paczero_core import (
    assert_balanced_membership,
    make_balanced_membership,
    mezo_step,
    paczero_zpl_release,
    paczero_zpl_sign_step,
    quadratic_per_sample_loss,
    subset_means_from_fd,
    two_point_per_sample_fd,
    zpl_release_from_subset_means,
)


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-8) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def test_balanced_membership() -> dict:
    membership = make_balanced_membership(num_examples=17, num_subsets=10, seed=123)
    assert_balanced_membership(membership)
    col_counts = membership.astype(int).sum(axis=0).tolist()
    row_counts = membership.astype(int).sum(axis=1).tolist()
    return {
        "name": "balanced_membership",
        "shape": list(membership.shape),
        "column_counts_unique": sorted(set(col_counts)),
        "row_count_min": min(row_counts),
        "row_count_max": max(row_counts),
        "pass": True,
    }


def test_two_point_fd_linearized_quadratic() -> dict:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(11, 4))
    y = rng.normal(size=11)
    theta = rng.normal(size=4)
    direction = rng.normal(size=4)
    direction = direction / np.linalg.norm(direction)
    mu = 1e-5
    per_sample_loss = quadratic_per_sample_loss(x, y)
    fd = two_point_per_sample_fd(theta, direction, per_sample_loss, mu)
    analytic = (x @ theta - y) * (x @ direction)
    max_abs_err = float(np.max(np.abs(fd - analytic)))
    if max_abs_err > 1e-6:
        raise AssertionError(f"finite-difference error too large: {max_abs_err}")
    return {
        "name": "two_point_fd_linearized_quadratic",
        "max_abs_err": max_abs_err,
        "pass": True,
    }


def test_zpl_unanimous_and_disagreement() -> dict:
    rng = np.random.default_rng(9)
    positive = np.array([1.0, 2.0, 3.0])
    release_pos = zpl_release_from_subset_means(positive, rng)
    if not release_pos.unanimous or release_pos.sign != 1:
        raise AssertionError("positive unanimous release failed")

    negative = np.array([-1.0, -0.1, -2.0])
    release_neg = zpl_release_from_subset_means(negative, rng)
    if not release_neg.unanimous or release_neg.sign != -1:
        raise AssertionError("negative unanimous release failed")

    mixed = np.array([-1.0, 0.2, 2.0, -0.3])
    release_mixed = zpl_release_from_subset_means(mixed, rng)
    if release_mixed.unanimous:
        raise AssertionError("mixed signs should be non-unanimous")
    if release_mixed.sign not in (-1, 1):
        raise AssertionError("mixed release sign must be +/-1")

    return {
        "name": "zpl_unanimous_and_disagreement",
        "positive_sign": release_pos.sign,
        "negative_sign": release_neg.sign,
        "mixed_unanimous": release_mixed.unanimous,
        "mixed_released_sign": release_mixed.sign,
        "pass": True,
    }


def test_subset_means_and_zpl() -> dict:
    per_sample_fd = np.array([1.0, 2.0, -5.0, 4.0, -3.0, 0.5])
    membership = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1],
    ], dtype=bool)
    means = subset_means_from_fd(per_sample_fd, membership, clip=2.0)
    assert_close("subset mean 0", float(means[0]), 1.5)
    assert_close("subset mean 1", float(means[1]), -2.0)
    assert_close("subset mean 2", float(means[2]), 1.25)
    rng = np.random.default_rng(5)
    release = paczero_zpl_release(per_sample_fd, membership, clip=2.0, rng=rng)
    if release.unanimous:
        raise AssertionError("expected disagreement for hand-built subset means")
    return {
        "name": "subset_means_and_zpl",
        "subset_means": means.tolist(),
        "release_sign": release.sign,
        "unanimous": release.unanimous,
        "pass": True,
    }


def test_toy_optimization(smoke_steps: int) -> dict:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(32, 6))
    true_theta = rng.normal(size=6)
    y = x @ true_theta
    loss_fn = quadratic_per_sample_loss(x, y)
    theta_mezo = np.zeros(6)
    theta_zpl = np.zeros(6)
    membership = make_balanced_membership(num_examples=x.shape[0], num_subsets=8, seed=12)
    assert_balanced_membership(membership)

    mezo_losses = []
    zpl_losses = []
    unanimity = []
    for _ in range(smoke_steps):
        mezo = mezo_step(theta_mezo, loss_fn, lr=0.2, mu=1e-3, rng=rng)
        theta_mezo = mezo.theta
        mezo_losses.append(mezo.loss_after)

        zpl = paczero_zpl_sign_step(theta_zpl, loss_fn, membership, lr=0.02, mu=1e-3, clip=10.0, rng=rng)
        theta_zpl = zpl.theta
        zpl_losses.append(zpl.loss_after)
        unanimity.append(zpl.release.unanimous)

    if not np.isfinite(mezo_losses).all():
        raise AssertionError("MeZO toy losses contain non-finite values")
    if not np.isfinite(zpl_losses).all():
        raise AssertionError("ZPL toy losses contain non-finite values")
    if mezo_losses[-1] >= mezo_losses[0]:
        raise AssertionError(f"MeZO toy loss did not decrease: {mezo_losses[0]} -> {mezo_losses[-1]}")

    return {
        "name": "toy_optimization",
        "steps": smoke_steps,
        "mezo_loss_first": mezo_losses[0],
        "mezo_loss_last": mezo_losses[-1],
        "zpl_loss_first": zpl_losses[0],
        "zpl_loss_last": zpl_losses[-1],
        "zpl_unanimity_rate": sum(unanimity) / len(unanimity),
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_smoke_results.json"))
    args = parser.parse_args()

    tests = [
        test_balanced_membership(),
        test_two_point_fd_linearized_quadratic(),
        test_zpl_unanimous_and_disagreement(),
        test_subset_means_and_zpl(),
        test_toy_optimization(args.steps),
    ]
    payload = {"success": all(t["pass"] for t in tests), "tests": tests}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
