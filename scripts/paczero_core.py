#!/usr/bin/env python3
"""Small PACZero-style primitives used by smoke tests.

This module intentionally has no MLX dependency.  It validates the parts of a
PACZero extension that are easy to get subtly wrong before we attach the logic
to a real LoRA parameter vector:

* balanced candidate-subset construction;
* per-sample two-point finite-difference scalars;
* clipping and subset-sign aggregation;
* PACZero-ZPL release semantics;
* a minimal zeroth-order vector update on a toy differentiable loss.

The real MLX LoRA trainer can import these pieces later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ZPLRelease:
    """One PACZero-ZPL sign release."""

    sign: int
    unanimous: bool
    subset_signs: Array
    subset_means: Array


@dataclass(frozen=True)
class ZOStepResult:
    """Diagnostics for a two-point zeroth-order step."""

    theta: Array
    direction: Array
    release: ZPLRelease
    loss_before: float
    loss_after: float
    per_sample_fd_mean: float


def _repair_empty_candidate_subsets(membership: Array, rng: np.random.Generator) -> Array:
    """Repair empty rows while preserving the M/2-per-example invariant.

    For very small reviewer-proof runs, random half-subset assignment can leave
    one of the M candidate subsets empty even though every example appears in
    exactly M/2 subsets.  Empty rows make subset means undefined and are not a
    meaningful PACZero candidate subset.  This repair swaps memberships from
    overfull rows into empty rows column-by-column, preserving each column count
    exactly while ensuring every row has at least one example whenever the total
    number of memberships is sufficient.
    """

    fixed = np.array(membership, dtype=bool, copy=True)
    num_subsets, num_examples = fixed.shape
    total_memberships = int(fixed.sum())
    if total_memberships < num_subsets:
        raise ValueError(
            "cannot make every candidate subset non-empty: "
            f"total memberships {total_memberships} < num_subsets {num_subsets}"
        )

    # Repeatedly move one column membership from a row with count > 1 to an empty row.
    # This keeps every example's membership count unchanged.
    for _ in range(num_subsets * max(1, num_examples) * 4):
        row_counts = fixed.astype(np.int64).sum(axis=1)
        empty_rows = np.flatnonzero(row_counts == 0)
        if empty_rows.size == 0:
            return fixed
        donor_rows = np.flatnonzero(row_counts > 1)
        if donor_rows.size == 0:
            break
        empty = int(empty_rows[0])
        donor_order = rng.permutation(donor_rows)
        repaired = False
        for donor in donor_order:
            donor = int(donor)
            candidate_cols = np.flatnonzero(fixed[donor] & ~fixed[empty])
            if candidate_cols.size == 0:
                continue
            col = int(rng.choice(candidate_cols))
            fixed[donor, col] = False
            fixed[empty, col] = True
            repaired = True
            break
        if not repaired:
            break

    row_counts = fixed.astype(np.int64).sum(axis=1)
    if np.any(row_counts == 0):
        raise ValueError("failed to repair empty PACZero candidate subsets")
    return fixed


def make_balanced_membership(num_examples: int, num_subsets: int, seed: int = 0) -> Array:
    """Return an M x N boolean membership matrix.

    Each example appears in exactly M/2 candidate subsets.  This is the key
    structural property used by PAC privacy: before observing the release, each
    example has prior membership probability 1/2.  The constructor also ensures
    every candidate subset is non-empty whenever possible, so tiny smoke tests
    with large M still produce defined subset means.

    Args:
        num_examples: N, the number of training examples.
        num_subsets: M, the number of candidate subsets. Must be even.
        seed: RNG seed.

    Returns:
        Boolean array of shape (M, N).
    """

    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if num_subsets <= 0 or num_subsets % 2 != 0:
        raise ValueError("num_subsets must be a positive even integer")

    rng = np.random.default_rng(seed)
    membership = np.zeros((num_subsets, num_examples), dtype=bool)
    half = num_subsets // 2
    for example_idx in range(num_examples):
        chosen = rng.choice(num_subsets, size=half, replace=False)
        membership[chosen, example_idx] = True

    membership = _repair_empty_candidate_subsets(membership, rng)
    return membership


def assert_balanced_membership(membership: Array) -> None:
    """Validate the M/2-per-example invariant plus non-empty subsets."""

    if membership.ndim != 2:
        raise AssertionError(f"membership must be 2D, got shape {membership.shape}")
    num_subsets, num_examples = membership.shape
    if num_subsets % 2 != 0:
        raise AssertionError("membership has odd number of subsets")
    if num_examples == 0:
        raise AssertionError("membership has no examples")
    col_counts = membership.astype(np.int64).sum(axis=0)
    expected = num_subsets // 2
    if not np.all(col_counts == expected):
        raise AssertionError(f"not balanced: expected {expected}, got {col_counts.tolist()}")
    row_counts = membership.astype(np.int64).sum(axis=1)
    if np.any(row_counts == 0):
        raise AssertionError("at least one candidate subset is empty")


def two_point_per_sample_fd(
    theta: Array,
    direction: Array,
    per_sample_loss_fn: Callable[[Array], Array],
    mu: float,
) -> Array:
    """Compute per-sample two-point finite-difference scalars.

    Returns `(loss_i(theta + mu*z) - loss_i(theta - mu*z)) / (2*mu)` for each
    sample i.  This equals each per-sample gradient projected onto z up to
    finite-difference error.
    """

    if mu <= 0:
        raise ValueError("mu must be positive")
    theta = np.asarray(theta, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if theta.shape != direction.shape:
        raise ValueError(f"theta and direction shapes differ: {theta.shape} vs {direction.shape}")

    plus = np.asarray(per_sample_loss_fn(theta + mu * direction), dtype=np.float64)
    minus = np.asarray(per_sample_loss_fn(theta - mu * direction), dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 1:
        raise ValueError("per_sample_loss_fn must return 1D arrays with matching shapes")
    return (plus - minus) / (2.0 * mu)


def subset_means_from_fd(per_sample_fd: Array, membership: Array, clip: float) -> Array:
    """Clip per-sample finite differences and average inside each subset."""

    if clip <= 0:
        raise ValueError("clip must be positive")
    fd = np.asarray(per_sample_fd, dtype=np.float64)
    if fd.ndim != 1:
        raise ValueError("per_sample_fd must be 1D")
    if membership.ndim != 2 or membership.shape[1] != fd.shape[0]:
        raise ValueError("membership shape must be (num_subsets, len(per_sample_fd))")

    clipped = np.clip(fd, -clip, clip)
    counts = membership.astype(np.float64).sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("all candidate subsets must be non-empty")
    return (membership.astype(np.float64) @ clipped) / counts


def sign_nonzero(values: Array) -> Array:
    """Return -1 for negative values and +1 for zero or positive values."""

    values = np.asarray(values, dtype=np.float64)
    return np.where(values < 0.0, -1, 1).astype(np.int64)


def zpl_release_from_subset_means(subset_means: Array, rng: np.random.Generator) -> ZPLRelease:
    """Apply PACZero-ZPL sign release semantics.

    If every candidate subset agrees, release that unanimous sign. Otherwise
    release a fresh uniform random sign, which is the zero-privacy-leakage branch.
    """

    means = np.asarray(subset_means, dtype=np.float64)
    if means.ndim != 1 or means.size == 0:
        raise ValueError("subset_means must be a non-empty 1D array")
    subset_signs = sign_nonzero(means)
    unanimous = bool(np.all(subset_signs == subset_signs[0]))
    if unanimous:
        sign = int(subset_signs[0])
    else:
        sign = int(rng.choice(np.array([-1, 1], dtype=np.int64)))
    return ZPLRelease(sign=sign, unanimous=unanimous, subset_signs=subset_signs, subset_means=means)


def paczero_zpl_release(
    per_sample_fd: Array,
    membership: Array,
    clip: float,
    rng: np.random.Generator,
) -> ZPLRelease:
    """Compute a PACZero-ZPL release from per-sample finite differences."""

    means = subset_means_from_fd(per_sample_fd, membership, clip)
    return zpl_release_from_subset_means(means, rng)


def quadratic_per_sample_loss(x: Array, y: Array) -> Callable[[Array], Array]:
    """Return per-sample squared-error loss function for toy smoke tests."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x must be N x D and y must be N")

    def loss(theta: Array) -> Array:
        theta = np.asarray(theta, dtype=np.float64)
        pred = x @ theta
        return 0.5 * (pred - y) ** 2

    return loss


def total_loss(per_sample_loss_fn: Callable[[Array], Array], theta: Array) -> float:
    return float(np.mean(per_sample_loss_fn(theta)))


def mezo_step(
    theta: Array,
    per_sample_loss_fn: Callable[[Array], Array],
    lr: float,
    mu: float,
    rng: np.random.Generator,
) -> ZOStepResult:
    """One non-private MeZO-style step on a toy vector parameter."""

    theta = np.asarray(theta, dtype=np.float64)
    direction = rng.normal(size=theta.shape)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    loss_before = total_loss(per_sample_loss_fn, theta)
    fd = two_point_per_sample_fd(theta, direction, per_sample_loss_fn, mu)
    scalar = float(np.mean(fd))
    theta_next = theta - lr * scalar * direction
    release = ZPLRelease(
        sign=1 if scalar >= 0 else -1,
        unanimous=True,
        subset_signs=np.array([1 if scalar >= 0 else -1], dtype=np.int64),
        subset_means=np.array([scalar], dtype=np.float64),
    )
    loss_after = total_loss(per_sample_loss_fn, theta_next)
    return ZOStepResult(theta_next, direction, release, loss_before, loss_after, float(np.mean(fd)))


def paczero_zpl_sign_step(
    theta: Array,
    per_sample_loss_fn: Callable[[Array], Array],
    membership: Array,
    lr: float,
    mu: float,
    clip: float,
    rng: np.random.Generator,
) -> ZOStepResult:
    """One PACZero-ZPL sign-only toy step.

    This is intentionally minimal: it validates the control flow and sign update,
    not model quality.  A real LoRA trainer would replace theta with the flattened
    adapter vector and per_sample_loss_fn with model losses.
    """

    theta = np.asarray(theta, dtype=np.float64)
    direction = rng.normal(size=theta.shape)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    loss_before = total_loss(per_sample_loss_fn, theta)
    fd = two_point_per_sample_fd(theta, direction, per_sample_loss_fn, mu)
    release = paczero_zpl_release(fd, membership, clip, rng)
    theta_next = theta - lr * release.sign * direction
    loss_after = total_loss(per_sample_loss_fn, theta_next)
    return ZOStepResult(theta_next, direction, release, loss_before, loss_after, float(np.mean(fd)))
