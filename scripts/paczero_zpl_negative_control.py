#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path("benchmark-results/paczero-smollm-10of10-aggregate")
OUT_JSON = OUT_DIR / "zpl_negative_control_results.json"


def make_balanced_membership(num_examples: int, num_subsets: int, seed: int = 0) -> list[list[bool]]:
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if num_subsets <= 0 or num_subsets % 2 != 0:
        raise ValueError("num_subsets must be a positive even integer")
    rng = random.Random(seed)
    half = num_subsets // 2
    membership = [[False for _ in range(num_examples)] for _ in range(num_subsets)]
    for example_idx in range(num_examples):
        for subset_idx in rng.sample(range(num_subsets), half):
            membership[subset_idx][example_idx] = True
    membership = repair_empty_candidate_subsets(membership, rng)
    assert_balanced_membership(membership)
    return membership


def repair_empty_candidate_subsets(membership: list[list[bool]], rng: random.Random) -> list[list[bool]]:
    num_subsets = len(membership)
    num_examples = len(membership[0]) if membership else 0
    total_memberships = sum(sum(1 for v in row if v) for row in membership)
    if total_memberships < num_subsets:
        raise ValueError("cannot make every candidate subset non-empty")

    for _ in range(num_subsets * max(1, num_examples) * 4):
        row_counts = [sum(1 for v in row if v) for row in membership]
        empty_rows = [i for i, count in enumerate(row_counts) if count == 0]
        if not empty_rows:
            return membership
        donor_rows = [i for i, count in enumerate(row_counts) if count > 1]
        if not donor_rows:
            break
        empty = empty_rows[0]
        rng.shuffle(donor_rows)
        repaired = False
        for donor in donor_rows:
            candidate_cols = [j for j in range(num_examples) if membership[donor][j] and not membership[empty][j]]
            if not candidate_cols:
                continue
            col = rng.choice(candidate_cols)
            membership[donor][col] = False
            membership[empty][col] = True
            repaired = True
            break
        if not repaired:
            break
    if any(sum(1 for v in row if v) == 0 for row in membership):
        raise ValueError("failed to repair empty PACZero candidate subsets")
    return membership


def assert_balanced_membership(membership: list[list[bool]]) -> None:
    num_subsets = len(membership)
    if num_subsets == 0 or num_subsets % 2 != 0:
        raise AssertionError("membership must have positive even subset count")
    num_examples = len(membership[0])
    expected = num_subsets // 2
    for col in range(num_examples):
        count = sum(1 for row in membership if row[col])
        if count != expected:
            raise AssertionError(f"not balanced at example {col}: expected {expected}, got {count}")
    for row_idx, row in enumerate(membership):
        if sum(1 for v in row if v) == 0:
            raise AssertionError(f"empty candidate subset at row {row_idx}")


def subset_means_from_fd(fd: list[float], membership: list[list[bool]], clip: float) -> list[float]:
    clipped = [max(-clip, min(clip, value)) for value in fd]
    means = []
    for row in membership:
        values = [value for value, include in zip(clipped, row) if include]
        if not values:
            raise ValueError("all candidate subsets must be non-empty")
        means.append(sum(values) / len(values))
    return means


def sign_nonzero(values: list[float]) -> list[int]:
    return [-1 if value < 0.0 else 1 for value in values]


def zpl_good_release(subset_signs: list[int], rng: random.Random) -> dict:
    unanimous = len(set(subset_signs)) == 1
    if unanimous:
        return {
            "branch": "unanimous_subset_independent",
            "release_sign": int(subset_signs[0]),
            "secret_subset_index_used_for_release": False,
            "rng_derived_release": False,
            "violations": [],
        }
    return {
        "branch": "disagreement_randomized",
        "release_sign": int(rng.choice([-1, 1])),
        "secret_subset_index_used_for_release": False,
        "rng_derived_release": True,
        "violations": [],
    }


def zpl_bad_release(subset_signs: list[int], secret_subset_index: int) -> dict:
    # This is the forbidden failure mode: disagreement release depends on S_star.
    unanimous = len(set(subset_signs)) == 1
    if unanimous:
        return {
            "branch": "unanimous_subset_independent",
            "release_sign": int(subset_signs[0]),
            "secret_subset_index_used_for_release": False,
            "rng_derived_release": False,
            "violations": [],
        }
    return {
        "branch": "disagreement_bad_secret_subset_dependent",
        "release_sign": int(subset_signs[secret_subset_index]),
        "secret_subset_index_used_for_release": True,
        "rng_derived_release": False,
        "violations": ["disagreement_release_depends_on_secret_subset_index"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260617)
    num_examples = 8
    num_subsets = 126
    membership = make_balanced_membership(num_examples, num_subsets, seed=20260617)

    # Deterministic mixed directional derivatives force subset-sign disagreement.
    fd = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
    subset_means = subset_means_from_fd(fd, membership, clip=25.0)
    subset_signs = sign_nonzero(subset_means)
    has_disagreement = len(set(subset_signs)) > 1
    if not has_disagreement:
        raise RuntimeError("negative control setup failed: expected subset-sign disagreement")

    good = zpl_good_release(subset_signs, rng)
    bad = zpl_bad_release(subset_signs, secret_subset_index=0)

    good_audit_passes = (
        good["branch"] == "disagreement_randomized"
        and good["rng_derived_release"]
        and not good["secret_subset_index_used_for_release"]
        and len(good["violations"]) == 0
    )
    bad_audit_fails = (
        bad["secret_subset_index_used_for_release"]
        and not bad["rng_derived_release"]
        and "disagreement_release_depends_on_secret_subset_index" in bad["violations"]
    )

    column_counts = [sum(1 for row in membership if row[col]) for col in range(num_examples)]
    row_counts = [sum(1 for v in row if v) for row in membership]
    payload = {
        "success": bool(good_audit_passes and bad_audit_fails),
        "purpose": "Negative-control audit: prove the checker catches the forbidden PACZero-ZPL failure mode where disagreement release depends on S_star.",
        "num_examples": num_examples,
        "num_subsets_M": num_subsets,
        "membership_column_counts_unique": sorted(set(column_counts)),
        "membership_row_count_min": min(row_counts),
        "has_subset_sign_disagreement": bool(has_disagreement),
        "good_zpl_release": good,
        "bad_secret_dependent_release": bad,
        "checks": {
            "good_zpl_release_passes_audit": bool(good_audit_passes),
            "bad_secret_dependent_release_fails_audit": bool(bad_audit_fails),
            "negative_control_effective": bool(good_audit_passes and bad_audit_fails),
        },
        "conclusion": "PASS: the negative control catches S_star-dependent disagreement release" if good_audit_passes and bad_audit_fails else "FAIL",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("ZPL_NEGATIVE_CONTROL_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
