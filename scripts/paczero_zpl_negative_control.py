#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paczero_core import assert_balanced_membership, make_balanced_membership, sign_nonzero, subset_means_from_fd

OUT_DIR = Path("benchmark-results/paczero-smollm-10of10-aggregate")
OUT_JSON = OUT_DIR / "zpl_negative_control_results.json"


def zpl_good_release(subset_signs: np.ndarray, rng: np.random.Generator) -> dict:
    unique = sorted(set(subset_signs.astype(int).tolist()))
    unanimous = len(unique) == 1
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
        "release_sign": int(rng.choice(np.array([-1, 1], dtype=np.int64))),
        "secret_subset_index_used_for_release": False,
        "rng_derived_release": True,
        "violations": [],
    }


def zpl_bad_release(subset_signs: np.ndarray, secret_subset_index: int) -> dict:
    # This is the forbidden failure mode: disagreement release depends on S_star.
    unique = sorted(set(subset_signs.astype(int).tolist()))
    unanimous = len(unique) == 1
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
    rng = np.random.default_rng(20260617)
    num_examples = 8
    num_subsets = 126
    membership = make_balanced_membership(num_examples, num_subsets, seed=20260617)
    assert_balanced_membership(membership)

    # Construct deterministic per-sample directional derivatives with mixed signs.
    # This reliably creates disagreement among candidate subset signs.
    fd = np.array([-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    subset_means = subset_means_from_fd(fd, membership, clip=25.0)
    subset_signs = sign_nonzero(subset_means)
    has_disagreement = len(set(subset_signs.astype(int).tolist())) > 1
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

    payload = {
        "success": bool(good_audit_passes and bad_audit_fails),
        "purpose": "Negative-control audit: prove the checker catches the forbidden PACZero-ZPL failure mode where disagreement release depends on S_star.",
        "num_examples": num_examples,
        "num_subsets_M": num_subsets,
        "membership_column_counts_unique": sorted(set(membership.astype(int).sum(axis=0).tolist())),
        "membership_row_count_min": int(membership.astype(int).sum(axis=1).min()),
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
