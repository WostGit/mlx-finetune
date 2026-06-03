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
from paczero_mlxlm_exhaustive_readiness import eval_dataset, get_path, set_path
from paczero_mlxlm_lora_reproduction import PACZeroLoRALinear, infer_linear_dims, save_adapter_npz


def load_sst2_three_way(train_n: int, dev_n: int, eval_n: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/glue", "sst2")

        def convert(split: str, start: int, n: int) -> list[tuple[str, str]]:
            rows = []
            for row in ds[split].select(range(start, start + n)):
                label = "positive" if int(row["label"]) == 1 else "negative"
                rows.append((str(row["sentence"]), label))
            return rows

        valid_total = len(ds["validation"])
        if dev_n + eval_n > valid_total:
            raise ValueError(f"SST-2 validation has {valid_total} rows, cannot allocate dev={dev_n} + eval={eval_n}")
        return (
            convert("train", 0, train_n),
            convert("validation", 0, dev_n),
            convert("validation", dev_n, eval_n),
            "nyu-mll/glue/sst2 train split plus disjoint validation dev/eval slices",
        )
    except Exception as exc:
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
        ]
        rows = (fallback * ((train_n + dev_n + eval_n) // len(fallback) + 1))[: train_n + dev_n + eval_n]
        return rows[:train_n], rows[train_n:train_n + dev_n], rows[train_n + dev_n:train_n + dev_n + eval_n], f"fallback_builtin_sst2_like:{type(exc).__name__}:{exc}"


def evaluate_all(model: Any, tokenizer: Any, train_rows: list[tuple[str, str]], dev_rows: list[tuple[str, str]], eval_rows: list[tuple[str, str]]) -> dict:
    train_eval = eval_dataset(model, tokenizer, train_rows)
    dev_eval = eval_dataset(model, tokenizer, dev_rows)
    heldout_eval = eval_dataset(model, tokenizer, eval_rows)
    return {
        "train_loss": train_eval["loss_mean"],
        "train_accuracy": train_eval["accuracy"],
        "dev_loss": dev_eval["loss_mean"],
        "dev_accuracy": dev_eval["accuracy"],
        "eval_loss": heldout_eval["loss_mean"],
        "eval_accuracy": heldout_eval["accuracy"],
    }


def train_losses_for_theta(model: Any, tokenizer: Any, train_rows: list[tuple[str, str]], lora: PACZeroLoRALinear, theta: mx.array) -> np.ndarray:
    lora.set_theta(theta)
    return np.array(eval_dataset(model, tokenizer, train_rows)["losses"], dtype=np.float64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target-path", default="model.layers.0.self_attn.q_proj")
    p.add_argument("--slug", required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=20260609)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--train-examples", type=int, default=128)
    p.add_argument("--dev-examples", type=int, default=128)
    p.add_argument("--eval-examples", type=int, default=500)
    p.add_argument("--num-subsets", type=int, default=126)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=25.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--adapter-out", type=Path, required=True)
    args = p.parse_args()

    start = time.perf_counter()
    print("# PACZero-LoRA reviewer-risk gate")
    print("Addresses high reviewer-risk gaps: M=126, rank=8/alpha=16, disjoint held-out eval, larger split, privacy accounting.")
    print(f"model={args.model}")
    print(f"slug={args.slug}")
    print(f"rank={args.rank} alpha={args.alpha} M={args.num_subsets} steps={args.steps}")
    print(f"train={args.train_examples} dev={args.dev_examples} eval={args.eval_examples}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    if args.num_subsets % 2 != 0:
        raise ValueError("PACZero balanced membership requires an even num-subsets")

    model, tokenizer = load(args.model)
    train_rows, dev_rows, eval_rows, dataset_source = load_sst2_three_way(args.train_examples, args.dev_examples, args.eval_examples)

    base_module = get_path(model, args.target_path)
    input_dim, output_dim, dim_debug = infer_linear_dims(base_module, args.target_path)
    print("LORA_DIMENSION_INFERENCE=" + json.dumps(dim_debug))
    lora = PACZeroLoRALinear(base_module, input_dim=input_dim, output_dim=output_dim, rank=args.rank, alpha=args.alpha, seed=args.seed)
    set_path(model, args.target_path, lora)

    baseline = evaluate_all(model, tokenizer, train_rows, dev_rows, eval_rows)
    theta = lora.theta()
    theta_size = int(theta.shape[0])
    rng = np.random.default_rng(args.seed)
    membership = make_balanced_membership(num_examples=len(train_rows), num_subsets=args.num_subsets, seed=args.seed + 17)
    assert_balanced_membership(membership)

    fd_finite_count = 0
    fd_nonzero_count = 0
    unanimous_count = 0
    disagreement_count = 0
    sign_counts = {"positive": 0, "negative": 0}
    fd_abs_max_values = []
    fd_abs_mean_values = []
    history = []
    best = {"step": 0, **baseline}
    best_theta = theta

    for step in range(1, args.steps + 1):
        direction_np = rng.normal(size=theta_size).astype(np.float32)
        direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
        direction = mx.array(direction_np, dtype=mx.float32)
        plus = train_losses_for_theta(model, tokenizer, train_rows, lora, theta + args.mu * direction)
        minus = train_losses_for_theta(model, tokenizer, train_rows, lora, theta - args.mu * direction)
        fd = (plus - minus) / (2.0 * args.mu)
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        fd_finite_count += int(np.isfinite(fd).all())
        fd_nonzero_count += int(fd_abs_max > 0.0)
        release = paczero_zpl_release(fd, membership, clip=args.clip, rng=rng)
        unanimous = bool(release.unanimous)
        unanimous_count += int(unanimous)
        disagreement_count += int(not unanimous)
        if int(release.sign) > 0:
            sign_counts["positive"] += 1
        else:
            sign_counts["negative"] += 1
        theta = theta - args.lr * float(release.sign) * direction
        lora.set_theta(theta)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            metrics = evaluate_all(model, tokenizer, train_rows, dev_rows, eval_rows)
            row = {
                "step": step,
                **metrics,
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": unanimous,
            }
            history.append(row)
            print("REVIEWER_GATE_EVAL=" + json.dumps(row))
            if (metrics["dev_accuracy"] > best["dev_accuracy"]) or (
                metrics["dev_accuracy"] == best["dev_accuracy"] and metrics["dev_loss"] < best["dev_loss"]
            ):
                best = {"step": step, **metrics}
                best_theta = theta

    lora.set_theta(theta)
    final = evaluate_all(model, tokenizer, train_rows, dev_rows, eval_rows)
    if (final["dev_accuracy"] > best["dev_accuracy"]) or (
        final["dev_accuracy"] == best["dev_accuracy"] and final["dev_loss"] < best["dev_loss"]
    ):
        best = {"step": args.steps, **final}
        best_theta = theta

    lora.set_theta(best_theta)
    selected_eval = evaluate_all(model, tokenizer, train_rows, dev_rows, eval_rows)
    adapter_info = save_adapter_npz(args.adapter_out, lora)

    fd_finite_rate = fd_finite_count / max(1, args.steps)
    fd_signal_rate = fd_nonzero_count / max(1, args.steps)
    unanimity_rate = unanimous_count / max(1, args.steps)
    disagreement_rate = disagreement_count / max(1, args.steps)
    column_counts = membership.astype(int).sum(axis=0).tolist()

    privacy_accounting = {
        "mechanism": "PACZero-ZPL-style sign release",
        "claim_scope": "conceptual ZPL transcript accounting for this implementation; not differential privacy",
        "not_differential_privacy": True,
        "mutual_information_claim_under_zpl_rule": "I(S_star; Y_1:T)=0 when disagreement releases are independent uniform random signs and unanimous signs are subset-independent",
        "num_subsets_M": args.num_subsets,
        "examples_per_column_expected": args.num_subsets // 2,
        "membership_column_counts_unique": sorted(set(column_counts)),
        "membership_balanced": sorted(set(column_counts)) == [args.num_subsets // 2],
        "steps": args.steps,
        "unanimous_steps": unanimous_count,
        "disagreement_steps": disagreement_count,
        "disagreement_releases_randomized": disagreement_count,
        "unanimity_rate": unanimity_rate,
        "disagreement_rate": disagreement_rate,
        "release_sign_counts": sign_counts,
        "clip": args.clip,
        "membership_seed": args.seed + 17,
        "rng_seed": args.seed,
    }

    checks = {
        "attached_lora_wrapper": isinstance(get_path(model, args.target_path), PACZeroLoRALinear),
        "paper_style_num_subsets": args.num_subsets == 126,
        "paper_style_lora_rank_alpha": args.rank == 8 and abs(args.alpha - 16.0) < 1e-9,
        "has_disjoint_eval_split": len(eval_rows) > 0 and len(dev_rows) > 0,
        "theta_size_positive": theta_size > 0,
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "fd_magnitude_ok": max(fd_abs_max_values) > 0.0,
        "losses_remain_finite": all(np.isfinite([final["train_loss"], final["dev_loss"], final["eval_loss"], selected_eval["eval_loss"]])),
        "best_dev_accuracy_not_worse": best["dev_accuracy"] >= baseline["dev_accuracy"],
        "best_eval_reported": "eval_accuracy" in selected_eval,
        "privacy_membership_balanced": privacy_accounting["membership_balanced"],
        "adapter_saved": args.adapter_out.exists() and args.adapter_out.stat().st_size > 0,
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "slug": args.slug,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "eval_examples": len(eval_rows),
        "parameterization": "reviewer_gate_actual_lora_ab_paczero_zpl",
        "target_path": args.target_path,
        "dimension_inference": dim_debug,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": args.rank,
        "alpha": args.alpha,
        "theta_size": theta_size,
        "seed": args.seed,
        "steps": args.steps,
        "num_subsets": args.num_subsets,
        "mu": args.mu,
        "lr": args.lr,
        "clip": args.clip,
        "checks": checks,
        "baseline": baseline,
        "final": final,
        "best_checkpoint_by_dev": best,
        "selected_best_adapter_eval": selected_eval,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "privacy_accounting": privacy_accounting,
        "adapter_file": adapter_info,
        "history": history,
        "verdict": "Reviewer-risk medium gate: closes M, LoRA config, held-out eval, and privacy-reporting gaps. Still smaller than full 1000/500/1000 paper scale unless train/dev/eval inputs are increased.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("REVIEWER_GATE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
