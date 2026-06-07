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

from paczero_mlxlm_faithful_adaptation import (
    attach_loras,
    best_key,
    evaluate_all,
    load_squad_rows,
    load_sst2_rows,
    multi_theta,
    resolve_layer_targets,
    save_multi_adapter_npz,
    set_multi_theta,
    train_losses_for_theta,
)


def print_kv(title: str, value: Any) -> None:
    print(f"UTILITY_CONTROL::{title}={json.dumps(value, sort_keys=True, default=str)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/SmolLM-135M-4bit")
    p.add_argument("--slug", default="smollm-135m-4bit")
    p.add_argument("--task", choices=["sst2", "squad"], required=True)
    p.add_argument("--projections", default="q_proj,v_proj")
    p.add_argument("--layers", default="all")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--train-examples", type=int, default=8)
    p.add_argument("--dev-examples", type=int, default=8)
    p.add_argument("--eval-examples", type=int, default=32)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--adapter-out", type=Path, required=True)
    args = p.parse_args()

    start = time.perf_counter()
    print("# PACZero SmolLM non-private ZO utility control")
    print("This is intentionally NOT private. It is a utility control for the ZPL I=0 run.")
    print_kv("cli_args", vars(args))
    print_kv("python", sys.version.splitlines()[0])
    for pkg in ["mlx", "mlx-lm", "datasets", "numpy"]:
        try:
            print_kv(f"package_{pkg}", md.version(pkg))
        except Exception as exc:
            print_kv(f"package_{pkg}", f"unavailable:{exc}")

    if args.task == "sst2":
        train_rows, dev_rows, eval_rows, dataset_source = load_sst2_rows(args.train_examples, args.dev_examples, args.eval_examples)
    else:
        train_rows, dev_rows, eval_rows, dataset_source = load_squad_rows(args.train_examples, args.dev_examples, args.eval_examples)
    print_kv("dataset_loaded", {"task": args.task, "source": dataset_source, "train": len(train_rows), "dev": len(dev_rows), "eval": len(eval_rows)})

    model, tokenizer = load(args.model)
    projections = [x.strip() for x in args.projections.split(",") if x.strip()]
    target_paths = resolve_layer_targets(model, projections, args.layers)
    print_kv("resolved_lora_targets", {"count": len(target_paths), "targets": target_paths, "layers_arg": args.layers})
    loras, target_infos = attach_loras(model, target_paths, rank=args.rank, alpha=args.alpha, seed=args.seed)
    theta, slices = multi_theta(loras)
    theta_size = int(theta.shape[0])
    print_kv("combined_lora_theta", {"theta_size": theta_size, "target_count": len(target_paths)})

    baseline = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
    print_kv("baseline_metrics", baseline)

    rng = np.random.default_rng(args.seed)
    best = {"step": 0, **baseline}
    best_theta = theta
    fd_finite = 0
    fd_nonzero = 0
    sign_counts = {"positive": 0, "negative": 0}
    fd_abs_max_values: list[float] = []
    fd_abs_mean_values: list[float] = []
    history = []

    for step in range(1, args.steps + 1):
        direction_np = rng.normal(size=theta_size).astype(np.float32)
        direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
        direction = mx.array(direction_np, dtype=mx.float32)
        plus = train_losses_for_theta(model, tokenizer, args.task, train_rows, loras, theta + args.mu * direction, slices)
        minus = train_losses_for_theta(model, tokenizer, args.task, train_rows, loras, theta - args.mu * direction, slices)
        fd = (plus - minus) / (2.0 * args.mu)
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        fd_finite += int(np.isfinite(fd).all())
        fd_nonzero += int(fd_abs_max > 0.0)
        mean_fd = float(np.mean(fd))
        release_sign = -1 if mean_fd < 0.0 else 1
        sign_counts["positive" if release_sign > 0 else "negative"] += 1
        theta = theta - args.lr * float(release_sign) * direction
        set_multi_theta(loras, theta, slices)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            metrics = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
            row = {"step": step, **metrics, "fd_abs_max": fd_abs_max, "fd_abs_mean": fd_abs_mean, "mean_fd": mean_fd, "release_sign": release_sign}
            history.append(row)
            print_kv("eval_checkpoint", row)
            if best_key(args.task, metrics) > best_key(args.task, best):
                best = {"step": step, **metrics}
                best_theta = theta
                print_kv("new_best_checkpoint", best)

    set_multi_theta(loras, theta, slices)
    final = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
    if best_key(args.task, final) > best_key(args.task, best):
        best = {"step": args.steps, **final}
        best_theta = theta
    set_multi_theta(loras, best_theta, slices)
    selected_eval = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
    adapter_info = save_multi_adapter_npz(args.adapter_out, loras, target_paths, int(best["step"]))

    fd_finite_rate = fd_finite / max(1, args.steps)
    fd_signal_rate = fd_nonzero / max(1, args.steps)
    utility_not_worse = best_key(args.task, best) >= best_key(args.task, baseline)
    checks = {
        "not_private_control": True,
        "paper_style_lora_rank_alpha": args.rank == 8 and abs(args.alpha - 16.0) < 1e-9,
        "faithful_projection_set_q_and_v": set(projections) == {"q_proj", "v_proj"},
        "all_layers_requested": args.layers == "all",
        "has_60_qv_targets_for_smollm": len(target_paths) == 60,
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "utility_selection_not_worse_than_baseline_on_selection_metric": utility_not_worse,
        "adapter_saved": args.adapter_out.exists() and args.adapter_out.stat().st_size > 0,
    }
    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "slug": args.slug,
        "task": args.task,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "eval_examples": len(eval_rows),
        "parameterization": "non_private_mean_sign_zo_lora_utility_control",
        "privacy_claim": "none; this control intentionally uses the full training mean FD sign and is not ZPL-private",
        "target_paths": target_paths,
        "target_count": len(target_paths),
        "target_infos_head": target_infos[:4],
        "layers": args.layers,
        "projections": projections,
        "rank": args.rank,
        "alpha": args.alpha,
        "theta_size": theta_size,
        "seed": args.seed,
        "steps": args.steps,
        "mu": args.mu,
        "lr": args.lr,
        "checks": checks,
        "baseline": baseline,
        "final": final,
        "best_checkpoint": best,
        "selected_best_adapter_eval": selected_eval,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "release_sign_counts": sign_counts,
        "adapter_file": adapter_info,
        "history": history,
        "limitations": [
            "This is a fast non-private utility control, not PACZero-ZPL and not I=0.",
            "Utility is smoke-scale; it contextualizes the ZPL run but does not reproduce paper utility numbers.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SMOLLM_NONPRIVATE_UTILITY_CONTROL_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
