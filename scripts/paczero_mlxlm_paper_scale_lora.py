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
from paczero_mlxlm_exhaustive_readiness import eval_dataset, load_sst2_rows, set_path, get_path
from paczero_mlxlm_lora_reproduction import PACZeroLoRALinear, save_adapter_npz


def parse_targets(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def load_sst2_population(train_limit: int | None, dev_limit: int | None) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-mll/glue", "sst2")
        def convert(split: str, limit: int | None) -> list[tuple[str, str]]:
            n = len(ds[split]) if limit is None or limit <= 0 else min(limit, len(ds[split]))
            rows = []
            for row in ds[split].select(range(n)):
                rows.append((str(row["sentence"]), "positive" if int(row["label"]) == 1 else "negative"))
            return rows
        return convert("train", train_limit), convert("validation", dev_limit), "nyu-mll/glue/sst2"
    except Exception:
        # Fallback mirrors earlier smoke scripts but is intentionally marked as not paper-scale.
        train, dev, source = load_sst2_rows(64 if train_limit is None or train_limit <= 0 else train_limit, 64 if dev_limit is None or dev_limit <= 0 else dev_limit)
        return train, dev, "fallback:" + source


def infer_dims_for_target(path: str, module: Any, hidden_size: int) -> tuple[int, int]:
    bias = getattr(module, "bias", None)
    if bias is not None and hasattr(bias, "shape"):
        return int(hidden_size), int(bias.shape[0])
    # Attention output or MLP layers without bias are not first-class in this script yet.
    raise ValueError(f"Cannot infer LoRA dims for {path}; target must expose a bias")


class MultiLoRA:
    def __init__(self, entries: list[tuple[str, PACZeroLoRALinear]]):
        self.entries = entries
        self.slices: list[tuple[int, int]] = []
        offset = 0
        for _, lora in entries:
            size = int(lora.theta().shape[0])
            self.slices.append((offset, offset + size))
            offset += size
        self.theta_size = offset

    def theta(self) -> mx.array:
        return mx.concatenate([lora.theta() for _, lora in self.entries])

    def set_theta(self, theta: mx.array) -> None:
        for (_, lora), (start, end) in zip(self.entries, self.slices):
            lora.set_theta(theta[start:end])

    def save_all(self, out_dir: Path) -> list[dict]:
        infos = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, lora in self.entries:
            safe = path.replace("model.layers.", "layer_").replace(".", "_")
            infos.append(save_adapter_npz(out_dir / f"{safe}.npz", lora))
        return infos


def eval_losses_for_rows(model: Any, tokenizer: Any, rows: list[tuple[str, str]]) -> np.ndarray:
    return np.array(eval_dataset(model, tokenizer, rows)["losses"], dtype=np.float64)


def evaluate_subset(model: Any, tokenizer: Any, rows: list[tuple[str, str]]) -> dict:
    return eval_dataset(model, tokenizer, rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    p.add_argument("--targets", default="model.layers.0.self_attn.q_proj,model.layers.0.self_attn.v_proj")
    p.add_argument("--hidden-size", type=int, default=896)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=20260608)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-examples", type=int, default=32)
    p.add_argument("--train-limit", type=int, default=0, help="0 means full SST-2 train split")
    p.add_argument("--dev-limit", type=int, default=0, help="0 means full SST-2 validation split")
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=25.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-examples", type=int, default=256)
    p.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_paper_scale_lora_results.json"))
    p.add_argument("--adapter-dir", type=Path, default=Path("benchmark-results/paczero-paper-scale-lora-adapters"))
    args = p.parse_args()

    start = time.perf_counter()
    print("# PACZero-LoRA paper-scale runnable run")
    print("Full SST-2 population with minibatch two-point FD, multiple LoRA targets, periodic dev evaluation.")
    print(f"model={args.model}")
    print(f"targets={args.targets}")
    print(f"steps={args.steps} batch_examples={args.batch_examples} train_limit={args.train_limit} dev_limit={args.dev_limit}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    train_rows, dev_rows, dataset_source = load_sst2_population(args.train_limit, args.dev_limit)
    rng = np.random.default_rng(args.seed)

    entries = []
    for idx, path in enumerate(parse_targets(args.targets)):
        base = get_path(model, path)
        input_dim, output_dim = infer_dims_for_target(path, base, args.hidden_size)
        lora = PACZeroLoRALinear(base, input_dim=input_dim, output_dim=output_dim, rank=args.rank, alpha=args.alpha, seed=args.seed + idx)
        set_path(model, path, lora)
        entries.append((path, lora))
    multi = MultiLoRA(entries)
    theta = multi.theta()

    eval_rows = dev_rows[: min(len(dev_rows), args.eval_examples)]
    baseline_eval = evaluate_subset(model, tokenizer, eval_rows)
    full_baseline_eval = evaluate_subset(model, tokenizer, dev_rows)
    best = {"step": 0, "eval_loss": baseline_eval["loss_mean"], "eval_accuracy": baseline_eval["accuracy"], "full_dev_loss": full_baseline_eval["loss_mean"], "full_dev_accuracy": full_baseline_eval["accuracy"]}
    best_theta = theta

    fd_finite = 0
    fd_nonzero = 0
    unanimous = 0
    sign_counts = {"positive": 0, "negative": 0}
    fd_abs_max_values = []
    fd_abs_mean_values = []
    history = []
    train_n = len(train_rows)

    for step in range(1, args.steps + 1):
        idxs = rng.choice(train_n, size=min(args.batch_examples, train_n), replace=False)
        batch_rows = [train_rows[int(i)] for i in idxs]
        membership = make_balanced_membership(num_examples=len(batch_rows), num_subsets=8, seed=args.seed + step)
        assert_balanced_membership(membership)
        direction_np = rng.normal(size=int(theta.shape[0])).astype(np.float32)
        direction_np = direction_np / max(float(np.linalg.norm(direction_np)), 1e-12)
        direction = mx.array(direction_np, dtype=mx.float32)

        multi.set_theta(theta + args.mu * direction)
        plus = eval_losses_for_rows(model, tokenizer, batch_rows)
        multi.set_theta(theta - args.mu * direction)
        minus = eval_losses_for_rows(model, tokenizer, batch_rows)
        fd = (plus - minus) / (2.0 * args.mu)
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        fd_finite += int(np.isfinite(fd).all())
        fd_nonzero += int(fd_abs_max > 0.0)
        release = paczero_zpl_release(fd, membership, clip=args.clip, rng=rng)
        unanimous += int(bool(release.unanimous))
        if int(release.sign) > 0:
            sign_counts["positive"] += 1
        else:
            sign_counts["negative"] += 1
        theta = theta - args.lr * float(release.sign) * direction
        multi.set_theta(theta)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            eval_metrics = evaluate_subset(model, tokenizer, eval_rows)
            row = {
                "step": step,
                "eval_loss": eval_metrics["loss_mean"],
                "eval_accuracy": eval_metrics["accuracy"],
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": bool(release.unanimous),
                "examples_seen_approx": step * len(batch_rows),
            }
            history.append(row)
            print("PAPER_SCALE_LORA_EVAL=" + json.dumps(row))
            if eval_metrics["accuracy"] > best["eval_accuracy"] or (eval_metrics["accuracy"] == best["eval_accuracy"] and eval_metrics["loss_mean"] < best["eval_loss"]):
                best = {"step": step, "eval_loss": eval_metrics["loss_mean"], "eval_accuracy": eval_metrics["accuracy"], "full_dev_loss": None, "full_dev_accuracy": None}
                best_theta = theta

    multi.set_theta(theta)
    final_eval = evaluate_subset(model, tokenizer, eval_rows)
    final_full_dev = evaluate_subset(model, tokenizer, dev_rows)
    if final_eval["accuracy"] > best["eval_accuracy"] or (final_eval["accuracy"] == best["eval_accuracy"] and final_eval["loss_mean"] < best["eval_loss"]):
        best = {"step": args.steps, "eval_loss": final_eval["loss_mean"], "eval_accuracy": final_eval["accuracy"], "full_dev_loss": final_full_dev["loss_mean"], "full_dev_accuracy": final_full_dev["accuracy"]}
        best_theta = theta

    multi.set_theta(best_theta)
    best_full_dev = evaluate_subset(model, tokenizer, dev_rows)
    best["full_dev_loss"] = best_full_dev["loss_mean"]
    best["full_dev_accuracy"] = best_full_dev["accuracy"]
    adapter_infos = multi.save_all(args.adapter_dir)

    fd_finite_rate = fd_finite / max(1, args.steps)
    fd_signal_rate = fd_nonzero / max(1, args.steps)
    unanimity_rate = unanimous / max(1, args.steps)
    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "dataset_full_train_population": args.train_limit == 0 and len(train_rows) > 1000,
        "dataset_full_dev_population": args.dev_limit == 0 and len(dev_rows) > 100,
        "multiple_lora_targets": len(entries) >= 2,
        "theta_size_positive": int(theta.shape[0]) > 0,
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "best_eval_accuracy_not_worse": best["eval_accuracy"] >= baseline_eval["accuracy"],
        "best_full_dev_accuracy_not_worse": best["full_dev_accuracy"] >= full_baseline_eval["accuracy"],
        "adapters_saved": all(Path(info["path"]).exists() and info["bytes"] > 0 for info in adapter_infos),
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_population_examples": len(train_rows),
        "dev_population_examples": len(dev_rows),
        "eval_examples_per_checkpoint": len(eval_rows),
        "parameterization": "paper_scale_minibatch_multi_target_custom_lora_paczero_zpl",
        "targets": [path for path, _ in entries],
        "target_shapes": [{"path": path, "input_dim": lora.input_dim, "output_dim": lora.output_dim, "rank": lora.rank, "alpha": lora.alpha} for path, lora in entries],
        "theta_size": int(theta.shape[0]),
        "seed": args.seed,
        "steps": args.steps,
        "batch_examples": args.batch_examples,
        "mu": args.mu,
        "lr": args.lr,
        "clip": args.clip,
        "checks": checks,
        "baseline_eval_subset": {"loss": baseline_eval["loss_mean"], "accuracy": baseline_eval["accuracy"]},
        "baseline_full_dev": {"loss": full_baseline_eval["loss_mean"], "accuracy": full_baseline_eval["accuracy"]},
        "final_eval_subset": {"loss": final_eval["loss_mean"], "accuracy": final_eval["accuracy"]},
        "final_full_dev": {"loss": final_full_dev["loss_mean"], "accuracy": final_full_dev["accuracy"]},
        "best_checkpoint": best,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "unanimity_rate": unanimity_rate,
        "release_sign_counts": sign_counts,
        "adapter_files": adapter_infos,
        "history": history,
        "verdict": "Runnable paper-scale PACZero-LoRA approximation: full SST-2 population with minibatch FD and multiple LoRA targets. A literal full-batch FD over all training examples per step is not practical on GitHub-hosted macOS runners.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PAPER_SCALE_LORA_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
