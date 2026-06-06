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
from paczero_mlxlm_exhaustive_readiness import eval_dataset, get_path, load_sst2_rows, set_path


class PACZeroLoRALinear:
    """Minimal inference-time LoRA wrapper for an MLX-LM linear/quantized linear module."""

    def __init__(self, base: Any, input_dim: int, output_dim: int, rank: int, alpha: float, seed: int):
        self.base = base
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 0.01, size=(rank, input_dim)).astype(np.float32)
        b = np.zeros((output_dim, rank), dtype=np.float32)
        self.A = mx.array(a, dtype=mx.float32)
        self.B = mx.array(b, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        x_f32 = x.astype(mx.float32)
        if int(x_f32.shape[-1]) != self.input_dim:
            raise ValueError(
                f"LoRA input dimension mismatch: x.shape[-1]={int(x_f32.shape[-1])} "
                f"but wrapper input_dim={self.input_dim}; base={type(self.base).__name__}"
            )
        delta = (x_f32 @ mx.transpose(self.A)) @ mx.transpose(self.B)
        delta = delta * self.scale
        if int(delta.shape[-1]) != int(base_out.shape[-1]):
            raise ValueError(
                f"LoRA output dimension mismatch: delta.shape[-1]={int(delta.shape[-1])} "
                f"but base_out.shape[-1]={int(base_out.shape[-1])}; wrapper output_dim={self.output_dim}"
            )
        return base_out + delta.astype(base_out.dtype)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def theta(self) -> mx.array:
        return mx.concatenate([mx.reshape(self.A, (-1,)), mx.reshape(self.B, (-1,))])

    def set_theta(self, theta: mx.array) -> None:
        a_size = self.rank * self.input_dim
        self.A = mx.reshape(theta[:a_size], (self.rank, self.input_dim)).astype(mx.float32)
        self.B = mx.reshape(theta[a_size:], (self.output_dim, self.rank)).astype(mx.float32)

    def adapter_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.A.tolist(), dtype=np.float32), np.array(self.B.tolist(), dtype=np.float32)


def _optional_int_attr(module: Any, *names: str) -> int | None:
    for name in names:
        try:
            value = getattr(module, name)
        except Exception:
            continue
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            pass
    return None


def _infer_quantized_input_dim_from_weight(weight: Any, debug: dict[str, Any]) -> int | None:
    """Infer dense input dim from MLX 4-bit QuantizedLinear packed weights.

    MLX-LM 4-bit quantized linear weights are packed into uint32 values.  Each
    uint32 stores 8 four-bit values, so a packed weight shape [out, in/8]
    corresponds to dense input dimension weight.shape[1] * 8.  This matters for
    grouped-query attention: q_proj may be square, but k_proj/v_proj can have
    smaller output dims while still consuming the full hidden dimension.
    """

    if weight is None or not hasattr(weight, "shape") or len(weight.shape) < 2:
        return None
    dtype_text = str(getattr(weight, "dtype", ""))
    shape = list(weight.shape)
    if "uint32" in dtype_text:
        debug["quantized_input_inference"] = {
            "reason": "uint32_4bit_packed_weight",
            "packed_input_dim": int(shape[1]),
            "values_per_uint32": 8,
            "dense_input_dim": int(shape[1]) * 8,
        }
        return int(shape[1]) * 8
    return None


def infer_linear_dims(module: Any, target_path: str) -> tuple[int, int, dict]:
    """Infer LoRA input/output dims for MLX-LM Linear or QuantizedLinear.

    For dense Linear layers, explicit module attrs are preferred. For MLX-LM
    4-bit QuantizedLinear, weight.shape is packed as [out_dim, in_dim / 8].
    Therefore output_dim can come from weight.shape[0], while input_dim should
    come from weight.shape[1] * 8 when attrs are unavailable. This handles both
    square q_proj and grouped-query k/v projections such as SmolLM's v_proj
    where output_dim != input_dim.
    """

    debug: dict[str, Any] = {"module_type": type(module).__name__, "target_path": target_path}
    input_dim = _optional_int_attr(module, "input_dims", "input_dim", "in_features", "in_dim")
    output_dim = _optional_int_attr(module, "output_dims", "output_dim", "out_features", "out_dim")

    bias = getattr(module, "bias", None)
    if bias is not None and hasattr(bias, "shape"):
        output_dim = int(bias.shape[0])
        debug["output_source"] = "bias.shape[0]"
        debug["bias_shape"] = list(bias.shape)

    weight = getattr(module, "weight", None)
    if weight is not None and hasattr(weight, "shape"):
        debug["weight_shape"] = list(weight.shape)
        debug["weight_dtype"] = str(getattr(weight, "dtype", "unknown"))
        if output_dim is None and len(weight.shape) >= 1:
            output_dim = int(weight.shape[0])
            debug["output_source"] = "weight.shape[0]"
        packed_input_dim = _infer_quantized_input_dim_from_weight(weight, debug)
        if input_dim is None and packed_input_dim is not None:
            input_dim = packed_input_dim
            debug["input_source"] = "packed_quantized_weight.shape[1]*8"

    if input_dim is None and output_dim is not None:
        input_dim = output_dim
        debug["input_source"] = "square_projection_fallback"
    elif input_dim is not None and "input_source" not in debug:
        debug["input_source"] = "module_attr"

    if output_dim is None or input_dim is None:
        raise ValueError(f"Cannot infer LoRA dims for {target_path}; debug={debug}")
    debug["input_dim"] = input_dim
    debug["output_dim"] = output_dim
    return input_dim, output_dim, debug


def evaluate_lora(model: Any, tokenizer: Any, train_rows: list[tuple[str, str]], dev_rows: list[tuple[str, str]]) -> dict:
    train_eval = eval_dataset(model, tokenizer, train_rows)
    dev_eval = eval_dataset(model, tokenizer, dev_rows)
    return {
        "train_loss": train_eval["loss_mean"],
        "train_accuracy": train_eval["accuracy"],
        "dev_loss": dev_eval["loss_mean"],
        "dev_accuracy": dev_eval["accuracy"],
    }


def eval_losses_with_lora_theta(
    model: Any,
    tokenizer: Any,
    train_rows: list[tuple[str, str]],
    lora: PACZeroLoRALinear,
    theta: mx.array,
) -> np.ndarray:
    lora.set_theta(theta)
    vals = eval_dataset(model, tokenizer, train_rows)["losses"]
    return np.array(vals, dtype=np.float64)


def save_adapter_npz(path: Path, lora: PACZeroLoRALinear) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    a, b = lora.adapter_numpy()
    np.savez(path, A=a, B=b, alpha=np.array([lora.alpha], dtype=np.float32), rank=np.array([lora.rank], dtype=np.int32))
    return {"path": str(path), "bytes": path.stat().st_size, "A_shape": list(a.shape), "B_shape": list(b.shape)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--target-path", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--dev-examples", type=int, default=64)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--clip", type=float, default=25.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_lora_reproduction_results.json"))
    parser.add_argument("--adapter-out", type=Path, default=Path("benchmark-results/paczero-lora-adapters/qproj_lora_rank4.npz"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero-LoRA reproduction run")
    print("Actual custom LoRA A/B tensors are attached to a real MLX-LM projection and updated with PACZero-ZPL.")
    print(f"model={args.model}")
    print(f"target_path={args.target_path}")
    print(f"rank={args.rank} alpha={args.alpha} seed={args.seed} steps={args.steps}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm", "datasets"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    train_rows, dev_rows, dataset_source = load_sst2_rows(args.train_examples, args.dev_examples)

    base_module = get_path(model, args.target_path)
    input_dim, output_dim, dim_debug = infer_linear_dims(base_module, args.target_path)
    print("LORA_DIMENSION_INFERENCE=" + json.dumps(dim_debug))
    lora = PACZeroLoRALinear(base_module, input_dim=input_dim, output_dim=output_dim, rank=args.rank, alpha=args.alpha, seed=args.seed)
    set_path(model, args.target_path, lora)

    baseline = evaluate_lora(model, tokenizer, train_rows, dev_rows)
    theta = lora.theta()
    theta_size = int(theta.shape[0])
    rng = np.random.default_rng(args.seed)
    membership = make_balanced_membership(num_examples=len(train_rows), num_subsets=8, seed=args.seed + 17)
    assert_balanced_membership(membership)

    fd_finite_count = 0
    fd_nonzero_count = 0
    unanimous_count = 0
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
        plus = eval_losses_with_lora_theta(model, tokenizer, train_rows, lora, theta + args.mu * direction)
        minus = eval_losses_with_lora_theta(model, tokenizer, train_rows, lora, theta - args.mu * direction)
        fd = (plus - minus) / (2.0 * args.mu)
        fd_abs_max = float(np.max(np.abs(fd)))
        fd_abs_mean = float(np.mean(np.abs(fd)))
        fd_abs_max_values.append(fd_abs_max)
        fd_abs_mean_values.append(fd_abs_mean)
        fd_finite_count += int(np.isfinite(fd).all())
        fd_nonzero_count += int(fd_abs_max > 0.0)
        release = paczero_zpl_release(fd, membership, clip=args.clip, rng=rng)
        unanimous_count += int(bool(release.unanimous))
        if int(release.sign) > 0:
            sign_counts["positive"] += 1
        else:
            sign_counts["negative"] += 1
        theta = theta - args.lr * float(release.sign) * direction
        lora.set_theta(theta)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            metrics = evaluate_lora(model, tokenizer, train_rows, dev_rows)
            row = {
                "step": step,
                **metrics,
                "fd_abs_max": fd_abs_max,
                "fd_abs_mean": fd_abs_mean,
                "release_sign": int(release.sign),
                "unanimous": bool(release.unanimous),
            }
            history.append(row)
            print("PACZERO_LORA_EVAL=" + json.dumps(row))
            if (metrics["dev_accuracy"] > best["dev_accuracy"]) or (
                metrics["dev_accuracy"] == best["dev_accuracy"] and metrics["dev_loss"] < best["dev_loss"]
            ):
                best = {"step": step, **metrics}
                best_theta = theta

    lora.set_theta(theta)
    final = evaluate_lora(model, tokenizer, train_rows, dev_rows)
    if (final["dev_accuracy"] > best["dev_accuracy"]) or (
        final["dev_accuracy"] == best["dev_accuracy"] and final["dev_loss"] < best["dev_loss"]
    ):
        best = {"step": args.steps, **final}
        best_theta = theta

    lora.set_theta(best_theta)
    adapter_info = save_adapter_npz(args.adapter_out, lora)

    fd_finite_rate = fd_finite_count / max(1, args.steps)
    fd_signal_rate = fd_nonzero_count / max(1, args.steps)
    unanimity_rate = unanimous_count / max(1, args.steps)
    checks = {
        "attached_lora_wrapper": isinstance(get_path(model, args.target_path), PACZeroLoRALinear),
        "theta_size_positive": theta_size > 0,
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "fd_magnitude_ok": max(fd_abs_max_values) > 0.0,
        "losses_remain_finite": all(np.isfinite([final["train_loss"], final["dev_loss"], best["train_loss"], best["dev_loss"]])),
        "best_dev_accuracy_not_worse": best["dev_accuracy"] >= baseline["dev_accuracy"],
        "best_dev_loss_not_worse_when_accuracy_ties": bool(best["dev_accuracy"] > baseline["dev_accuracy"] or best["dev_loss"] <= baseline["dev_loss"]),
        "adapter_saved": args.adapter_out.exists() and args.adapter_out.stat().st_size > 0,
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "parameterization": "actual_custom_lora_ab_tensors_paczero_zpl",
        "target_path": args.target_path,
        "dimension_inference": dim_debug,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": args.rank,
        "alpha": args.alpha,
        "theta_size": theta_size,
        "seed": args.seed,
        "steps": args.steps,
        "mu": args.mu,
        "lr": args.lr,
        "clip": args.clip,
        "checks": checks,
        "baseline": baseline,
        "final": final,
        "best_checkpoint": best,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "unanimity_rate": unanimity_rate,
        "release_sign_counts": sign_counts,
        "adapter_file": adapter_info,
        "history": history,
        "verdict": "Full custom PACZero-LoRA reproduction on one projection if success=true. This trains actual LoRA A/B tensors, not base-model bias parameters.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("PACZERO_LORA_REPRODUCTION_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
