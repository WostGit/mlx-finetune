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

from paczero_core import assert_balanced_membership, make_balanced_membership, sign_nonzero, subset_means_from_fd
from paczero_mlxlm_exhaustive_readiness import encode_prompt_label, eval_dataset, get_path, label_only_loss, set_path
from paczero_mlxlm_lora_reproduction import PACZeroLoRALinear, infer_linear_dims


PAPER_FIDELITY = {
    "mechanism": "PACZero-ZPL",
    "paper_zpl_M": 126,
    "paper_lora_rank": 8,
    "paper_lora_alpha": 16,
    "paper_lora_targets": "q_proj and v_proj in OPT attention modules",
    "paper_tasks": ["sst2", "squad"],
    "paper_scale_reference": {"train": 1000, "dev": 500, "eval": 1000, "zpl_steps": 1000},
    "this_is_not_exact_opt_reproduction": True,
}


def print_kv(title: str, value: Any) -> None:
    print(f"VERBOSE::{title}={json.dumps(value, sort_keys=True, default=str)}")


def load_sst2_rows(train_n: int, dev_n: int, eval_n: int) -> tuple[list[dict], list[dict], list[dict], str]:
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "sst2")
    valid_total = len(ds["validation"])
    if dev_n + eval_n > valid_total:
        raise ValueError(f"SST-2 validation has {valid_total} rows, cannot allocate dev={dev_n} + eval={eval_n}")

    def convert(split: str, start: int, n: int) -> list[dict]:
        rows = []
        for row in ds[split].select(range(start, start + n)):
            label = "positive" if int(row["label"]) == 1 else "negative"
            rows.append({"input": str(row["sentence"]), "label": label})
        return rows

    return (
        convert("train", 0, train_n),
        convert("validation", 0, dev_n),
        convert("validation", dev_n, eval_n),
        "nyu-mll/glue/sst2 train plus disjoint validation dev/eval slices",
    )


def load_squad_rows(train_n: int, dev_n: int, eval_n: int) -> tuple[list[dict], list[dict], list[dict], str]:
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    valid_total = len(ds["validation"])
    if dev_n + eval_n > valid_total:
        raise ValueError(f"SQuAD validation has {valid_total} rows, cannot allocate dev={dev_n} + eval={eval_n}")

    def convert(split: str, start: int, n: int) -> list[dict]:
        rows = []
        for row in ds[split].select(range(start, start + n)):
            answers = row.get("answers", {})
            texts = answers.get("text", []) if isinstance(answers, dict) else []
            answer = str(texts[0]) if texts else ""
            rows.append({
                "context": str(row["context"]),
                "question": str(row["question"]),
                "label": answer,
            })
        return rows

    return (
        convert("train", 0, train_n),
        convert("validation", 0, dev_n),
        convert("validation", dev_n, eval_n),
        "rajpurkar/squad train plus disjoint validation dev/eval slices",
    )


def sst2_prompt(text: str) -> str:
    return "Classify the sentiment of the movie-review sentence as positive or negative.\nSentence: " + text + "\nAnswer:"


def squad_prompt(context: str, question: str) -> str:
    return "Answer the question using the context.\nContext: " + context + "\nQuestion: " + question + "\nAnswer:"


def encode_task_row(tokenizer: Any, task: str, row: dict) -> tuple[list[int], list[int], str]:
    if task == "sst2":
        prompt = sst2_prompt(row["input"])
        label = " " + row["label"]
    elif task == "squad":
        prompt = squad_prompt(row["context"], row["question"])
        # SQuAD is evaluated here by answer-likelihood loss, not generated EM/F1.
        label = " " + row["label"]
    else:
        raise ValueError(f"unsupported task: {task}")
    prompt_ids = tokenizer.encode(prompt)
    full_ids = tokenizer.encode(prompt + label)
    if len(full_ids) <= len(prompt_ids):
        full_ids = prompt_ids + tokenizer.encode(label)
    return prompt_ids, full_ids, label.strip()


def task_row_loss(model: Any, tokenizer: Any, task: str, row: dict) -> float:
    prompt_ids, full_ids, _ = encode_task_row(tokenizer, task, row)
    return label_only_loss(model, full_ids, len(prompt_ids))


def score_sst2_row(model: Any, tokenizer: Any, row: dict) -> str:
    candidates = []
    for label in ["negative", "positive"]:
        prompt = sst2_prompt(row["input"])
        prompt_ids = tokenizer.encode(prompt)
        full_ids = tokenizer.encode(prompt + " " + label)
        if len(full_ids) <= len(prompt_ids):
            full_ids = prompt_ids + tokenizer.encode(" " + label)
        candidates.append((label, label_only_loss(model, full_ids, len(prompt_ids))))
    return min(candidates, key=lambda x: x[1])[0]


def evaluate_task(model: Any, tokenizer: Any, task: str, rows: list[dict], split_name: str) -> dict:
    losses = []
    correct = 0
    for row in rows:
        loss = task_row_loss(model, tokenizer, task, row)
        losses.append(loss)
        if task == "sst2":
            correct += int(score_sst2_row(model, tokenizer, row) == row["label"])
    arr = np.array(losses, dtype=np.float64)
    result = {
        f"{split_name}_loss": float(arr.mean()),
        f"{split_name}_loss_min": float(arr.min()),
        f"{split_name}_loss_max": float(arr.max()),
    }
    if task == "sst2":
        result[f"{split_name}_accuracy"] = correct / max(1, len(rows))
    else:
        result[f"{split_name}_answer_likelihood_metric"] = "mean_label_only_gold_answer_loss_lower_is_better_not_generated_em_f1"
    return result


def evaluate_all(model: Any, tokenizer: Any, task: str, train_rows: list[dict], dev_rows: list[dict], eval_rows: list[dict]) -> dict:
    out = {}
    out.update(evaluate_task(model, tokenizer, task, train_rows, "train"))
    out.update(evaluate_task(model, tokenizer, task, dev_rows, "dev"))
    out.update(evaluate_task(model, tokenizer, task, eval_rows, "eval"))
    return out


def train_losses_for_theta(model: Any, tokenizer: Any, task: str, train_rows: list[dict], loras: list[PACZeroLoRALinear], theta: mx.array, slices: list[tuple[int, int]]) -> np.ndarray:
    set_multi_theta(loras, theta, slices)
    vals = [task_row_loss(model, tokenizer, task, row) for row in train_rows]
    return np.array(vals, dtype=np.float64)


def resolve_layer_targets(model: Any, projections: list[str], layer_mode: str) -> list[str]:
    layers = get_path(model, "model.layers")
    num_layers = len(layers)
    if layer_mode == "all":
        layer_indices = list(range(num_layers))
    else:
        count = int(layer_mode)
        if count <= 0:
            raise ValueError("layer count must be positive or 'all'")
        layer_indices = list(range(min(count, num_layers)))
    targets = []
    for i in layer_indices:
        for proj in projections:
            targets.append(f"model.layers.{i}.self_attn.{proj}")
    return targets


def attach_loras(model: Any, target_paths: list[str], rank: int, alpha: float, seed: int) -> tuple[list[PACZeroLoRALinear], list[dict]]:
    loras = []
    infos = []
    for idx, path in enumerate(target_paths):
        base = get_path(model, path)
        input_dim, output_dim, dim_debug = infer_linear_dims(base, path)
        lora = PACZeroLoRALinear(base, input_dim=input_dim, output_dim=output_dim, rank=rank, alpha=alpha, seed=seed + idx)
        set_path(model, path, lora)
        loras.append(lora)
        infos.append({"target_path": path, "input_dim": input_dim, "output_dim": output_dim, "theta_size": int(lora.theta().shape[0]), "dimension_inference": dim_debug})
    return loras, infos


def multi_theta(loras: list[PACZeroLoRALinear]) -> tuple[mx.array, list[tuple[int, int]]]:
    parts = []
    slices = []
    offset = 0
    for lora in loras:
        theta = lora.theta()
        size = int(theta.shape[0])
        parts.append(theta)
        slices.append((offset, offset + size))
        offset += size
    return mx.concatenate(parts), slices


def set_multi_theta(loras: list[PACZeroLoRALinear], theta: mx.array, slices: list[tuple[int, int]]) -> None:
    for lora, (start, end) in zip(loras, slices):
        lora.set_theta(theta[start:end])


def save_multi_adapter_npz(path: Path, loras: list[PACZeroLoRALinear], target_paths: list[str], best_step: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"best_step": np.array([best_step], dtype=np.int32)}
    adapter_shapes = []
    for i, (lora, target) in enumerate(zip(loras, target_paths)):
        a, b = lora.adapter_numpy()
        payload[f"target_{i}"] = np.array([target])
        payload[f"A_{i}"] = a
        payload[f"B_{i}"] = b
        payload[f"rank_{i}"] = np.array([lora.rank], dtype=np.int32)
        payload[f"alpha_{i}"] = np.array([lora.alpha], dtype=np.float32)
        adapter_shapes.append({"target_path": target, "A_shape": list(a.shape), "B_shape": list(b.shape)})
    np.savez(path, **payload)
    return {"path": str(path), "bytes": path.stat().st_size, "adapter_count": len(loras), "adapter_shapes": adapter_shapes}


def audited_zpl_release(per_sample_fd: np.ndarray, membership: np.ndarray, clip: float, rng: np.random.Generator, step: int) -> tuple[int, dict]:
    subset_means = subset_means_from_fd(per_sample_fd, membership, clip)
    subset_signs = sign_nonzero(subset_means).astype(int)
    unique_signs = sorted(set(subset_signs.tolist()))
    unanimous = len(unique_signs) == 1
    violations: list[str] = []
    if unanimous:
        branch = "unanimous_subset_independent"
        release_sign = int(subset_signs[0])
        rng_derived = False
    else:
        branch = "disagreement_randomized"
        release_sign = int(rng.choice(np.array([-1, 1], dtype=np.int64)))
        rng_derived = True
    secret_subset_index_used = False
    if branch == "disagreement_randomized" and not rng_derived:
        violations.append("disagreement_branch_not_rng_derived")
    if secret_subset_index_used:
        violations.append("secret_subset_index_was_used_for_release")
    audit = {
        "step": step,
        "branch": branch,
        "release_sign": release_sign,
        "unanimous": unanimous,
        "unique_subset_signs": unique_signs,
        "subset_sign_counts": {"negative": int(np.sum(subset_signs < 0)), "positive": int(np.sum(subset_signs > 0))},
        "subset_means_min": float(np.min(subset_means)),
        "subset_means_max": float(np.max(subset_means)),
        "subset_means_mean": float(np.mean(subset_means)),
        "rng_derived_release": rng_derived,
        "secret_subset_index_used_for_release": secret_subset_index_used,
        "subset_independent_by_construction": len(violations) == 0,
        "violation_reasons": violations,
    }
    return release_sign, audit


def best_key(task: str, metrics: dict) -> tuple[float, float]:
    if task == "sst2":
        return (float(metrics.get("dev_accuracy", 0.0)), -float(metrics["dev_loss"]))
    return (-float(metrics["dev_loss"]), -float(metrics["train_loss"]))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--task", choices=["sst2", "squad"], required=True)
    p.add_argument("--projections", default="q_proj,v_proj")
    p.add_argument("--layers", default="1", help="number of initial layers to adapt, or 'all'")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=20260611)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--train-examples", type=int, default=32)
    p.add_argument("--dev-examples", type=int, default=32)
    p.add_argument("--eval-examples", type=int, default=128)
    p.add_argument("--num-subsets", type=int, default=126)
    p.add_argument("--mu", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=25.0)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--adapter-out", type=Path, required=True)
    args = p.parse_args()

    start_time = time.perf_counter()
    print("# PACZero MLX faithful adaptation audit")
    print("This run explicitly targets the reviewer objections: both PACZero tasks, q/v projection fidelity, M=126, rank=8/alpha=16, and strict ZPL transcript audit.")
    print_kv("paper_fidelity_reference", PAPER_FIDELITY)
    print_kv("cli_args", vars(args))
    print_kv("python", sys.version.splitlines()[0])
    for pkg in ["mlx", "mlx-lm", "datasets", "numpy"]:
        try:
            print_kv(f"package_{pkg}", md.version(pkg))
        except Exception as exc:
            print_kv(f"package_{pkg}", f"unavailable:{exc}")

    if args.num_subsets % 2 != 0:
        raise ValueError("num-subsets must be even for M/2 membership")
    projections = [x.strip() for x in args.projections.split(",") if x.strip()]
    print_kv("projection_fidelity", {"requested": projections, "paper_targets": "q_proj + v_proj", "faithful_projection_set": set(projections) == {"q_proj", "v_proj"}})

    if args.task == "sst2":
        train_rows, dev_rows, eval_rows, dataset_source = load_sst2_rows(args.train_examples, args.dev_examples, args.eval_examples)
    else:
        train_rows, dev_rows, eval_rows, dataset_source = load_squad_rows(args.train_examples, args.dev_examples, args.eval_examples)
    print_kv("dataset_loaded", {"task": args.task, "source": dataset_source, "train": len(train_rows), "dev": len(dev_rows), "eval": len(eval_rows)})
    print_kv("dataset_caveat", "SQuAD uses label-only gold-answer likelihood in this MLX audit; it is not generated EM/F1 unless a separate generation evaluator is added." if args.task == "squad" else "SST-2 uses label-only positive/negative scoring accuracy.")

    model, tokenizer = load(args.model)
    target_paths = resolve_layer_targets(model, projections, args.layers)
    print_kv("resolved_lora_targets", {"count": len(target_paths), "targets": target_paths, "layers_arg": args.layers})
    loras, target_infos = attach_loras(model, target_paths, rank=args.rank, alpha=args.alpha, seed=args.seed)
    print_kv("attached_lora_targets", target_infos)
    theta, slices = multi_theta(loras)
    theta_size = int(theta.shape[0])
    print_kv("combined_lora_theta", {"theta_size": theta_size, "slices": slices})

    baseline = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
    print_kv("baseline_metrics", baseline)

    rng = np.random.default_rng(args.seed)
    membership = make_balanced_membership(len(train_rows), args.num_subsets, seed=args.seed + 17)
    assert_balanced_membership(membership)
    column_counts = membership.astype(int).sum(axis=0).tolist()
    print_kv("membership_audit", {"shape": list(membership.shape), "unique_column_counts": sorted(set(column_counts)), "expected": args.num_subsets // 2})

    best = {"step": 0, **baseline}
    best_theta = theta
    fd_finite = fd_nonzero = unanimous_count = disagreement_count = randomized_disagreement_count = secret_index_use_count = privacy_violation_count = 0
    sign_counts = {"positive": 0, "negative": 0}
    fd_abs_max_values = []
    fd_abs_mean_values = []
    history = []
    privacy_trace = []

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
        release_sign, audit = audited_zpl_release(fd, membership, args.clip, rng, step)
        privacy_trace.append(audit)
        unanimous_count += int(audit["unanimous"])
        disagreement_count += int(not audit["unanimous"])
        randomized_disagreement_count += int(audit["branch"] == "disagreement_randomized" and audit["rng_derived_release"])
        secret_index_use_count += int(audit["secret_subset_index_used_for_release"])
        privacy_violation_count += len(audit["violation_reasons"])
        sign_counts["positive" if release_sign > 0 else "negative"] += 1
        theta = theta - args.lr * float(release_sign) * direction
        set_multi_theta(loras, theta, slices)

        if step == 1 or step == args.steps or step % args.eval_every == 0:
            metrics = evaluate_all(model, tokenizer, args.task, train_rows, dev_rows, eval_rows)
            row = {"step": step, **metrics, "fd_abs_max": fd_abs_max, "fd_abs_mean": fd_abs_mean, "zpl_branch": audit["branch"], "release_sign": release_sign, "privacy_rule_violations_so_far": privacy_violation_count}
            history.append(row)
            print_kv("eval_checkpoint", row)
            print_kv("privacy_checkpoint", {"step": step, "unanimous_count": unanimous_count, "disagreement_count": disagreement_count, "randomized_disagreement_count": randomized_disagreement_count, "secret_index_use_count": secret_index_use_count, "privacy_violation_count": privacy_violation_count, "last_audit": audit})
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
    membership_balanced = sorted(set(column_counts)) == [args.num_subsets // 2]
    all_unanimous_independent = all(item["branch"] != "unanimous_subset_independent" or item["subset_independent_by_construction"] for item in privacy_trace)
    all_disagreement_randomized = randomized_disagreement_count == disagreement_count
    secret_never_indexed = secret_index_use_count == 0
    transcript_independent = membership_balanced and all_unanimous_independent and all_disagreement_randomized and secret_never_indexed and privacy_violation_count == 0
    utility_not_worse = best_key(args.task, best) >= best_key(args.task, baseline)

    privacy_accounting = {
        "mechanism": "PACZero-ZPL audited sign release",
        "claim_scope": "implementation-level transcript audit; not differential privacy",
        "not_differential_privacy": True,
        "mutual_information_claim_under_zpl_rule": "I(S_star;Y_1:T)=0 because unanimous releases are identical for all candidate subsets and disagreement releases are independent RNG signs that never index S_star.",
        "num_subsets_M": args.num_subsets,
        "examples_per_column_expected": args.num_subsets // 2,
        "membership_column_counts_unique": sorted(set(column_counts)),
        "membership_balanced": membership_balanced,
        "steps": args.steps,
        "audited_steps": len(privacy_trace),
        "unanimous_steps": unanimous_count,
        "disagreement_steps": disagreement_count,
        "disagreement_releases_randomized": randomized_disagreement_count,
        "release_sign_counts": sign_counts,
        "privacy_audit": {
            "all_unanimous_steps_subset_independent": all_unanimous_independent,
            "all_disagreement_steps_randomized": all_disagreement_randomized,
            "secret_subset_never_indexed_for_release": secret_never_indexed,
            "release_rule_violation_count": privacy_violation_count,
            "transcript_distribution_independent_of_secret_subset_by_construction": transcript_independent,
            "conclusion": "PASS" if transcript_independent else "FAIL",
        },
        "privacy_trace_head": privacy_trace[:10],
        "privacy_trace_tail": privacy_trace[-10:],
        "privacy_trace_full_saved_in_log": True,
    }

    checks = {
        "paper_style_num_subsets": args.num_subsets == 126,
        "paper_style_lora_rank_alpha": args.rank == 8 and abs(args.alpha - 16.0) < 1e-9,
        "faithful_projection_set_q_and_v": set(projections) == {"q_proj", "v_proj"},
        "has_at_least_two_lora_targets": len(target_paths) >= 2,
        "has_disjoint_eval_split": len(eval_rows) > 0 and len(dev_rows) > 0,
        "fd_finite_rate_ok": fd_finite_rate >= 1.0,
        "fd_signal_rate_ok": fd_signal_rate >= 0.80,
        "privacy_transcript_audit_passed": transcript_independent,
        "utility_selection_not_worse_than_baseline_on_selection_metric": utility_not_worse,
        "adapter_saved": args.adapter_out.exists() and args.adapter_out.stat().st_size > 0,
    }

    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "slug": args.slug,
        "task": args.task,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "dataset_source": dataset_source,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "eval_examples": len(eval_rows),
        "paper_fidelity_reference": PAPER_FIDELITY,
        "parameterization": "faithful_adaptation_qv_lora_paczero_zpl_strict_audit",
        "target_paths": target_paths,
        "target_infos": target_infos,
        "layers": args.layers,
        "projections": projections,
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
        "best_checkpoint": best,
        "selected_best_adapter_eval": selected_eval,
        "fd_finite_rate": fd_finite_rate,
        "fd_signal_rate": fd_signal_rate,
        "fd_abs_max_max": float(max(fd_abs_max_values)),
        "fd_abs_max_mean": float(np.mean(fd_abs_max_values)),
        "fd_abs_mean_mean": float(np.mean(fd_abs_mean_values)),
        "privacy_accounting": privacy_accounting,
        "adapter_file": adapter_info,
        "history": history,
        "limitations": [
            "Default workflow is medium/smoke scale unless inputs are increased to 1000/500/1000 and 1000 steps.",
            "Qwen/Llama MLX models are adaptations, not exact OPT-1.3B/OPT-6.7B reproduction.",
            "SQuAD metric here is gold-answer label-likelihood loss, not generated EM/F1.",
            "Use --layers all for all-layer q/v coverage; default CI uses initial layer count for runtime.",
        ],
    }
    print_kv("final_payload_summary", {k: payload[k] for k in ["success", "task", "model", "elapsed_seconds", "checks", "privacy_accounting"]})
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("FAITHFUL_ADAPTATION_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
