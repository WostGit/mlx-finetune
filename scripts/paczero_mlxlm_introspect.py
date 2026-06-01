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
from mlx_lm import load


def brief_value(value: Any) -> dict:
    info: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        try:
            info["shape"] = list(shape)
        except Exception:
            info["shape"] = str(shape)
    if dtype is not None:
        info["dtype"] = str(dtype)
    return info


def safe_children(obj: Any) -> list[tuple[str, Any]]:
    children = []
    # MLX nn.Module stores child modules/arrays as attributes.  Use vars first,
    # then fall back to public dir entries.  Keep this defensive because module
    # internals change across mlx-lm versions.
    try:
        for name, value in vars(obj).items():
            if not name.startswith("_"):
                children.append((name, value))
    except Exception:
        pass
    if children:
        return children
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        children.append((name, value))
    return children


def walk_modules(root: Any, max_depth: int = 5, max_nodes: int = 400) -> list[dict]:
    rows = []
    seen = set()

    def rec(obj: Any, path: str, depth: int) -> None:
        if len(rows) >= max_nodes:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        row = {
            "path": path,
            "type": type(obj).__name__,
            "module": str(type(obj))[:200],
        }
        attrs = {}
        for attr in ["weight", "bias", "scales", "biases", "bits", "group_size", "input_dims", "output_dims"]:
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            attrs[attr] = brief_value(value)
        if attrs:
            row["attrs"] = attrs
        rows.append(row)
        if depth >= max_depth:
            return
        for name, child in safe_children(obj):
            if name in {"training"}:
                continue
            if isinstance(child, (str, int, float, bool, bytes, type(None))):
                continue
            child_type = type(child).__name__.lower()
            looks_module = any(k in child_type for k in ["linear", "model", "layer", "attention", "mlp", "embedding", "norm", "quantized"])
            has_weight = hasattr(child, "weight")
            is_container = isinstance(child, (list, tuple, dict))
            if is_container:
                items = child.items() if isinstance(child, dict) else enumerate(child)
                for k, v in list(items)[:80]:
                    if hasattr(v, "weight") or "layer" in type(v).__name__.lower() or "block" in type(v).__name__.lower():
                        rec(v, f"{path}.{name}.{k}", depth + 1)
            elif looks_module or has_weight:
                rec(child, f"{path}.{name}", depth + 1)

    rec(root, "model", 0)
    return rows


def pick_candidate_targets(rows: list[dict]) -> list[dict]:
    candidates = []
    for row in rows:
        lower_path = row["path"].lower()
        lower_type = row["type"].lower()
        if any(key in lower_path for key in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
            candidates.append(row)
        elif "linear" in lower_type or "quantized" in lower_type:
            candidates.append(row)
    return candidates[:80]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_introspect_results.json"))
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM introspection smoke")
    print("This discovers model/module paths for the next real adapter hook.")
    print(f"model={args.model}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    rows = walk_modules(model, max_depth=args.max_depth)
    candidates = pick_candidate_targets(rows)
    top_level = [name for name, _ in safe_children(model)[:80]]
    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "found_modules": len(rows) > 0,
        "found_candidate_targets": len(candidates) > 0,
    }
    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "checks": checks,
        "model_type": type(model).__name__,
        "tokenizer_type": type(tokenizer).__name__,
        "top_level_children": top_level,
        "module_count_sampled": len(rows),
        "candidate_target_count_sampled": len(candidates),
        "candidate_targets_head": candidates[:30],
        "modules_head": rows[:80],
        "note": "Use candidate target paths to choose a real layer for the first custom PACZero adapter perturbation smoke.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("INTROSPECT_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
