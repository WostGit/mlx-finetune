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
    try:
        if isinstance(value, (list, tuple, dict)):
            info["len"] = len(value)
    except Exception:
        pass
    return info


def is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def flatten_tree(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if is_array_like(tree):
        rows.append((prefix or "<root>", tree))
    elif isinstance(tree, dict):
        for k, v in tree.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(flatten_tree(v, p))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            p = f"{prefix}.{i}" if prefix else str(i)
            rows.extend(flatten_tree(v, p))
    else:
        try:
            items = list(tree.items())
        except Exception:
            items = []
        for k, v in items:
            p = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(flatten_tree(v, p))
    return rows


def call_noarg(obj: Any, name: str) -> Any:
    try:
        fn = getattr(obj, name)
    except Exception:
        return None
    if not callable(fn):
        return None
    try:
        return fn()
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def collect_api_outputs(model: Any) -> dict:
    outputs = {}
    for name in [
        "parameters",
        "trainable_parameters",
        "leaf_modules",
        "children",
        "modules",
        "named_modules",
        "state",
    ]:
        out = call_noarg(model, name)
        if out is None:
            outputs[name] = {"available": False}
        elif isinstance(out, dict) and "__error__" in out:
            outputs[name] = {"available": True, "error": out["__error__"]}
        else:
            flat = flatten_tree(out)
            outputs[name] = {
                "available": True,
                "type": type(out).__name__,
                "flat_array_count": len(flat),
                "flat_arrays_head": [
                    {"path": p, **brief_value(v)} for p, v in flat[:80]
                ],
            }
    return outputs


def safe_attr_names(obj: Any) -> list[str]:
    names = set()
    try:
        names.update(vars(obj).keys())
    except Exception:
        pass
    try:
        names.update(dir(obj))
    except Exception:
        pass
    return sorted(names)


def collect_attrs(obj: Any, path: str, max_depth: int, max_nodes: int) -> list[dict]:
    rows = []
    seen = set()

    def rec(value: Any, current: str, depth: int) -> None:
        if len(rows) >= max_nodes:
            return
        oid = id(value)
        if oid in seen:
            return
        seen.add(oid)
        row = {"path": current, "type": type(value).__name__, "module": str(type(value))[:200]}
        interesting_attrs = {}
        for attr in ["weight", "bias", "scales", "biases", "bits", "group_size", "input_dims", "output_dims", "layers", "model", "self_attn", "mlp"]:
            try:
                attr_value = getattr(value, attr)
            except Exception:
                continue
            interesting_attrs[attr] = brief_value(attr_value)
        if interesting_attrs:
            row["attrs"] = interesting_attrs
        rows.append(row)
        if depth >= max_depth:
            return

        names = safe_attr_names(value)
        # Include private names too because MLX nn.Module may keep registered
        # modules in hidden containers rather than public __dict__ entries.
        priority = [
            n for n in names
            if any(k in n.lower() for k in ["module", "layer", "model", "attn", "mlp", "proj", "linear", "children", "modules", "parameters"])
        ]
        for name in priority[:120]:
            if name in {"__class__", "__dict__", "__doc__", "__module__", "__weakref__"}:
                continue
            try:
                child = getattr(value, name)
            except Exception:
                continue
            if callable(child) or isinstance(child, (str, int, float, bool, bytes, type(None))):
                continue
            if is_array_like(child):
                rows.append({"path": f"{current}.{name}", "type": type(child).__name__, "module": str(type(child))[:200], "array": brief_value(child)})
                continue
            if isinstance(child, dict):
                for k, v in list(child.items())[:80]:
                    if not isinstance(v, (str, int, float, bool, bytes, type(None))):
                        rec(v, f"{current}.{name}.{k}", depth + 1)
            elif isinstance(child, (list, tuple)):
                for i, v in list(enumerate(child))[:80]:
                    if not isinstance(v, (str, int, float, bool, bytes, type(None))):
                        rec(v, f"{current}.{name}.{i}", depth + 1)
            else:
                rec(child, f"{current}.{name}", depth + 1)

    rec(obj, path, 0)
    return rows


def pick_candidate_targets(parameters_flat: list[dict], attr_rows: list[dict]) -> list[dict]:
    candidates = []
    keys = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "linear", "quantized"]
    for row in parameters_flat:
        p = row.get("path", "").lower()
        if any(k in p for k in keys):
            candidates.append({"source": "parameters", **row})
    for row in attr_rows:
        p = row.get("path", "").lower()
        t = row.get("type", "").lower()
        if any(k in p for k in keys) or any(k in t for k in ["linear", "quantized"]):
            candidates.append({"source": "attrs", **row})
    return candidates[:120]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_introspect_results.json"))
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM introspection smoke")
    print("This discovers model/module paths for the next real adapter hook.")
    print("Uses MLX module APIs plus hidden/private attribute inspection.")
    print(f"model={args.model}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    api_outputs = collect_api_outputs(model)
    parameters_head = api_outputs.get("parameters", {}).get("flat_arrays_head", [])
    attr_rows = collect_attrs(model, "model", max_depth=args.max_depth, max_nodes=800)
    candidates = pick_candidate_targets(parameters_head, attr_rows)
    attr_names = safe_attr_names(model)
    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "found_api_or_attr_rows": any(v.get("flat_array_count", 0) > 0 for v in api_outputs.values() if isinstance(v, dict)) or len(attr_rows) > 1,
        "found_candidate_targets": len(candidates) > 0,
    }
    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "checks": checks,
        "model_type": type(model).__name__,
        "tokenizer_type": type(tokenizer).__name__,
        "model_dir_head": attr_names[:120],
        "api_outputs": api_outputs,
        "attr_row_count_sampled": len(attr_rows),
        "attr_rows_head": attr_rows[:120],
        "candidate_target_count_sampled": len(candidates),
        "candidate_targets_head": candidates[:60],
        "note": "Use candidate target paths to choose a real layer for the first custom PACZero adapter perturbation smoke. If candidates remain empty, inspect api_outputs.parameters.flat_arrays_head manually.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("INTROSPECT_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
