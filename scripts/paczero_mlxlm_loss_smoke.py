#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def safe_float(x) -> float:
    return float(np.array(x.tolist()).reshape(()))


def encode_chat(tokenizer, messages: list[dict]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return tokenizer.encode(rendered)
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return tokenizer.encode(text)


def per_sample_lm_loss(model, token_ids: list[int]) -> dict:
    if len(token_ids) < 3:
        raise ValueError("Need at least three tokens to compute next-token loss")
    x = mx.array(token_ids[:-1], dtype=mx.int32)[None, :]
    y = mx.array(token_ids[1:], dtype=mx.int32)[None, :]
    logits = model(x)
    # logits: [1, T, vocab], y: [1, T]
    loss_tokens = mx.losses.cross_entropy(logits, y)
    loss = mx.mean(loss_tokens)
    mx.eval(loss, loss_tokens)
    token_losses_np = np.array(loss_tokens.tolist(), dtype=np.float64).reshape(-1)
    return {
        "tokens": len(token_ids),
        "prediction_tokens": len(token_ids) - 1,
        "loss": safe_float(loss),
        "token_loss_min": float(token_losses_np.min()),
        "token_loss_max": float(token_losses_np.max()),
        "token_loss_mean": float(token_losses_np.mean()),
    }


def build_examples() -> list[list[dict]]:
    return [
        [
            {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
            {"role": "user", "content": "Sentence: A charming and warm little film. Sentiment?"},
            {"role": "assistant", "content": "positive"},
        ],
        [
            {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
            {"role": "user", "content": "Sentence: The movie was dull, slow, and joyless. Sentiment?"},
            {"role": "assistant", "content": "negative"},
        ],
        [
            {"role": "system", "content": "Classify sentiment. Answer only positive or negative."},
            {"role": "user", "content": "Sentence: Excellent acting and a moving ending. Sentiment?"},
            {"role": "assistant", "content": "positive"},
        ],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--json-out", type=Path, default=Path("benchmark-results/paczero_mlxlm_loss_smoke_results.json"))
    args = parser.parse_args()

    start = time.perf_counter()
    print("# PACZero MLX-LM per-sample loss smoke")
    print("This validates real model forward passes and per-sample LM loss extraction.")
    print("It does not yet perturb/update adapter tensors.")
    print(f"model={args.model}")
    print(f"python={sys.version.splitlines()[0]}")
    for pkg in ["mlx", "mlx-lm"]:
        try:
            print(f"package_{pkg}={md.version(pkg)}")
        except Exception as exc:
            print(f"package_{pkg}=unavailable:{exc}")

    model, tokenizer = load(args.model)
    examples = build_examples()
    sample_results = []
    for idx, messages in enumerate(examples):
        token_ids = encode_chat(tokenizer, messages)
        result = per_sample_lm_loss(model, token_ids)
        result["index"] = idx
        result["assistant_label"] = messages[-1]["content"]
        sample_results.append(result)
        print(f"sample={idx} tokens={result['tokens']} loss={result['loss']:.6f}")

    losses = [r["loss"] for r in sample_results]
    checks = {
        "loaded_model": model is not None,
        "loaded_tokenizer": tokenizer is not None,
        "all_losses_finite": all(math.isfinite(x) for x in losses),
        "all_losses_positive": all(x > 0 for x in losses),
        "all_prediction_lengths_positive": all(r["prediction_tokens"] > 0 for r in sample_results),
    }
    elapsed = time.perf_counter() - start
    payload = {
        "success": all(checks.values()),
        "model": args.model,
        "elapsed_seconds": round(elapsed, 3),
        "checks": checks,
        "num_examples": len(examples),
        "loss_mean": float(np.mean(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "samples": sample_results,
        "note": "Real MLX-LM per-sample loss extraction smoke. Next step is applying finite-difference perturbations to a real adapter parameter vector.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("LOSS_SMOKE_RESULT_JSON=")
    print(json.dumps(payload, indent=2))
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
