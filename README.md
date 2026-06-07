# PACZero-ZPL MLX Port Demo

This repository contains a compact MLX port demo of the PACZero-ZPL mechanism using `mlx-community/SmolLM-135M-4bit`.

The package is intended to demonstrate mechanism portability, not to reproduce the full paper-scale OPT utility results.

## What is included

- PACZero-ZPL sign-release audit with `M=126`.
- Balanced candidate subset construction with each example in `M/2 = 63` subsets.
- Per-sample two-point zeroth-order finite differences.
- Rank-8 / alpha-16 LoRA adapters.
- `q_proj + v_proj` adapters across all 30 SmolLM layers, for 60 LoRA targets.
- SST-2 and SQuAD data paths.
- Utility preservation checks against a frozen baseline at smoke scale.
- Non-private zeroth-order utility controls.
- A negative-control audit that deliberately makes the release depend on `S*` and verifies the audit catches it.

## Main result

The final aggregate result is:

```text
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_aggregate_results.json
```

The human-readable report is:

```text
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_release_report.md
```

Current aggregate status:

```text
success = true
```

## Reproduce the validation run

Run the GitHub Actions workflow:

```text
PACZero SmolLM release validation
```

Default settings:

```text
model = mlx-community/SmolLM-135M-4bit
tasks = SST-2 and SQuAD
train/dev/eval = 8/8/32
steps = 30
M = 126
LoRA = rank 8 / alpha 16
targets = q_proj + v_proj
layers = all
```

## Files to include in a zip release

```text
README.md
.zenodo.json
PACZERO_MLX_ZENODO_RELEASE.md
.github/workflows/paczero-smollm-release.yml
scripts/paczero_core.py
scripts/paczero_mlxlm_lora_reproduction.py
scripts/paczero_mlxlm_faithful_adaptation.py
scripts/paczero_smollm_nonprivate_utility_control.py
scripts/paczero_smollm_10of10_aggregate.py
scripts/paczero_zpl_negative_control.py
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_aggregate_results.json
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_release_report.md
benchmark-results/paczero-smollm-10of10-aggregate/zpl_negative_control_results.json
benchmark-results/paczero-smollm-10of10/smollm-135m-4bit-sst2/smollm_10of10_results.json
benchmark-results/paczero-smollm-10of10/smollm-135m-4bit-squad/smollm_10of10_results.json
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-sst2/nonprivate_utility_control_results.json
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-squad/nonprivate_utility_control_results.json
```

Adapter files are optional but useful if the release size allows them:

```text
benchmark-results/paczero-smollm-10of10-adapters/smollm-135m-4bit-sst2/all_layers_qv_lora_rank8_alpha16.npz
benchmark-results/paczero-smollm-10of10-adapters/smollm-135m-4bit-squad/all_layers_qv_lora_rank8_alpha16.npz
benchmark-results/paczero-smollm-utility-control-adapters/smollm-135m-4bit-sst2/nonprivate_all_layers_qv_lora_rank8_alpha16.npz
benchmark-results/paczero-smollm-utility-control-adapters/smollm-135m-4bit-squad/nonprivate_all_layers_qv_lora_rank8_alpha16.npz
```

## Scope and limitations

This package does not claim:

- paper-scale OPT-1.3B / OPT-6.7B reproduction;
- paper-scale `1000/500/1000` data or 1000 ZPL steps;
- generated SQuAD EM/F1;
- utility parity with the PACZero paper;
- differential privacy.

The MLX demo uses normalized zeroth-order directions and fixed `mu/lr` for fast, stable execution. It preserves the PACZero-ZPL release mechanism and transcript-audit claim, but it is not a byte-for-byte optimizer reproduction of the reference trainer.
