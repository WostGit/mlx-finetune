# PACZero-ZPL MLX Port Demo: Zenodo Release Package

## Claim boundary

This artifact is a fast, reviewer-facing **MLX port demo of the PACZero-ZPL privacy mechanism**. It is not a full paper-scale reproduction of PACZero utility numbers.

The release demonstrates that the PACZero-ZPL concepts can be ported to MLX by running the mechanism on `mlx-community/SmolLM-135M-4bit` across both SST-2 and SQuAD task paths with strict transcript auditing.

## Paper-concept to MLX-evidence map

| PACZero paper concept | MLX port evidence in this release |
|---|---|
| PACZero-ZPL release rule | `scripts/paczero_mlxlm_faithful_adaptation.py` audits unanimous releases as subset-independent and disagreement releases as RNG-derived signs. |
| `I(S*;Y_1:T)=0` mechanism | Aggregate result requires zero release-rule violations and transcript independence by construction across SST-2 and SQuAD. |
| `M=126` ZPL setting | Aggregate result requires `all_use_M_126 = true`. |
| Prior membership probability 1/2 | Candidate membership builder guarantees each example appears in `M/2 = 63` subsets; aggregate requires `membership_counts = [63]`. |
| Per-sample two-point zeroth-order finite differences | The runner evaluates per-example losses at `theta + mu*z` and `theta - mu*z`, then computes `(plus - minus) / (2*mu)`. |
| Sign-quantized subset aggregation | Per-sample FD values are clipped, averaged by candidate subset, and sign-quantized before ZPL release. |
| LoRA parameterization | Rank-8 / alpha-16 custom LoRA A/B tensors are attached in MLX. |
| q/v projection target fidelity | The 10/10 SmolLM run targets `q_proj + v_proj` across all 30 SmolLM layers, for 60 LoRA targets. |
| Both paper task families | The release runs SST-2 and SQuAD data paths. |
| Saved adapters | Each task saves an `.npz` adapter artifact. |
| Audit sanity check | `scripts/paczero_zpl_negative_control.py` deliberately violates ZPL by making disagreement releases depend on `S*` and checks that the audit catches it. |

## Primary release files

After the workflow completes, include these in the Zenodo release:

```text
PACZERO_MLX_ZENODO_RELEASE.md
.zenodo.json
scripts/paczero_core.py
scripts/paczero_mlxlm_lora_reproduction.py
scripts/paczero_mlxlm_faithful_adaptation.py
scripts/paczero_smollm_10of10_aggregate.py
scripts/paczero_zpl_negative_control.py
.github/workflows/paczero-smollm-10of10-reviewer-proof.yml
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_aggregate_results.json
benchmark-results/paczero-smollm-10of10-aggregate/smollm_10of10_reviewer_report.md
benchmark-results/paczero-smollm-10of10-aggregate/zpl_negative_control_results.json
benchmark-results/paczero-smollm-10of10/smollm-135m-4bit-sst2/smollm_10of10_results.json
benchmark-results/paczero-smollm-10of10/smollm-135m-4bit-squad/smollm_10of10_results.json
benchmark-logs/paczero-smollm-10of10/smollm-135m-4bit-sst2/smollm-10of10-latest.txt
benchmark-logs/paczero-smollm-10of10/smollm-135m-4bit-squad/smollm-10of10-latest.txt
```

Adapter `.npz` files are also useful evidence, but they may be larger:

```text
benchmark-results/paczero-smollm-10of10-adapters/smollm-135m-4bit-sst2/all_layers_qv_lora_rank8_alpha16.npz
benchmark-results/paczero-smollm-10of10-adapters/smollm-135m-4bit-squad/all_layers_qv_lora_rank8_alpha16.npz
```

## Reproduction command

The easiest reproduction path is the GitHub Actions workflow:

```text
PACZero SmolLM 10-of-10 reviewer proof
```

Default workflow parameters:

```text
model = mlx-community/SmolLM-135M-4bit
tasks = SST-2 and SQuAD
steps = 30
train/dev/eval = 8/8/32
M = 126
LoRA = rank 8 / alpha 16
targets = q_proj + v_proj
layers = all
privacy audit = strict ZPL transcript audit
```

## Negative-control audit

The negative control intentionally replaces the safe ZPL disagreement branch:

```text
disagreement -> independent random sign
```

with the forbidden branch:

```text
disagreement -> sign from secret subset S*
```

The release is stronger if the negative control result reports:

```text
good_zpl_release_passes_audit = true
bad_secret_dependent_release_fails_audit = true
negative_control_effective = true
```

## Explicit limitations

This release does not claim:

- paper-scale OPT-1.3B / OPT-6.7B reproduction;
- paper-scale `1000/500/1000` data or 1000 ZPL steps;
- generated SQuAD EM/F1;
- utility parity with the PACZero paper;
- differential privacy.

The MLX port uses a normalized zeroth-order direction and fixed `mu/lr` for a fast, stable MLX demo. That preserves the PACZero-ZPL release mechanism and transcript-audit claim, but it is not a byte-for-byte numerical reproduction of the reference trainer.

## Recommended release wording

> This artifact demonstrates a mechanism-faithful MLX port of PACZero-ZPL. It validates the key privacy-mechanism ingredients: `M=126`, `M/2` candidate membership, per-sample two-point finite differences, sign-quantized subset aggregation, rank-8/alpha-16 q/v LoRA across all SmolLM layers, strict ZPL transcript auditing, and a negative control showing that an `S*`-dependent disagreement release is caught. It is intended as a compact MLX port demo, not a full paper-scale utility reproduction.
