# PACZero SmolLM Release Validation Report

Overall success: **true**

## Claim boundary

Fast SmolLM MLX adaptation of the PACZero-ZPL mechanism, with smoke-scale utility preservation checks; not paper-scale utility reproduction.

## Paper-concept to MLX-evidence map

| Paper concept | MLX evidence |
|---|---|
| `PACZero_ZPL_I_Sstar_Y_1T_equals_0_mechanism` | Strict transcript audit: unanimous releases are subset-independent, disagreement releases are RNG-derived, release-rule violations are zero. |
| `M_126_ZPL_setting` | Both SST-2 and SQuAD use `M=126`. |
| `M_over_2_membership` | Both tasks have `membership_column_counts_unique = [63]`, i.e. each example appears in `M/2` candidate subsets. |
| `LoRA_rank8_alpha16` | Both ZPL tasks and non-private utility controls use rank 8 / alpha 16 LoRA. |
| `q_proj_v_proj_target_fidelity` | SmolLM run targets `q_proj + v_proj` across all 30 layers, for 60 LoRA targets. |
| `SST2_and_SQuAD_task_coverage` | Both SST-2 and SQuAD data paths run; SQuAD uses gold-answer likelihood, not generated EM/F1. |
| `utility_preservation_smoke` | ZPL selected checkpoints and non-private ZO controls are not worse than the frozen baseline on the selection metric. |
| `audit_soundness_negative_control` | Negative control deliberately makes disagreement releases depend on `S*`, and the audit catches it. |

## Aggregate checks

| Check | Pass |
|---|---:|
| `both_task_results_present` | true |
| `both_tasks_successful` | true |
| `all_use_M_126` | true |
| `all_membership_M_over_2` | true |
| `all_rank8_alpha16` | true |
| `all_qv_projection_set` | true |
| `all_layers_requested` | true |
| `all_have_60_qv_targets_for_smollm` | true |
| `all_fd_finite` | true |
| `all_fd_signal` | true |
| `all_privacy_audits_passed` | true |
| `all_release_rule_violations_zero` | true |
| `all_transcripts_independent_by_construction` | true |
| `all_zpl_utility_not_worse_than_baseline` | true |
| `all_adapters_saved` | true |
| `negative_control_present` | true |
| `negative_control_successful` | true |
| `negative_control_good_zpl_passes` | true |
| `negative_control_bad_secret_release_fails` | true |
| `utility_controls_present` | true |
| `utility_controls_successful` | true |
| `utility_controls_not_worse_than_baseline` | true |
| `utility_controls_fd_signal` | true |

## ZPL task summaries

| Task | Success | Runtime s | M | Membership | Targets | Steps | FD finite | FD signal | I=0 audit | Utility >= baseline | Violations |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | true | 44.826 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | true | true | 0 |
| SQuAD | true | 63.347 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | true | true | 0 |

## Non-private ZO utility controls

| Task | Success | Runtime s | Targets | Steps | FD finite | FD signal | Utility >= baseline | Adapter saved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2 | true | 46.794 | 60 | 30 | 1.0 | 1.0 | true | true |
| SQuAD | true | 64.042 | 60 | 30 | 1.0 | 1.0 | true | true |

## Negative-control audit

| Check | Result |
|---|---:|
| good ZPL release passes audit | true |
| bad secret-dependent release fails audit | true |
| negative control effective | true |

## Limitations explicitly not claimed

- Not a full OPT-1.3B/OPT-6.7B reproduction.
- Not paper-scale `1000/500/1000` data or 1000 ZPL steps.
- SQuAD metric is label-only gold-answer likelihood, not generated EM/F1.
- Utility numbers are smoke-scale; they demonstrate preservation relative to frozen baseline, not paper-level utility parity.
- The MLX demo uses normalized ZO directions and fixed `mu/lr` for fast stable execution; it is not a byte-for-byte optimizer reproduction.
