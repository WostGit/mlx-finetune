# PACZero SmolLM 10/10 reviewer-proof aggregate report

Overall success: **False**

## Reviewer claim boundary

Fast reviewer-proof SmolLM MLX adaptation of PACZero-ZPL mechanism, not paper-scale utility reproduction.

## Paper-concept to MLX-evidence map

| Paper concept | MLX evidence |
|---|---|
| `PACZero_ZPL_I_Sstar_Y_1T_equals_0_mechanism` | addressed by strict transcript audit: unanimous releases subset-independent, disagreement releases RNG-derived, zero release-rule violations |
| `M_126_ZPL_setting` | addressed across both tasks |
| `LoRA_rank8_alpha16` | addressed across both tasks |
| `q_proj_v_proj_target_fidelity` | addressed across all SmolLM layers; 60 q/v targets |
| `SST2_and_SQuAD_task_coverage` | addressed with small-scale data paths; SQuAD uses gold-answer likelihood, not generated EM/F1 |
| `audit_soundness_negative_control` | addressed by deliberately making disagreement releases depend on S_star and requiring the audit to fail that case |

## Aggregate checks

| Check | Pass |
|---|---:|
| `both_task_results_present` | True |
| `both_tasks_successful` | True |
| `all_use_M_126` | True |
| `all_membership_M_over_2` | True |
| `all_rank8_alpha16` | True |
| `all_qv_projection_set` | True |
| `all_layers_requested` | True |
| `all_have_60_qv_targets_for_smollm` | True |
| `all_fd_finite` | True |
| `all_fd_signal` | True |
| `all_privacy_audits_passed` | True |
| `all_release_rule_violations_zero` | True |
| `all_transcripts_independent_by_construction` | True |
| `all_adapters_saved` | True |
| `negative_control_present` | False |
| `negative_control_successful` | False |
| `negative_control_good_zpl_passes` | False |
| `negative_control_bad_secret_release_fails` | False |

## Task summaries

| Task | Success | Runtime s | M | Membership | Targets | Steps | FD finite | FD signal | Privacy audit | Violations |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| sst2 | True | 49.024 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | True | 0 |
| squad | True | 62.08 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | True | 0 |

## Negative-control audit

Present: **False**
Success: **False**
Good ZPL release passes audit: **None**
Bad secret-dependent release fails audit: **None**

## Limitations explicitly not claimed

- Not a full OPT-1.3B/OPT-6.7B reproduction.
- Not paper-scale 1000/500/1000 data or 1000 ZPL steps.
- SQuAD metric is label-only gold-answer likelihood, not generated EM/F1.
- Utility numbers are smoke-scale; privacy mechanism is the reviewed claim.
- The MLX demo uses normalized ZO directions and fixed mu/lr for fast stable execution; it is not a byte-for-byte optimizer reproduction.
