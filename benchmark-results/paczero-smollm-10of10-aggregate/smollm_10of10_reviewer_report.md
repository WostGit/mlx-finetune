# PACZero SmolLM 10/10 reviewer-proof aggregate report

Overall success: **True**

## Reviewer claim boundary

Fast reviewer-proof SmolLM MLX adaptation of PACZero-ZPL mechanism, not paper-scale utility reproduction.

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

## Task summaries

| Task | Success | Runtime s | M | Membership | Targets | Steps | FD finite | FD signal | Privacy audit | Violations |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| sst2 | True | 43.964 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | True | 0 |
| squad | True | 61.725 | 126 | [63] | 60 | 30 | 1.0 | 1.0 | True | 0 |

## Limitations explicitly not claimed

- Not a full OPT-1.3B/OPT-6.7B reproduction.
- Not paper-scale 1000/500/1000 data or 1000 ZPL steps.
- SQuAD metric is label-only gold-answer likelihood, not generated EM/F1.
- Utility numbers are smoke-scale; privacy mechanism is the reviewed claim.
