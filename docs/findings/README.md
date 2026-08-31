# Findings

Durable experiment findings and diagnoses, one file per finding. Migrated out of local agent
memory 2026-08-31 so subagents and the server agent can read them (`lore/adr/0008`, whose
"Alternatives considered" rejects keeping memory as the index).

**Read `../../experiments/LEDGER.md` first for anything post-780** — the LEDGER is the one-line-per-
verdict record, and its preamble carries the framework-epoch caveats (E-a…E-f) and the TASK_882
instrument discipline that decide whether a verdict is citable as "closed". These files are the
long-form diagnoses behind those lines, plus the pre-780 history the LEDGER delegates.

## Scope warning — read before citing anything below as closed

Every "CLOSED / NULL / EXHAUSTED / marginal" verdict in the **scalar-architecture** section is
scoped to the OLD architecture: a single-**scalar**-action planner plus single-mechanism encoder
tweaks, built largely by a pre-Opus-4.8 model over ~6 months. They are honest evidence about
*those formulations* — not laws of nature, not a verdict on the vector-action rebuild (untested by
all of them), and not licence to default to "ship the linear rule / trading is impossible".
Prune the fatalism; keep the measurements.

## Live / forward

| Finding | One line |
|---|---|
| [imprint_oos_artifact](imprint_oos_artifact.md) | Imprint lane has NO OOS edge (n=5, settled 2026-08-06); scoped to hard-target BC. |
| [vector_env_cash_wall](vector_env_cash_wall.md) | `step_vector` snapshot-cash makes breadth lethal at any cost tier — the real root of HOLD collapse. |
| [planner_closed_clean_cell](planner_closed_clean_cell.md) | TASK_821 scalar planner converges to 100% HOLD; an action-space defect, not a market verdict. |
| [oracle_design_intent](oracle_design_intent.md) | The oracle is a measurement instrument, never a teacher; misuse is the suspected root of anti-correlation. |
| [stageb_reward_head_diseased](stageb_reward_head_diseased.md) | Stage-B reward head measured action-blind; eval-MCTS absolutes are joint-decode artifacts. |
| [verdict_contamination_audit](verdict_contamination_audit.md) | 2026-07-05 audit 734→819: 758/740/743 reconciled to NULL; the −16.54pp "catastrophe" was an argmax artifact. |
| [diversified_trend_shipcandidate](diversified_trend_shipcandidate.md) | Linear trend CTA real but marginal (Sharpe +0.67, CI crosses zero); at paper-live gate 3. |
| [bull_blindness](bull_blindness.md) | Bull failure = oracle HOLD imbalance + oracle-as-teacher. NOT encoder or search. |
| [deployment_gate_ladder](deployment_gate_ladder.md) | Four gates: all-regime → unseen → paper → live. Nothing promotes straight to deploy. |

## Old scalar-architecture negatives (SCOPED — see the warning above; none tested vector-action)

| Finding | One line |
|---|---|
| [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md) | Sharpening the classifier anti-correlates with Alpha; the bottleneck is downstream of the encoder. |
| [single_mechanism_exhausted](single_mechanism_exhausted.md) | Single-mechanism encoder tweaks 734–810 all null-or-marginal. |
| [modeling_surface_exhausted](modeling_surface_exhausted.md) | Same verdict at modeling-surface scope. |
| [encoder_bella_class_closed](encoder_bella_class_closed.md) | Bella-as-encoder class 0-for-3 null. |
| [candle_language](candle_language.md) | Candle-language build CPU-verified; `docs/candle_language/` + `tools/candle_lang/`. |
| [763_767_closed_null](763_767_closed_null.md) | Entropy regularisation + limit-order action space: closed null. |
| [collapse_fixed_predictability_wall](collapse_fixed_predictability_wall.md) | TASK_818 fixed the 812 reward wiring; the load-bearing recall≈0.02 was a pre-fix count-proxy. |
| [benchmark_live_gap](benchmark_live_gap.md) | 2026-05-20 eval-operator recalibration; eval-MCTS sims=16 joint is the canonical scorecard. |
| [task741_result](task741_result.md) | Mint injection regressed Alpha; abandoned. |
| [task741_staged](task741_staged.md) | The 741 FiLM implementation staged in-tree 2026-05-13, never committed or smoke-tested. |
| [flow_matching_promising](flow_matching_promising.md) | Flow-matching dynamics loss REVERSED at 5 seeds after looking good at 3 — a seed-variance cautionary tale. |
| [bella_column_mismatch](bella_column_mismatch.md) | TASK_740 distillation invalid: Bella's columns are DB-ID order, the trainer's are rotation-specific, no remap. |

## Ops / infra

| Finding | One line |
|---|---|
| [workstation_syncflood](workstation_syncflood.md) | Hard crashes = AMD data-fabric sync flood; fixed 2026-07-06, RAM replaced 2026-08-01. Summary in `docs/infrastructure.md`. |
| [seed7_stageb_wedge](seed7_stageb_wedge.md) | Seed-7 log-freeze wedge; kill and rerun solo. |
| [benchmark_gap](benchmark_gap.md) | Train↔benchmark trade-agree gap, root-caused. |
| [apothecary_marketplace](apothecary_marketplace.md) | The public plugin marketplace `github.com/apoage/apothecary`. |

## Not migrated

`train_easy.py --bella` 48-min stall (encoder exonerated 2026-07-05; suspect is Bella's 302M
`sample` at untested seq length, `setup.py:188,293`), the TASK_742 ≤2-concurrent cap, the
`zstd→rsync` corruption-blindness note, and the Foxglove SegmentPool facts are one-liners that
stay in the memory index rather than earning a file each.
