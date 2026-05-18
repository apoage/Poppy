# TASK_741 Mint — Status Audit (2026-05-18)

## Verdict
**COMPLETE.** Mint is fully implemented, merged, regression-validated, and abandoned. The QUEUE "needs implementation" row is stale.

## Evidence trail (code is present)

- `mcts_trader/jax_agent/trm_encoder_v3.py:9-29` — `FiLMProjector` (zero-init `film_out`)
- `mcts_trader/jax_agent/trm_encoder_v3.py:60,89-95` — `per_asset_scalar_dims` field + conditional instantiation
- `mcts_trader/jax_agent/trm_encoder_v3.py:252-256` — FiLM applied to `per_asset_feats` post-`pre_proj_norm` (the exact injection point the spec called for)
- `mcts_trader/jax_agent/muzero_model.py:320-323,340-345,489-532` — `use_per_asset_injection` + `_extract_per_asset_scalars` plumbed through `encode`/`encode_with_features`
- `mcts_trader/jax_agent/trm_flax.py` — `per_asset_scalars` kwarg forwarded (commit `1957a6c` modified +26 lines)
- `easy_config.py:136` and `phase2_config.py:212-214` — config mirror flags
- `train_easy.py:2997,3065-3075` and `train_phase2.py:2230,2517-2526` — CLI flag `--per-asset-injection {none,cheat,xs,both}` with `[MINT]` log tag
- `tools/benchmark_easy.py:855-860` — cheat-safety rail honored
- `tools/tests/test_trm_encoder_v3.py:8,159-226` — `test_film_zero_init_byte_identical`, `test_film_nonzero_changes_output`, `test_film_shape_correctness`

## Commit history

- `1957a6c` (2026-05-14) "Land TASK_741 Mint, TASK_744 Bella path-fix, and 2026-05 experiment infrastructure" — +7768 / -369 LoC across 22 files. Mint code: `trm_encoder_v3.py +241`, `muzero_model.py +406 / -... `, `trm_flax.py +26`, configs + CLI + tests. Matches every "Files to Modify" row in the spec.
- `51500a7` (2026-05-14) "TASK_741 Mint xs done — CLEAR REGRESSION (mean ΔAlpha −6.98pp)" — updates QUEUE + lore + memory only (no code).

## Did the experiment actually run?

**Yes — on the server, not synced back to workstation.** `runs/741_mint/` has 12 named subdirs (`{baseline,mint_xs}_seed_{7,13,23}` + `eval_*`) but they are **all empty on workstation** (only `.`/`..`). The result.md (7962 bytes, modified 2026-05-14 21:58, matching the regression commit) cites specific artifacts (`stage_b_metrics.csv`, `phase2_stageB_best.pkl`, per-seed `benchmark.json`) that live on the server. The per-seed numbers in result.md (totals 6.6581/6.1451/6.6935 baseline vs 5.7152/5.5858/5.7598 mint_xs; ΔAlpha −14.62/−1.22/−5.08; mean −6.98) are unfabricable specificity and match the commit message verbatim. Verdict: numbers real, artifacts unsynced.

## Deprecation / cleanup status

- Mint code is **kept as opt-in** (default `--per-asset-injection none` → `FiLMProjector` not instantiated, zero-cost, byte-identical). No code should be ripped out; the FiLM surface may be reused for future per-asset scalars (Bella tag, regime tag).
- **`experiments/QUEUE.md:149` is stale** — the "Tasks needing implementation" row for `TASK_741_PER_ASSET_INJECTION_FIX.md` (Medium, ~1-2 days) was true on 2026-05-13 and obsoleted by commit `1957a6c` 2026-05-14. Same row in the Priority 2 table (line 44) already says "DONE — REGRESSION", so the file contradicts itself within ~100 lines.
- `tasks/TASK_741_PER_ASSET_INJECTION_FIX.md` header still says `Status: PROPOSED (2026-04-25)` — also stale; should read `DONE — REGRESSION 2026-05-14`.

## Recommended action

- Remove `TASK_741_PER_ASSET_INJECTION_FIX.md` row from `experiments/QUEUE.md:149` ("Tasks needing implementation" table); optionally update the TASK_741 spec header to `DONE — REGRESSION 2026-05-14` for archival hygiene. Optional but useful: rsync `runs/741_mint/**/*.{csv,json,log,pkl}` from server so the numbers in `result.md` are reproducible locally.
