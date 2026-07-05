# TASK_737: Ablation Suite on TASK_736 Baseline

**Status**: ACTIVE (2026-04-23 evening).
**Priority**: HIGHEST for next steps — tells us which of 730 / 732 / 735 actually contribute once MCTS is working.
**Depends on**: 32-ch Phase 1 checkpoint + TASK_736 reference result (Alpha +3.87%, Bal.Acc 26.5%, 6/6 beats-random).
**Artifact**: `tools/run_737_ablations.sh` (sequential runner).

---

## Motivation

TASK_736 established that `--value-weight 0.5` was the silent blocker for 6 prior Stage B runs. Those runs included TASK_730 (rank head), TASK_732 (cross-sectional channels), and TASK_735 (aux log-return head). With a broken ExIt, none of those results tell us if those pivots actually help.

TASK_736 itself ran with *all three* pivots enabled (rank head + cross-sectional channels + aux price head + value weight). Its positive result is therefore ambiguous: it could be the value fix alone doing all the work, or the aux heads meaningfully contributing, or some stacking.

This suite isolates contributions via flag-flip ablations.

---

## Test Matrix

All runs use the TASK_736 reference config (32-ch V3 P1 checkpoint + `--value-weight 0.5` + everything else identical). The only differences are the head-enable flags.

| Tag | Encoder ch | `rank_head` | `aux_price_head` | `direction_head` | Checkpoint | What's tested |
|---|---|---|---|---|---|---|
| **Reference (TASK_736)** | 32 | ON | ON | ON | xs32 P1 | Already done — Alpha +3.87% AVG |
| **no_aux** | 32 | ON | **OFF** | ON | xs32 P1 | Does TASK_735 contribute on working ExIt? |
| **no_rank** | 32 | **OFF** | ON | ON | xs32 P1 | Does TASK_730 contribute on working ExIt? |
| **minimal** | 32 | **OFF** | **OFF** | ON | xs32 P1 | Is value-weight fix alone sufficient? (keep direction head — it was baseline pre-730) |
| **legacy_27ch** | **27** | ON | ON | ON | **v3 P1 (27-ch)** | Does TASK_732 (cross-sectional channels) contribute on working ExIt? |

All other config identical to TASK_736 (factored T=0.1, value-target=nstep, etc.). The `legacy_27ch` row uses a different warmstart checkpoint because shape-compatibility requires 27-ch encoder weights; every other dim matches.

### Out of scope (documented for later)

- **TASK_727 ablation (direction head off)**: `--no-direction-head`. Cheap but direction head has been on since 727 and is structurally different from rank/price heads (it's per-asset binary, uses direction_policy_refiner). Risk of unrelated destabilization. Defer.

### Related tasks

Split out of this suite for bookkeeping:
- **TASK_738** (legacy-27ch ablation) — same ablation as Run 4 of this suite, with its own spec for when the 27-ch outcome needs follow-up analysis. See `tasks/TASK_738_LEGACY_27CH_ABLATION.md`.

### Downstream tasks (ground prepared, not yet launched)

After 737 completes, the top-ranked forward pivots are:
- **TASK_739** (Tier A1) — dense sign-of-return oracle. Replaces top-k oracle with per-asset directional labels. See `tasks/TASK_739_DENSE_SIGN_ORACLE.md`.
- **TASK_740** (Path β) — Bella distillation head. Richer prior than 60-step OHLCV alone. See `tasks/TASK_740_BELLA_DISTILL_HEAD.md`.

These are **not** run in sequence with 737 — they are next-pivot candidates gated on the 737 decision matrix.

---

## Success Criteria

For each run, collect:
- Bal.Acc AVG across 6 periods
- Alpha AVG across 6 periods
- ModelPnL AVG
- Beats-random count (x/6)
- Action distribution (how diverse vs oracle?)

**Decision matrix after all three runs complete:**

- `no_aux` ≈ reference → TASK_735 (aux price head) contributes nothing measurable. Remove the ~60 LoC and associated config fields.
- `no_aux` significantly worse than reference → TASK_735 contributes. Keep as permanent.
- `no_rank` ≈ reference → TASK_730 (rank head) contributes nothing. Keep flag for later stacking but don't rely on it.
- `no_rank` significantly worse than reference → TASK_730 contributes. Keep.
- `minimal` ≈ reference → value-weight was doing all the work. 730 + 735 were placebo. Simplify training config and all future work.
- `minimal` significantly worse than reference → stacked aux heads genuinely help. Keep the full stack as baseline for future pivots.
- `legacy_27ch` ≈ reference → TASK_732 cross-sectional channels don't help with a working ExIt. Keep them as input richness (no cost) but don't rely on them for ceiling-breaking.
- `legacy_27ch` significantly worse than reference → TASK_732 genuinely contributes. 32-ch remains the baseline; consider Tier D2 (external features) as the next input-expansion bet.

"Significantly" = ≥ 3 pp change in Bal.Acc or ≥ 2 pp change in Alpha or ≥ 1 change in beats-random.

---

## Execution

Shell script `tools/run_737_ablations.sh` runs all three variants sequentially and benchmarks each. Designed to run overnight.

```bash
# Pre-launch hygiene (verify before starting)
[redacted: host service detail]

# Launch suite (runs 3 trainings + 3 benchmarks = ~6.5 hours)
bash tools/run_737_ablations.sh > ablation_737.log 2>&1 &
```

Output directories:
- `jax_ckpt_easy_phase2_737_no_aux/` — Run 1
- `jax_ckpt_easy_phase2_737_no_rank/` — Run 2
- `jax_ckpt_easy_phase2_737_minimal/` — Run 3

Each contains: `phase2_stageB_best.pkl`, `stage_b_nohup.log`, `benchmark.log`, `benchmark.json`.

After all four complete, a summary table is printed to the master log + saved to `ablation_737_summary.md`.

---

## Timing

- Each training run: ~2h (E199 at ~26s/epoch + ~15min pool/JIT)
- Each benchmark: ~7min
- Total suite: ~8.5 hours (4 runs)

If launched at 19:00: completes ~03:30. Safe overnight window. Lead with cheapest variants (1–3) first so a crash mid-suite still leaves useful data; `legacy_27ch` is Run 4 because it rebuilds the pool with 27-ch shape which costs one extra pool init.

---

## Risks

- **VRAM fragmentation** across sequential runs. Between runs, the script explicitly `jax.clear_caches()` isn't available from shell — we rely on process-exit to free VRAM. Should be clean.
- **Slot pool caching**: the 17-slot data pool is rebuilt each run (~15 min each). Total pool load overhead: 45 min across 3 runs. Could be reduced with a shared data pool but not worth the complexity for this one-off.
- **If Run 1 crashes (e.g. OOM)**: `set -e` in script aborts the suite. Rerun from the failing variant. Earlier runs' checkpoints are preserved.
- **Disk space**: each output dir is ~2-3 GB (checkpoints + logs). 3 runs = ~9 GB. Verify available space before launch.
