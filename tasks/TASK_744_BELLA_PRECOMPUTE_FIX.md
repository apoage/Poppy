# TASK_744 — Bella precompute: bug fix + minimum-viable design

**Status:** OPEN (2026-05-14)
**Blocks:** TASK_740 (Bella distill training) until resolved.
**Owner:** Agent investigation → workstation operator decides design knobs.

## Context

Server agent attempted TASK_740 Step 1 (Bella precompute) on 2026-05-13
night. Two problems surfaced:

### Problem 1 — Path manipulation bug (single-line fix)

`tools/precompute_bella_predictions.py` crashes on manifest assembly:

```
File "tools/precompute_bella_predictions.py", line 210, in _process_period
    return {'path': str(out_path.relative_to(REPO_ROOT)),
ValueError: 'data/bella_preds/bella_preds_2018-10-17__2019-04-15.npz'
is not in the subpath of '/home/operator/Trader' OR one path is relative
and the other is absolute.
```

Root cause: line 244 constructs `out_dir = Path(args.output_dir)` as
**relative** (default `data/bella_preds`). `out_path = out_dir / "..."`
inherits the relative-ness. `REPO_ROOT` (line 52) is absolute. `Path
.relative_to()` requires both same-kind.

**Fix (one line):**
```python
# line 244
out_dir = Path(args.output_dir).resolve()
```

Also worth audit-fixing for symmetry:
- Line 155 (same `.relative_to(REPO_ROOT)` pattern in the cache-hit path)
- Line 304 already uses `is_relative_to` guard — good pattern; can apply
  same defensive pattern to the two `.relative_to` calls in `_process_period`.

The 250 MB completed shard `data/bella_preds/bella_preds_2018-10-17__2019-04-15.npz`
is **preserved** on server disk. Re-run will skip it via `out_path.exists()`
short-circuit at line 153.

### Problem 2 — Scale catastrophe (the real blocker)

- `config/foxglove_periods.json`: **270 periods**.
- One period (2018-10-17 to 2019-04-15) processed at 8.6 windows/sec for
  258,803 windows × k=4 samples = **8.3 hours of GPU compute**.
- Naive projection: 270 × 8.3 ≈ **93 days of GPU**. Storage ~68 GB.
- SPEC's "30-60 min" estimate was off by ~3 orders of magnitude.

Cannot ship TASK_740 at current scale. Need minimum-viable design.

## Investigation required (before fix is applied)

An agent should answer:

1. **Consumption density** — read how `train_phase2.py` uses
   `--bella-distill-head --bella-preds-manifest` and the underlying shard
   files. Specifically:
   - How are Bella preds sampled into a training batch?
   - Is every minute-window consumed, or does training sub-sample
     anyway? (If sub-sampled, precompute density is wasted.)
   - What's the loss formulation? (KL on log-returns at H horizons? Affects
     whether K=1 vs K=4 averaging matters.)

2. **Dimensionality** — current shape per shard is `(n_windows, n_horizons, n_assets)`.
   - If `train_phase2` reads only random batches per epoch, what's the
     minimum stride that still gives full coverage of training segments?
   - Are all 270 periods used in training, or only a subset?

3. **Reduction levers** — quantify expected cost of each:
   - K samples: 4 → 1 (likely ~4× speedup, modest quality loss; ablate later)
   - Stride: every minute (1) → every hour (60) (60× speedup, but does
     training need minute density?)
   - Period subset: 270 → e.g. 17 (the SegmentPool size?) or 50 (~6× reduction)
   - Horizons trim: current `1,5,10,20`; can we drop or coarsen?

4. **Quickest viable proof-of-concept** — what's a precompute that gives
   the TASK_740 hypothesis a real test in ≤6 hours of GPU? Even if it's
   a subset of periods with stride>1, the experiment can answer "does
   Bella distillation help dir_acc at all" before scaling out.

## Hard constraints for the agent

- **Do not start any precompute or training runs.** Investigation +
  fix only. Workstation operator decides knobs after reading the report.
- **Apply the path fix** (line 244 + lines 155/210 defensive pattern) —
  this is safe and unblocks any future precompute.
- **Do not commit, push, or deploy.** Leave changes in working tree.
- Do not change any other files than `tools/precompute_bella_predictions.py`.

## Deliverable

A report containing:

1. Path fix patch summary (line numbers + before/after diff).
2. Answers to investigation questions 1-4 above, with file:line citations
   from `train_phase2.py`, the bella consumer code, etc.
3. **Concrete minimum-viable precompute proposal** — proposed args:
   ```
   --k-samples ?
   --stride ?  (if such a flag needs to be added — say so)
   --limit-periods ?
   --horizons ?
   ```
   with the projected wall-time and disk usage.
4. Risks / open questions.

After the report, operator decides:
- whether to add a `--stride` CLI flag (likely yes)
- whether to ship the design as TASK_744 done + restart precompute
- or to defer TASK_740 entirely until other experiments yield

## Files to read

- `tools/precompute_bella_predictions.py` (the broken script)
- `train_phase2.py` — search for `bella_distill`, `bella-preds-manifest`,
  `bella_preds`, `BellaDistill`
- `mcts_trader/jax_agent/` — head implementations for `bella_distill`
- `experiments/740_bella_distill/SPEC.md` (the current SPEC, may need amendment)

## Out of scope

- Re-implementing Bella distillation training itself
- Changing what Bella predicts (horizons, log-returns vs prices)
- Modifying any non-tooling files
