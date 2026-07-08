# TASK_836 — E4 value-ranking probe (does the learned value rank states above momentum?)

**Status**: READY 2026-07-08. **Size: S–M. Risk: low-medium** (read-only instrument, zero canon
diffs; `experiments/` module — 834/835 pattern).
**Pathway**: rung R2 (`docs/pathway.md` — E4 is specced there, unbuilt). Hypothesis: H1/H2.

## Why
E3 (834) tests the reward head; 835 grades full rollouts. Neither isolates the VALUE net —
if a trained policy underperforms, we can't currently tell "value can't rank states" from
"search/policy can't exploit a good value". E4 closes that diagnostic gap, and it runs on the
phase-1 cost0 snapshots the moment they exist.

## Scope — `experiments/836_value_ranking/rank_probe.py` + SPEC + fixture smoke
1. **Checkpoint load** — same config-then-`trainer.load_checkpoint` pattern as 834 (reuse its
   fail-loud restore guards; do not re-derive them).
2. **State sample**: N states drawn along deterministic env walks on held windows (the four
   831-EXT windows + full window), observations via the training obs rule (panel-consistency —
   the 832 PIN applies).
3. **Value ranking**: predicted root value per state vs the REALIZED forward portfolio-relevant
   return over horizon k (pre-register k=3 primary — matches the 831x/C value-horizon finding —
   plus k=1, k=6 secondary; realized return computed by the true env, 831 machinery).
4. **Metric**: Spearman rank correlation, per window per seed. **Baseline**: the same
   correlation for a momentum ranking (trailing m-day return, pre-register m in SPEC).
   PASS/interpretation (pre-registered): value beats momentum's correlation on ≥3/5 windows =
   the value net carries signal beyond the obvious; below-momentum everywhere = the value net
   is the bottleneck (feeds the phase-2 postmortem either way — this probe is DIAGNOSTIC, it
   gates nothing by itself).
5. **Output JSON + LEDGER-convention row**; fixture smoke on an untrained model (plumbing
   proof: correlations ≈ 0, no threshold applied).

## Gates
Fixture smoke green (CPU); zero diffs outside `experiments/836_value_ranking/`; selftest 7/7.

## Out of scope
Running on real snapshots (scheduled/planner, with the phase-1 flags), any trainer/model edits.
