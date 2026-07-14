> **STATUS (2026-07-14): SHIPPED as `task844-short-bias-probe`.** Read-only diagnostic
> (`experiments/844_short_bias/`, zero canon diffs, D13-independent). Three probes: mirror-book
> value (calibrated), flat-book policy marginals, rollout leg decomposition + fixture smoke on an
> untrained model. Medium panel: regression clean; **2 MAJOR FIXED** — (1) probe-1 mirror-Δ was
> uncalibrated (independently-initialized BUY/SELL embedding columns → a FRESH model trips |t|~4;
> now gated on a matched fresh-init null z-statistic, smoke robust across seeds 0/7/8: z
> +0.10/−0.85/−1.12), (2) probe-3 mislabeled an all-close SUPERSET as the EXACT `reward_attributable`
> term → relabeled + an action-attributed subset now gates `credit_misassign`. MINOR (smoke
> tolerance false-alarm) fixed. NIT → planner: `open_count` also counts limit-fill opens. selftest
> 7/7. **Real numbers = the scheduled planner run** on p1c s0/s1(+s2) + p1b s1 control with the
> weight-env run flags (this ships the instrument + fixture plumbing proof only).

# TASK_844 — Short-bias postmortem probe (where does the ~10-short book come from?)

**Status**: READY 2026-07-12 (valid under EITHER D13 ruling — diagnostic, zero canon diffs).
**Size: S–M. Risk: low** (read-only instrument, `experiments/` module — 834/836 pattern; reuse
their checkpoint-restore + walk machinery, do not re-derive).

## Why
Both p1b AND p1c policies converge to persistently SHORT-heavy standing books (≈10 shorts vs
≈1–6 longs) that profit only in bear windows and lose everywhere else (LEDGER 835/S4-p1b,
835/S4-p1c) — across two different env semantics. The objective is absolute return (checked:
reward-hybrid = loss recipe, not market-relative), so the bias is a learning artifact with an
unknown locus: policy prior, value preference, or realized-reward asymmetry. D13's ruling (and
the H6-vs-E5 fork) deserves this measurement either way.

## Scope — `experiments/844_short_bias/bias_probe.py` + SPEC + fixture smoke
1. **Mirror-book value probe** (primary): sample N states via the 836 walk (weight env, run
   flags from `--config-json`); for each, construct book B and its sign-mirror −B (longs↔shorts,
   same magnitudes, same cash); report V(B) − V(−B) distributions per checkpoint. Pre-registered
   read: mean significantly < 0 (prefers short books) on ≥2 checkpoints ⇒ VALUE-locus bias;
   ≈0 ⇒ value not the carrier.
2. **Flat-book policy marginals**: at flat-book states (episode starts), the policy's per-asset
   marginal mass on Sell+LimitSell vs Buy+LimitBuy. Pre-registered read: sell-mass/buy-mass
   ratio > 1.5 across seeds ⇒ POLICY-prior locus.
3. **Rollout leg decomposition**: re-run the 835 deterministic rollout (same operator flags)
   capturing per-leg realized PnL; report long-legs vs short-legs totals and counts, and the
   per-step reward the TRAINING objective would have assigned each class. Read: if short legs
   REALIZE losses but the policy keeps opening them, the training signal (value/reward credit)
   mis-assigns; if short legs realized profits during TRAINING-era self-play but not at eval,
   it is distribution shift.
4. Checkpoints: p1c s0/s1 (+ s2 when it lands) primary; p1b s1 as the cross-env control.
   Output JSON + LEDGER-convention row per checkpoint; fixture smoke on an untrained model
   (plumbing proof: mirror-Δ≈0, marginals≈uniform).

## Gates
Fixture smoke green (CPU); zero diffs outside `experiments/844_short_bias/`; selftest 7/7;
medium panel (instrument integrity: the mirror construction and leg attribution must be
leak-checked like 836's alignment audit).

## Out of scope
Fixing the bias (that is the NEXT task, informed by this), training changes, env changes,
running on real checkpoints (scheduled/planner with the run flags — the 836 precedent).
