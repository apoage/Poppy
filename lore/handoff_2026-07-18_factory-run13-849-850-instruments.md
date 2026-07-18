# Factory run 13 handoff — 2026-07-18 (849 + 850, the D18/OOS instrument builds)

Both Ready rows shipped. Zero OPEN decisions on entry, zero on exit. Clean tree, nothing
pushed. Two tagged commits; both are read-only / loader-only instruments with the real
runs left to the planner.

## Shipped

| Commit | Tag | Task | Verification outcome |
|--------|-----|------|----------------------|
| `d836d46` | `task849-leaf-eval-autopsy` | TASK_849 — read-B leaf-eval autopsy probe (D18 / Amendment 6 FAIL branch) | correctness + flow-verifier: 1 MAJOR + 1 MINOR fixed + re-verified clean; acceptance (i)–(iv) PASS; zero canon diffs; selftest 8/8 |
| `97a2710` | `task850-daily-panel-source` | TASK_850 — `--panel-source {minute,daily}` for the 835/848 eval (unlocks the OOS benchmark) | correctness + fail-closed + flow-verifier: **1 BLOCKER + 1 MAJOR + 1 MINOR** fixed + re-verified clean; acceptance (i)–(iv) PASS; default-minute byte-identical; selftest 8/8 |

## TASK_849 — leaf-eval autopsy probe

`experiments/849_leaf_autopsy/leaf_probe.py` (new, read-only). Walks the 848 policy-argmax
operator trajectory of a frozen ckpt and, at every Nth decision (default N=25), runs the
SHIPPED `SampledMuZeroVectorSearch` candidate-gen + scoring WITHOUT stepping the env,
dumping per state: the K candidates (flagged policy-argmax/explore/dither) with prior mass
/ r̂ / v̂ / Q=r̂+γv̂ / Gumbel score / visits / selected?, plus the TRUE one-step env reward
per candidate (env stepped on a copy = the calibration column). Aggregate: (a) rank
distribution of the in-set prior under each signal, (b) per-candidate signal-vs-true-reward
Pearson+Spearman, (c) selected-vs-prior churn. Both `policy-prior` and `dithered-book`.

- **Zero canon diffs**: model/env/search/panel all imported by-path from the frozen 835
  harness + 832 imprint runner; `eval_vector.py` unedited by this task.
- **The MAJOR (fixed)**: rank column (a) was corrupted by batched(M=16)-vs-unbatched(M=1)
  XLA float32 inference noise (~1e-7, one-directional) — the in-set prior was ranked via a
  separate M=1 recompute, biasing its rank systematically WORSE (produced impossible rank
  K+1), worst precisely in the near-tie regime the autopsy exists to resolve → a diagnostic
  biased toward its own hypothesis. Fix: in-set case reads the prior's own batched row
  `r_hat[pa_idx]`/`v_hat[pa_idx]`/`Q[pa_idx]`; the M=1 recompute is retained only for the
  not-in-set (dithered-book) reference. Re-verify: k=8 rank 15→13, k=24 17→16, 0 in-set
  rank>K violations, policy_argmax block bit-identical to the candidate row.
- **The read** (why this exists): with the policy-argmax GUARANTEED in-set (847, 40/40),
  search-eval STILL scores −7.84 vs +0.839 (LEDGER 832/E5-inversion) — the SELECTION layer
  rejects the in-set prior. The probe measures WHICH signal misranks it and by how much;
  the real-run numbers decide value-repair vs reward-repair vs selection-mechanics as a
  fresh DECISIONS row (D18), BEFORE any further training spend.

**Planner-owned real run (NOT launched)**: frozen `runs/832_e5/ckpt_imprint_full`, fee16 env
flags = the ExIt set, BOTH schemes. That is the deliverable — the fixture smoke only proves
the instrument is faithful (selection cross-check 100% vs the 835 operator — traces the real
path, not a fork).

## TASK_850 — daily-table panel source

`experiments/835_stage4_eval/eval_vector.py` (single file, +451/−1). `--panel-source
{minute,daily}` (default minute = byte-identical); `daily` reads `ohlcv_daily` directly
(current to 2026-06) instead of resampling the dying minute table, valid only with
`--bar-minutes 1440`. `--equivalence-gate` = the trust anchor (CLOSE leg rtol≤1e-4 +
FLOOR leg ±0.02; divergence → data-integrity finding, exit 2, no artifact).

- **The BLOCKER (fixed) — the load-bearing one**: the no-`--symbols` daily branch ranked its
  universe by coverage over `ohlcv_daily` → on the scheduled OOS command it resolved an
  arbitrary alphabetical/meme-coin tail overlapping the minute majors baseline
  (XBT/ETH/LTC/XMR/ZEC/ETC/DASH/REP) by ONE asset (ZEC). The ~240-decision OOS benchmark
  would have been scored on the WRONG, non-reproducible universe, and the equivalence gate
  was blind to it (it resolved a shared minute-top-N universe and fed THAT to both sources
  while the shipped daily loader ran a different set). Fix: `_minute_ranked_symbols`
  coverage-ranks the top-N over the MINUTE `ohlcv` table (exact parity with the env minute
  loader — same `GROUP BY a.id ORDER BY COUNT(*) DESC LIMIT n`, same `_to_epoch` bounds,
  `cache_path=None`) and reads those symbols from `ohlcv_daily`; `_resolve_universe`
  delegates to the same resolver → daily==minute universe by construction (re-verified
  IDENTICAL ORDER on the real-DB OOS window); fail-loud if the minute table is absent.
- **MAJOR (fixed)**: gate FLOOR leg built the panels without pinning `args.symbols` →
  compared floors on disjoint universes; now pins `args.symbols = universe`.
- **MINOR (fixed)**: gate printed PASS/exit 0 when zero legs actually ran → now
  INCONCLUSIVE/exit 3 (real-divergence still exit 2, CLOSE-pass still exit 0).
- **Default byte-identical**: `--smoke` reproduces `net@21 −43.972249581758106` (md5
  unchanged); daily branch fully guarded; committed smoke JSON untouched. imprint.py stays
  minute-sourced (training-panel change out of scope).

**Planner-owned real run (NOT launched)**: `--panel-source daily --start 2025-07-03
--end 2026-06-30` (~240 OOS decisions vs the thin 31; FLOOR leg needs `--vector-weight-rebalance`,
runs on `trading_data.db`). Replaces the preliminary 832/E5-oos ledger row. Still a BENCHMARK
read only (operator D17 correction). **Watch item**: `ohlcv_daily.source` mixes `agg1m` /
`kraken_rest` — watch `kraken_rest`-sourced recent days when interpreting any residual close
diff on the real run.

## Where it stands

- Queue Ready is now EMPTY. Remaining rows: the Gated Stage-4 scheduled server run (not a
  factory job) + Parked (D1/D4). Zero OPEN decisions.
- The two real runs above are the planner's to schedule — they are the actual deliverables;
  the factory shipped the instruments that make them possible/correct.
- Both real runs feed the SAME open question tree: 849's numbers decide the D18 fix locus
  (value vs reward vs selection mechanics); 850 unlocks the real OOS benchmark of the imprint
  (832/E5-oos, benchmark-only per D17). The short ExIt re-try stays gated on 847's real-panel
  leg (which FAILED post-ship at −7.84 → that's what 849 autopsies).

## Questions that unblock the highest-value track

1. **Schedule 849's real run?** (frozen `ckpt_imprint_full`, fee16 ExIt flags, both schemes) —
   its per-signal rank + calibration numbers are the gate on the D18 fix decision and on any
   further training spend. This is the single highest-value next action.
2. **Schedule 850's OOS run?** (`--panel-source daily`, 2025-07-03→2026-06-30) — gives the
   binding ~240-decision OOS benchmark of the imprint, replacing the thin n=31 preliminary.
   Independent of 849; can run in parallel.

Neither was launched by the factory (expensive real-data runs are scheduled, not polled
in-loop — invariant). Tree clean, nothing pushed.
