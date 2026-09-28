# Handoff — 2026-09-28 factory run 57 (profitability-recomposition window)

Armed 2026-09-28 12:41 (session d31db1e0), base HEAD `aa24f5a`. Steering 918 → 924 → 923 → 915
per `lore/factory_prompt_2026-09-28.md`. All four rows taken; all compute CPU-only on gpu-server,
`safe_run`-wrapped, private `/var/tmp/<task>-<role>/` scratch. Zero DB writes (fixture
`fixtures/trading_fixture.db` sha `b0f8de8b…` read-only throughout, unchanged). 907 tree untouched.

## Per-row outcomes

| Row | Commit / tag | Cycles | Verdict line |
|---|---|---|---|
| 918 | `ebe9cd5` / `task918-hull-step0` | 2 | prereg STEP-0 kill NOT FIRED (hull net@21 +5.409); post-hoc honest-tier hulls −1.307 SC / −0.897 T+1 → D86 |
| 924 | `745df70` / `task924-exec-feasibility` | 2 | G1 PASS on B1 snapshot; prereg gate NON-DISCRIMINATING; gap-clipped frozen-mid BTC 30.42% → D87 |
| 923 | `2e5e32f` / `task923-listing-book-A` | 3 (last = transcription closing pass) | UNDERPOWERED every exit; gross separates from placebo (thin), net cost-dominated → D88 |
| 915 | `ebdc3eb` / `task915-ic-ceiling-GA-GB` | 2 (+ closing-pass claim downgrades) | G-A 128/160 prereg, 109 survive a 1-day skip; G-B FAIL for all 109 at 80 bp → D89 |

### TASK_918 — tactic-library hull STEP-0
- Pre-registered (net@21, same-close): hull +5.409 (CI +4.186..+6.658; n 912 daily / 30 monthly)
  vs +0.97 → NOT FIRED. Carried by `reversal_21d_xs` (net21 +4.445, hull weight 0.60).
- Post-hoc (`experiments/918_tactic_hull/results/VERDICT.json`): hull re-optimised at net80 +
  borrow-low −1.307 same-close / −0.897 T+1. T+1 (all arms), hull at net21 +1.329 (CI lo +0.073,
  inside MDE95 1.753 of the bar). reversal_21d_xs T+1 net21 +0.733. Stale-close hypothesis refuted
  (1.64% of |PnL|). Cause of the timing sensitivity not established.
- Data: fixture ohlcv_daily (through 2026-07-06) + 31 staged USD CSVs to 2026-08-31, seam 488 rows
  / 0 mismatches. 398 assets (363 + 35 dead); the task's "273" is unattributed.
- Remainder not built: walk-forward / Hedge / random-null / B0 comparators (D86).

### TASK_924 — execution-feasibility instrument (Option B pre-gate)
- Capture path correction: srv `~/Trader/runs/907_live_fill_probe/capture_907_main.jsonl` (FINAL,
  1,440,479,339 B, sha `8b3f748d…`), not `~/captures/`.
- B1's 46.5% is from the 1.248 GB SNAPSHOT; the snapshot = first 1,248,413,957 bytes of the final
  file (sha `7aede82e…` both). G1 on that prefix: 127,070.878 / 273,233.044 s — PASS.
- The shipped frozen-mid numerator is gap-blind (`capture.py:2422-2436`). 924's gap-clipped
  (split-at-gap) frozen-mid: BTC 30.42 / ETH 12.44 / SOL 3.66 / XRP 1.01% (gap-blind 43.39 / 23.36 /
  14.38 / 9.79). Time-weighted quote survival p50/p90 BTC 32.5/160.5 s. Median rel half-spread
  0.0064 / 0.22 / 0.94 / 0.76 bp.
- Pre-registered disqualifiers were practically unfireable: `recorder.py:527-531` never emits a
  one-sided row; "AS(1h)" as specified = mean half-spread, not markout. Status
  `PREREG_UNBLOCKED__GATE_NON_DISCRIMINATING`.

### TASK_923 — post-listing event book phase A
- Direction SHORT (P3 profile: age 0–9d residual −78.3 bp/day), β=1 hedge, T+1 entry, 80 bp/leg +
  borrow-high. 386–388 events, ~134 cohort-week clusters.
- Spec-time MDE 470 / 640 / 801 / 1296 / 1573 bp vs +100 bp band → UNDERPOWERED at every exit.
- G2 as first implemented (|real| > max|placebo|, zero-centred) is mis-denominated under a
  constant cost; "NULL" withdrawn. Corrected (task wording): gross h21 real +1021.3 (CI
  +284.6..+1758.0) vs placebo ≈ 0; placebo interval from correlated draws is ~2–3× too narrow —
  pooled-SE z 1.59 / 2.68 / 3.17 / 3.11 / 2.91 vs 2.576. Population-mismatch confound refuted
  (1/397 events).
- Capacity median ≈ $3.6k/event. Shortability at listing unverified.

### TASK_915 — IC ceiling sweep (G-A) + fundamental-law bound (G-B)
- SPEC `d9b7d405…` (errata appended → `59fb144b…`). G-C (GP book, two instruments) NOT built —
  named remainder.
- Data: fixture ohlcv_daily carries true O/H/L/C/V; seam re-verified on all five columns (2,440
  cells, 0 mismatches). 384 assets.
- Official run (srv `/var/tmp/915-run`, 160/160, 3433.7 s, `grid_results.json` sha `19eef63a…`):
  G-A 128 pass; G-B binding (80 bp, borrow NOT charged) 127 FAIL / 1 MARGINAL
  (`y4__h1__ALL__ridge` +0.693) / 0 PASS.
- Same-close alignment (`features.py:31-33`, `targets.py:38-43`): F1 features end at close_t, label
  starts at close_t; the null families cannot see it. One-day-skip re-score: 109/128 survive
  (F2 24/24, F3 24/24, F4 24/27, ALL 24/26, F1 13/27; y4 0/10). The MARGINAL cell dies (IC 0.0039).
  All 109 survivors FAIL G-B (best `y1__h21__ALL__gbm` +0.037, skip1 IC 0.1201, N_eff 57.2).
- MDE at t=3.0 (pinned) vs t=3.6047 (Bonferroni): h1 0.0057/0.0069, h3 0.0128/0.0154, h7
  0.0207/0.0248, h21 0.0367/0.0441 — h7/h21 UNDERPOWERED either way.
- Pre-committed consequence: "G-A passes / G-B fails → record the IC/N_eff/turnover triple;
  breadth/cost are operator decisions; no trader build."

## Orchestrator errors this run (D83 record)
1. 918 brief: kill set at net@21 with a kill-side-only direction argument ("can only under-kill");
   no NOT-FIRED-side protection. Caught by the brief-direction lens.
2. 918 cycle-2 brief premise: the "+0.805 T+1 hull" was the judge's reversal-only-delayed number;
   all arms delayed = +1.329. Caught by the fix implementer.
3. 924 brief: carried B1's 46.5% and "longest gap-free stretch 561.8 s" from the snapshot to the
   final capture (the error class run 53 recorded). G1 failed on the final file first.
4. 924 cycle-2 brief: asked for both the split-at-gap rule and reproduction of the unsplit 33.31%.
5. D86 first draft claimed a selector cannot beat the best fixed mixture; corrected before commit
   (a time-varying selector is not bounded by the fixed-mixture hull).
6. Watcher bug: `until … ! pgrep -f run_full_pass.py` over ssh matched its own command line; 924's
   pass finished 14:52 and was noticed 19:27 — ~4.5 h of the window idle. Use a done-marker only.
7. 923: the zero-centred G2 criterion was flagged by the orchestrator from the judge's arithmetic;
   the judge itself had passed it.

## Ops notes
- 918 fix agent's first posthoc launch spun ~61 cores (default BLAS threads); it killed its own
  process and relaunched with 4-thread caps. Every later brief carried OMP/OPENBLAS/MKL caps.
- 924 implementer wrote 3 files to srv `/tmp` outside its scratch; deleted them itself.
- 915 implementer's dry run `cd`'d into srv `~/Trader` to use its `safe_run.sh` (read-only use; the
  prereg judge confirmed no writes to the srv tree).
- gpu-server GPU 0 is a new driver-unrecognised card (operator 2026-09-28); not used.
- Srv scratch left in place: `/var/tmp/{918,923,924,915}-*`, `/var/tmp/915-run`. Safe to delete
  once the operator has read this.

## Open decisions seeded (operator)
D86 (918 selector class), D87 (Option B after 924), D88 (Option A after 923), D89 (915 G-B result).

## Addenda (end of window)
- 915 cycle-2 re-verify: `make_verdict.py`'s "status rule fixed before the skip1 table was read" is
  contradicted by srv mtimes (22:18:45 vs 22:17:57) — downgraded to POST-HOC in a closing pass
  (comment + README only; the consequence key never depended on it). Skip1 purge margin shrinks h→h−1
  at one boundary row per refit — "leak-free" wording downgraded.
- `grid_results_annotated.json` (9.4 MB, sha `1d475186…`) kept OUT of the tree: srv
  `/var/tmp/915-fix/results/`, workstation `/var/tmp/915-archive/`.
- 915's real run took 57 min, not the ≤2 CPU-days budgeted — read in-window (fix agent waited on the
  done-marker); the "scheduled, not polled" instruction was written for a multi-day run.
- Cross-row synthesis: 918, 915 and 923 all show gross signal that the 916 cost stack removes
  (918 reversal +4.45 → −0.37 at 80 bp; 915 best survivor +0.04 at 80 bp; 923 h21 gross +1021 bp vs
  ~664 bp all-in). 918 and 915 both carry a same-close timing fragility in daily reversal
  features. Execution cost is the common binding constraint; D87(a) (measure adverse selection via the
  D81 re-cut) is the one experiment that bears on all three.
