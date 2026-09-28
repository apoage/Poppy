# TASK_918 — tactic library + convex-hull ceiling (the selector-class kill test)

**Status**: PROPOSED 2026-09-05 (planner; = tactics-panel row 1, `lore/fable_tactics_panel_2026-09-04.md`
§"Proposed rows", upgraded with the dead-panel refinements in `lore/harvest_6b32_appendix/tactics_gaps.md`).
**AMENDED 2026-09-17 (planner, operator handed unattended time): the data precondition is
satisfiable READ-ONLY — see the amendment block under Precondition below. TAKEABLE now.**
**Size XS/S. Minutes of CPU. Risk: LOW — new module only, no canon surface. High closure value.**

## What it decides, one sentence

If the ex-post best fixed mixture of a pre-registered tactic library is below the bar, **no
selector of any kind — MuZero included — can clear the goalposts with these arms**, and the
entire selector/loop class closes in minutes instead of batteries.

## The library (K = 8 position matrices on the PIT 273-asset true-daily panel)

flat · 819 trend · long-flat trend · equal-weight constant-mix · beta-hedged constant-mix ·
vol-targeted trend (covers the parked vol-target row — its analytic ceiling is
√(1+CV²), net Δ ∈ [−0.10,+0.08]) · **7d reversal = TIME-SERIES −sign(7d) equal-weight**
(label corrected per tactics_gaps: NOT cross-sectional) · **21d cross-sectional reversal
(tercile L/S)** — the never-measured arm and the first direct H2 read (`docs/pathway.md:38`).

## Scoring (own module; do NOT use `cell_metrics` — it cannot score these windows; use the 898
`score_window` semantics)

Window 2024-01→2026-06 (**GATE: 899 daily ingest landed first** — else the unseen window
shrinks to ~24 decisions; if 899 is blocked, run with window truncated at 2026-03 and DISCLOSE,
never silently).

**AMENDMENT 2026-09-17 (planner) — preferred fallback, read-only, NO DB writes:** the staged
daily catch-up (`runs/858_backfill/api_2026-08-31/daily/`, 56/56 canon pairs, seam-verified
2026-08-31: 62 overlap rows vs the DB, zero mismatches; manifest.json carries per-pair
first/last) is readable as FILES. The loader concatenates DB-daily (through 2026-06-21) +
staged CSVs, RE-VERIFIES the seam at load time (re-derives the overlap-row check and refuses
on any mismatch — the 2026-08-31 verification is a recorded fact, not a load-time
assumption), and pins the loader + seam hash in the artifact. This satisfies the gate's
substance (the 2026H1 seam) with zero DB writes; the DB INGEST itself stays gated (RO02 /
operator go-word) and is NOT needed for this row. The task file's own truncate-at-2026-03
fallback remains the LAST resort if even the staged files are unavailable. Fees: 916 SHIPPED
(`18d42d3`) — the 916 tiers + borrow columns are mandatory, not optional. Log AND simple returns both (the scorer's log-sum cannot see a rebalancing
premium by construction — `trader/backtest.py:221-227`). Fees at the tiers from 916 + borrow
model when 916 ships; at minimum legacy 21 bp + a borrow-inclusive column. By-year reads
(2024/2025/2026H1).

## Comparators (all pre-registered)

best fixed mixture (the hull) · ex-post oracle single-tactic selector · walk-forward
best-single · Hedge (theory η) · random-selector null (200 seeds) · **equal-weight blend of
the arms (the BINDING null — absent comparators are how hindsight luck sneaks in)** ·
**B0: 1-month tactic-return momentum (the strategy-momentum spurious-IC control — a selector
that cannot beat B0 has learned nothing but momentum-of-tactics)**.

## Statistics discipline

Block-CI certification FIRST (block_len=5 under-covers monthly switches; effective n = number
of decisions ≈ 30, not daily rows). Report **SR0(N)** beside every "best cell" (oracle-null
selection inflation ≈ +0.5–0.75; fires ≥ +0.97 30–45% by chance at this N — the loops-panel
standing rule). 819 TEST B2 is the receipt to cite against any "adaptive refit" arm idea
(per-regime refit lost 4/4 OOS: −2.34/−2.41/−1.60/−2.09 vs TSMOM).

## Gates (binary, pre-registered before the first run)

- **STEP-0 KILL: hull < +0.97 ⇒ record hull, oracle, blend and the 21d-reversal book numbers
  and STOP the selector class** (MuZero-as-selector included). 85% prior it closes here.
- **PASS conjuncts** (all required): a selector ≥ +0.97 with CI lo > 0, beats the equal-weight
  blend AND walk-forward best-single by ≥ 0.20 each, 2026H1 > 0, flat ≤ 50% of days, cadence
  sign agreement with its arms.

## Consequences

- STEP-0 kills ⇒ everything rides on 915's IC ceiling (prediction half) and 917's instruments;
  the "MuZero re-enters as selector" idea (tactics file, "MuZero's place") is CLOSED.
- PASS (≈15%): the winning selector owes a 3-seed walk-forward battery before any further
  claim; switching costs then decide whether the selector needs to be sequential at all.

## Out of scope

Learning inside the selector (that is 915-downstream), execution tuning (919), scorer changes
(920 — this module carries its own honest accounting instead).
