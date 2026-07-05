# TASK_817 — Cleverer-algorithm probe (attack the cost side; untested OHLC structure)

**Status**: DRAFT — operator direction (2026-06-26). **The bleeding-edge take**, sibling to 814 (richer
data) and 816 (better operating point). Operator-flagged as research-frontier / lower-confidence. The
honest framing up front: the *prediction* axis is **closed** (811 proved the OHLC cross-section is
linear-optimal — a smarter *predictor* won't help), so this task does NOT chase a cleverer prediction
model. It probes the two axes the campaign never attacked: **execution (the cost side)** and **untested
OHLC structure**.

## The question (one sentence)
The binding constraint is **cost (≈42 bp round-trip) ≫ edge (≈3 bp)** — so can a **cleverer execution
algorithm** (passive/maker fills, adverse-selection-aware, optimal liquidation) drive the *effective* cost
below the edge, and/or does an **untested OHLC structure** (explicit stat-arb / cointegration mean-reversion)
carry edge the campaign's cross-sectional-momentum framing missed — OOS, leak-free, honest, FDR-surviving?

## Why this, now (what the campaign did NOT try)
- 806→813 attacked the **edge** side (better signal, better model, vol-harvesting) and lost to cost.
  811 closed the prediction frontier (linear-optimal). The campaign **assumed naive taker execution
  (21 bp/side)** throughout and **only tested cross-sectional-momentum-style + vol/trend structures.**
- The single most logical untried lever: **the problem IS cost, so attack cost, not edge.** Execution
  alpha is exactly that, and it's a "cleverer algorithm," not more data.

## The angles (ranked by prior; #1 is the real one)
1. **Execution alpha (PRIMARY — attacks the binding constraint).** Replace taker execution with modeled
   **passive/maker** execution: queue-position modeling, adverse-selection-aware limit placement, optimal
   liquidation (Almgren-Chriss-style) over the ride. Taker 21 bp/side → maker ~0 to −rebate when fills are
   favorable; a realistic maker/passive program can cut the round-trip from ~42 bp to ~10 bp or below →
   the **14× gap becomes ~3×**, and combined with a cleaner universe (816) could close. This is the highest-
   leverage untested idea in the whole campaign. NB: distinct from 816's "re-price at maker fees" — that
   asks *if cost were lower*; this **builds the execution to achieve it** (and must model fill probability +
   adverse selection honestly — the leak surface is assuming you get the maker fill you wouldn't have).
2. **Untested OHLC structure (SECONDARY).** Explicit **stat-arb / cointegration / pairs mean-reversion**
   on the cross-section (spread reversion between cointegrated pairs) — a *different* structure from the
   cross-sectional momentum 806/811 tested. Lower prior (it's still HF on this universe → likely cost-bound)
   but it is a genuinely different OHLC-derivable edge. Cost-aware portfolio/turnover optimization beyond
   810 (turnover penalty in the *allocation*, not just the objective) sits here too.
3. **Frontier ML (LOW prior — mostly closed).** GNN on the asset-correlation graph, cross-sectional
   attention, etc. — but these are still *prediction*, and 811 says prediction is linear-optimal. Park
   unless 1–2 surprise.

## Method
- **Angle 1 (execution):** needs **order-book / microstructure data** (L2 snapshots or trade prints) the
  DB lacks — Phase 0 is sourcing it (the hard part; overlaps 814's data-sourcing problem). Then a **fill-
  honest backtest**: model maker fill probability + adverse selection (you don't get filled on the good
  side for free), re-price the 806/813 signal under modeled passive execution, gross AND net, OOS. The #1
  audit item: **the fill model must not assume favorable fills** (that manufactures fake cost savings —
  the execution analogue of a timestamp leak).
- **Angle 2 (stat-arb):** reuse the 806 OHLC path; cointegration test on the train split, spread-reversion
  signal, per-decision H-strided Sharpe, gross AND net, BH-FDR, OOS. Leak-free.

## Gate (kill or go)
- **GO**: (Angle 1) modeled passive execution drives effective round-trip cost below the OHLC edge with a
  fill-honest backtest → net post-cost Sharpe > 0, CI-separated, FDR, gross-beat → **cost is beatable; the
  campaign's taker assumption, not OHLC, was the wall.** OR (Angle 2) a cointegration structure clears cost
  OOS. → a deployable strategy + execution layer.
- **KILL**: fill-honest passive execution still can't get cost below the edge, AND no untested structure
  clears → the cost wall is real even against clever execution; OHLC on Kraken spot is closed against the
  algorithm axis too. STOP honestly.

## Honest prior
**Low-to-moderate, and explicitly bleeding-edge.** The prediction axis is closed (811), so most "cleverer
model" ideas are dead on arrival. **Execution alpha is the one genuinely-promising clever angle** — it
attacks the actual binding constraint (cost) — but it is operationally hard (needs microstructure data, a
fill-honest backtest is easy to fool yourself with), and getting consistent maker fills on the *informed*
side is exactly what adverse selection prevents. Stat-arb is a real but likely-also-cost-bound structure.
This is the highest-ceiling, highest-risk, highest-effort of the three sibling tasks (814/816/817).

## Constraints / audit
Leak-free (causal features ≤ t; forward = label); **fill-honest execution model (no assumed favorable
fills)** — the #1 audit item for Angle 1; honest cost, gross AND net (810); per-decision H-strided Sharpe
(809); BH-FDR; OOS only; reuse `probe`/`backtest`; server-side, OOM-safe, never the workstation display GPU.
STOP for audit; the auditor checks the fill model is adverse-selection-honest and the edge survives it.

## References
- `runs/811_ic_probe/` (prediction is linear-optimal → don't chase a smarter predictor), `runs/808/810/813`
  (the cost wall + the gross-vs-net audit standard this must clear).
- `tasks/TASK_814_RICHERINFO_PROBE.md` (richer data) + `tasks/TASK_816_OPERATING_POINT_PROBE.md` (better
  operating point) — the three sibling reads of the OHLC close. 816 asks *if* cost were lower; 817 Angle 1
  *builds the execution* to make it lower.
- `[[project_modeling_surface_exhausted]]`.
