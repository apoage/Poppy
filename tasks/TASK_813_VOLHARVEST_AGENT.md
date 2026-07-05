# TASK_813 — Volatility-harvesting agent (cost-compatible by construction)

**Status**: DRAFT — operator direction (2026-06-26). The working-agent fork. Same venue (Kraken), same
1-min data, no market-mechanism bending — but a different SIGNAL and FREQUENCY, reusing the proven
806–812 apparatus. **Probe-gated: do not build on an untested signal (the 806–812 lesson).**

## Mandate
A working agent that profits from market **volatility** — large moves (expansion / trend / breakout) —
not high-frequency tilts. The design principle that overrides everything: **cost-compatibility by
construction** — sparse trades, large per-trade edge (target moves ≫ 21 bp), so Kraken's 21 bp/side is
negligible. This is the opposite of the cost-bound path (whole-book re-ranking every minute → turnover ×
cost > edge → death, proven 806/808/809/810/812).

## Why this fork can work (honestly)
- The binding constraint was never the model class or the venue — it was **turnover × cost** on a signal
  type (cross-sectional re-ranking) that trades the whole book every bar.
- Volatility-harvesting (large moves) is **structurally cost-compatible**: you trade rarely (when a large
  move forms), ride it for hours–days, exit on exhaustion. Few trades, large per-trade edge → 21 bp is
  noise. Trend-following CTAs have done this for decades; crypto's regime swings make it plausible.
- **The 812 planner already learned the right policy** — it abstains when edge < cost. On a large-move
  signal where edge > cost, it trades. The infrastructure is built; the **signal** is the change.

## Signal candidates (vol-harvesting — detect from 1-min bars, TRADE sparsely)
1. **Trend-following (CTA-style)** — enter on established trend, ride it, exit on reversal. Sparse, large moves.
2. **Volatility breakout** — enter on vol expansion (range/ATR breakout), position for the large move.
3. **Lead-lag vol propagation (the ORIGINAL Oct concept)** — when asset A's vol spikes, B's vol follows →
   trade B before it expands. (This is the lead-lag fork folded in as a vol signal.)
> Detect from 1-min bars, but the TRADE frequency is sparse (hold hours–days). 1-min is the raw
> material, not the trade cadence. Re-ranking the whole book every minute is the dead path — do not
> revert to it.

## Architecture — complete triggering map + profitability gate (the operator principle)
The agent is **detector + gate**, not a single predictor:
- **Layer 1 — complete triggering map (detector, complete recall):** recognize the FULL space of
  candidate trade triggers — vol expansion, trend onset, breakout, lead-lag vol propagation, regime
  shift. Broad coverage — miss no opportunity. This is the *map*.
- **Layer 2 — profitability gate (high precision):** for each detected trigger, estimate the expected
  **post-cost** edge (predicted move − 21 bp) and **fire ONLY when predicted edge > 0**. Act only on
  profitable trades. This is the 812 value-head + threshold — 812 abstained because its map (cross-
  sectional) contained *no* profitable patterns; on large moves, profitable triggers exist, so the gate
  opens on those and stays shut on the rest.
- **Layer 3 — execution:** sparse, ride the move, exit on exhaustion.

The hard part is Layer 2: "trigger only on profitable" requires **predicting ex ante which triggers
yield move > cost**. That's the value head's job (predict move size net of 21 bp), and it's genuinely
hard — but it's exactly the post-cost value estimation 812 already does. The gate exists; the missing
piece is the complete map (Layer 1) of large-move triggers for it to gate on.

## Build — phased, probe-gated

### PHASE 1 — edge-vs-cost probe (cheap, days, kill-gated)
806-style, but framed as **detector + gate**: build the **complete triggering map** (Layer 1 — all
candidate vol/trend/breakout/lead-lag triggers), then measure across that map **what fraction of
detected triggers are profitable (per-trade edge > 21 bp) OOS, leak-free, FDR-surviving**, and whether
trading **only the profitable subset** (Layer 2 gate applied) clears the post-cost bar. Inherit 806's
guards (causal features, forward-only target, H-strided per-decision Sharpe, gross AND net reported).
- **Gate:** the profitable subset of the complete map yields post-cost Sharpe > 0, CI-separated, on at
  least one trigger family. **Report the map's precision** (profitable / total triggers) — that's the
  quantity that determines whether a learned gate can isolate the good trades. **If no trigger family has
  a profitable subset → vol-harvesting is also cost-bound on Kraken; STOP honestly.** If one clears →
  Phase 2 (the agent = a learned version of this gate on the surviving map).

### PHASE 2 — the agent (only if Phase 1 passes)
Reuse the **812 planner harness** on the surviving vol signal, post-cost objective (it will trade only
when edge > cost — exactly sparse vol-harvesting), leak guards, ≤24 GB. If the planner is overkill for a
sparse signal, a simpler RL agent on post-cost reward is acceptable — but the post-cost objective and
the abstain-when-edge-<cost behavior are non-negotiable (812's lesson).
- **Gate:** beat a simple trend-following rule AND buy-and-hold **POST-COST**, OOS, CI-separated,
  gross-beat reported. That's "working."

## Constraints (hard — from the campaign)
- **Cost-compatible by construction** — target moves ≫ 21 bp, sparse trades. If it drifts back to
  high-frequency whole-book re-ranking, it's the dead path; kill it.
- **Post-cost objective** (812 — honor the abstain behavior; the reward IS post-cost P&L, never gross/oracle).
- **Leak-free** (806 guards — features ≤ t, target strictly future; no oracle/lookahead).
- **Honest 21 bp costs; gross AND net reported** (810 — gross-beat must accompany any net-beat claim).
- **Per-decision (H-strided) Sharpe, not per-bar × √525600** (809).
- Server-side, OOM-safe (≤24 GB, cgroup cap, persist VRAM/RSS peaks), never the workstation display GPU.
- Audit discipline: re-derive from artifacts; an independent auditor checks edge > cost, leak-freedom,
  and gross-beat before any "working" claim.

## Honest prior (state it in the result)
Vol-harvesting (trend/breakout) on crypto has real historical edge in trending regimes and real
drawdowns in chop. It is **cost-compatible** (sparse large trades) — structurally unlike the cost-bound
cross-sectional signal — and **genuinely untested** in this campaign. Moderate prior: a better shot than
anything in 806–812, but not a sure thing. **Phase 1 settles it in days, cheaply.** If trend/breakout
edge clears 21 bp at sparse frequency, you have the first cost-compatible signal of the campaign and a
real path to a working agent; if not, vol-harvesting on Kraken spot is closed too — also a clean answer.

## First action
Phase 1 probe on Kraken 1-min (reuse 806's data + guard path): per-signal (trend / breakout / lead-lag)
gross edge vs 21 bp at sparse frequency, OOS, leak-free, FDR. Report which (if any) clears. **Stop for
audit before Phase 2.**

## References
- `lore/research/Master Orchestration Prompt.md` (the original concept — lead-lag + vol banking).
- `runs/806_predictability_probe/` (probe methodology + guards), `runs/812_mcts_trader/` (planner harness
  — abstains correctly, reuse on a high-edge signal), `runs/808_linear_backtest/` (cost model + per-rebalance Sharpe).
- The binding-constraint finding: 806→812 closed high-freq cross-sectional on Kraken taker; this task
  changes signal + frequency, not venue.
