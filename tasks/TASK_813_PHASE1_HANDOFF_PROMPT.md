# HANDOFF PROMPT — TASK_813 Phase 1 — complete triggering map + profitability-gate probe

> Paste everything below. Self-contained.

---

You're launching **Phase 1 of the volatility-harvesting agent (TASK_813)** — the cheap, kill-gated probe
that decides whether a working vol-harvesting agent is even possible on Kraken before any build.
**Read first:** `tasks/TASK_813_VOLHARVEST_AGENT.md` (the spec — esp. the **Architecture: detector +
gate**), `runs/806_predictability_probe/` + `runs/808_linear_backtest/` (the audited data path, leak
guards, cost model, per-decision Sharpe you REUSE).

## Context (you don't have this)
806→812 closed the high-frequency cross-sectional surface on Kraken 1-min taker — the signal is real,
linear-optimal, and cost-bound (turnover × 21 bp > edge, proven 5×); the optimal MCTS planner abstains
(812). The binding constraint was never the model class — it was **turnover × cost** on a signal that
re-ranks the whole book every bar. **The fork (TASK_813): harvest VOLATILITY (large moves) via a
complete triggering map (detector) + a profitability gate (fire only when edge > cost).** Large moves are
the one place edge exceeds cost; sparse trades make 21 bp negligible. **Phase 1 is the probe: does the
complete map CONTAIN a profitable subset?**

## Your mission (Phase 1): complete-map + profitability-gate probe
Build the **complete triggering map** (Layer 1 detector) and measure, OOS, leak-free, whether it
contains a **profitable subset** (triggers whose realized move > 21 bp cost) that clears the post-cost bar.

### Layer 1 — the complete triggering map (broad coverage; PRE-REGISTER before measuring)
Enumerate candidate large-move triggers across these families — each a **causal condition over 1-min
bars ≤ t**, with swept params:
- **Trend / breakout** — Donchian N-bar high/low breakout; MA-channel breakout; trend-confirmation (MA
  slope, consecutive HH/HL). Sweep N / MA window.
- **Volatility expansion** — bar-range or ATR expansion (range > k × trailing ATR); Bollinger breakout;
  realized-vol-regime shift (vol crossing a threshold). Sweep k / threshold / lookback.
- **Lead-lag vol propagation** (the original Oct concept) — asset A's recent vol/return spike → asset B
  predicted to follow. Detect A-spike ∧ B-not-yet-moved. Sweep lead asset, lag, spike threshold.
- **Regime** — high-vol-regime entry (where large moves cluster).

Completeness matters — the map must cover the space of large-move opportunities. Pre-register the full
trigger × param grid before measuring.

### Layer 2 — the profitability gate (measurement, leak-free)
For each trigger fire (detected on features ≤ t), measure the **realized forward move** over the ride
horizon (sweep {1 h, 4 h, 1 d, 4 d}) — this is the **LABEL (future only, no leak)**. A fire is
**profitable** if the realized move (in the trade direction) > round-trip cost (42 bp = 21 bp/side × 2).
Per trigger × param × horizon, compute: coverage (# fires) · **precision** (profitable fraction) · mean
gross edge/fire · post-cost Sharpe of ALL fires · **post-cost Sharpe of the PROFITABLE SUBSET** (the
ceiling a perfect gate achieves). Block-bootstrap 95% CI on Sharpe; BH-FDR across the full grid. OOS test
only (reuse 806's train/val/test split).

### Gate (kill or go)
- **GO:** ≥1 trigger family has a profitable subset with **post-cost Sharpe > 0, CI-separated,
  FDR-surviving** — the map contains enough profitable trades for a learned gate (Phase 2) to isolate.
  Report the map's best precision.
- **KILL:** no trigger family has a profitable subset → vol-harvesting is also cost-bound on Kraken 1-min.
  STOP honestly (a clean answer, not a failure).

## Audit items (independent auditor re-derives from code, not the summary)
- **Leak-free detection**: every trigger fires on features ≤ t only; the realized move is the LABEL
  (future), never fed back into detection. Assert it.
- **Honest 21 bp/side cost** (42 bp round-trip); **gross AND net both reported** (810 — a "profitable"
  claim on net-only with no gross-beat is the turnover artifact).
- **Per-decision Sharpe, H-strided** (the ride horizon), NOT per-bar × √525600 (809).
- **The profitable-subset is a MEASUREMENT CEILING** (uses realized move as label), NOT an ex-ante
  trading result — state it honestly. The LEARNED gate (predicting profitability ex ante) is Phase 2.
- Reuse 806's data path + leak guards + split; cite them.

## Constraints (non-negotiable)
- Server-side (CPU / light GPU for detection), OOM-safe (cgroup cap, persist RSS/VRAM peaks), **never the
  workstation display GPU**.
- Leak-free (806 guards), honest 21 bp costs, gross + net, per-decision Sharpe, FDR.
- Audit discipline: re-derive from artifacts.

## Hand-back
`runs/813_volharvest/PHASE1_MAP_PROBE.md` — the complete map (families + params), per-cell coverage /
precision / mean-edge / post-cost-Sharpe(all fires) / post-cost-Sharpe(profitable subset) + CI, the
GO/KILL verdict, and the map's best precision. **STOP for audit.** Do NOT start Phase 2 (the agent /
learned-gate build) until the auditor confirms the profitable subset is real (leak-free, honest-cost,
gross+net, CI-separated). If KILL, that's the clean close of vol-harvesting on Kraken spot — report it
plainly.
