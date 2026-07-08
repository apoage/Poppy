> **STATUS (2026-07-08): COMPLETE — E0 official.** Real-panel run done (safe_run CPU, db
> read-only, 279 assets × 1704 bars 2021-01-01..2025-09-01): net@21 corridor **[floor_inenv
> +0.424, ceiling +29.698]**, engine ref **+0.751**; BEAR22 floor −0.143/ref +1.049. Identical
> to the fixture smoke ⇒ fixture daily==db daily confirmed. Gap |floor−ref|=0.327>0.2 ⇒ **H6
> ARMED** (per D6; conditional — see LEDGER 830/E0-REAL). Floor leverage-invariant. `step_vector`
> re-cleared. Artifacts: `experiments/830_corridor/results_real_panel.json`,
> `runs/830_corridor/run_real.log`. D6 = DECIDED (dual floor).

# TASK_830 — Instrument corridor: linear book + oracle bound through the vector env (E0, R0)

**Status**: GATED on 828 Stage 1 banner (needs `step_vector`).
**Pathway**: `docs/pathway.md` rung R0. Hypothesis: none — this calibrates the instrument.
**Size: S–M. Risk: medium** (read-only vs canon code; its OUTPUT gates everything above it).

## Purpose
Two numbers that every later experiment is scored against, both produced INSIDE the vector env
so the env itself is validated:
1. **Corridor floor** — replay the TASK_819 linear trend book as a sequence of `action_vec`s
   through `step_vector` on the daily panel, 2021–2025, cost=0 then fee16/slip5. PASS: net@21
   Sharpe reproduces the `trader/` engine's +0.97 within tolerance (pre-register the tolerance
   in SPEC.md; expected slippage of a few bps from simultaneous-rebalance semantics — assert
   and document any divergence, don't absorb it silently).
2. **Corridor ceiling** — per-asset max-extractable under the SAME cost model (reuse
   `mcts_trader/jax_agent/path_solver.py` per asset, sum the book). Oracle stays in its valid
   role: measurement instrument, upper bound. NEVER a teacher (H5).

## Deliverables
- `experiments/830_corridor/SPEC.md` (pre-registered tolerance + windows) and a small runner
  importing the env + `trader/` baseline — no edits to monster files (token-size mandate).
- `experiments/LEDGER.md` line: verdict + the corridor `[floor, ceiling]` per window/cost tier.
- If the floor does NOT reproduce: that is the finding — file the discrepancy against 828
  Stage 1, do not tune until it matches.

## Factory split
Factory job = implement runner + fixture smoke (fixture daily panel, truncated corridor fine).
Real-panel numbers = scheduled CPU run (24GB db read-only), not a factory-loop job.

## Gates
- Fixture smoke green; `python -m trader selftest` green; zero diffs outside `experiments/830_corridor/`.
- Real run: both corridor numbers recorded in LEDGER with windows + cost tiers + seed n/a (deterministic).
