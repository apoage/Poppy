> **STATUS (2026-07-08): factory half SHIPPED.** `experiments/830_corridor/` runner + SPEC +
> fixture smoke landed; panel-green (runner correct, zero diffs outside `experiments/`, env
> byte-identical). Fixture floor does NOT reproduce the +0.97 bar within tol — verified as a
> **replay-mapping artifact, NOT a `step_vector` bug** (see CHANGELOG / SPEC Attribution). Seeded
> **DECISIONS D6** (official floor mapping). **Remainder:** real-panel corridor numbers = SCHEDULED
> CPU run (24GB db read-only), not a factory-loop job — the load-bearing E0 verdict comes from there.

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
