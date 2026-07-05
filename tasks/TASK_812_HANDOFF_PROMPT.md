# HANDOFF PROMPT — TASK_812 — build the MCTS trader (Phase A inventory → Phase B minimal loop)

> Paste everything below. Self-contained.

---

You're pivoting from the closed linear/cross-sectional surface — **806-811 established the 1-min Kraken
signal is real, linear-optimal, and cost-bound; no model beats linear's IC; the binding constraint is
cost, not modeling** — to the operator's consistent conviction: **build the MCTS-based trader (MuZero
ExIt), trainable in ≤24 GB VRAM, reusing the existing `mcts_trader/` tooling.** This is genuinely
broader: multi-step *planning* optimizing cumulative post-cost P&L via search, vs the single-step
prediction surface that's closed.

**Read first:** `tasks/TASK_807_GENUINE_RL_REBUILD_SCOPE.md` (the 807 WALL verdict — your map of the
known walls on THIS exact path); `runs/806_predictability_probe/` + `runs/808_linear_backtest/` (the
audited linear baseline + leak-guard discipline you must reuse AND beat); `runs/810_postcost_headroom/`
(the post-cost objective + the gross-beat-metric lesson).

## The known walls — go in eyes-open, address them, don't rediscover them
1. **Death-spiral (807, UNSOLVED)** — the MCTS-sharpening → over-trade → collapse that killed pre-666
   ExIt; "routed around, never solved." **#1 risk; it's research, not engineering.** The build MUST carry
   a death-spiral diagnostic + mitigation from day 1.
2. **24 GB VRAM (807, was intractable)** — pre-666 was "single-3090-intractable." Fitting model +
   vectorized mctx + replay buffer in 24 GB is the first engineering win. **Hard constraint.**
3. **Leak-free + post-cost objective (666/TRM bug, now solvable)** — value head learns **post-cost
   realized P&L**, NOT leaky oracle-PnL. Reward = the trading objective. Reuse the 806/810 leak guards.
4. **811 (linear-optimal)** — prediction/representation is bounded; MCTS's value can ONLY come from
   *planning* — multi-step action sequences that decouple edge from per-step turnover (the thing 810's
   single-step cost-awareness couldn't).
5. **Cost ceiling (808-810)** — the gate is POST-COST.

## The only gate that counts
Beat `linear_xs` (808) **POST-COST** on the audited OOS test, CI-separated, with the gross-beat +
per-rebalance-Sharpe + leak-free discipline. **Nothing else is success.**

---

## PHASE A — grounding inventory (READ-ONLY; report, then proceed to B)
Inventory what exists so the build starts on real ground:
1. **`mcts_trader/` tooling** — MuZero model (`muzero_model.py`), `mctx_search.py`, `self_play.py`,
   `replay_buffer.py`, TRM (`trm_flax.py`). For each: reusable-as-is / needs-repair / bit-rotted.
   `muzero_model.py` drifted ~9 commits (807) — reconcile.
2. **`train_jax_fast.py` (the ExIt entry)** — does it import/run today? Enumerate the dep-reconciliation
   needed (807 est. ~2-4 days).
3. **Locate the death-spiral** — where in code, the mechanism, what mitigations were tried and why they
   "routed around, never solved" it. Propose a concrete day-1 mitigation.
4. **VRAM feasibility (single-GPU first, two-GPU escalation)** — primary target: fit a minimal ExIt
   (small TRM + vectorized mctx + replay) in **≤24 GB on one GPU**. Estimate the footprint; what must
   shrink; is it feasible? **If not**, scope the two-GPU fallback — the natural ExIt split: **GPU-0 =
   trainer** (fwd/bwd, JIT step), **GPU-1 = self-play/mctx search + replay**. Cover: which workloads go
   where, the cross-device data flow (episodes GPU-1 → trainer GPU-0), the JAX multi-device approach
   (device placement / pmap), the added orchestration cost, and **which two GPUs are actually available**
   (gpu-server GPU map; recall GPU 0 is VirtualBox-reserved/forbidden). Note explicitly: two-GPU relaxes
   the MEMORY wall only — it does not touch the death-spiral, the leak/post-cost objective, the
   linear-optimal ceiling, or the cost-bound gate.
5. **The leak/oracle surface** — where the value head gets its target today (leaky oracle-PnL), and the
   concrete wiring to make reward = post-cost realized P&L.
6. **Write the formal TASK_812 spec** from the above: minimal-loop design (model size, search width,
   reward, leak guards, death-spiral mitigation), 24 GB budget breakdown, the gate.

**Deliverable A:** `runs/812_mcts_trader/INVENTORY.md` — the inventory + derived minimal-loop plan + the
24 GB feasibility verdict + the death-spiral mitigation strategy. Report it, then proceed to Phase B.

## PHASE B — minimal loop ONLY (prove the walls clear; then STOP)
**Do NOT build the full trader.** Build the minimal ExIt loop that proves the load-bearing unknowns:
- Small model that **fits 24 GB** (per Phase A's budget). Leak-free (806/810 guards). Reward = post-cost
  realized P&L (not oracle). Death-spiral mitigation active (per Phase A).
- Train on Kraken 1-min (audited split).
- **Check: (a) trains stably — no death-spiral collapse; (b) fits 24 GB; (c) produces any post-cost
  signal at all.**
- Benchmark vs `linear_xs` POST-COST (the gate), gross-beat + per-rebalance Sharpe + leak-free.

**Deliverable B:** `runs/812_mcts_trader/MINIMAL_LOOP_RESULT.md` — stability (death-spiral y/n), 24 GB
fit (peak VRAM), post-cost Sharpe vs linear + CI, gross-beat. **Then STOP for audit.** Do NOT commit to
the full trader build until the minimal loop clears the walls and the auditor confirms.

## Constraints (non-negotiable)
- All training/compute on gpu-server, **never the workstation display GPU** (hard-locks the 45 GB box —
  burned 3×). Workstation = edits / `py_compile` only.
- **OOM safety by cgroup construction** (cap = peak × ~1.5, `python -u`, opt out of oomd pressure,
  persist RSS + VRAM peaks).
- **VRAM: ≤24 GB single-GPU is the primary target**; the two-GPU executor/support split (Phase A item 4)
  is the staged escalation if it won't fit — not the default. Report peak VRAM either way.
- **Audit discipline** — every result re-derived from artifacts; leak-free asserted (value-identity
  checks like 806/810); gross-beat reported alongside net.

## Hand-back
Phase A inventory → report + proceed → Phase B minimal loop → **STOP for audit** before any full-build
commitment. Report: the inventory, the 24 GB feasibility verdict, the death-spiral mitigation, the
minimal loop's stability + 24 GB fit + post-cost-vs-linear gate verdict, RSS/VRAM peaks. The auditor
verifies leak-freedom, death-spiral-clearance, and whether it honestly beats linear post-cost before the
full build is greenlit.
