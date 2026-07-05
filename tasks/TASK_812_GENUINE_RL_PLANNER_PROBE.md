# TASK_812 — Genuine-RL planner probe (post-cost, leak-guarded, death-spiral-gated)

**Status**: DRAFT — Phase A inventory COMPLETE (`runs/812_mcts_trader/INVENTORY.md`, 2026-06-25).
Phase B (minimal loop) **awaits operator go/no-go** (the Leg-A WALL finding is premise-changing; see §1).
**Lineage**: executes the never-run `TASK_807_GENUINE_RL_REBUILD_SCOPE.md`; sits downstream of the
closed 806→811 modeling surface (`[[project_modeling_surface_exhausted]]`).

## The question (one sentence)

> 811 proved the 1-min Kraken signal is **linear-optimal at single-step prediction** and cost-bound; the
> one untested hypothesis is whether a **post-cost, leak-guarded, genuine-RL multi-step planner** (MuZero
> ExIt, MCTS-in-training) extracts path-dependent structure — entry/exit/hold timing that decouples edge
> from turnover — that single-step scoring cannot.

## Why now, and the honest prior

- **Premise correction (load-bearing):** the pivot assumed pre-666 RL was *promising*. Phase-A Leg A
  found it is a **WALL** — every "promising" alpha (1753%/526%/76%/50,000%) was a **leak artifact**;
  leak-free random α ≈ −0.02%; the one honest trained benchmark was **−4.3% in bull, 0% hold**
  (`runs/812_mcts_trader/inventory/02_*.md`). BUT 666 was a **speed pivot** ("4 weeks not 4 months") off
  a flat, **prediction-dominated (97% of loss)**, 40-bug, **never-cleanly-run** objective — *not* a proof
  RL fails. The clean experiment (post-cost reward + leak guards + ε-greedy from step 0 + no prediction-
  loss domination) **has never been run.**
- **Prior: LOW.** 811 linear-optimal + cost-bound; Leg-A wall; death-spiral unsolved. Even a gross-beat
  still faces the cost arithmetic. This is a **probe with a days-not-months kill gate**, not a campaign.

## Scope guardrails

- **Phase A (DONE):** read-only inventory. No code changed, no run. Deliverable `INVENTORY.md`.
- **Phase B (this task, on go):** build the **minimal** loop ONLY — not the full trader. Train on the
  audited Kraken 1-min split, STOP for audit. Do NOT commit to a full build until the minimal loop clears
  the walls AND the auditor confirms leak-freedom + death-spiral-clearance + honest post-cost gross-beat.
- **Server-only, OOM-safe:** gpu-server, single verified-idle GPU (1/2/3 — **never GPU 0** VirtualBox-
  reserved, **never** the workstation display GPU). `safe_run`+tmux, `python -u`, persist RSS+VRAM peaks.
- **Sub-agents** may commit + report; **never** call `deploy.sh` (main agent owns server writes/deploy).

## The minimal loop (Phase B build)

1. **Spine** — reintroduce `train_jax_fast.py`'s genuine-RL loop into the current codebase (NOT a
   git-revert; keep encoder/eval/data infra). Pure self-play MCTS RL: visit-count policy targets +
   bootstrapped/MC value targets (`self_play.py` + `trainer.py` MuZero core).
2. **Objective** — reward = **post-cost realized P&L**: `fee_bps=16` + `slippage_bps=5` (21 bp/side
   all-in), **fill lagged to t+1**. **Remove/down-weight the prediction loss** (the 97% trap; 811 closed
   prediction). Pure trading gradient.
3. **Death-spiral guard from step 0** — ε-greedy target mixing ON (`EpsilonSchedule 0.25→0.05`,
   default is 0.0=OFF, the trap); log the triad every step (`policy_entropy`, `mcts_visit_entropy`,
   `trade_rate`); kill-on-onset wired. Adaptive entropy bonus as backup. **No burn controllers** as the
   anti-HOLD lever.
4. **Leak discipline** — RIP the oracle (`path_solver.greedy_lookahead_path` + `oracle_distillation.py` +
   BC/imprint/oracle-mask loss terms); value target from `compute_returns` only; causal features ≤ t;
   forward return = label never input; carry 810's value-identity + gross-vs-net asserts.
5. **Fit** — small=7M, sims=32, B=256, **replay off-device** (`window_starts`, not observations) →
   ~6–7 GB on one 3090. bf16 optional. Two-GPU NOT needed.

## Falsifiable convergence gate + kill condition

- **Death-spiral kill:** triad (entropy↓ + visit_entropy↓ + trade_rate→rail) trends N steps → kill.
- **Convergence signal:** held-out OOS post-cost realized alpha trends **positive AND gross-beats the
  linear 808 baseline** under the eval-MCTS scorecard, with **gross(cost=0) on H-strided non-overlapping
  rows** (the discipline that caught 809/810). **Net-beat without gross-beat = turnover artifact → kill.**
- **Abandon:** no stability win AND no post-cost gross-beat within the budgeted days → the planner
  hypothesis closes; lever is unambiguously a **data/problem decision** (operator), not modeling.

## Deliverables

1. `runs/812_mcts_trader/INVENTORY.md` — **DONE** (Phase A: inventory + minimal-loop plan + 24 GB
   verdict + death-spiral mitigation + Leg-A WALL + go/no-go recommendation).
2. `runs/812_mcts_trader/MINIMAL_LOOP_RESULT.md` (Phase B): stability, 24 GB VRAM peak, post-cost Sharpe
   vs linear + CI, **gross-beat**, leak-freedom asserts, death-spiral triad traces. Then STOP for audit.

## Acceptance checklist (audited separately)

- [ ] Phase A: no code changed, no GPU run; Leg-A verdict backed by cited pre-666 metric (DONE).
- [ ] Phase B reward is post-cost (16+5 bp, t+1 fill); prediction loss removed/down-weighted; verified.
- [ ] Oracle ripped — value target provably from `compute_returns`, never `path_solver`/distillation.
- [ ] Death-spiral triad logged from step 0; ε-greedy mixing ON; kill-on-onset demonstrated or clean.
- [ ] Result reports **gross(cost=0) Sharpe on H-strided rows** alongside net; net-beat-without-gross
      flagged as artifact.
- [ ] 24 GB VRAM peak recorded; ran on gpu-server GPU 1/2/3 (never 0, never workstation display GPU).
- [ ] STOP for audit before any full-build commitment.

## References

- `runs/812_mcts_trader/INVENTORY.md` + `inventory/0{1..5}_*.md` (the Phase-A evidence + cites).
- `tasks/TASK_807_GENUINE_RL_REBUILD_SCOPE.md` (the read-only scope this executes).
- `runs/806_predictability_probe/`, `runs/808_linear_backtest/`, `runs/810_postcost_headroom/`,
  `runs/811_ic_probe/` (the audited linear baseline + post-cost discipline to reuse AND beat).
- `[[project_modeling_surface_exhausted]]`, `[[project_classifier_alpha_anti_correlation]]`,
  `[[project_bull_blindness]]` (the failure patterns the probe must escape).
