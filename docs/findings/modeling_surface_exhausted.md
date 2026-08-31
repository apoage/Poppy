# modeling surface exhausted

> 806→811 CLOSED the MODEL-SIDE on Kraken 1-min. 810: post-cost objective = turnover artifact. 811: signal is LINEAR-OPTIMAL (IC-max model gets 40–61% of linear's IC, loses at every H). Signal real but linear + cost-bound. 812 (2026-06-26): genuine-RL MuZero/ExIt planner ALSO NULL — greedy policy ABSTAINS (100% HOLD on OOS). 813 (2026-06-26): VOL-HARVESTING fork (trend/breakout/vol-expansion/lead-lag) ALSO closed — the one apparent GO (C1 BTC-lead-lag) was selection luck on n=6; cross-regime hold-out (n=25) it INVERTS to net -4.08/gross -3.27. ALL signal classes (prediction, planning, vol-harvesting) are cost-bound on Kraken 1-min spot. 816 (2026-06-26): the OPERATING-POINT escape ALSO closed — the cross-sectional signal stays cost-bound across cleaner universe (drop 31 junk pairs) × futures cost (3bp) × lower frequency (to daily); KILL 0/54. So OHLC is cost-bound across the operating-point axis too, not just at 1-min spot taker. BUT the 816 residual — low-turnover TSMOM trend-HOLD — found the campaign's FIRST REAL EDGE: cross-regime-robust (positive shorting the 2022 −75% bear 13/15 cells, survives LUNA-out, positive in all trending regimes, bleeds only in chop = genuine trend-following NOT bull-beta), turnover ~0 so cost-immune, beats buy-and-hold; modest (FULL Sharpe ~0.7-1.0), crowded (CTA), chop-fragile, fails strict 15-way FDR (power limit not falsification). So: cross-sectional/high-turnover OHLC is cost-bound, but DIRECTIONAL trend-following is real + tradeable on low-cost perps. CTA premium DOES live on Kraken crypto. The cross-sectional vs directional distinction is the key. Cleanest cell L14d/H3d. Lever beyond TSMOM = richer DATA (814) / on-chain (Iris).

*Migrated from local agent memory 2026-08-31 (`project_modeling_surface_exhausted.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**The 806→810 wave EXHAUSTED the modeling surface (verdict 2026-06-25, Fork-C).** The 1-min Kraken
cross-sectional signal is REAL (clean-OOS IC 0.06–0.15; gross cost=0 Sharpe positive) but **cost-floor-
bound post-cost** at every level tried:
- **806**: signal exists (clean-OOS linear IC 0.06–0.15).
- **808**: linear dies to Kraken taker cost (0/72 cells post-cost).
- **809**: smarter rule (conviction gate + min-hold) + maker — ceiling-bound. The headline "+5.08 Sharpe"
  was a **per-bar-Sharpe annualization artifact** (caught by the audit; aggregate to holding period).
- **810**: a small model trained on a differentiable **post-cost trading objective** (the right objective,
  which 666's oracle-distillation never had) — beats linear post-cost on NET **only by trading less**; on
  GROSS (cost=0) it scores WORSE than plain linear (TEST_oos: model gross +15.43 vs linear +24.43, both
  positive). **Turnover-reduction artifact, NOT durable headroom.** Caught by the audit + the resolving
  gross-at-cost=0 measurement on H-strided non-overlapping rows.

**The POST-COST-OBJECTIVE branch is closed (only that branch):** edge and turnover are COUPLED — the edge
IS in the high-turnover trading; you can't cut cost without cutting the edge (810 proved it: lower turnover
dropped gross proportionally). A bigger model on the SAME post-cost objective inherits the trap → wasted
run. **So do NOT run "full encoder on the post-cost objective" (the original 811 — DROPPED).**

**Revised TASK_811 (operator 2026-06-25) — RAN; verdict LINEAR-OPTIMAL → model-side FULLY CLOSED.** Trained
an IC-maximizing model (real-capacity MLP 4032→512→256→64, cross-sectional correlation loss — PURE
prediction, NOT post-cost Sharpe) on the 806 features/split, benchmarked vs linear's IC. **The model is
WORSE at every horizon: IC 0.093/0.064/0.024 (H=1/5/20) vs linear 0.151/0.141/0.060 — 40–61% of linear,
the >20%-relative kill criterion missed in the WRONG direction.** A fair loss, not undertrained (improved
0.022→0.093 @ H=1 with full data; feature value-identity 9.89e-07; linear IC == 806). **There is no
nonlinear cross-sectional structure to exploit — the closed-form linear ridge is optimal. No model fixes
the 1-min Kraken signal.** `runs/811_ic_probe/` (ic_model.py, cache_close.py, result.md). Infra: the
recurring ~2.5GB-load oomd is fixed by caching `close` to a 513MB npz (`cache_close.py`) — VRAM-resident
runs load that, no build peak.

**TASK_812 (operator pivot 2026-06-25; RAN 2026-06-26) — PLANNING side CLOSED NULL. The last open seam.**
811 closed single-step PREDICTION; the one untested hypothesis was multi-step PLANNING (genuine-RL MuZero/
ExIt). Phase A inventory found the pre-666 RL was a WALL (the operator's "promising models" = leak artifacts;
honest trained α was −4.3% bull), BUT the clean experiment (post-cost reward + leak guards + ε-greedy from
step 0 + prediction loss removed) had NEVER been run — so we ran it. Minimal loop: `train_jax_fast.py`
genuine-RL path, small=7M, sims=32, post-cost fee16/slip5 (asserted), value from `compute_returns` (oracle
ripped), ε-greedy 0.25→0.05 in-loop target mix, death-spiral triad logged, 2000 epochs/~8h GPU2.
**Result: the planner ABSTAINS.** Trained: plateaued ~94% HOLD, no convergence (in-sample α ~−2%, flat since
ep ~1050); entropy fell then plateaued (mitigations CONTAINED the spiral but couldn't manufacture a
profitable policy) = the operator's pre-named "spiral-contained-but-no-learning = definitive RL close."
OOS eval on the final checkpoint (greedy, vs linear_xs, H-strided gross+post-cost): **policy turnover
0.000, NaN Sharpe — 100% HOLD (action histogram [0,5000,0,0,0], non-HOLD 0.000%) → zero gross edge → NO
gross-beat** (linear gross +108/+24 @ H=5/20). Verified genuine (not a repricing bug): same harness scores
linear with real turnover; matches the HOLD-plateau. **Mechanism: on a cost-bound signal under a post-cost
objective, every trade loses net of 21bp/side, so the optimal learnable policy is don't-trade — and the
planner found it.** `runs/812_mcts_trader/` (INVENTORY.md, eval_postcost_vs_linear.py, MINIMAL_LOOP_RESULT.md),
`tasks/TASK_812_*`, `lore/notes_2026-06-26.md`. Infra: oomd killed the eval's linear build 3× (8G PSI
ceiling despite 389G free) — fixed via the 811 close cache + H=5/20 (H=1 feature matrix too big). Code edits
flag-gated (canon-identical OFF); the smoke gate caught 2 launch bugs (forgot `--jax-env`; new CLI flags
dropped by the TrainingConfig dataclass filter) before GPU-hours burned.

**TASK_813 (operator pivot 2026-06-26; RAN same day) — VOL-HARVESTING fork CLOSED. The data/problem fork.**
Premise: 806-812 closed high-FREQUENCY cross-sectional re-ranking (turnover×cost>edge); maybe LARGE MOVES
(sparse, edge≫21bp) are cost-compatible. Phase 1: a pre-registered COMPLETE triggering map (trend/breakout
Donchian+MA, vol-expansion ATR+Bollinger+RV-regime, BTC-lead-lag, regime) = 132 cells, OOS, leak-free.
Fire-level: nominal GO (C1 lead-lag +5.81, D1 regime +1.03). **Event-level audit (collapse co-fires into the
episode they rode) DEFLATED it:** D1 = clustering artifact (CI includes 0 at 28 events); C1 = real-looking
but only **n=6 BTC-spike episodes** (under-powered, best-of-132 selection). Phase 1.5 (operator): re-test C1
on the 2023-2024 HOLD-OUT (disjoint from selection), pre-registered, full C1 grid × {BTC,ETH} = 48 cells,
event-level, FDR. **DISSOLVES: 0/48 GO; the exact n=6 winner (BTC w5 th0.02 H5760) INVERTS to net -4.08 /
gross -3.27 / precision 0.28 at n_events=25.** The n=6 test GO was selection luck in the post-ETF regime.
`runs/813_volharvest/` (MAP_PREREGISTERED, phase1_map_probe, PHASE1_MAP_PROBE, audit_go_eventlevel,
PHASE1_5_PREREG, phase1_5_leadlag, PHASE1_5_RESULT). Method win: the fire→event→hold-out ladder caught a
selection-luck GO a naive pipeline would have built Phase 2 on. Infra: oomd 8G-PSI ceiling fought 3× — fixed
via close/OHLC caches + test+lookback slicing (full-grid cumsum was the leak).

**Venue = Fork-C (operator): stay Kraken 1-min, do NOT chase a rebate venue ("not bending market
mechanism").** The data/problem fork (different instrument/frequency, or harvest the methodology) opens
only IF 811 confirms the signal is linear-optimal. Extends [single_mechanism_exhausted](single_mechanism_exhausted.md). Detail:
`lore/notes_2026-06-25.md`.

**Audit discipline note:** re-derive-from-artifacts caught TWO measurement artifacts in a row (809 per-bar
Sharpe, 810 per-bar-rows + cost-saving), both authored in good faith. The standard for any "model beats
baseline post-cost" claim on this stack: independent re-derivation from code + the **gross (cost=0)
Sharpe on holding-period-strided rows** — net-beat without gross-beat is a turnover artifact.
