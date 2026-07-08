# DECISIONS — the single blocked-on-human surface

One row per call. Recommendation FIRST — the human should be able to answer with one word.
OPEN rows are walls: the factory never guesses past one; the planner seeds rows here instead of
scattering open questions across task files. Answered rows move to DECIDED with the date.

## OPEN

| # | Decision | Recommendation | Context |
|---|----------|----------------|---------|
| _(none)_ | | | |

## DECIDED

| # | Decision | Outcome | Date |
|---|----------|---------|------|
| D6 | Which book defines the official E0 corridor FLOOR — env-native **hold-shares** replay or **constant-$-weight daily-rebalance** (matches the +0.97 engine bar)? | **DUAL-FLOOR (planner ruling; operator may override).** (a) RUNG experiments (E1 gate, E5 teacher-beat) grade against the env-native HOLD-SHARES in-env floor — apples-to-apples: same discrete action space the agent lives in. (b) The **+0.97 engine bar stays UNCHANGED as the Stage-4/R4 graduation bar** (definition-of-working untouched — we must beat the thing we could actually ship). (c) The scheduled real-panel run reports BOTH plus the **EXPRESSIBILITY GAP** (best in-env replay vs engine bar); if the gap is material (pre-registered: >0.2 net@21 Sharpe), seed **H6: action-granularity** (the {0..4} per-asset set is too coarse — weight-ladder actions) as a structural hypothesis row. Context: the fixture floor↔bar gap is a verified replay-MAPPING artifact, not a `step_vector` bug (830 panel, 2026-07-08); the discrete action space may make exact reproduction of a daily-rebalanced book inexpressible in-env — that is a measurement to take, not a reason to move the graduation bar. Detail: `experiments/830_corridor/SPEC.md`. | 2026-07-08 |
| D1 | 824/825/826 active alongside 828? | 828 is the focus. The others are "interesting outview" only — PARKED, non-binding. | 2026-07-07 |
| D2 | TASK_827 reward-head fix: dead or fallback? | PROMISING — not dead. Its substance (action-conditional reward head; kill the 60× action-variance gap) is carried explicitly in 828 Stage 3. | 2026-07-07 |
| D3 | Keep the TASK_820 paper loop timer? | KILL — "not really anything to do with MCTS nor ML trading." Timer disabled 2026-07-07. `trader/` package stays (factory test baseline + the +0.97 eval bar). | 2026-07-07 |
| D4 | Iris-DEX / on-chain path (TASK_826)? | SHELVED — "we are OHLC, we have OHLC, live in that constraint; we have the 24GB DB." | 2026-07-07 |
| D5 | Retire Foxglove and/or Bella? | Retire FOXGLOVE (→ TASK_829, factory job — needs import retargeting first). KEEP BELLA — operator's call: not useful for trading, but the one thing in the repo that's interesting and actually works. Resident, not load-bearing. | 2026-07-07 |
| D0 | Deploy the linear CTA to the VPS (TASK_823)? | REJECTED — too trivial to deploy; operator 2026-07-07: "meaningless, same as D3." | 2026-07-06 |
