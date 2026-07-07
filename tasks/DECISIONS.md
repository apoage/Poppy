# DECISIONS — the single blocked-on-human surface

One row per call. Recommendation FIRST — the human should be able to answer with one word.
OPEN rows are walls: the factory never guesses past one; the planner seeds rows here instead of
scattering open questions across task files. Answered rows move to DECIDED with the date.

## OPEN

| # | Decision | Recommendation | Context |
|---|----------|----------------|---------|
| D6 | Which book defines the official E0 corridor FLOOR the vector agent is graded against — the env-native **hold-shares** replay (+0.42 net@21 on fixture) or a **constant-dollar-weight daily-rebalance** floor (matches the +0.97 engine bar)? | **Report BOTH on the scheduled real-panel run; grade the agent against the hold-shares in-env floor** (it is what `step_vector` natively does when a position is left unchanged), keep +0.97 as the engine-native reference ceiling-of-the-rule. Rationale: the corridor must be the linear strategy run through the SAME env the agent uses. | TASK_830/E0 panel (2026-07-08) proved the fixture floor↔+0.97 gap is a REPLAY-MAPPING artifact, not a `step_vector` bug (env reproduces an independent cash-account sim to 3–4 sig figs; +0.013 residual). Dominant term = fixed-weight vs renormalized `1/n_present`; BEAR22 = faithful hold-shares de-leveraging vs the book's daily constant-dollar rebalance. **Deeper flag for Stage 2/3:** the discrete per-asset action space {sell/hold/buy/limit×2} cannot express an exact continuous weight target, so exact reproduction of a daily-rebalanced +0.97 book may be INEXPRESSIBLE in-env — relevant to what "beat the bar" can mean. Does not block Stage 2 (search) build; DOES gate the interpretation of the scheduled corridor number. Detail: `experiments/830_corridor/SPEC.md`. |

## DECIDED

| # | Decision | Outcome | Date |
|---|----------|---------|------|
| D1 | 824/825/826 active alongside 828? | 828 is the focus. The others are "interesting outview" only — PARKED, non-binding. | 2026-07-07 |
| D2 | TASK_827 reward-head fix: dead or fallback? | PROMISING — not dead. Its substance (action-conditional reward head; kill the 60× action-variance gap) is carried explicitly in 828 Stage 3. | 2026-07-07 |
| D3 | Keep the TASK_820 paper loop timer? | KILL — "not really anything to do with MCTS nor ML trading." Timer disabled 2026-07-07. `trader/` package stays (factory test baseline + the +0.97 eval bar). | 2026-07-07 |
| D4 | Iris-DEX / on-chain path (TASK_826)? | SHELVED — "we are OHLC, we have OHLC, live in that constraint; we have the 24GB DB." | 2026-07-07 |
| D5 | Retire Foxglove and/or Bella? | Retire FOXGLOVE (→ TASK_829, factory job — needs import retargeting first). KEEP BELLA — operator's call: not useful for trading, but the one thing in the repo that's interesting and actually works. Resident, not load-bearing. | 2026-07-07 |
| D0 | Deploy the linear CTA to the VPS (TASK_823)? | REJECTED — too trivial to deploy; operator 2026-07-07: "meaningless, same as D3." | 2026-07-06 |
