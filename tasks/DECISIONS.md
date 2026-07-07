# DECISIONS — the single blocked-on-human surface

One row per call. Recommendation FIRST — the human should be able to answer with one word.
OPEN rows are walls: the factory never guesses past one; the planner seeds rows here instead of
scattering open questions across task files. Answered rows move to DECIDED with the date.

## OPEN

| # | Decision | Recommendation | Context |
|---|----------|----------------|---------|
| — | (none) | | |

## DECIDED

| # | Decision | Outcome | Date |
|---|----------|---------|------|
| D1 | 824/825/826 active alongside 828? | 828 is the focus. The others are "interesting outview" only — PARKED, non-binding. | 2026-07-07 |
| D2 | TASK_827 reward-head fix: dead or fallback? | PROMISING — not dead. Its substance (action-conditional reward head; kill the 60× action-variance gap) is carried explicitly in 828 Stage 3. | 2026-07-07 |
| D3 | Keep the TASK_820 paper loop timer? | KILL — "not really anything to do with MCTS nor ML trading." Timer disabled 2026-07-07. `trader/` package stays (factory test baseline + the +0.97 eval bar). | 2026-07-07 |
| D4 | Iris-DEX / on-chain path (TASK_826)? | SHELVED — "we are OHLC, we have OHLC, live in that constraint; we have the 24GB DB." | 2026-07-07 |
| D5 | Retire Foxglove and/or Bella? | Retire FOXGLOVE (→ TASK_829, factory job — needs import retargeting first). KEEP BELLA — operator's call: not useful for trading, but the one thing in the repo that's interesting and actually works. Resident, not load-bearing. | 2026-07-07 |
| D0 | Deploy the linear CTA to the VPS (TASK_823)? | REJECTED — too trivial to deploy; operator 2026-07-07: "meaningless, same as D3." | 2026-07-06 |
