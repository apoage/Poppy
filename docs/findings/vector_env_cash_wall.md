# vector env cash wall

> 828 vector env's snapshot-cash accounting makes breadth lethal at any cost tier — the real root of HOLD collapse; fix = D7/TASK_839 budget-constrained step

*Migrated from local agent memory 2026-08-31 (`project_vector_env_cash_wall.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**Finding (2026-07-09, LEDGER 828s3/env-wall):** `step_vector`
(`mcts_trader/env/jax_trading_env.py:1391-1491`) charges EVERY buying/shorting asset
`min(0.1×PV, 0.99×cash)` against ONE cash snapshot with no aggregate budget constraint;
overdraft → `process_margin_call` → long liquidation → bankruptcy (PV→0, done). Reproduced: a
near-uniform 64-asset action vector bankrupts envs at step 0.

Supersedes the 828s3/preflight reading: HOLD collapse is NOT primarily "fees punish
exploration" — **cost0 collapses too** (phase-1 battery: 98% hold, negative self-play rewards,
and the trained policies STILL bankrupt full eval rollouts — LEDGER 835/S4-p1). The H6/H1
breadth thesis (EV = Σ64 small edges) is structurally inexpressible under 10%-of-PV sizing
(~10-position cap). E1/E2 perfect-model (+5.29) unaffected (hb3.0 keeps vectors narrow).

**Why:** six months of collapse diagnoses kept blaming costs/exploration/teachers; the env's
cash mechanics were the invariant nobody re-derived after the scalar→vector pivot (scalar-era
sizing: one 10% trade per step was safe by construction; 64 simultaneous ones are not).

**How to apply:** any vector-action training/eval/probes before TASK_839 ships are measuring
the wall, not the model — gate phase 2, imprints, and DEC-1/E4 re-runs on the budget-constrained
step (D7). When designing env changes, re-derive cash-flow invariants under the WIDEST legal
action, not the typical one. See [planner_closed_clean_cell](planner_closed_clean_cell.md) (the scalar HOLD-collapse
finding this reframes a second time).

**Update 2026-07-11 (phase-1b):** the 839 budget step fixed step-0 death but NOT rollout death —
sparse trained books still bankrupt (10–11 unhedged shorts, margin cascade; LEDGER 835/S4-p1b).
Meanwhile **the nets demonstrably learn** (834 DEC-1 PASS ×2; 836 value>momentum ×2 — first
positive learning signal). Residual killer = incremental-order semantics (positions accumulate,
nothing re-trims). Fix = TASK_840 target-weight rebalance step (D8 escalation, pre-registered).
Also: post-checkpoint-save wedge pattern (2/6 runs freeze after orbax "Finished synchronous
save") → TASK_841 watchdog+hunt.

**Update 2026-07-14 (arc closed):** TASK_840 weight-rebalance step FIXED the env — phase-1c:
zero bankruptcies, first-ever complete 1670-step trained-policy rollouts, 3/3 seeds (at fee0
the weight path hard-asserts cash ≥ 0 and margin-call is a no-op, env:1584-1587). Policy still
FAILS the corridor (net@21 −0.83/−1.12/−1.39 vs floor +0.424) with a short-heavy book — but
the TASK_844 postmortem cleared all three suspected loci (value / policy prior / reward
credit; LEDGER 844/postmortem): no learned short preference; the book is weight-env flow
mechanics (shorts exit only by BUY-flip to long — the probe's leg tracker is flip-blind in
weight semantics, exact arithmetic 3/3). Live fork = D13 (redirect to imprint→ExIt vs H6
widening), operator's call. **D13 RULED 2026-07-14: REDIRECT** — imprint→ExIt (TASK_845
compat + DEC-2 schedule) is the critical path; H6 widening kept as the named "rainy day"
contingency (activates only if the imprinted policy can't hold the floor).
