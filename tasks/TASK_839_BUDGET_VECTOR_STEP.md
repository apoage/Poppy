# TASK_839 — Budget-constrained vector step (the env-wall fix)

**Status**: READY 2026-07-09 — **D7 RATIFIED option (a)** (operator, 2026-07-09). Take from
the top.
**Size: M. Risk: high** (touches `mcts_trader/env/jax_trading_env.py` `step_vector` — canon-adjacent;
full flag-gate + parity discipline, the 838 pattern).
**Pathway**: unblocks R2/R3 — every phase-1 instrument and phase-2 training is gated behind this.

## Why
LEDGER **828s3/env-wall** (2026-07-09): `step_vector` charges EVERY buying/shorting asset
`min(0.1×PV, 0.99×cash)` against ONE cash snapshot (`jax_trading_env.py:1391-1396`), sums the
deltas (:1489-1491), and lets `process_margin_call` clean up the overdraft — long liquidation,
then BANKRUPTCY (PV→0, episode done). Measured consequences:
- A near-uniform action vector (≈13 buys + 13 shorts of 64) bankrupts envs at **step 0** —
  exploration is lethal from the first uniform-policy step ⇒ HOLD collapse **at cost0** (the
  fees-punish-exploration reading of 828s3/preflight was the wrong root; fees only accelerate).
- The trained 98%-HOLD phase-1 policies STILL bankrupt full rollouts (835/S4-p1: PV→0 at step
  ~750/1670) via the non-hold tail + 3–4 unhedged shorts + margin cascade.
- The H6/H1 breadth thesis (EV = Σ 64 small edges) is structurally inexpressible: 10%-of-PV
  sizing caps the book at ~10 concurrent positions.

## Scope (per D7 option (a) — adjust only if the operator ratifies differently)
1. Flag-gated budget sizing (config `vector_budget_step` / `--vector-budget-step`, default OFF =
   today's behavior BIT-IDENTICAL — parity test pinned like the Stage-1/2/838 gates).
2. ON: one pass computes `n_trades = Σ(a_buy | a_sell-opening-short)` per env-step; each trade
   sizes at `PV × position_size_pct_effective` where `position_size_pct_effective =
   min(position_size_pct, budget_pct / max(n_trades, 1))`, `budget_pct` default 0.99 of available
   cash — Σ spend ≤ cash by construction, margin call becomes the exception path again, not the
   normal path. Closing trades (sell-of-long, buy-of-short) are proceeds, not spend — excluded
   from `n_trades`. All `jnp.where`, no dynamic shapes (JAX pitfalls).
3. Shorts: opening-short spend counts against the same budget (margin reserve = spend, current
   semantics otherwise unchanged). No change to `process_margin_call` itself.
4. **Acceptance experiment (fixture, pre-registered)**: (i) the 834-repro walk (zero-logit
   dithered-book, hb0.0, 4 envs) survives ≥64 steps with ZERO bankruptcies under ON (it dies at
   step 0 under OFF — reproduce that too, as the control); (ii) E1-pattern perfect-model probe
   ON vs OFF at K16: report net@21 — pre-registered read: ON ≥ OFF − 0.10 (budget sizing must
   not destroy the narrow-vector regime E1 lives in; a tie is a PASS, a collapse is a FAIL).
5. Thread the flag through 835 (eval operator symmetry — the 838 MAJOR precedent), 834/836
   walks, and the training CLI.

## Gates
OFF-path parity hash == pinned; new tests alongside the existing vector-step tests (do not
modify existing); fixture acceptance (i)+(ii) green; selftest 7/7; full high-tier panel
(canon-adjacent).

## Out of scope
Weight-ladder action-space changes (D7 option (c) = H6 territory, separate decision), reward
shaping, proposal-scheme changes, `process_margin_call` redesign, phase-1 re-training (scheduled
run AFTER this ships).
