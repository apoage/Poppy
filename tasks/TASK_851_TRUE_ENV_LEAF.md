# TASK_851 — True-env leaf rollouts in the vector search (D19 opt (a); the E1 +5.29 operator)

> **STATUS (2026-07-21): SHIPPED as `task851-true-env-leaf`.** `--leaf-eval {model,true-env}`
> on `SampledMuZeroVectorSearch`, default `model` bit-identical; wired into training self-play
> + the 835 eval path, reusing the 831 search-probe operator. Fixture inversion leg (i) PASSES
> (+6.801 vs −6.778 net@21); E1-consistency anchor (ii) PASSES (fresh-init net: +2.198 true-env
> vs −10.287 model, floor −3.437). Medium panel, two rounds: **2 MAJOR found + fixed +
> independently re-verified** (unique-per-env prices dispatch crash; unreproducible headline
> numbers, root-caused to a concurrent TASK_852 edit transiently mutating the shared
> `eval_vector.py` file during review — now pinned via LEDGER rows + committed artifacts). Zero
> regressions (selftest 10/10, full regression suite 28/28). Real binding leg + the Amendment-6
> ExIt re-try re-arm are the planner's scheduled post-ship chain.

**Status**: READY 2026-07-20 (D19 ruled — operator: "ok seed 851").
**Size: M. Risk: medium** (search LEAF EVALUATION layer, flag-gated; canon default unchanged —
every existing invocation bit-identical; touches BOTH training self-play and 835 eval plumbing).

## Why (LEDGER 849/leaf — the measured locus)
The search leaf `v̂(g(s,a))` is ACTION-BLIND: +0.03 correlation with true one-step reward across
candidates (r̂ works at +0.29; the policy prior's −0.45 is the expected fee-now/payoff-later
profile of a trend teacher). With a flat v̂, maximizing r̂+γv̂ sees only the fee cost of trading
⇒ search rationally converges to HOLD/churn — the attractor behind preflight, p1b, p1c, and the
ep100 imprint erosion. The program's own record holds the counter-evidence: **E1/831x — the
same proposal machinery with a TRUE-ENV R-step rollout leaf scored +5.29 net@21** (R=1 already
beats the engine ref +0.751, 831x/C). The entire −6.92 ↔ +5.29 gap is the learned leaf. We own
a perfect simulator (historical replay env); plan through it, not through a broken model.

## Scope
1. `--leaf-eval {model,true-env}` (default `model` = today, bit-identical) on the vector search
   used by BOTH training self-play and 835 eval:
   - `true-env`: at each candidate leaf, step a COPY of the true env forward with the candidate
     action, then `--leaf-rollout-r` (default 3; support 1) further steps under HOLD; leaf value
     = truncated discounted TRUE return, plus `--leaf-vhat-tail` (default ON) adding γ^R·v̂ at
     the horizon. Reuse the 831 search-probe machinery (`experiments/831_search_probe/`) — this
     operator already exists there against `step_vector`; the task is WIRING it into the shipped
     Gumbel search path, not reinventing it.
   - Eval-time determinism: the true-env rollout uses the same panel rows the search is deciding
     on (no lookahead beyond the R-step horizon — that horizon is the sanctioned planning
     lookahead INSIDE the simulator, identical in kind to E1's probe; document this scoping in
     the SPEC).
2. Training-loop integration: self-play targets (visit distributions, value targets) computed
   under the true-env leaf when the flag is ON; no other objective change.
3. Cost guard: vectorize the K-candidate × R-step rollouts through the existing JAX env vmap
   path; a fixture timing note in the report (bar: search step ≤ 3× the model-leaf time at
   K16/R3 — if it blows past that, report, don't silently ship a 10× slower trainer).

## Acceptance (gates)
- (i) **The inversion, third try (fixture leg)**: fixture search-eval of the frozen fixture
  ckpt with `--leaf-eval true-env` produces a book that preserves-or-beats the policy-argmax
  replay on the same rows (directional, fixture-scale). The REAL binding leg (frozen
  `runs/832_e5/ckpt_imprint_full`, bar net@21 ≥ +0.74) = planner's scheduled run.
- (ii) E1 consistency anchor: with a ZERO-knowledge net (fresh init), `true-env` search on the
  fixture reproduces the 831-probe behavior class (reaches the fixture floor) — the probe and
  the shipped path must agree now that they share the operator.
- (iii) Default `model` leaf BIT-IDENTICAL (canon + 835 result JSONs reproduce); selftest green
  + new leaf property tests (R=0 degenerates to r̂+γv̂; tail-OFF = pure truncated return).
- (iv) Medium panel (correctness + regression lenses; efficiency note per the cost guard).

## After ship (planner-owned, pre-registered chain)
1. Real inversion read: 835 search-eval of the frozen imprint, `--leaf-eval true-env`, bar
   ≥ +0.74 → LEDGER.
2. PASS ⇒ Amendment-6 ExIt re-try RE-ARMS with `--leaf-eval true-env` (same battery shape:
   3 seeds × 300 ep, prior-survival read) — the first training run where search can see why
   trading pays.

## Out of scope
Distilling the true-env leaf back into v̂ (D19 opt (c), the long-term MuZero path — its own
decision after the re-try); any reward-model change; retuning K/sims/gumbel.
