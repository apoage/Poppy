# E1/E2 — True-model search-capacity probe (TASK_831, pathway rung R1)

**Hypothesis**: **H1 (search side)** — one decision is a per-asset action VECTOR whose EV = Σ(64
edges) is above noise, so a *perfect-model* K-sample search should be able to FIND at least the
819 linear book (the corridor floor). This experiment isolates SEARCH capacity from LEARNING: it
runs the Stage-2 Sampled-MuZero *sampling* machinery with the ENV SIMULATOR as dynamics and the
TRUE env reward — **no learned nets**. If a perfect-model search cannot reach the floor, no learned
model will, and 828 Stage 3 training must not launch.

## What "perfect model" means here (and why it is not leakage)

The search is granted a PERFECT model on purpose:
- **dynamics** = `mcts_trader.env.jax_trading_env.step_vector` (the just-shipped Stage-1 vector env),
- **reward** = the env's own whole-portfolio log-return reward (symlog-scaled, canon formula),
- **value** = a *truncated true-env rollout* (see below) — the model-based value MuZero would learn.

The search therefore "sees" forward prices *through the model* when it scores candidate actions.
That is the DEFINITION of a search-capacity ceiling, not train/test leakage: nothing is fit, no
label crosses a train/test boundary. The realized return series that is scored for Sharpe is
produced by stepping the REAL env one step per decision (identical accounting to TASK_830's floor).
A *working* perfect search should EXCEED the floor and approach the oracle ceiling; the science of
E1 is whether the SAMPLER ever proposes book-quality actions at all.

## Value heuristic (pre-registered, config knob `--value-rollout-steps`)

**Chosen: truncated env rollout** (default 3 steps), NOT the 830 oracle bound. From each candidate's
post-action state, roll the true env forward under an all-HOLD policy for `value_rollout_steps` and
take the discounted sum of true rewards; Q(candidate) = r₀ + γ·Σ γᵗ rₜ. **Justification**: the 830
per-asset oracle bound is a hindsight upper bound that depends only on prices, not on the candidate's
resulting portfolio — it would assign the *same* value to every candidate at a node and give the
search zero discrimination. The truncated rollout differentiates candidates by their resulting state
(positions), which is exactly what guides action selection. It is also the natural "perfect-model"
value (model-based bootstrap). `--value-rollout-steps 0` degrades to a pure one-step-reward greedy;
the knob lets the scheduled run sweep depth.

## Reuse-vs-reimplement decision (flagged for the panel)

The canon Stage-2 `SampledMuZeroVectorSearch` (`mcts_trader/jax_agent/sampled_search.py`) is wired to
a LEARNED `model_apply_fn` and decodes value/reward through the 401-bin categorical support
(`get_value_support`). The **primary, dispositive** reason the canon `_search` cannot be reused is
structural: it *requires* a learned `model_apply_fn` for its recurrent step, and this probe passes
`model_apply_fn=None` (there is no net — the whole point is a net-free perfect-model ceiling). So a
true-model search must be reimplemented regardless. **Secondarily**, the canon value/reward decode
through the 401-bin categorical support (bin resolution ≈ 0.05 over [-10,+10]) is *lossy* for the
symlog rewards at play here: the env reward is `symlog(log_return·100)`, so a ~0.1% daily log-return
maps to a value of order ~0.1, and the measured true-model Q std across candidates is ≈0.14 —
*larger* than one 0.05 bin, but still coarse relative to the discrimination the search needs. (An
earlier draft of this SPEC claimed the round-trip would "destroy scalar returns of order 1e-3"; that
conflated the raw daily log-return with the symlog-scaled value that would actually be binned and is
corrected here.) Per the task's design call, the probe **REUSES the canon SAMPLING + visit-correction
machinery** — it imports
`SampledMuZeroVectorSearch` and calls its `_sample_action_vectors` (per-asset factored draws) and
`_empirical_prior` (β̂ = nₐ/K) unchanged — and **REIMPLEMENTS the minimal true-model search loop**
inside this experiment module (zero diffs to canon). The reimplemented search is a *root planning*
over the K sampled candidates: score every candidate by Q = r₀ + γ·V and pick argmax.

**FLAG**: the reimplemented search evaluates ALL K candidates (full expansion, argmax-Q), rather than
mctx Gumbel Sequential-Halving over a sub-budget. For a *capacity* probe this is deliberate and
STRICTLY STRONGER than halving (halving is a budget-saving approximation that can only lose value):
if full-eval perfect search cannot reach the floor, halving certainly cannot, so a FAIL is a true
capacity ceiling. The visit-analog reported to the ledger is softmax(Q/τ) over the sampled set.

## E2 — sampling-scheme × K grid (the design input Stage 3 needs)

Each scheme is injected purely by constructing the per-asset `policy_logits` fed to the canon sampler
(no new model code). Logits are laid out `[asset, action]` (index = asset*5 + action), reshaped
(A,5); each asset row is an independent categorical marginal drawn K times i.i.d.

| Scheme | Maps to policy_logits as | Intent |
|--------|--------------------------|--------|
| **independent** (Stage-2 default) | all-zeros ⇒ uniform per-asset draws (no learned policy exists) | the factored-policy default with an untrained net |
| **linear-prior mixture** | `bias` on the 819 book's target action per asset (via `_to_action_vec(target_sign, current_side)`), 0 elsewhere | proposal seeded by the working book (H1 prior) |
| **dithered-book** | `bias` on HOLD(1) per asset, 0 elsewhere ⇒ mostly hold, occasional local flip | local moves around the held portfolio |

`bias` = `--proposal-bias` (default 3.0). K ∈ {8, 16, 32}. Output: which scheme×K reaches the floor
cheapest, where **cost** = K (and, as a tiebreak, K·(1+value_rollout_steps) model evaluations).
That scheme×K is carried into Stage 3.

## What E1 measures (semantics — read before the PASS/FAIL)

E1 measures **PROPOSAL CAPACITY**: does the per-asset factored sampler place at least one
floor-quality action-vector in its top-K candidate set, such that a perfect-model argmax can pick
it? It does **not** measure whether a *learned* selector will find that vector.

- A **FAIL across all schemes ⇒ the proposal mechanism is the defect ⇒ Stage-3 does NOT launch.**
  This is the sound, load-bearing kill: if a *perfect* value cannot find a floor-quality vector in
  any sampler's top-K, no learned value will, so the sampler must be respec'd before any training
  GPU is spent.
- A **PASS is NECESSARY-BUT-NOT-SUFFICIENT** for Stage-3. It proves only that the search SPACE
  *contains* the book (the sampler can propose it); it does **not** prove a learned selector will
  actually select it under an imperfect learned value.
- **E1 is a WEAK PASS-detector.** Because the probe scores candidates with a *perfect* R=3 truncated
  true-env rollout, even the uniform `independent` scheme reaches the floor at K≥16 in the fixture
  smoke. A PASS is therefore cheap; E1's kill-power is confined to a *catastrophically-broken*
  sampler (one that cannot propose a floor-quality vector under any scheme at any K). The real
  deliverable of E1 is the **E2 scheme×K ranking**, carried into Stage 3 as the sampler design input.
- **Not a bound on the as-built Stage-2 search.** The probe substitutes a different value estimator
  (perfect truncated rollout) for the learned Gumbel-search value, so it is **neither a strict upper
  nor a strict lower bound** on the as-built Stage-2 Gumbel Sequential-Halving search. It bounds the
  *proposal space*, not the *learned search*.

## Pre-registered PASS/FAIL (E1)

- **Metric**: net@21 (fee16 + slip5) H-strided per-decision Sharpe of the search's realized return
  series (`mean/std·√(525600/1440)`, the 819/backtest annualization), computed by `trader.backtest`
  via the reused TASK_830 scorecard, over the SAME panel rows as the corridor.
- **PASS**: at least one scheme×K achieves `search.net21 ≥ floor.net21`, where `floor.net21` is the
  TASK_830 corridor floor recomputed on the identical rows (apples-to-apples).
- **FAIL**: no scheme×K reaches the floor.
- Reported as a **corridor position** inside `[floor, ceiling]`, where the 830 per-asset oracle
  ceiling is recomputed on the **same first-N decision rows** as the floor/search (window-consistent;
  on a capped/smoke run the ceiling is NOT scored on the full panel — that would mix windows).

## Kill logic (from pathway rung R1)

E1 FAIL **across all E2 schemes** ⇒ the action-proposal (sampling) mechanism is the defect ⇒ **828
Stage 3 does NOT launch**; the planner respecs the sampler (a new structural hypothesis) before any
training GPU is spent. A single scheme×K reaching the floor PASSES E1 (NECESSARY-BUT-NOT-SUFFICIENT
for Stage-3 — see semantics above) and names the candidate Stage-3 sampler via the E2 ranking.

## Seeds / compute / kill date

- **Seeds**: `--seeds` (default `0,1,2` for the smoke; ≥3 for any promotion claim). Sampling is the
  only stochastic element (per-asset categorical draws); the env walk is deterministic.
- **Compute budget**: ≤1 GPU-day for the SCHEDULED full-panel sweep (invariant #7). The factory
  deliverable is the CPU fixture SMOKE only (`JAX_PLATFORMS=cpu`, `fixtures/trading_fixture.db`,
  never GPU, never `trading_data.db`); the smoke caps decisions (`--max-decisions`, default 100) to
  stay in seconds–minutes and is a SMOKE SIGNAL, not the real verdict.
- **Kill date**: 2026-07-22. If the runner cannot produce a finite corridor position by then, E1 is
  re-scoped.

## Deliverables

- `search_probe.py` — the runner (this dir). Imports the canon sampler helpers
  (`SampledMuZeroVectorSearch`), the env (`step_vector`), and the TASK_830 corridor module (panel
  loader, book replay, oracle ceiling, scorecard) by path. No edits to any canon/monster file.
- Fixture smoke: `python search_probe.py` → writes `results_fixture_smoke.json`.
- `experiments/LEDGER.md` lines per scheme×K with corridor position + the E1 verdict, marked
  **FIXTURE-SMOKE — real verdict PENDING the scheduled GPU sweep** (mirrors the 830 convention).
