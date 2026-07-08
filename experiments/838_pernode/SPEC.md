# 838 — Per-node action resampling: acceptance probe (TASK_838)

**Hypothesis**: the shipped `SampledMuZeroVectorSearch` reuses the ROOT's K sampled action
vectors at every interior node (Stage-2 documented simplification, 828 banner carry-note 4),
which caps depth-2+ planning to root-proposed portfolios. `per_node_resampling=True`
(the TASK_838 flag) samples K fresh vectors from the factored policy at EACH expanded
interior node, so deeper search can consider portfolios the root never proposed.
**Pre-registered read: ON ≥ OFF at equal K.** A TIE at today's sims/depth is a FINDING,
not a failure — record it verbatim.

## The Sampled-MuZero correction stays exact per node (Hubert et al. 2021, arXiv:2104.06303)

Sampled MuZero replaces the intractable joint prior π(a)=Πᵢπᵢ(aᵢ) with the sample-based
prior β̂(a)=n_a/K, valid when the K candidates at a node are drawn i.i.d. from that node's
proposal β. In the fixed-K slot representation this is the uniform-1/K slot prior
(zero prior_logits); duplicates accumulate n_a/K mass inside mctx's visit allocation.

- **OFF (root-reuse)**: all nodes share the root's samples; the uniform slot prior is the
  documented Stage-2 approximation at interior nodes.
- **ON (per-node)**: each expanded node's K slots are drawn i.i.d. from that node's OWN
  proposal (built from the factored policy at that node's latent, same `_build_proposal`
  dithered-book machinery). β=π therefore holds PER NODE, and the uniform-1/K slot prior is
  again the exact sample-based correction at every node — the correction is local to the
  node that sampled. Unit-tested in `tools/tests/test_pernode_resampling.py`
  (per-node prior uniformity on the search tree).

## Per-node book + mask decision (pre-registered)

The latent does NOT carry a book. Per DEC-3 (v1 book = HOLD; HOLD maintains position), the
per-node proposal uses the HOLD-book: `_build_proposal(..., book=None)` ⇒ `hold_bias` lands
on the HOLD slot directly — identical semantics to the v1 all-HOLD book at the root. No
`action_mask` at interior nodes: interior vectors are planning candidates that are never
executed (only the root's selected vector reaches the env), and node-level legality is
unknowable without a simulated env state.

## Per-node rng derivation (pre-registered)

mctx passes a fresh `expand_key` to `recurrent_fn` at every expansion (mctx 0.0.5
`search.py:90,101,226`). The per-node stream is `node_rng = fold_in(expand_key, 838)`,
split once into `(dither_rng, sample_rng)`. Deterministic given the search seed; namespaced
away from any other consumer of the expand key; the TOP-LEVEL rng split
(sample, dither, search) is untouched, so the OFF path consumes no new rng and the ROOT
candidate set is bitwise-identical ON vs OFF at equal seed (tested).

## Acceptance approach: the CANON search class, true-model apply_fn (adapted 831 pattern)

TASK_831 built its true-model probe by REIMPLEMENTING the search loop, forced because the
canon `_search` requires a learned `model_apply_fn`. For 838 the thing under test IS the
canon search class (the code that was modified), so reimplementation would prove nothing.
**Pre-registered choice: wrap a true-model apply_fn** — a callable satisfying the exact
`model_apply_fn` contract the canon `_search` uses:

- `(variables, observation, method='initial_inference')` → `{policy_logits, latent_state, value}`
- `(variables, embedding, vec, mode='recurrent_inference_vector')` → `{latent_state, policy_logits, value, reward}`

backed by the REAL env:
- **latent_state** = the true `TradingState` pytree (mctx embeddings are pytrees; the canon
  search treats latents as opaque). The root observation is the flat-encoded state
  (exact uint32 bitcast codec, round-trip asserted at startup).
- **dynamics/reward** = `step_vector` and the env's own reward.
- **value** = truncated true-env all-HOLD rollout (R=3, γ=0.997 — the 831 heuristic,
  pre-registered there).
- **value/reward encoding**: TWO-HOT over the canon 401-bin support, so the canon
  categorical decode (`_categorical_to_scalar`) recovers the scalar near-exactly
  (linear decode of a two-atom distribution; error ~1e-7) — this removes the 0.05-bin
  quantization worry 831 documented, WITHOUT touching canon decode.
- **policy_logits** = zeros at every node (there is no learned policy). Proposal shaping
  comes from the canon search's own dithered-book knobs (hold_bias=3.0, dither_scale=1.0 —
  the E2-carried scheme), identically in both arms.

### What each half proves (honesty section)

- **This probe (true model, canon class)**: isolates the SEARCH MECHANISM at zero model
  error — does giving interior nodes fresh per-node proposals (vs the root's frozen K)
  change realized planning quality, with the canon `_search`/mctx/gumbel machinery running
  verbatim? Because the true model's policy is constant (zeros), ON here tests the
  *fresh-candidate-diversity-at-depth* mechanism, NOT the *policy-improvement-at-better-
  latents* mechanism — the latter needs a trained factored policy and is explicitly out of
  scope for the fixture (that is the scheduled Stage-3/4 GPU evidence stream).
- **The unit/parity half (learned model, fresh-init)**: `tools/tests/test_pernode_parity.py`
  (pre-edit pin `22b0c7f5…f23e`) proves OFF is bit-identical to the shipped tree through the
  REAL learned-model apply path; `tools/tests/test_pernode_resampling.py` proves the ON
  integration (per-node sets genuinely differ, per-node prior exactness, determinism,
  rng isolation) on the real `MuZeroNetworks`. Neither half claims anything about trained-
  policy performance.

## Pre-registered protocol

- **Arms**: OFF (`per_node_resampling=False`) vs ON (`True`), EQUAL K, same seeds, same
  proposal knobs, `deterministic=True` selection (carry-note 3 eval semantics).
- **Depth>1 sims**: `sims=32`, `K=8`, `max_considered=8` — the Sequential-Halving budget
  (32 sims over 8 root candidates) forces interior expansions, so resampling has depth to
  bite on.
- **Metric**: net@21 (fee16+slip5) per-decision Sharpe of the realized return series
  (real env stepped once per decision), gross (fee 0) reported alongside; TASK_830 corridor
  floor recomputed on the same rows for context (831 convention). Seeds 0,1,2; mean over
  seeds is the headline.
- **Read**: `net21(ON) ≥ net21(OFF)` ⇒ mechanism confirmed-or-neutral at this depth; record
  the delta either way. A TIE (|Δ| small relative to seed spread) is a finding: at
  sims=32/K=8 with an uninformed constant policy, root Sequential-Halving may still
  dominate — that bounds where per-node resampling can pay, it does not un-ship the flag
  (default stays OFF regardless; out-of-scope to change it).
- **Smoke scope**: CPU + `fixtures/trading_fixture.db` ONLY (never trading_data.db, never
  GPU); decisions capped (`--max-decisions`, default 60). FIXTURE-SMOKE signal — the
  trained-policy answer is the scheduled Stage-3/4 stream.

## Deliverables

- `pernode_probe.py` (this dir) — the runner; imports the CANON `SampledMuZeroVectorSearch`
  (both arms run the canon `_search` verbatim), the env, and the 830 corridor module by
  path. Zero canon diffs.
- `python pernode_probe.py` → `results_fixture_smoke.json` (ON vs OFF net@21 per seed +
  mean, corridor floor context, the pre-registered read applied).

## RESULT (fixture smoke, 2026-07-09 — recorded verbatim from `results_fixture_smoke.json`)

Run: 279 assets, 60 decisions, warm=31, K=8, sims=32, seeds 0,1,2, hold_bias=3.0,
dither_scale=1.0, deterministic selection. Per-decision Sharpe, mean over seeds [per-seed]:

| arm | gross | net@21 |
|-----|-------|--------|
| OFF (root-reuse) | +7.700 [+10.542, +4.711, +7.848] | **+6.421** [+7.093, +5.959, +6.210] |
| ON (per-node)    | +6.493 [+6.039, +6.224, +7.217]  | **+3.634** [+2.912, +2.427, +5.563] |

Corridor floor (same rows): gross +5.721, net@21 +5.591. Both arms ran green; OFF clears
the net@21 floor; ON does not clear the net@21 floor on any seed (best +5.563 < +5.591);
all ON seeds clear the gross floor (+5.721).

**Read applied — the pre-registered read did NOT hold**: Δnet@21 (ON−OFF) = **−2.787**
(Δgross −1.207), ON seed wins 0/3. On THIS smoke **OFF > ON**, recorded verbatim as the
FINDING. Interpretation (hypothesis, not measured): with an UNINFORMED constant policy,
per-node resampling injects fresh random dithered portfolios into every subtree, adding
variance to the deep value backups that rank the root candidates — root-reuse gives each
root candidate a consistent (if narrower) subtree and a cleaner ranking at sims=32/K=8.
The mechanism the flag exists for — informed per-node proposals from a TRAINED factored
policy at better latents — is exactly what a constant-policy true-model probe cannot
exhibit; whether it reverses the sign is the scheduled Stage-3/4 (trained-policy) question.
Consequence: **default stays OFF** (as shipped; also the task's out-of-scope rule), and any
Stage-3 run flipping `--vector-per-node-resample` ON should treat this smoke as evidence
that ON at K=8/sims=32 with a weak policy is NOT free — re-measure, don't assume.
