# TASK_844 — short-bias postmortem probe (where does the ~10-short standing book come from?)

**Status**: instrument + fixture smoke shipped (factory). Real numbers = a SCHEDULED run on the
p1c/p1b snapshots with the run flags (planner-owned — the 836 precedent). This is a **read-only
diagnostic** (830/834/836 precedent): it localizes the bias, it fixes nothing and gates nothing.

## Why

Both p1b AND p1c policies converge to persistently SHORT-heavy standing books (≈10 shorts vs ≈1–6
longs) that profit only in bear windows and lose everywhere else (LEDGER 835/S4-p1b, 835/S4-p1c) —
across two different env semantics. The objective is absolute return (reward-hybrid = a loss
recipe, not market-relative), so the bias is a LEARNING ARTIFACT with an unknown locus: policy
prior, value preference, or realized-reward asymmetry. This probe measures which.

## Interpretation — quoted VERBATIM (emitted in every JSON as `interpretation.rule_verbatim`)

> mirror-book value mean Δ significantly < 0 (prefers short books) ⇒ VALUE locus; flat-book
> sell/buy policy-mass ratio > 1.5 ⇒ POLICY-prior locus; short legs realize losses but the policy
> keeps opening them ⇒ REWARD/credit mis-assignment (profited in training-era self-play but not at
> eval ⇒ distribution shift). DIAGNOSTIC — this probe localizes the ~10-short bias; it gates
> nothing and fixes nothing by itself. The ≥2-checkpoint aggregation is the planner's job across
> scheduled runs.

Applied only in checkpoint mode — NEVER to the untrained fixture smoke (834/835/836 precedent).

## The three probes

All three run on the SAME loaded checkpoint over one deterministic, seed-fixed panel (single env,
fixed start at the first obs-valid row — the 835/836 pattern). Checkpoint restore is the 834
FAIL-LOUD guard verbatim (`trainer_metadata.json` cross-check + step_count>0 + params-differ-from-
fresh-init — flax `restore_checkpoint(step=None)` silently returns the target UNCHANGED on a bad
dir, which the 834 panel caught as a FALSE PASS).

### 1. Mirror-book value (primary) — VALUE locus

Sample N books via the 834 zero-logit dithered-book walk (seed-fixed, legality-masked,
checkpoint-INDEPENDENT — the identical states are probed across every checkpoint). Each state's
book has a SUPPORT (the assets holding a position). From the state's OWN market latent
(`initial_inference` on the leak-free 27ch obs `[t-window, t-1]`, which is **price-only /
position-blind** — so book value can only enter through the dynamics), take ONE
`recurrent_inference_vector` step and decode the value with the canon
`SampledMuZeroVectorSearch._categorical_to_scalar`:

- **B** = open the support LONG: `BUY` on support, `HOLD` elsewhere.
- **−B** = its EXACT element-wise long↔short mirror: `SELL` on support, `HOLD` elsewhere.

`Δ = V(B) − V(−B)`. B is canonicalized to the LONG side so the sign is interpretable: mean Δ < 0 ⇒
the value net prefers SHORT exposure. States with an empty book are excluded (they force Δ≡0).

**CALIBRATION (MAJOR-1 — the raw within-state t is UNCALIBRATED).** `Δ` is NOT antisymmetric:
`VectorActionEmbedding.vec_act_proj` (`muzero_model.py:306-314`) has independently-initialized BUY
vs SELL columns, so even a FRESH untrained model produces a non-zero per-checkpoint mirror-Δ offset
— an arbitrary architectural zero-point whose sign/size varies by init seed (measured: raw
within-state |t| up to ~4; raw mean up to ~±0.6 across init seeds on the fixture). The
within-state t-test (SEM over states) uses the WRONG null and cannot separate a learned
short-preference from that zero-point on a single checkpoint. So the VALUE-locus verdict gates on a
**matched fresh-init null**: for the SAME book states/latents, measure the mirror-Δ offset of a few
fresh-init models of the SAME architecture (built from the run's own cfg). The null distribution is
OVER INITS; the inference unit is the per-checkpoint offset (its mean-Δ). The calibrated statistic
is `z = (Δ_ckpt_offset − null_mean) / null_std_over_inits` (the null bootstrapped over inits for a
CI). **Value-locus ⇔ calibrated `z < −2` (n_null ≥ 3)**, NOT the raw t. The JSON emits BOTH the raw
`delta_stats` (reference) and the `calibrated` block (verdict-bearing). `--mirror-null-inits`
controls the null size (smoke 5, checkpoint 6; 0 disables).

**Sign-mirror is asserted EXACT** (the "leak-check like 836's alignment audit"): off-support both
`HOLD`; on-support B=`BUY` vs −B=`SELL`; per-state long-leg count == short-leg count; identical
active-asset set. Same support + same starting latent ⇒ same magnitudes and same "cash", so the
only thing that differs between B and −B is the SIDE.

### 2. Flat-book policy marginals — POLICY-prior locus

At M flat-book states (the all-cash reset book relocated to M rows spread across the panel;
asserted position-free), the RAW per-asset softmax of the factored policy head = the learned PRIOR.
`sell-mass = p(SELL)+p(LimitSell)`, `buy-mass = p(BUY)+p(LimitBuy)`. Read: `sell/buy > 1.5` ⇒
policy-prior locus. A legality-masked variant is reported as a diagnostic secondary. Deterministic
(no RNG).

### 3. Rollout leg decomposition — REWARD / credit-misassignment vs distribution shift

Re-run the 835 deterministic rollout (real Sampled-MuZero vector search, visit-argmax, operator
flags from `--config-json`) instrumented for per-leg REALIZED PnL by side:

- **counts**: long vs short legs opened / closed.
- **`realized_log_pnl_by_side`** = ALL-CLOSE realized log-PnL (long = `log(exit/entry)`, short =
  `−log(exit/entry)`, `exit` = decision-row price, `entry` = pre-close average entry). This is a
  **SUPERSET of the env `reward_attributable` term** (`jax_trading_env.py:1249-1259`), NOT that term
  exactly (MAJOR-2): it sweeps in limit-fill / TP-SL / margin closes, because it snapshots the book
  BEFORE this step's limit/TP-SL/margin, whereas the env snapshots `prev_active_attr` POST
  limit/TP-SL/margin, pre-action (`env:1401`/`:1191`) → the env credits ACTION-caused closes only.
- **`action_attributed_log_pnl_by_side`** = the closes the market SELL/BUY on that asset directly
  commanded (SELL closes a long, BUY closes a short) — a clean, **conservative subset** of the env
  term (it excludes indirect post-action margin liquidations the env would also credit, so it never
  over-credits). This is the reward the TRAINING objective actually credits by side, and it is what
  **credit-misassignment gates on** — so the flag can never fire on realized PnL training never
  credited. (The env symlog-compresses the per-step SUM; the base reward is portfolio log-return,
  side-agnostic.)
- **`realized_dollar_by_side`** / **`unrealized_dollar_by_side`** — the $ views (closed legs, and
  legs still open at rollout end marked-to-market).

**Alignment-clean** (audited like 836): SELL closes a long, BUY closes a short — a side never flips
in ONE step, so each asset closes at most once per step (no off-by-one, no double-count); realized
uses PRE-step entry/size × the decision-row exit price.

Read: short legs realize losses but the policy keeps opening them ⇒ credit mis-assignment; short
legs profited in training-era self-play but not at eval ⇒ distribution shift.

## Fixture smoke (the committed deliverable)

`python experiments/844_short_bias/bias_probe.py --fixture-smoke` — fresh-init UNTRAINED model on
`fixtures/trading_fixture.db`, CPU, exercising the REAL `save_checkpoint`→`load_checkpoint`
roundtrip (params proven different before load, identical after) then all three probes end-to-end.
Plumbing proof: an UNBIASED instrument reads ≈0 / ≈1 when the model is symmetric —

- CALIBRATED mirror-Δ `z` ≈ 0 (`|z| < 3`): a fresh-init "checkpoint" is drawn from the SAME init
  distribution as the matched null, so its null-subtracted, init-standardized z is ≈0 by
  construction and seed-robust (verified across init seeds 0,1,7,8 — max |z| ≈ 1.1). The RAW mirror-Δ
  mean is NOT ≈0 on a fresh init (the vec_act_proj zero-point), so the smoke gates on z, not the raw
  mean,
- flat-book sell/buy ≈ 1.0 (in `[0.60, 1.60]` — comfortably below the >1.5 signal),
- leg decomposition runs and is finite,

so a non-zero signal on a real checkpoint then MEANS something. No interpretation applied to the
untrained case. Determinism: same seed ⇒ bit-identical (verified).

Config (pre-registered, mirrors the 834/836 smoke): `bar_minutes=60` (the fixture's 1-min table
spans ~90 days; hourly bars give enough rows for the walk while exercising the resample path),
`fee16/slip5`, `reward_attributable=True`, `vector_hold_bias=3.0` (the battery operator; the CLI
default 0.0 is a documented trap), small search operator (`vector_k=8`, `sims=8`) for CPU speed —
the checkpoint run takes the FULL battery operator from `--config-json`.

## Gates

Fixture smoke green (CPU); zero diffs outside `experiments/844_short_bias/`; `python -m trader
selftest` 7/7; medium panel (instrument integrity: the mirror construction and leg attribution are
leak-checked like 836's alignment audit).

## Out of scope

Fixing the bias (the NEXT task, informed by this); training/env changes; running on real
checkpoints (SCHEDULED/planner with the run flags — the 836 precedent).
