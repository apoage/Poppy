# TASK_818 — MuZero planner collapse: diagnosis + wiring fix

**Status**: DRAFT — operator direction (2026-06-27). The 812 planner collapses to ~95% HOLD under ALL
cost regimes (incl. cost=0). This task documents the diagnosis (3 distinct failure modes from the full DS24
parameter set) and the fix. Goal: a planner that does NOT collapse — that captures the available gross edge
when one exists.

## Diagnosis (from `logs/training_812_mcts_probe.csv`, all 70 params)

### VERIFIED root cause (workflow + adversarial, 2026-06-27)
`reward_loss` is pinned at the **marginal-entropy floor (~5.1 nats, computed — matches 5.26→4.87)**: the
reward head is trained with **pure categorical CE, NO MSE anchor, NO PopArt** (trainer.py:920-925, fixed
×100 scale), while the VALUE head uses HYBRID (0.8 MSE/Huber + 0.2 CE, trainer.py:404-443) whose docstring
credits the MSE term with "preventing bin-50 collapse" + PopArt-normalized targets (trainer.py:1389-1413).
Pure CE on a near-zero noisy target → the minimizer is the MARGINAL distribution (≈0 for every action) → no
gradient toward the signed conditional mean. In MCTS `Q = reward + γ·value`, the reward term **dominates
value ~10×** (reward ±5 un-PopArt'd via mcts_search.py:163-173; value ±0.5 normalized) but a marginal head
emits a CONSTANT offset that cancels in action ranking → ZERO discrimination → HOLD even at cost=0. (Precedent:
mcts_search.py:117 / TASK_723 — a reward-scale bug caused this exact "MCTS blind to reward → HOLD" before.)

**Adversarial convergence:** the per-step reward is action-invariant noise (mark-to-market of already-held
positions, jax_trading_env.py:1196-1203; per-bar IC→R²~0.01, 806→816). LOAD-BEARING fix = **multi-step,
action-attributable reward** (credit realized PnL on CLOSE / horizon return — the TSMOM trend, where signal
lives). Loss-wiring is necessary-not-sufficient; needs a **policy-prior entropy floor** (else the collapsed
prior, policy_loss −46%, never lets MCTS consider trade branches in Gumbel's top-16/320). **Validate on
pct_hold/alpha, NOT reward_loss** (part-2 normalization shrinks the loss number trivially).

### Fix (tiered, verified)
- **Tier A (wiring, flag --reward-hybrid):** hybrid reward loss (add Huber on `_logits_to_scalar(reward)` vs
  target, mirror trainer.py:404-443) + a `reward_popart` running-standardize of the target (trainer.py:920)
  + CONSISTENT inference denorm in `_reward_to_scalar` (mcts_search.py:163-173) to avoid the TASK_723
  train/inference scale mismatch. Un-sticks reward_loss + makes the head discriminate. Canon-identical OFF.
- **Tier B (load-bearing):** multi-step action-attributable reward (env, jax_trading_env.py:1196-1203 — credit
  realized log-PnL on True→False position flip; down-weight mark-to-market) + policy-prior entropy floor
  (mcts_search.py:237-247 — keep trade actions in the considered set) + reward-head own MLP (capacity,
  muzero_model.py:454-457).
- **Validate on pct_hold/alpha in a FRESH run** (not a collapsed checkpoint — the prior-gating loop is
  self-sealing). Decisive cheap pre-check: fit linear(latent+action → per-step reward) OOS R²; if ~0,
  Tier B (horizon reward) is mandatory.

**Mode 1 — the reward model barely learns (the WIRING fault, the headline).**
`reward_loss` is the DOMINANT loss and the LEAST-improving: policy_loss 4.35→2.35 (−46%), value_loss
1.28→1.17 (−9%), **reward_loss 5.26→4.87 (−7%, stuck, biggest)**. In MuZero, MCTS plans by rolling imagined
trajectories through the learned reward head — a poorly-trained reward head means the search can't tell a
good imagined trade from a bad one, so it defaults to the action with predictable ~0 reward: **HOLD**. This
is why it collapses even at cost=0 (a broken reward model can't find the +24 gross edge for free). Likely
cause: the reward target is a symlog-compressed *differential* return (~±0.05, tiny+noisy) — a brutal
regression target the head can't fit above the noise.

**Mode 2 — exit asymmetry.** n_buy 37→15.6, n_sell 36→**10.0** (sells fall faster). The agent half-learns
to ENTER and barely learns to EXIT — the round-trip's close side is under-incentivized.

**Mode 3 — no stability scaffolding + half-blind logging.** `redo`, `sac`, `bc`/expert all **0/2000 (OFF)**.
Grad-norms were NEVER logged (can't see if the reward head's gradient vanished). The generalization eval
never ran (eval_* all blank). The "better runs" the operator remembers almost certainly had ReDo/SAC on +
grad-norms logged.

**The rational floor (not over-claiming):** avg_value is genuinely negative (−0.47) and value_mse improving
— the value head correctly learned the 21bp cross-sectional signal is unprofitable. So PART of the HOLD is
correct; but Mode 1 means it can't exploit even the gross edge. Both the planner AND the signal are broken.

## Oracle-fitness metric (operator's idea — use as DIAGNOSTIC/selection, NEVER as training reward)
Oracle = perfect foresight, pays MAX fees, takes ONLY net-profitable longs/shorts → the count of genuinely
tradeable big moves (moves that clear 42bp round-trip). **Fitness = fraction of the oracle's net-profitable
trades the model takes.** The model currently trades ~5% of steps (95% HOLD) → captures ≈0 of them. The
*realizable* ceiling is the LINEAR RIDGE's recall, not the oracle's (the oracle cheats) — report model vs
BOTH. **CAUTION:** training the model to match the oracle's trades is the oracle-distillation trap (the very thing 666's Phase-1 oracle-imitation tried and failed at; 666 itself = Easy Mode's FROZEN-model staged ExIt, not an "oracle trap")
trap (the model can't foresee which moves clear cost → cloning foresight doesn't transfer → classifier
anti-correlates with alpha). Use the oracle to MEASURE + SELECT, train on realized post-cost PnL.

## Fix plan (tiered)

### Tier 1 — turn the scaffolding ON + instrument (config; this run)
- `--redo` (recycle dormant neurons), `--sac-entropy` (adaptive entropy — anti-collapse), `--atc-loss`
  (temporal-contrastive stability), `--ds24-level 2` (LOG GRAD-NORMS — the missing Mode-1 instrument), and
  ensure the generalization eval runs. Run at **cost=0** so the +24 gross edge IS available → tests whether
  the fixes let the planner CAPTURE it (no collapse) rather than re-confirming the cost-bound signal.
- **Watch:** does `reward_loss` finally drop? does `value_grad_norm` vs `reward_grad_norm` reveal a vanishing
  reward-head gradient? does `trade_rate` stay up + `avg_value` go positive (collapse averted)?

### Tier 2 — re-wire the reward model (code; the real fix, gated on Tier-1 grad-norm evidence)
If Tier-1 grad-norms confirm the reward head is the weak link: replace the symlog-compressed differential
reward target with a cleaner, PopArt-normalized post-cost-PnL target (so it's not micro-scale noise);
consider more reward-head capacity. Symmetrize the enter/exit incentive (Mode 2). Re-run; reward_loss must
fall for the planner to plan.

### Tier 3 — point it at a learnable, profitable signal (only after the wiring plans)
The genuinely net-profitable trades are the BIG MOVES (trend/vol), not the cross-sectional re-rank the 812
planner was aimed at. The durable fix combines a working reward model + the TSMOM-trend regime (coarser
bars so episodes span trends, low/zero cost). See `runs/816_operating_point/` (TSMOM is the real edge).

## RESULTS (2026-06-27)

**Tier-A SHIPPED + validated (`--reward-hybrid`).** Smoke: reward_loss OFF 5.5 (canon) → ON 1.1; OFF
bit-identical, value path untouched. Validation run `812fix3` (modules + reward-hybrid, cost=0, 184 ep):

| metric | 812@21bp | 812fix@cost0 | **812fix3 (Tier-A)** |
|---|---|---|---|
| reward_loss | ~5 stuck | ~5 stuck | **1.07 — UN-STUCK** |
| pct_hold | 89→95 | 90.5→94.6 | **89.9→91.6 — spiral HALTED** |
| avg_value | −.33→−.48 | worse | **−.25→−.22 — stable** |
| alpha | neg | neg | **−1.1, still neg** |

**Verdict: Tier-A breaks the death-spiral but does NOT make it trade.** reward_loss un-stuck → planner can
rank → stops collapsing (hold stable ~91% vs baselines→95; avg_value stable vs →−.48). BUT still ~91% HOLD,
alpha −1.1 — halted the bleed without learning to trade, because the per-step reward is still
action-invariant. **Both adversarial verdicts confirmed simultaneously:** V2 (reward fix prevents the loop
from the start) AND V1 (reward_loss drops but pct_hold stays — per-step noise). This is the textbook Tier-B
case. `812fix2` (modules-only isolation arm) died at epoch 0 (transient, no error) — arm lost, but
reward_loss=1.07 proves Tier-A is active.

**Tier-B SHIPPED (`--reward-attributable`), canon-OFF bit-identical proven (compiled HLO md5 match).** Gate:
NEVER without `--reward-hybrid` (realized-PnL ~2.4/close spike clips the 401-bin support without Tier-A
PopArt). Validation run `812fix4` (Tier A+B, cost=0) vs `812fix3` (Tier A) isolates Tier-B.

**LATE VERDICT (fix3 @510 ep / fix4 @317 ep):** Tier-B DELAYS but does NOT prevent the drift. fix3 (Tier A)
resumes collapsing — hold 91→**95.1%** by ep 510, trades 74→38, reward_loss ~1.05. fix4 (Tier A+B) drifts
hold to **93.7%** by ep 317 (~30-50 ep slower than fix3), trades 74→48, reward_loss **0.65** (lower —
attributable PnL is learnable structure), alpha stays negative throughout. Both arms converge back toward
HOLD. **Conclusion:** with the wiring honest (reward_loss un-stuck), the model STILL drifts to HOLD because
at cost=0 with no learnable edge trading is zero-EV (slightly neg vs a rising market) → no gradient to trade
→ relaxes to the low-entropy HOLD default. The persistent HOLD is no longer blindness (fixed) — it's the
absence of anything to see. Confirms the PREDICTABILITY-wall conclusion below.

## ORACLE-FITNESS + FINAL CONCLUSION (2026-06-27)
Operator metric built (`runs/818_oracle_fitness/RESULT.md`): max-fee (42bp RT) foresight selective oracle on
the 812 eval window. **~1024 net-profitable >42bp trades/episode** (median 870, ~16/asset, 92% of
asset-episodes) → opportunities **ABUNDANT, not scarce**; selective oracle nets +754%/ep. **Model recall
≈0.02** (812 takes ~20/1024) → operator's 0.25 **REFUTED** (~10× high). DECISIVE cross-check: non-selective
oracle trading the big moves blindly at 21bp/side = **+115% gross → −85% net** (ruin).

**The wall is PREDICTABILITY, not scarcity or wiring.** The catchable moves exist + are huge but only
realizable with foresight; the causal 1-min signal can't forecast WHICH swings clear 42bp ex-ante. This
locates the 806→816 cost-floor precisely: it's a FORECASTING floor.

**Arc concluded:** (1) collapse WAS a real wiring bug (operator's "badly wired" — correct); (2) FIXED by
Tier A+B (reward_loss 5→0.66, spiral broken, ~70 trades retained, action-attributable signal in place);
(3) fixing it exposed the predictability wall — repaired planner trades cleanly but can't profit at 1-min
cost=0. **Recommendation: STOP reward/planner surgery (wiring done). Lever = more-predictable regime — feed
the FIXED planner the 816 TSMOM daily-trend edge (survived 2022 bear + LUNA-out), or a richer data source
(Iris-DEX per-block / futures), NOT 1-min cross-sectional.** Don't blindly clone the foresight oracle (the −85%
blind result IS the proof the model can't foresee which moves clear cost).

## Gate
- **Fixed** iff the planner at cost=0 captures meaningful gross edge (trade_rate stays up, avg_value>0,
  reward_loss drops, alpha trends positive) — i.e. it does NOT collapse when an edge is available.
- **Still broken** iff it collapses even with the scaffolding + grad-norms show the reward head is dead →
  the reward representation is the bug (do Tier 2 immediately).

## Audit / artifacts
`logs/training_812fix_*.csv` (the fix runs), DS24 at localhost:8051 (workstation). Reuse the 812 infra
(runs-as-is). Server-side training, never the workstation display GPU. Per-decision discipline; oracle as
metric not reward.
