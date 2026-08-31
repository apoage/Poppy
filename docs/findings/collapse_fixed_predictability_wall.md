# collapse fixed predictability wall

> TASK_818 — the 812 planner HOLD-collapse was a real reward-wiring bug (now fixed); fixing it exposed that the wall is PREDICTABILITY, not scarcity or wiring.

*Migrated from local agent memory 2026-08-31 (`project_collapse_fixed_predictability_wall.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
**TASK_818 (2026-06-27): the 812 MuZero planner's 95%-HOLD collapse was a REAL reward-wiring bug — diagnosed, fixed in two flag-gated tiers, and fixing it relocated the campaign wall onto PREDICTABILITY.**

**The bug (verified, workflow + adversarial):** reward head trained with PURE categorical cross-entropy, NO MSE anchor, NO PopArt — unlike the value head (hybrid 0.8 Huber + 0.2 CE + PopArt). On the action-invariant per-step reward (mark-to-market of already-held positions → SNR≈0 at cost=0), pure CE just predicts the marginal → constant reward per action → MCTS can't rank → HOLD. reward_loss pinned at the marginal-entropy floor (~5.1, computed; matched 5.26→4.87). Reward term also out-scaled value ~10× in the MCTS Q backup (un-PopArt'd). Precedent: TASK_723 (a reward-scale bug caused this exact "MCTS blind → HOLD" before).

**The fix (shipped, deployed, validated):**
- Tier A `--reward-hybrid` (trainer.py): hybrid Huber+CE reward loss + host-side reward PopArt (unit-variance, also rebalances Q). Smoke: reward_loss OFF 5.5 (canon) → ON 1.1. Canon-OFF bit-identical.
- Tier B `--reward-attributable` (jax_trading_env.py): credit realized log-PnL on a position CLOSE (snapshot-diff `closed = prev_active & ~new_pos_active`), JAX/vmap-safe. Canon-OFF proven (compiled HLO md5 match). HARD GATE: never run without `--reward-hybrid` (the ~2.4/close spike clips the 401-bin support without Tier-A PopArt).
- Result: `812fix3` (A) halts the death-spiral (hold 89.9→91.6 stable vs baselines→95, reward_loss 1.07) but resumes drifting late; `812fix4` (A+B) holds ~91%, retains ~70 trades, reward_loss 0.66 — but alpha STILL negative. Wiring fixed; not profitable.

**The real wall = PREDICTABILITY (oracle-fitness, `runs/818_oracle_fitness/`):** max-fee (42bp RT) foresight selective oracle finds **~1024 net-profitable >42bp trades/episode** (ABUNDANT, not scarce; nets +754%/ep). Model recall **≈0.02** (operator's 0.25 guess refuted ~10×). DECISIVE: non-selective oracle trading the big moves BLINDLY at 21bp/side = **+115% gross → −85% net** (ruin). So the catchable moves exist + are huge but only realizable WITH foresight — the 1-min causal signal can't forecast WHICH clear cost. This locates [modeling_surface_exhausted](modeling_surface_exhausted.md) precisely: a FORECASTING floor, not scarcity or wiring.

**CAVEAT (2026-07-05 forensic audit):** the load-bearing **recall ≈ 0.02 was a PRE-fix trade-COUNT
proxy** (model non-hold ~20 / oracle ~1024, read from the ORIGINAL 812 CSV, 1-min only) — the true
post-fix direction-aligned recall was NEVER measured (no fix4 checkpoint on workstation; open gap
flagged in `lore/notes_2026-06-27.md`). And the coarse-bar epilogue test was confounded (bull-only
812 window + differential reward). So the wall is PROVEN at 1-min, but the planner-on-daily-bars
cell is EMPTY → TASK_821 tests it cleanly (absolute reward, multi-regime, oracle eval-only).

**Decision:** STOP reward/planner surgery (wiring is done + correct). Lever = a more-predictable regime — feed the FIXED planner the [modeling_surface_exhausted](modeling_surface_exhausted.md)-adjacent 816 TSMOM daily-trend edge (survived 2022 bear + LUNA-out, see `runs/816_operating_point/`), or a richer source (Iris-DEX per-block / futures). NOT 1-min cross-sectional. NEVER blindly clone the foresight oracle as a training target — a causal agent can't foresee which moves clear cost (the −85% blind-oracle result is the proof; cf. [classifier_alpha_anti_correlation](classifier_alpha_anti_correlation.md)). (NB: TASK_666 = Easy Mode's frozen-model staged ExIt, NOT an "oracle trap" — earlier notes mislabeled this.)
