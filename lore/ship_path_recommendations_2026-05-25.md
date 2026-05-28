# Ship-Path Recommendations — 2026-05-25

**Operative framing (per request):** MCTS is core. `benchmark_easy.py --eval-mcts-sims 16 joint` is the deploy-truthful scorecard. Canon to beat = a1b1 (with optional rs050 reward-scale, +0.27pp). "Ship" = paired Δα ≥ +3pp at 3-seed with no seed < −5pp.

**State on 2026-05-25**: workstation tip `79430e3`, server tip `3eb0d5f`. 762 W2 closed FAIL. 767/769/763 authorized on server. 764 k8_s7 done on server (165 min, oracle agree 60.4%), k8_s13 in flight; k8_s23 status uncertain. Single-mechanism class CLOSED per `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md`. The "things that move" map is bare: encoder/aux/distill/data-aug/memory/dynamics/value-target/search-depth all null-or-worse under eval-MCTS sims=16 joint.

This memo is a triage of what is left to try. Where I can, I point to *partially-built* mechanisms with code already in tree to lower wall-time, and to *combinations* of individually-tested mechanisms that have never been jointly tested. I flag every candidate against the anti-correlation finding (`project_classifier_alpha_anti_correlation.md`).

---

## Ranked candidates

| # | Candidate | Class | EV/h | Wall (server) | New code | Anti-corr risk | Falsifiable on canon scorecard? |
|---|---|---|---|---|---|---|---|
| 1 | TASK_695 oracle-mask aux head (Phase-1 re-imprint) | Phase-1 imprint | High | ~12h imprint + 8h P2 | ~0 (already in model+trainer; needs `--oracle-mask-head` flag) | LOW (planning-side aux, not classifier sharpening) | Yes — paired Δα vs 760 a1b1 |
| 2 | rs050 + entropy_coef sweep (combination, never jointly run) | Reward + entropy | High | ~18h | ~0 (both flags shipped) | LOW (entropy *de-sharpens* the classifier) | Yes |
| 3 | TASK_756 mixture value target (oracle/nstep blend) | Planning-side | Mid-High | ~12h | ~50 LOC (insertion point known, `train_phase2.py:1257`) | LOW (planning-side, same class as 746 +0.72pp) | Yes |
| 4 | TASK_722 Bella-as-rollouts for MCTS dynamics (closed-loop replacement of dynamics MLP) | Planning + closed-loop | Mid (high ceiling, high risk) | ~12h paired | Large (~500 LOC new module + train loop wiring) | LOW (touches dynamics, not classifier) | Yes once built |
| 5 | rs050 + 767 (action restructure) stacking | Reward + branching | Mid | ~12h | 0 (both shipped, never combined) | LOW (767 prunes search, not classifier) | Yes |
| 6 | TASK_697 oracle-as-MCTS distillation (Phase A: train Phase 1 on MCTS-shaped visit-count targets) | Phase-1 rewrite | Mid (largest unexecuted design) | ~12h imprint | Very large (new bridge_trainer.py per spec) | UNKNOWN | Yes once built |
| 7 | TASK_725-D return-weighted BC + 767 + rs050 (3-stack) | BC weighting + action + reward | Mid | ~12h | ~10 LOC (725-D never landed) | MEDIUM (BC return-weighting may sharpen classifier; mitigated by 767 mask) | Yes |
| 8 | Encoder unfreeze schedule at canon Stage-B | Phase-2 fine-tune | Low-Mid | ~9h | 0 (`encoder_unfreeze_epoch` / `encoder_lr` already wired) | MEDIUM (touches encoder; flagged by anti-correlation) | Yes |
| 9 | Class-balance + entropy joint sweep (canon vs balanced+entropy) | BC weighting + entropy | Mid | ~15h | 0 (both shipped) | MEDIUM (balanced BC is classifier-side; entropy partially counteracts) | Yes |
| 10 | TASK_756c smoothed-oracle value target | Planning-side | Low (likely null) | ~8h | ~15 LOC (`train_phase2.py:619-624`) | LOW | Yes |

EV ordering reflects (a) preferring partially-built mechanisms over greenfield, (b) preferring combinations of mechanisms whose individual run was indeterminate over standalone retries of closed mechanisms, (c) cost discount: 165-min imprint on server (TASK_764 k8_s7 actual, `lore/handoff_2026-05-24_764-launch.md`) means Phase-1 re-imprints are no longer a 12h-per-seed sink.

---

## 1. TASK_695 Oracle-Mask Aux Head — Phase-1 re-imprint with the head ON

**Hypothesis**: `docs/easy_mode.md` Phase 3 ("The Thought") explicitly designs an internalized-oracle head so MCTS can self-evaluate without the external oracle (Klissarov 2025 RL²F, `docs/arXiv-2602.16066v1/`). The mechanism class is *planning-aware* — it adds a per-asset selection-mask predictor that downstream MCTS will use as a learned prior over which assets to consider. Distinct from classifier sharpening: the head is trained on a binary "would the oracle pick this asset" target, not on action-class probabilities — same mechanism family as 746/734 (planning-side), not 738/740/741/757 (encoder-sharpening).

**Mechanism + evidence**:
- Per `docs/easy_mode.md:96-127`, the head is the autodidact handle. Without it Phase 3 cannot exist on this stack.
- `project_classifier_alpha_anti_correlation.md` line 27: "the bottleneck is downstream of the encoder ... look at planning side." This head IS planning-side aux.

**Already done**:
- Model wiring: `mcts_trader/jax_agent/muzero_model.py:286,417-420,564-568,662-663,1003-1004,1052-1053` (`use_oracle_mask_head`, `oracle_mask_prediction`, `oracle_mask_dense{1,2}`).
- Phase-1 trainer: `easy_trainer.py:60-61,216-231,298` (loss term, accuracy metric, weighted into total).
- CLI flag for Phase-1: `train_easy.py:1885-1886` (`args.oracle_mask_head` → `config.use_oracle_mask_head = True`).
- Phase-2 inheritance: `train_phase2.py:392-393,2650` (factory passes `use_oracle_mask_head` through; freeze logic at `:919,2217-2220` keeps it frozen during Stage B).
- Config defaults: `easy_config.py:86-88` (`use_oracle_mask_head: bool = False`, `oracle_mask_weight: float = 0.1`).

**What's missing**:
- The current base imprint `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl` was trained with `use_oracle_mask_head: false` (verified: `config.json:44`). Every Phase-2 run in the campaign inherits this — confirmed across `runs/{749,752,757,758}/stage_b_config.json` all line 44.
- **No code missing.** The experiment is purely a Phase-1 re-imprint with `--oracle-mask-head` + a Phase-2 retrain on top.
- Optional consumer wiring for Phase 3 (MCTS uses the head as a per-asset prior) — out of scope for the ship gate; the head's *gradient signal during imprint* is the immediate experiment.

**Cost**:
- Phase-1 re-imprint: ~12 h workstation per seed (older estimate) or **~165 min server** per seed (TASK_764 k8_s7 actual, `lore/handoff_2026-05-24_764-launch.md`).
- Phase-2 Stage-B: ~2.5 h × 3 seeds × 2-wide ≈ 4-5 h wall.
- Total: ~3 seeds × 165 min imprint + 4-5 h Stage-B = ~13 h server.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint. No seed < −5pp. Mechanism diagnostic: oracle-mask Bal.Acc on benchmark > 60% (the head learned anything at all).

**Anti-correlation check**: The head outputs per-asset *binary selection*, not per-action class probability. It does not sharpen the classifier output that the `[[classifier-alpha-anticorrelation]]` finding implicates (the policy head's per-asset 5-class softmax). It produces a *separate* gradient signal that shapes the encoder toward "represent which assets are worth attending to" — a different feature. Closest prior: 740_bella_distill (encoder aux, regression −1.92pp). Critical distinction: 740 distilled *price predictions*; 695 distills *selection-set membership*. Selection-set is what MCTS searches over; price predictions are what dynamics simulates. Different surface.

---

## 2. rs050 + entropy_coef sweep — the never-jointly-run combination

**Hypothesis**: 762 W1 (`lore/notes_2026-05-23.md` §167) found reward-scale 50 produces paired Δα **+0.27pp** with Δtrade-rate **−0.243%** — a small but reliable mechanism gate PASS. 762 W2 found stacking `fee_penalty_bps` on top does NOT extend it. The remaining unexplored combination is **rs050 + entropy regularization** at the policy head. 763 was queued *standalone* (sweep entropy_coef at canon reward_scale=100), never with reward_scale=50.

**Mechanism + evidence**:
- rs050 reduces over-trading by reducing the magnitude (and thus the reward-clip incidence) of the per-step reward signal — a *gradient-scale* mechanism.
- entropy_coef constrains policy mass distribution — a *probability-distribution* mechanism.
- These are orthogonal levers attacking the same `[[classifier-alpha-anticorrelation]]` pathway (over-confident wrong trades).
- Both are already promotion-relevant individually: rs050 +0.27pp paired (Wave 1), 763 not yet run.

**Already done**:
- `phase2_config.py:30,52-53` — `reward_scale: float = 100.0`, `entropy_coef: float = 0.0`, `min_entropy: float = 0.0` (all CLI-overridable, code shipped commit `b4e0158`).
- `train_phase2.py:632-634, 646` — reward-scale wiring through `SegmentBatchGenerator`.
- `train_phase2.py:1235-1237` — policy entropy term in joint-MCTS loss (factored-path skip is documented as a known limitation; sweep must run with `use_factored_search=False` *or* use the joint-path subset of the loss).
- 763 SPEC at `experiments/763_entropy_reg/SPEC.md` (status READY).

**What's missing**:
- Add a 2D sweep: cells `{rs050 × entropy_coef ∈ {0.005, 0.02, 0.05}}` = 3 cells × 3 seeds. SPEC delta from 763: add `--reward-scale 50` to every variant cell. Comparison baseline: 760 a1b1 sims=32 (canon) OR rs050 alone (`runs/762_reward_shaping/rs050_seed_*`) to isolate the entropy contribution.
- One caveat to check first: 762 W1 cells ran with `use_factored_search: true` (confirmed `runs/762_reward_shaping/*/stage_b_config.json:147`). 763's entropy loss is gated to `not use_factored=True` (`train_phase2.py:1237`). Either flip the production runs to joint MCTS for the entropy variant, or expose a factored-path entropy term first (~10 LOC, per-asset Shannon entropy averaged across the 64-asset axis). The factored-path extension is small and load-bearing; without it 763 is a one-line null risk.

**Cost**:
- 3 cells × 3 seeds = 9 Stage-B runs, ~2.5 h/run = ~22.5 h serial, **~12-15 h wall at 2-wide on server**.
- Optional small code: ~10 LOC for factored-path entropy term.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1, no seed < −5pp, under eval-MCTS sims=16 joint. Anti-correlation diagnostic: the cell must show Δ-Bal.Acc ≤ 0 (classifier de-sharpened) AND Δα > 0 — per 763 SPEC's anti-correlation early-exit gate.

**Anti-correlation check**: LOW risk by construction — entropy regularization is the *direct attack* on the anti-correlation pattern (its purpose is to keep the classifier softer). The risk is that high entropy_coef pushes the policy toward uniform random, collapsing Alpha to ~0 — mitigated by sweeping {0.005, 0.02, 0.05} and exiting if the cell hits min_entropy floor consistently (TASK_763 §risk).

---

## 3. TASK_756 oracle/nstep mixture value target

**Hypothesis**: 746_value_oracle (full-episode oracle PnL as value target) gave paired Δα **+0.72pp** at 5-seed, only positive paired signal in the entire campaign aside from 734_flow_matching (+0.20pp, also reversed at 5-seed). A *convex combination* `α·oracle + (1-α)·nstep` interpolates between high-variance/bias-correct oracle and low-variance/biased nstep. If 746's +0.72pp is variance-limited, an intermediate α should reduce per-seed STD and lift the mean.

**Mechanism + evidence**:
- 746 STD across 5 seeds was 2.28pp (`runs/746_value_oracle/result.md`).
- Per-seed spread in 746 3-seed: s7 −3.99pp vs s23 +8.69pp (`runs/746_value_oracle/result.md:13-23`). The seed-fragility is the variance signature.
- Mixture rationale + math fully spec'd in `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.1.

**Already done**:
- Oracle infrastructure: `mcts_trader/jax_agent/path_solver.py:39-150`, `train_easy.py:595,725,762-766` (oracle_cum_pnl precompute).
- Both value-target read sites: `train_phase2.py:619-633` (oracle_values, nstep_rewards both in batch dict).
- Value-loss dispatch: `train_phase2.py:1257-1297` (current branches: `oracle`, `nstep`, `td0`; both `pre_popart` / `post_popart` clip modes shipped via TASK_769).
- CLI + config skeleton: `phase2_config.py:206-220` (`value_target_mode`, `oracle_value_horizon`, `nstep_horizon`, `value_clip_mode`).

**What's missing**:
- ~50 LOC across 3 files per `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §6:
  - Add `"mixture"` to `value_target_mode` choices (~3 LOC).
  - 3 new CLI flags: `--value-mixture-alpha`, `--oracle-horizon`, `--oracle-smoothing` (~12 LOC).
  - Config fields: `value_mixture_alpha: float = 0.5`, `oracle_smoothing: float = 0.0` in `Phase2Config` (~6 LOC).
  - Argument-translation + validation (~9 LOC).
  - New `elif value_target_mode == "mixture":` block at `train_phase2.py:1257-1297` (~15 LOC). Spec'd line-for-line in §4.1.
- α=1.0 must reproduce 746 bit-identical (the smoke gate in §7a is essential — if it doesn't, the mixture insertion has a sign or clip-order bug).

**Cost**:
- Code: 1-2 h on workstation.
- Smoke: ~30 min (the bit-identity α=1.0 vs α=0.0 reproductions).
- Experiment: 3 α points × 3 seeds × ~3 h = 27 h serial, **~12-14 h wall at 2-wide**.

**Falsifier**: paired Δα ≥ +3pp absolute vs 760 a1b1 sims=32 (the canonical gate), no seed < −5pp. The TASK_756 spec proposes paired-vs-746 with a +1pp gate (which maps to ~+1.7pp absolute given 746's +0.72pp paired), but the deploy-truthful gate stays at +3pp absolute vs canon.

**Anti-correlation check**: LOW. Same mechanism class as 746 — touches the value target, not the policy/classifier output. 746's per-period breakdown (`runs/746_value_oracle/result.md`) showed bear/ftx_crash gains and no trade-rate explosion. Worst-case: smoothing across α produces a flat sweep, which is informative (closes the value-target family). Best-case: variance reduction + sign of value-target mechanism is real but not strong enough to clear +3pp.

---

## 4. TASK_722 Bella-conditioned MCTS rollouts (closed-loop replacement of the dynamics MLP)

**Hypothesis**: The dynamics MLP (`mcts_trader/jax_agent/muzero_model.py:415-420` dynamics_dense1/2, 3.3M params) has been stuck at consistency loss ~1.34 since the start of Phase-2 training. This is the *core* search bottleneck — MCTS expansion happens by running a learned dynamics step in latent space, and an inaccurate dynamics means search "dreams" of futures the encoder/value head don't recognize. **Replace the dynamics step with Bella's flow-matching generator**: encode current window → Bella generates K plausible future windows → encode each → MCTS searches over the real encoded latents instead of dynamics MLP outputs. Dynamics MLP becomes the consistency *filter*, downweighting branches where Bella hallucinated.

**Mechanism + evidence**:
- `tasks/TASK_722_BELLA_DYNAMICS_ROLLOUTS.md` is a complete design — every encoder/decoder/freeze decision is specified.
- 758_bella_augment showed catastrophic failure at −16.54pp (`lore/notes_2026-05-19.md` §148-149), but that was using Bella as *replacement training data* (synthetic slots in the BC loss). TASK_722 uses Bella as *runtime rollouts during MCTS search*, never seen by the policy as training data. Different surface entirely.
- The "Bella class CLOSED" finding (`project_encoder_bella_class_closed.md`) explicitly says: *"Re-opening would require a fundamentally different ansatz (e.g., Bella as world-model rollouts for MCTS, not as supervision signal or data)"* — TASK_722 is literally that ansatz.
- Bella standalone works (verified by 28a122e server-side smoke). The mechanism is a known-good generator wired into a never-tested consumer.

**Already done**:
- Bella generator: `diff_teacher/` subtree, `generator.generate_on_device(...)` (per TASK_722 §Step 2, current ckpt produces 768d/12L conv + 10 ODE steps per `lore/notes_2026-05-18.md` §134).
- Bella checkpoint: deployable per `MEMORY.md` workstation↔server contract.
- Skeleton extraction: `diff_teacher/skeleton/extraction.py` (referenced by TASK_722 §Step 1 alternative).
- Encoder: `mcts_trader/env/jax_encoder.py:491-535` produces `(32, ws, n)` from observed prices — reusable for encoding each Bella future.
- MCTS expansion site: `mcts_trader/jax_agent/factored_search.py:_search` (or `mctx_search.py:_recurrent_fn`) is the swap point.

**What's missing**:
- New module `mcts_trader/jax_agent/bella_dynamics.py` (~500 LOC per TASK_722 §"Step 3 Bella-MCTS Integration").
- Encoder modification to accept pre-generated price windows (sliding-window encoding of Bella outputs): ~50 LOC.
- Phase-2 trainer integration: replace the `recurrent_inference` call in the search inner loop with `BellaMCTSDynamics.expand_node` (~30 LOC in `train_phase2.py`).
- Consistency score: dynamics MLP scores Bella's encoded latents — ~30 LOC.
- `generator.generate(seq_length=1440)` JAX pad bug — TASK_722 §134 documents the workaround (chained `generate_on_device` at native 400 with a seam tolerance). This is a known issue, not new work.

**Cost**:
- Code: bella_dynamics module + integration + smoke, two-gate review protocol per `lore/notes_2026-05-18.md` §138.
- Wall: per TASK_722 §"Performance Budget", expect ~200-500 ms per train batch (vs ~105 ms canon), so ~2-3× slower training step. Stage-B at ~5-7 h/seed instead of 2.5. Paired 3-seed × 2-wide = **~12-15 h wall** for the experiment itself.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1, no seed < −5pp. Mechanism diagnostic: consistency score distribution must concentrate (Bella futures the dynamics MLP "agrees with" should be a non-trivial subset); if every Bella future is rejected as inconsistent, Bella's flow isn't aligned with the learned latent dynamics and the integration is meaningless.

**Anti-correlation check**: LOW direct risk — touches dynamics + search, not classifier output. Indirect risk: if Bella's generated futures are systematically *easier* (the 7× oracle-PnL ratio finding from `lore/notes_2026-05-18.md` §134), the policy could over-anchor to easy rollouts and Alpha collapses on hard regimes. Mitigation: tag the result.md if per-window oracle-PnL ratio on Bella branches > 2× historical, same protocol as 758.

**Why this is here**: this is the *largest unexecuted designed mechanism* in the campaign. `tasks/TASK_722` is a complete spec from February 2026 that never got implementation budget because the campaign was burning down small-mechanism probes. Single-mechanism class is now closed; the budget is freed.

---

## 5. rs050 + 767 (action restructure) — combination

**Hypothesis**: 762 W1 found rs050 reliable at +0.27pp. 767 action-space restructure (drop limit orders, 320→192 effective actions) is queued but not yet run. The two attack different bottlenecks: rs050 attenuates reward magnitude (the *gradient-magnitude* failure mode); 767 prunes the action space (the *search-branching-factor* failure mode). Both `disable_limit_orders` and `reward_scale` are CLI-orthogonal — combining them in one cell costs the same wall as either alone.

**Mechanism + evidence**:
- 767 oracle audit (`mcts_trader/jax_agent/path_solver.py:406,473`): "LimitSell/LimitBuy are never emitted by this oracle." So the Phase-1 imprint never trained the policy toward limit actions — the limit logits are essentially noise at imprint time. Phase-2 search can still drift into them; 767 prevents that drift.
- rs050 mechanism is independent: it scales the per-step PnL signal that feeds the value target, not the action logits.

**Already done**:
- 767 wiring: `phase2_config.py:45` (`disable_limit_orders: bool = False`), `train_phase2.py:1060-1081, 1742-1754, 2332-2333` (factored-path mask + MCTS expansion guard + CLI). Production-deployed per `lore/notes_2026-05-23.md` §165 (with the TASK_771/F1 mean-shift fix).
- rs050: `phase2_config.py:30`, `train_phase2.py:632-634`. Production-deployed per commit `b4e0158`.
- AUTHORIZATION for 767 standalone exists at `runs/767_action_space/AUTHORIZATION_2026-05-23.md`.

**What's missing**:
- New cell: `--reward-scale 50 --disable-limit-orders` × 3 seeds, paired against 760 a1b1 sims=32 (full-canon baseline). Add to the 767 SPEC or write a thin SPEC `experiments/767plus_rs050/SPEC.md`.
- No new code.

**Cost**:
- 3 seeds × 2.5 h = 7.5 h serial, **~4-5 h wall at 2-wide**. Lowest-cost candidate in the table.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1, no seed < −5pp. Both individual contributions (rs050 +0.27pp, 767 TBD) are known small; the combination either super-adds (mechanism interaction, ship-worthy) or doesn't (both are small standalone, sum is still small).

**Anti-correlation check**: LOW. 767 prunes options that the oracle never selects; rs050 attenuates magnitude. Neither sharpens a per-asset classifier output.

---

## 6. TASK_697 Oracle-as-MCTS distillation — Phase-1 architectural redo

**Hypothesis**: Foxglove Phase-1 trains the encoder/TRM with *flat one-hot oracle action labels* and *scalar oracle PnL*. Poppy Phase-2 then expects the same network to produce MCTS-shaped outputs (visit-count distributions, backed-up Q-values, dynamics rollouts). There is a **representation gap**: the network learns classification features in Phase 1, not search-compatible features. TASK_697 closes the gap: train Phase-1 with *MCTS-shaped* targets from day one (multi-pass oracle visit distributions, per-step policies, Q-values).

**Mechanism + evidence**:
- Multi-pass oracle (`mcts_trader/jax_agent/path_solver.py` `multipass_oracle_policy`, batched variant) already produces N rollouts with noise perturbation per TASK_673 — exact analog of MCTS simulations.
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` is a 270-line complete design. Mapping table is concrete:
  - N oracle passes ↔ N MCTS simulations
  - per-pass selections ↔ visited actions
  - aggregated frequency ↔ visit-count distribution
  - mean return ↔ root value
- Per `docs/easy_mode.md` Phase 2 "The Dream" is the dynamics warmup that 697 Phase A would replace with oracle-derived multi-step trajectories.

**Already done**:
- Multi-pass oracle infrastructure exists.
- `easy_trainer.py` and `train_easy.py` are the imprint loop to be replaced by `bridge_trainer.py`.

**What's missing**:
- New `bridge_trainer.py` (per TASK_697 §"Recommended approach").
- New `oracle_to_mcts_targets()` function.
- Dynamics training in bridge (replaces the consistency-only Phase-2 Stage-A).
- Buffer-format change: from `(observations, oracle_targets)` to `(observations, actions, rewards, visit_distributions, values, dones)`.
- Action-space alignment decision (TASK_697 §A.4): 320 vs 192 — operator pick.
- Phase A imprint + Phase B freeze + Phase C unfreeze pipeline.

**Cost**:
- Phase A imprint: ~165 min server (k8_s7 baseline) × 3 seeds = ~9 h.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1 at end-of-Phase-C. Earlier gate: Phase-A imprint must hit a balanced-accuracy floor comparable to current xs32 imprint (~85%) within same epoch budget.

**Anti-correlation check**: UNKNOWN. The mechanism redesigns the *target* the encoder is trained against, but does so by enriching it with planning structure rather than sharpening the classifier. Closest prior: 740 (encoder distill from Bella, regression) and 738 (drop cross-sectional, catastrophe). TASK_697 doesn't drop signal, it adds structure — should not trigger the anti-correlation, but should be smoke-gated by reading Phase-A's per-asset Bal.Acc vs canon imprint at matched epoch.

**Why this is in the table**: this is the largest unexecuted design in the project that is *not* already a roadmap entry. TASK_768 (channel restructure) is now done as Option A bit-identical (`tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log"). TASK_697 has been "ON HOLD" since February 2026 (`tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` line 6) and was the most carefully designed Phase-1 rewrite.

---

## 7. TASK_725-D return-weighted BC + 767 + rs050 (3-stack)

**Hypothesis**: The class-balanced BC loss (TASK_726, shipped) gives trades up to 64× more weight than holds, but treats all trades as equally informative. **Return-weighted BC** (TASK_725-D) weights each oracle decision by the magnitude of its expected forward return — bull-period BUY signals get more gradient than sideways BUY signals. This directly attacks `project_bull_blindness.md` root cause. Combined with rs050 (reward-magnitude attenuation) and 767 (search-branching prune), the stack covers three failure surfaces.

**Mechanism + evidence**:
- TASK_725-D in `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment D" specifies the exact ~10 LOC change at `train_phase2.py` BC computation.
- `project_bull_blindness.md`: bull-regime failure persists across all encoder/aux mechanisms tried. The BC loss never weights by return magnitude.

**Already done**:
- TASK_726 class-balanced BC: `train_phase2.py:1053-1054,1339` (`use_bc_balance`, `bc_max_trade_weight`).
- 725-D draft: lives in TASK_725 doc; **not landed**.

**What's missing**:
- ~10 LOC in `train_phase2.py:make_stage_b_train_step` per TASK_725 spec.
- `bc_return_weighted: bool = False` in `phase2_config.py`.
- `--bc-return-weighted` CLI flag.

**Cost**:
- Code: ~2 h on workstation including smoke.
- Experiment: 1 cell × 3 seeds × 2.5 h = 7.5 h serial, ~4-5 h wall at 2-wide. Add to 762 W1's rs050 result for paired Δα.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1, no seed < −5pp. Mechanism diagnostic: bull-regime Bal.Acc Δ should be ≥ +5pp on the 3-stack vs canon (per `project_bull_blindness.md`'s implied target).

**Anti-correlation check**: MEDIUM. Return-weighted BC could sharpen the classifier in the high-return regime — exactly the failure mode 738 hit (−18.44pp). Mitigation: 767's action-space prune narrows what the classifier can choose, and rs050 attenuates the reward-magnitude signal. If the stack still triggers anti-correlation (Δ-Bal.Acc > 0 AND Δα < 0), peel rs050 off first (gives the classifier more reward signal back to compensate). Flag this candidate explicitly — if running, watch the trade-rate diagnostic on epoch 50 of seed 7 and abort the cell if trade-rate > 2× canon (early warning).

---

## 8. Encoder unfreeze schedule at canon Stage-B (TASK_725-C, no new code)

**Hypothesis**: Encoder is frozen during Stage-B by default (`phase2_config.py:225` `encoder_lr: float = 0.0`). The Phase-1 imprint shaped it for one-hot oracle classification; Stage-B's MCTS-derived targets may want different features. Unfreezing the encoder at a very low LR after a warmup lets Stage-B adapt encoder features without destroying the Phase-1 imprint.

**Mechanism + evidence**:
- TASK_725 §"Experiment C" specifies the exact lever.
- `phase2_config.py:225-226` confirms `encoder_unfreeze_epoch` is already a wired field.
- `train_phase2.py:957-970, 2173` wires the unfreeze logic.

**Already done**:
- 100% of the code wiring. Just needs to be turned on via `--encoder-lr 3e-5 --encoder-unfreeze-epoch 100` (or similar).

**What's missing**:
- An experiment cell. Choose a single unfreeze schedule (e.g. `encoder_lr=3e-5` from epoch 0, or epoch 100), 3 seeds × 2.5 h.

**Cost**:
- 0 code.
- 1 cell × 3 seeds × 2.5 h = 7.5 h serial, **~4-5 h wall at 2-wide**.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1. Anti-correlation diagnostic: encoder gradient norm in Stage-B must stay bounded (< 2× canon at convergence) — if encoder destabilizes, peel back the LR.

**Anti-correlation check**: MEDIUM. The whole `project_classifier_alpha_anti_correlation.md` and `project_encoder_bella_class_closed.md` story is that encoder-side changes anti-correlate with Alpha. This candidate *changes the encoder*. It is here because (a) the change is bidirectional (gradient flows in from MCTS targets, not just from oracle), (b) it is implementable in 0 LOC so cost-of-falsification is minimal, (c) the prior null findings were on *encoder design changes* (architecture, channels, distillation targets), not on *encoder LR scheduling* — that surface is genuinely untouched per `lore/notes_2026-05-19.md` §149.

---

## 9. Class-balance + entropy joint sweep (canon vs balanced+entropy)

**Hypothesis**: `bc_class_balanced=True` is current canon (`train_phase2.py:1053`). Adding entropy regularization on top *should* counteract the over-confidence that class-balancing creates — but the two have never been swept jointly. The canonical claim is "balanced BC + entropy reg trades classifier sharpness for trade selectivity at the right rate."

**Mechanism + evidence**:
- TASK_726 (class-balanced BC) already in production.
- 763 SPEC entropy_coef sweep ready (`experiments/763_entropy_reg/SPEC.md`).
- Joint sweep never spec'd.

**Already done**:
- Both flags wired (covered in candidate #2).

**What's missing**:
- Spec writeup. 3 cells × 3 seeds (canon=baseline, bc_balanced+ec0.02, bc_balanced+ec0.05).

**Cost**:
- 9 runs × 2.5 h = 22.5 h serial, **~12-15 h wall at 2-wide**.

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1. Same anti-correlation diagnostic as candidate #2.

**Anti-correlation check**: MEDIUM — class-balanced BC is in the encoder-direct gradient pathway. Entropy is the counterweight. Net direction depends on coefficient balance.

---

## 10. TASK_756c smoothed-oracle value target

**Hypothesis**: 746's high seed-variance (STD 2.28pp) might be driven by per-window peakiness of the oracle PnL signal. EWMA over the per-step oracle deltas smooths this. Likely-null candidate, listed for completeness because TASK_756c is fully spec'd at minimal code cost.

**Mechanism + evidence**: Per `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.3.

**Already done**: All oracle/cum_pnl precompute infrastructure.

**What's missing**: ~15 LOC EWMA in `train_phase2.py:619-624` + `--oracle-smoothing` CLI flag + `oracle_smoothing: float = 0.0` config field.

**Cost**: ~1h code, ~8h experiment (3 λ × 3 seeds).

**Falsifier**: paired Δα ≥ +3pp vs 760 a1b1. Mechanism diagnostic: per-period Δα must show STD reduction vs 746 (if it doesn't, smoothing didn't reduce variance).

**Anti-correlation check**: LOW (planning-side, value target).

---

## Partially-implemented mechanisms (higher priority than greenfield)

These are mechanisms with code already in tree that has *never been activated in a campaign run*. Listed in cost order.

| Mechanism | Code site | Last active in campaign | What activates it |
|---|---|---|---|
| Oracle-mask aux head | `muzero_model.py:286-1053`, `easy_trainer.py:60-298`, `train_easy.py:1885-1886` | Never — `use_oracle_mask_head: false` in every recorded run config | `--oracle-mask-head` flag on Phase-1 re-imprint |
| Encoder unfreeze schedule | `phase2_config.py:225-226`, `train_phase2.py:957-970,2173` | Default OFF (encoder_lr=0.0) | `--encoder-lr 3e-5 --encoder-unfreeze-epoch N` |
| Return-weighted BC (TASK_725-D) | Designed in `tasks/TASK_725_BULL_EXPERIMENTS.md` §D | Never landed | Add ~10 LOC at `train_phase2.py` BC site |
| Factored-path entropy term | Joint-only at `train_phase2.py:1235-1237` | 763 SPEC requires it | ~10 LOC per-asset entropy averaged over 64-asset axis |
| `--bella-augment-ratio` low setting | `runs/758_bella_augment/` ran at default (presumably ratio=1.0) | One catastrophic run | Run at ratio 0.1-0.25 (low historical-dominant blend) |
| `oracle_smoothing` (TASK_756c) | Spec'd, ~15 LOC | Never landed | Land the EWMA |

Each of these is closer to "ship" than any greenfield mechanism because (a) compile failures and dependency surprises are already resolved, (b) the design has been audited at least once, (c) the wall-time estimate is bounded by known prior runs.

---

## Combination experiments (individually run, never jointly run)

The campaign has tested mechanisms one-axis-at-a-time. Combinations are the largest unexplored space; specifically, *combinations of mechanisms that individually had small-positive or neutral results.*

| Combination | Individual results | Joint hypothesis | Anti-corr risk |
|---|---|---|---|
| rs050 + entropy_coef (no canonical entropy ever run) | rs050: +0.27pp paired W1. entropy: never run | Both reduce trade-rate by different surfaces; combined should super-add | LOW |
| rs050 + 767 (limit-orders off) | rs050: +0.27pp. 767: pending | Reward-magnitude × search-branching, orthogonal | LOW |
| 746 oracle-value + 734 flow-matching (both planning-side small-positive) | 746: +0.72pp 5-seed. 734: +0.20pp 5-seed | Both touch the planning-target gradient; if they're hitting different aspects of value learning, super-additive | LOW |
| TASK_726 balanced BC + return-weight (TASK_725-D) + 767 | 726 in production. 725-D never landed. 767 pending | Three different attacks on `[[bull-blindness]]` | MEDIUM |
| Oracle-mask aux head (Phase-1) + factored search (Phase-2) | Mask: never run. Factored: in production | If mask head adds search-prior signal, factored Q-values benefit | LOW |
| Encoder unfreeze + 746 oracle-value target | Both never combined | If encoder rep is the bottleneck for value learning, unfreezing helps the oracle target stick | MEDIUM |

The highest-leverage combinations are the LOW-anti-corr ones with both legs already plus-or-zero individually: **rs050 + entropy** and **rs050 + 767** lead.

---

## Closed-loop / Phase-3 status

`docs/easy_mode.md` describes a 3-phase pipeline: Phase 1 "The Cheat" (supervised imprint), Phase 2 "The Dream" (dynamics + MCTS ExIt), Phase 3 "The Thought" (frozen-model MCTS at inference, autodidact self-correction via oracle-mask head).

**Phase 1**: implemented in `train_easy.py` + `easy_trainer.py`. SHIPPED.

**Phase 2**: implemented in `train_phase2.py` with `--stage A` (dynamics warmup, `train_phase2.py:11-12` docstring example, `:1586` Stage A loop) and `--stage B` (MCTS ExIt, `:1671` Stage B loop, `:1682` "Factored or MCTS ExIt"). SHIPPED.

**Phase 3 "The Thought"**: NOT IMPLEMENTED on the current Foxglove/Poppy stack.
- `grep -n "Stage C\|stage_c\|stage == .C\|phase3\|phase_3"` across `train_phase2.py`, `train_easy.py`, `mcts_trader/jax_agent/`, `diff_teacher/` returns zero matches in the active pipeline.
- The autodidact mechanism (`docs/easy_mode.md:96-127`) requires the oracle-mask head's outputs to feed back into MCTS as a per-asset selection prior — **the head is built in the model (`muzero_model.py:564-568`) but no consumer code exists**.
- Self-play in `mcts_trader/training/self_play.py` is part of the *legacy* (pre-Foxglove) pipeline (`mcts_trader/CLAUDE.md` lists `mcts_trader/training/` as the old `train_manager.py` entry). It is not invoked from `train_phase2.py` or `train_easy.py`.
- ExIt loop (Phase-3 §C "self-play generates improved targets → retrain") does not exist on the current stack. Every campaign experiment treats the Phase-1 imprint and Phase-2 trainer as the end of the pipeline.

**This is the largest unexecuted design in the project.** Real-trade inference does use MCTS (`real_trade_jax.py:131,326`) — so the *runtime* of Phase 3 is implicit in live trading. But the *training-time refinement loop* of Phase 3 (ExIt with MCTS-generated targets, autodidact via oracle-mask head) is missing.

For "ship", Phase 3 is not strictly required — `benchmark_easy.py --eval-mcts-sims 16 joint` already runs MCTS at eval, so a model that ships under the canon scorecard has its inference loop covered. But for breaking the +3pp ceiling, Phase 3 ExIt is the standard ML answer (closed-loop refinement), and the current campaign has never tried it.

The minimal Phase-3 implementation that matches `docs/easy_mode.md`:
1. Activate oracle-mask head (Candidate #1).
2. Inference-time MCTS using the mask head as per-asset prior — small modification to `mctx_search.py` or `factored_search.py` `_q_to_policy` to mix in the mask head's output.
3. Generate fresh training targets from MCTS rollouts on real data.
4. Retrain Phase 2 Stage B on these targets (ExIt round 2).

Combines candidates #1 + #4 (Bella rollouts) naturally.

---

## Fundamental constraints (things the current stack imposes that block ship)

| Constraint | Where | Implication |
|---|---|---|
| **Factored search is canon** | `runs/762_reward_shaping/*/stage_b_config.json:147` `"use_factored_search": true` | TASK_759's MCTS-sims sweep is *only meaningful on the joint path*. Factored search is 1-step lookahead, no rollouts. Any mechanism that targets MCTS depth implicitly forks the stack. |
| **Cross-sectional channels load-bearing** | `mcts_trader/env/jax_encoder.py:491-535` (TASK_732 channels), 738 ablation −18.44pp | Universe-coupled. Bakes 64-asset assumption into the input tensor. TASK_768 Option A landed bit-identical (`tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log") but the variable-universe path (n as runtime arg in weights) is still ahead. |
| **Phase-1 imprint frozen** | `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`, every Phase-2 run starts from same ckpt | Every paired Δα measurement is conditional on this imprint. Re-imprinting (candidates #1, #6, TASK_764, TASK_768) breaks comparability with the entire 738-769 campaign. |
| **eval-MCTS sims=16 joint = canon scorecard** | `benchmark_easy.py --eval-mcts-sims 16` since 2026-05-19 | Asymmetric-lift finding: this lifts the *baseline* α more than the variants on average (`project_single_mechanism_exhausted.md`). Any variant whose Phase-2 was trained against argmax eval has the wrong "deployment target" and a deeper retraining of Phase-2 against eval-MCTS at training-time may unlock additional gain (this is what TASK_759 P1 tried at sims=64 train-time, closed +0.35pp). |
| **8.5M parameter model, encoder = 3.5%** | `mcts_trader/jax_agent/muzero_model.py:243` | TRM reasoning is 93% of params. Scale-up tried per TASK_683/729, did not break the ceiling. Don't scale; redesign structure. |
| **Single-asset inference impractical** | Cross-sectional channels (constraint #2) + factored Q's averaged across batch axis | Cannot deploy on a different universe size without TASK_768's variable-n extension. |
| **No deploy-time uncertainty estimate** | `real_trade_jax.py:131,326` MCTS used for action selection only | The model produces a single best action; there's no calibrated confidence to gate live trades. A confidence-gated inference path is missing. |

These are deployment-relevant constraints. The biggest is that **factored search is the production path** — many proposed "MCTS sweep" experiments are arguments against the current stack rather than refinements of it. Acknowledge or fork.

---

## Early-notes salvage — ideas surfaced in 2025-12 → 2026-02 that were never tasked

The operator asked for this explicitly. Reading the 25 oldest lore notes plus the design docs in `lore/`, the dormant ideas are:

1. **Adversarial market generator self-play** (`lore/ALPHAZERO_TRADING_VISION.md` Phase 2, 2025-12-19, never tasked beyond design): an agent-vs-market game where the market generator learns to attack the agent's portfolio. Could be implemented today using Bella as the generator backbone. *Concrete attack surface*: Bella's flow-matching loss + a reward signal "Bella loss when conditioned to produce a sequence agent loses on." Never tested. Combines with TASK_722 (Bella as MCTS rollouts) — Bella becomes both world-model AND adversarial generator. Hard to falsify on the canon scorecard (would need a paired comparison: agent trained with adversarial Bella vs canon, both eval'd on real-data benchmark).

2. **Self-Play Dojo with population + ELO** (`lore/SELF_PLAY_DOJO_VISION.md` 2026-01-11, never tasked): N=8 to 32 agents competing on shared markets, winner-loser distillation, exploiters/champions. Spec'd at the architecture level (`SelfPlayDojo`, `MultiAgentMarketEnv`), never implemented. **Most aligned with the AlphaZero pedigree of MuZero.** Falsifiable: a self-play champion benchmarks against canon on eval-MCTS sims=16. Risk: market-impact modelling complexity (the multi-agent env needs to model how agents' actions affect prices). Could start without market impact (just independent agents on same data) for cheap.

3. **Synthetic curriculum Levels 4-22** (`lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`, 2025-12-19, partially implemented at Levels 0-14 per `lore/notes_2026-01-13.md` §1): pairs trading, triangle arbitrage, index arbitrage, momentum-reversal patterns. The dojo has been deprecated (current campaign uses `tools/orchestration/e-dojo_v1.py` per `tools/orchestration/CLAUDE.md`). Levels 15-22 (volatility clustering GARCH, microstructure noise, etc.) never implemented. *This is curriculum design*, separate from architecture experiments.

4. **NeuralOGCM physics core** (`lore/NEURAL_MARKET_PHYSICS_DESIGN.md`, 2025-12-19, never tasked): differentiable market physics with learnable parameters (α, β, σ, γ per market). Inspired by Wu et al. 2025 ocean modelling paper. Provides long-term stability and prevents hallucination. Never implemented. Estimated weeks of design + integration. Aligns with TASK_722 (Bella as world-model) and could replace the dynamics MLP with a physics-informed dynamics block.

5. **Multi-pass oracle for visit-count distillation** (referenced in TASK_697, multi-pass code exists per `mcts_trader/jax_agent/path_solver.py:multipass_oracle_policy`): the bridge that TASK_697 needs to actually distill MCTS-shaped targets from oracle. The multi-pass code itself is implemented; the *bridge_trainer.py* that uses it is not.

6. **k=4 specifically never tested vs k=8/12** (`lore/notes_2026-02-18.md:39`): the original k-sweep was skipped during a code revert; canon stayed at k=4 by default. TASK_764 SPECs this explicitly; the experiment is now in flight (k8_s7 done, k8_s13 in flight per `lore/handoff_2026-05-24_764-launch.md`).

7. **Reward Spike + reward dropoff** investigations (TASK_126a, TASK_126b, etc., 2025-12 to 2026-01, all closed as bugs): the reward-hacking pathway. The fixes landed but the broader question of *whether the reward shaping is correctly aligned with the deployment-truthful operator* was never re-examined post-eval-MCTS pivot. 762 W1 partially addressed this; 769 partially addressed the clip; the **clip-mode × reward-scale × eval-decoder joint sweep** has never been run.

The highest-impact salvage candidates are **(1) adversarial market generator** (combines with TASK_722, attacks the data distribution that the canon scorecard tests on) and **(2) self-play dojo** (this is what an "AlphaZero-for-trading" pitch actually means and has been parked since January). Both are large efforts. Both align with the operator-stated framing: "MCTS is core."

---

## No-rewrite verdict (the honest version)

**Under the constraint that single-mechanism class is closed AND the eval-MCTS sims=16 joint scorecard is the deploy-truthful target:**

A small-perturbation candidate from the table above (#1, #2, #3, #5, #7, #8, #10) *might* ship. The honest prior, given the campaign's 14 evaluations 0 promotions and the eval-MCTS recal asymmetric-lift finding, is that each has 10-30% individual probability of clearing +3pp. None are slam-dunks. The **combinations** (#2 rs050+entropy and #5 rs050+767) are the most prior-favorable because both legs were independently zero-to-positive.

If none of the small candidates ship, the project is at a structural choice:
- **TASK_722 Bella rollouts** is the smallest *rewrite* on the table — already-designed unrun mechanism.
- **TASK_697 oracle-as-MCTS** is a larger already-designed unrun mechanism that touches the imprint layer.
- **Phase 3 ExIt closed-loop** (combining candidate #1 + candidate #4) is the design `docs/easy_mode.md` originally specified; the project has never implemented its own final phase.
- **Self-play dojo** (`lore/SELF_PLAY_DOJO_VISION.md`) is the AlphaZero-aligned rewrite.

**Rewrite identified**: **the rewrite is Phase 3 / ExIt closed-loop, with the oracle-mask aux head as its gating prerequisite**. Every other "rewrite" candidate on this list is a partial step toward Phase 3. The project's original 3-phase design has only ever shipped Phases 1 and 2; Phase 3 — the mechanism that distinguishes ExIt-based training from supervised + RL — is unbuilt. If small-perturbation candidates close null, that is the path. Cost of the experiment after build is the same as any candidate (~12 h paired 3-seed).

The "no rewrite, ship from small candidate" path is real but its probability mass is concentrated in the combinations (#2, #5) not the standalone retries. Recommend running those first as the cheap last small-mechanism probes before committing to Phase 3 ExIt build.

---

## Recommended execution order

1. **Now**: launch candidate #5 (rs050 + 767). 0 new code, 4-5 h wall, the cheapest probe on the list. Production already deployed both flags.
2. **Concurrent (if 762 → 763 queue clears a GPU)**: candidate #2 (rs050 + entropy, with the small factored-path entropy term first). ~12 h wall.
3. **If #5 or #2 clear +3pp**: 5-seed extension, ship.
4. **If both null**: candidate #1 (oracle-mask aux head). Phase-1 re-imprint × 3 seeds on server (~165 min/seed) + Stage-B paired 3-seed (~5 h wall). Mid-week deliverable.
5. **Concurrent or next**: candidate #3 (TASK_756 mixture value target). ~50 LOC code + ~14 h wall.
6. **If everything null**: commit to the Phase-3 ExIt build (oracle-mask head + closed-loop training with MCTS-generated targets). This is the project's original design and the campaign's most expensive untaken step.

The Bella-rollouts candidate (#4) is held until candidates #1, #2, #3, #5 are decided — it is high-risk-high-reward and worth the build cost only if the cheap probes confirm the planning-side is the right surface but the available levers are too small.

---

## References (load-bearing files cited)

- `mcts_trader/jax_agent/muzero_model.py:286-1053` — oracle-mask head wiring
- `mcts_trader/jax_agent/path_solver.py:39-150, 406, 473` — oracle, multipass, limit-order claim
- `mcts_trader/jax_agent/factored_search.py:128-157` — factored Q-to-policy (with TASK_771/F1 fix)
- `mcts_trader/jax_agent/mctx_search.py:7-55` — joint MCTS knobs (untouched class)
- `mcts_trader/env/jax_encoder.py:491-535` — cross-sectional channels (load-bearing, 738 ablation)
- `train_phase2.py:619-633` — value-target read sites (oracle_values, nstep_rewards)
- `train_phase2.py:1042-1297` — value-loss dispatch + clip-mode gates (TASK_769)
- `train_phase2.py:1053-1054, 1339` — TASK_726 class-balanced BC
- `train_phase2.py:1235-1237` — TASK_763 entropy term (joint-only)
- `train_phase2.py:1060-1081, 1742-1754, 2332-2333` — TASK_767 limit-order mask
- `train_phase2.py:957-970, 2173` — encoder unfreeze (TASK_728)
- `train_phase2.py:1685-1690, 1719-1736` — search creation (factored vs joint)
- `phase2_config.py:30, 37, 45, 52-53, 56-60, 220-226, 229-231` — config field inventory
- `easy_config.py:64, 86-88, 158` — `oracle_k`, `use_oracle_mask_head`, `oracle_mask_weight`
- `easy_trainer.py:60-298` — oracle-mask loss term
- `train_easy.py:1885-1886` — `--oracle-mask-head` CLI
- `jax_ckpt_easy_phase1_xs32/easy_medium/config.json:44` — base imprint has `use_oracle_mask_head: false`
- `runs/746_value_oracle/result.md` — 5-seed +0.72pp paired
- `runs/762_reward_shaping/*/stage_b_config.json:147` — production uses `use_factored_search: true`
- `tasks/TASK_695_ORACLE_MASK_PREDICTION_HEAD.md` — autodidact head spec
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` — bridge_trainer design
- `tasks/TASK_722_BELLA_DYNAMICS_ROLLOUTS.md` — Bella-as-rollouts design
- `tasks/TASK_725_BULL_EXPERIMENTS.md` §C, §D — encoder unfreeze + return-weighted BC
- `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` — mixture value target
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — roadmap framework
- `tasks/TASK_768_BASE_MODEL_REFRESH.md` — Option A shipped, broader refresh future
- `docs/easy_mode.md:96-127` — Phase 3 autodidact design
- `lore/notes_2026-05-19.md` §149 — Bella class closed, planning pivot trigger
- `lore/notes_2026-05-20.md` §151 — single-mechanism class closed verdict
- `lore/notes_2026-05-23.md` §167 — 762 W2 FAIL verdict
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server imprint baseline
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md`
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`
- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md`
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md`
- `~/.claude/projects/-home-lukas-Trader/memory/project_benchmark_live_gap.md`
