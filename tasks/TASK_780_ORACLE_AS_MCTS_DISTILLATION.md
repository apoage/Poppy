# TASK_780 — Oracle-as-MCTS Distillation (Phase-1 rewrite via bridge_trainer)

**Status**: DRAFT — revival of TASK_697 (parked Feb 2026 as "ON HOLD"). The Feb 2026 hold-rationale ("not compatible with current easy_trainer path") is unchanged; what changed in May 2026 is the campaign context (single-mechanism class CLOSED per `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md`) and the imprint economics (165 min/seed on server per TASK_768 implementation log + `lore/handoff_2026-05-24_764-launch.md`). Campaign budget that was previously locked to small-mechanism probes is now freed for the largest unexecuted design.

**Created**: 2026-05-25
**Ship-path candidate**: #6 of 10 per `lore/ship_path_recommendations_2026-05-25.md` ("largest unexecuted design in the project not already on the roadmap").
**Promotion gate (canon)**: Phase B promotion — paired Δα ≥ +3pp eval-MCTS sims=16 joint vs 760 a1b1, no seed < −5pp.
**Code-budget gate (pre-code)**: operator authorization for ~2-3 weeks code budget — this is the largest re-imprint and trainer rewrite in the project. Greenlight required before any implementation.

## Hypothesis

Foxglove Phase-1 (`easy_trainer.py`) trains the encoder/TRM/value head against **flat one-hot oracle action labels** (`easy_trainer.py:160-194` per-asset 5-class softmax CE on the policy head; targets are one-hot or soft-one-hot via TASK_673 multipass averaging) and **scalar oracle PnL** (`easy_trainer.py:207-213` categorical CE on a 401-bin discretization of the remaining oracle return). Poppy Phase-2 (`train_phase2.py`) then expects the *same network* to produce MCTS-shaped outputs: visit-count distributions over the joint action space, backed-up Q-values from search rollouts, and latent dynamics that compose under multi-step `recurrent_inference`.

There is a **representation gap**: Phase-1 shapes the latent space for classification (per-asset action argmax against a near-one-hot target) and a scalar regression (remaining PnL), neither of which exercises the multi-step latent transition manifold or the visit-distribution-aggregation surface that MCTS searches over. When Phase-2 Stage-A starts (`train_phase2.py:1613` `train_stage_a`), it has to relearn the dynamics from scratch against a frozen encoder whose representations were never selected to compose under unrolled `recurrent_inference`. Phase-2 Stage-B then has to convert per-asset argmax logits into joint visit-distribution policy targets without ever having seen one during pretraining.

**TASK_780 closes the gap**: train Phase-1 directly against MCTS-shaped targets — multi-pass oracle visit distributions over the joint action space (or per-asset, depending on the decoding-path choice), per-step policies along the trajectory, and Q-values derived from per-action mean returns. The dynamics block (currently `use_prediction_head=False` in Foxglove per `easy_trainer.py:52`) is trained inside the bridge trainer using the oracle trajectory's multi-step structure as the dynamics-target signal — replacing the consistency-only Stage-A warmup with an oracle-derived dynamics imprint that is target-rich from the first epoch.

**Mechanism + evidence**:
- Multi-pass oracle (`mcts_trader/jax_agent/path_solver.py:222` `multipass_oracle_policy`, `:345` `batched_multipass_oracle_policy`, both TASK_673) already produces N rollouts with noise perturbation per the algorithm in lines 236–264 — the exact analog of MCTS simulations. Per-pass selections are recorded; aggregated `all_policies` at `:320` is `(n_passes, T, 320)` (the per-step per-pass soft policy). Currently the trainer collapses this to a single soft policy at `:324` via `jnp.mean(all_policies, axis=0)`. TASK_780 keeps the per-pass structure to derive visit-count distributions and Q-values per the TASK_697 §A.1 mapping table.
- Mapping (from TASK_697 §A.1, verbatim verified):
  - N oracle passes ↔ N MCTS simulations
  - Per-pass asset selections ↔ visited actions per simulation
  - Aggregated selection frequency ↔ visit-count distribution
  - Per-pass returns ↔ backed-up leaf values
  - Mean return across passes ↔ root value estimate
  - Per-step oracle decisions ↔ dynamics transition targets
- `docs/easy_mode.md` §"Phase 2: The Dream" (lines 45–63) describes Phase 2 as the *dynamics warmup* with consistency loss `||predicted_next_latent − real_next_latent||²`. TASK_780 Phase A *replaces* this with oracle-derived multi-step dynamics targets: each oracle pass produces a (T, n_actions) trajectory; the bridge trainer trains `recurrent_inference(h_{t−1}, a_{t−1}) → (h_t, r_t)` against the encoder's latent at step t and the oracle's step PnL. Stage-A's consistency-only warmup becomes redundant if Phase A converges the dynamics block to a usable starting point.

**Anti-correlation hypothesis**: UNKNOWN by direct precedent. The closest priors on the encoder/representation surface are:
- 740_bella_distill (Δα **−1.92pp**) — encoder regression aux distilling Bella price predictions onto the latent. Closed negative.
- 757_bella_shape (Δα **−5.80pp**) — shape distillation. Closed negative.
- 738_legacy_27ch (Δα **−18.44pp**) — dropping cross-sectional channels. Catastrophic.

TASK_780 differs structurally from all three: it does **not** add a regression aux loss alongside the policy/value targets (740/757), nor does it remove signal (738). It **replaces the target shape** the encoder is trained against — enriching it with planning structure (visit-counts, Q-values, multi-step dynamics) while preserving every information surface the current targets carry (the per-action softmax is recoverable from the visit-count distribution; the scalar root value is the mean of per-pass values that currently produces it). The mechanism is target-shape enrichment, not auxiliary-head addition.

**The honest unknown**: because the campaign's anti-correlation pattern is "any mechanism that lowers Stage-B training loss + raises Bal.Acc collapses Alpha via trade-rate explosion" (`project_classifier_alpha_anti_correlation.md`), and Phase-A in TASK_780 introduces a richer policy target that *should* produce a smaller policy CE during Phase-1 (and possibly during Stage-B if the head transfers), there is non-zero risk of triggering the same pathway through an unmodeled coupling. The Phase-A sanity gate (below) is designed to detect this before paying the Phase-B cost.

## Verified code state (workstation tip `79430e3`)

### Multi-pass oracle infrastructure (verified present)

- `mcts_trader/jax_agent/path_solver.py:222` — `multipass_oracle_policy(returns, key, k=1, lookahead=20, ..., n_passes=10, noise_scale=0.1, n_actions_per_asset=5, ...)`. Algorithm at lines 281–316: each pass perturbs returns with noise (`single_pass`), solves a binary greedy path via `greedy_lookahead_path`, converts the binary path to a per-asset 5-class soft policy via `optimal_path_to_policy`. `jax.lax.map` over all passes at `:320` yields `all_policies` shape `(n_passes, T, n_assets * 5)`. **Current end-state collapses to mean at `:324`** — TASK_780 keeps the un-collapsed per-pass tensor for visit-count + Q-value derivation. Lines 326–339 apply optional post-averaging temperature sharpening (TASK_677) and return `(soft_policy, clean_path.cumulative_pnl, clean_path.total_alpha, clean_path.selections)` — the fourth return is the **clean** oracle's per-step selections (TASK_689), reusable as a per-step trajectory target.
- `mcts_trader/jax_agent/path_solver.py:345` — `batched_multipass_oracle_policy(returns_batch, key, ...)`. Double-vmap: outer over N episodes, inner over n_passes per episode (per docstring at lines 360–375). With N=8 episodes and n_passes=10, runs 80 oracle solves in one JIT call. Returns `(soft_policies, avg_pnls, avg_alphas, clean_selections)` shapes `(N, T, n_assets*5), (N, T), (N,), (N, T, n_assets)`. The bridge trainer's batch generator will call this and post-process to expose per-pass detail.

Both functions are JAX-compiled (`@partial(jax.jit, static_argnames=[...])` at `:342-344` for `batched_multipass_oracle_policy`; the inner `multipass_oracle_policy` runs under `jax.lax.map` for JIT-compatible per-pass loops at `:320`). The multi-pass infrastructure is **load-bearing for TASK_780 and verified at the cited line ranges**.

### Current Phase-1 imprint loop shape

- `easy_trainer.py:46-65` — `MuZeroNetworks` constructed with `use_prediction_head=False` (`:52`). **Dynamics is NOT trained in Foxglove Phase-1** — only `initial_inference` produces policy + value. TASK_780 flips this: Phase A constructs the same model with `use_prediction_head=True` and trains the dynamics block.
- `easy_trainer.py:131-302` — `_train_step_impl`. Loss structure:
  - Policy loss (`:165-205`): per-asset softmax CE between `policy_reshaped (B, 64, 5)` and `targets_reshaped (B, 64, 5)`. TASK_149 class-balancing (`:174-194`) gives trades up to 64× more weight than holds. TASK_713 return-weighting (`:196-203`) further multiplies trade gradients by `1 + min(|return|/mean, 10)`. Loss is computed via `optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)` — the target is a per-asset 5-class soft-one-hot distribution (averaged across multi-pass at the call site).
  - Value loss (`:207-213`): CE on a 401-bin categorical of `scaled_returns = batch['returns'] * config.value_scale`. Targets are one-hot bins (`_scalar_to_target` at `:118-129`). **`batch['returns']` is the *remaining oracle PnL from step t to episode end*** — not the per-step reward, not the root-value estimate of a search tree.
  - Oracle-mask loss (`:215-225`): TASK_695 binary BCE on a 64-d selection-set head. Default OFF; not relevant to TASK_780's mechanism but inherited as the planning-side aux. TASK_775 (in flight design) tests this in isolation.
  - Total loss (`:228-232`): `policy_weight * policy_loss + value_weight * value_loss + oracle_mask_weight * oracle_mask_loss`. **No dynamics-consistency term, no reward-prediction term, no multi-step unroll.**
- `easy_trainer.py:317-333` — `train_step`: runs the JIT-compiled `_train_step` once per batch.

The buffer format consumed by `_train_step_impl` is the dict assembled at `train_easy.py:924-935`: `{'observations', 'actions', 'returns', 'per_asset_returns', 'level', 'n_samples', ['oracle_mask']?, ['latent_states']?}`. `actions` is the per-asset 5-class soft policy from `policy_targets[i, start_idx:end_idx]` (i.e., the multi-pass-averaged soft policy, shape `(T, 320)`). `returns` is `remaining_pnl[i, start_idx:end_idx]` (`:927`). There is **no per-step reward**, **no visit-count distribution**, **no Q-values**, **no done flag**, **no action trajectory** in the current buffer — those are the buffer-format changes TASK_780 must introduce.

### Current Phase-1 entry

- `train_easy.py:1740` — `def main()`. Argparse at lines 1741–1845; `args = parser.parse_args()` at `:1847`.
- `train_easy.py:2117` — `trainer = create_trainer(config, seed=config.seed)`. Constructs `EasyTrainer` from `easy_trainer.py:570`.
- `train_easy.py:2202-2256` — the training loop body. `train_epoch(trainer, generator, config, epoch)` per epoch (`:2206`), then `validate(...)` every `eval_every` epochs (`:2223`). Best-checkpoint promotion at `:2273-2281` writes `phase1_best.pkl` to `output_dir`.
- `train_easy.py:2117` is the load site `EasyTrainer(config, rng_key)` — TASK_780's `bridge_trainer.py` is the replacement at this call site (or, depending on the operator's preferred path, an alternate entry `train_easy.py --bridge` that branches on a flag).

### Phase-2 expected interface (the load site)

- `train_phase2.py:96-100` — `load_phase1_checkpoint(ckpt_path)`. Returns `(params, config_dict, epoch, metrics)` from the pickled checkpoint. The pickle format (`easy_trainer.py:434-445`) is `{'params', 'opt_state', 'epoch', 'metrics', 'config', 'n_params'}`. **This is the contract TASK_780 must preserve**: the bridge trainer produces a pickle with these top-level keys (`params` is the `MuZeroNetworks` parameter tree, which must match the Phase-2 model factory's expected shape — including the dynamics block).
- `train_phase2.py:2261-2267` — `main()` opens, loads the Phase-1 checkpoint at `:2267`. The `config_dict` is round-tripped through `get_phase2_config(p1_config_path)` at `:2275` (when a sibling `config.json` exists) — so the bridge trainer must also write a sibling `config.json` with the standard `EasyConfig`-derived schema, including all flags Phase-2 reads (`oracle_k`, `use_oracle_mask_head`, encoder/embed dims, etc.).
- `train_phase2.py:2650` — `initialize_missing_aux_heads(...)` runs at load time, so a Phase-1 checkpoint that has *extra* heads (e.g. an MCTS-target-derived value head distinct from the canonical PopArt categorical head — TBD whether TASK_780 introduces such a head) is non-problematic, but a checkpoint that *lacks* a head Phase-2 needs is auto-filled. This makes the bridge trainer's surface tolerant in both directions.

**Critical interface invariant**: if TASK_780 Phase A trains dynamics, the dynamics parameters in the resulting checkpoint **must use the same parameter tree structure** as `MuZeroNetworks` with `use_prediction_head=True`. Otherwise Phase-2 Stage-A would treat them as fresh-initialized at load time. The bridge trainer should construct the model with `use_prediction_head=True` from the start; the Phase-A pickle then carries the trained dynamics directly and Phase-2 Stage-A becomes a fine-tune (or skip) instead of a from-scratch warmup.

## Code-change scope (~2 weeks total per TASK_697 §"Implementation Estimate")

Estimates are from TASK_697 §"Implementation Estimate" (lines 232–242); cross-checked against current source.

| Component | Effort | Notes |
|---|---:|---|
| `oracle_to_mcts_targets()` helper | 1 day | Core mapping function per TASK_697 §A.1. Input: per-pass selections + actions + returns (i.e., do not collapse `all_policies` to mean in `multipass_oracle_policy`). Output: `{'policy_target': visit-count distribution, 'value_target': root_value, 'q_values': per-action mean return, 'step_policies': (T, n_actions) per-step visit distributions, 'step_values': (T,) per-step value targets, 'step_actions': (T,) trajectory for dynamics targets}`. JIT-compatible. Lives next to the multi-pass oracle in `mcts_trader/jax_agent/path_solver.py` OR in a new `mcts_trader/jax_agent/oracle_to_mcts.py` (operator choice; bridge trainer imports the helper). |
| New `bridge_trainer.py` (top-level, alongside `easy_trainer.py`) | 3-5 days | Replaces or supplements `EasyTrainer`. Uses Foxglove's data pipeline (`OnTheFlyGenerator` in `train_easy.py:376` plus the multi-pass oracle helper) but Poppy's loss function and target structure. Loss: `policy_weight * CE(policy_logits, visit_distribution) + value_weight * hybrid_value_loss(value_logits, root_value) + reward_weight * CE(reward_logits, oracle_step_reward) + consistency_weight * ‖encode(s_t) − dynamics(h_{t−1}, a_{t−1})‖²` (per TASK_697 §A.2). Multi-step unroll loop modeled on `train_phase2.py:1613-1700` Stage-A `train_stage_a` and the joint-MCTS loss at `:1235-1545`. |
| Dynamics training inside the bridge | 2 days | Multi-step `recurrent_inference` unroll against the oracle trajectory. Replaces Stage-A's consistency-only warmup. The oracle's per-step PnL becomes the reward target; the encoder's latent at step t becomes the dynamics-output target at step t (consistency loss). Smoke-gate: `consistency_loss < 0.01` matches the Stage-A success criterion at `train_phase2.py:1679-1680`. |
| Pass count schedule (epoch-conditional) | 0.5 day | Per TASK_697 §A.3: `n_passes` anneals from 32 → 16 → 8 across epochs. Config field `bridge_pass_schedule: list[tuple[int, int]] = [(0, 32), (50, 16), (150, 8)]`. Annealing happens in the generator before calling `batched_multipass_oracle_policy`. |
| Buffer-format change | (included in `bridge_trainer.py` budget above) | From `(observations, actions [as soft policy], returns [as scalar remaining PnL])` to `(observations, actions [trajectory], rewards [per-step], visit_distributions, values [per-step root], dones)`. Touches `OnTheFlyGenerator._generate_batch_episodes` (`train_easy.py:659-945`) at the `ep_data` assembly site (`:924-935`) — likely a new `_generate_batch_episodes_bridge` method or a config-gated branch. **Do not break the EasyTrainer's existing buffer consumption** during the transition; bridge_trainer.py is a parallel path. |
| Action-space alignment decision | (operator decision; not coding effort) | Per TASK_697 §A.4: keep 320 (5 actions × 64 assets, current Foxglove + Poppy alignment via the LimitOrder mask in TASK_767) **OR** drop to 192 (3 actions × 64 assets, no Limit*). The current campaign has TASK_767 production-deployed as the Limit-order mask at Phase-2 (`runs/767_action_space/AUTHORIZATION_2026-05-23.md` per memory) — the oracle never emits LimitSell/LimitBuy (`path_solver.py:406, 473`), so the visit-count over those slots is identically zero in the multipass output. **Recommendation**: keep 320 throughout; the LimitSell/LimitBuy visit-counts are zero-by-construction, which exactly matches the TASK_767 mask. Reusing the existing model surface minimizes downstream integration cost. Operator confirms in the SPEC. |
| Phase A imprint loop + checkpoint emission | 1 day | Per-epoch loop, save `phase1_best.pkl` in the same pickle format as `easy_trainer.save_checkpoint` (`easy_trainer.py:428-447`) so Phase-2 `load_phase1_checkpoint` (`train_phase2.py:96-100`) can load it without modification. Sibling `config.json` with all `EasyConfig` fields populated (so `get_phase2_config` path works). |
| Phase B freeze + Phase C unfreeze pipeline | 1 day | Phase B per TASK_697 §B: freeze encoder/TRM/value/policy heads, train dynamics only against real MCTS rollouts (Phase-2 Stage-B with `factored_search=False`, joint MCTS at `mcts_sims=32`; consistency loss against real next-latent). Phase C per §C: gradual unfreeze schedule (dynamics → value+policy → TRM → encoder), discriminative LR (10× reduction per newly-unfrozen group). The freeze mechanism reuses the existing `optax.multi_transform` infrastructure at `train_phase2.py:919-921, 2217-2253`. |
| Smoke + Gate-1 + Gate-2 validation | 2 days | Per the two-gate review protocol (`lore/notes_2026-05-18.md` §138). Includes Phase-A sanity gate (below) before Phase-B paired 3-seed. |

**Total**: ~2 weeks code (TASK_697 §"Implementation Estimate" estimate, line 242). Workstation owns the implementation; server runs the experiment.

### Action-space alignment decision

Per TASK_697 §A.4 the choice is 320 vs 192. Operator pick at SPEC time. Default-recommended: **keep 320 throughout** (Option C in TASK_697). Rationale:
- Foxglove's current `n_actions_per_asset = 5` is the default everywhere in `easy_trainer.py:163`, `path_solver.py:232`, `train_easy.py:749, 831, 852, 900`. Switching to 3 requires rewiring all of those.
- The oracle never emits actions 3/4 (LimitSell/LimitBuy) — confirmed at `path_solver.py:406, 473` per `ship_path_recommendations_2026-05-25.md` §5. The visit-count distribution from `oracle_to_mcts_targets()` will have zero mass on slots 3 and 4 by construction.
- TASK_767 (Limit-order mask, production-deployed) already prunes search expansion to actions 0/1/2 at Phase-2 (`train_phase2.py:1060-1081, 1742-1754` per memory). The 320-slot policy head with hard-zero priors on 3/4 is operationally equivalent to a 192-slot head, **without** a model-architecture change.
- Decision deferred to the SPEC; default is 320.

### Two-gate review protocol

Per `lore/notes_2026-05-18.md` §138, the protocol that successfully landed P2/P4/Stage-A flow_matching/etc. with clean smokes:

- **Gate 1 (code review, pre-implementation)**: a read-only sub-agent reads the SPEC + the cited code sites + the proposed `bridge_trainer.py` design, returns ≤ 1 page of:
  1. The proposed code-change locations + estimated line counts.
  2. Two hard risks (similar to the 768 deep-dive's float32 + channel-order risks).
  3. Two computation sites (similar to the 768 deep-dive's "temporal AND tiled both need the update").
  4. Out-of-scope items.
  5. Effort estimate refinement.
- **Gate 2 (smoke discipline, post-implementation)**: per the 749 OOM rule, the smoke test must exercise the full bridge_trainer step at production batch and temporal window. Verify:
  - Phase-A `total_loss` is finite, no NaN in `policy_loss / value_loss / consistency_loss / reward_loss / grad_norm`.
  - `oracle_to_mcts_targets()` returns the expected shapes given a 10-pass oracle output.
  - Multi-step unroll reaches `unroll_steps=5` (Stage-A's default) without VRAM OOM at the production `batch_size=64`.
  - Phase-A `phase1_best.pkl` loads under `train_phase2.py:2267` `load_phase1_checkpoint` without raising — i.e. the parameter-tree structure matches `MuZeroNetworks(use_prediction_head=True, ...)`.

Sub-agent (Gate 1 reviewer) writes verdict; operator approves in-step mitigations; implementing agent proceeds to clean implementation. Time cost: ~25% of total wall, per §138's empirical finding.

## Variants (Phase A imprint × Phase B paired Stage-B)

### Phase A — 3-cell minimal (default)

| cell | n_passes_schedule | action_space | seed | wall (server, est.) |
|---|---|---:|---:|---:|
| 780A-default_s7  | (0:32, 50:16, 150:8) | 320 | 7  | ~165 min |
| 780A-default_s13 | (0:32, 50:16, 150:8) | 320 | 13 | ~165 min |
| 780A-default_s23 | (0:32, 50:16, 150:8) | 320 | 23 | ~165 min |

Wall: 3 × 165 min ≈ **8.25 h serial, ~5 h at 2-wide** on server.

### Phase B — paired 3-seed Stage-B retrain at Phase A best ckpt

After Phase-A sanity gate passes, retrain Phase-2 Stage-B from each Phase-A checkpoint using the canon a1b1 config (`runs/762_reward_shaping/rs050_seed_*/stage_b_config.json` for baseline reference). The bridge-trained dynamics block is loaded as-is (Stage-A becomes either a short fine-tune or is skipped — operator decision at smoke time).

| cell | base ckpt | seed | wall (server, est.) |
|---|---|---:|---:|
| 780B-best_s7  | 780A-default_s7  | 7  | ~2.5 h |
| 780B-best_s13 | 780A-default_s13 | 13 | ~2.5 h |
| 780B-best_s23 | 780A-default_s23 | 23 | ~2.5 h |

Wall: 3 × 2.5 h ≈ **7.5 h serial, ~4-5 h at 2-wide**.

### Phase C (out of scope for ship gate)

Gradual unfreeze + discriminative LR per TASK_697 §C.1. Only triggered if Phase B clears the +3pp gate at 3-seed and the operator wants the ExIt loop closed. Estimated 1-2 days additional wall, no new code (reuses the freeze/unfreeze infrastructure).

## Phase-A sanity gate (early-exit, mirrors TASK_775 pattern)

Two metrics, both required to advance to Phase-B:

1. **Per-asset Bal.Acc on benchmark ≥ canon imprint − 3pp at matched epoch**. The canon reference is the current `jax_ckpt_easy_phase1_xs32/easy_medium/` imprint's final-epoch balanced agreement (the trainable `oracle_agreement` metric reported at `easy_trainer.py:287`, which decomposes into `agree_trade + agree_hold`). The bridge-trained Phase-A model is evaluated on the same validation split (segment-pool's held-out cursor positions) at the same epoch budget. **The argmax of `visit_distribution[asset, :]` over the 5 actions per asset is the predicted action** — comparable to the current per-asset argmax. A regression > 3pp means the visit-distribution targets are degrading the encoder's classification surface, which is the leading anti-correlation indicator. **If FAIL → diagnose loss shape before paying Phase-B cost.**
2. **Dynamics consistency loss converged** (per TASK_697 §B.3). At Phase A end, the consistency loss `||encode(s_t) − dynamics(h_{t−1}, a_{t−1})||²` should be below 0.01 (the Stage-A success criterion at `train_phase2.py:1679-1680`). If above, the dynamics block is not learning multi-step composition from the oracle trajectory, which means Phase B Stage-A from this checkpoint will be no faster than from a vanilla Foxglove checkpoint — the whole point of bridging Phase 1 → 2 fails. Operator decision: either extend Phase-A training, or fall back to running Phase-2 Stage-A as canon and treating Phase-A as a policy/value-only imprint.

If both gates pass on ≥ 2/3 seeds, proceed to Phase-B. Otherwise diagnose.

## Phase-B promotion gate (canon)

- Paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint.
- No seed < −5pp.
- Eval command per the canon contract: `python tools/benchmark_easy.py --checkpoint <p2_ckpt> --periods-file config/benchmark_periods.json --eval-mcts-sims 16 --eval-mcts-decode joint --output runs/780_oracle_as_mcts/eval_p2_<cell>/benchmark.json`.

5-seed extension (seeds 31, 42) only if 3-seed clears.

## Cost estimate

- **Code (workstation, pre-launch)**: ~2 weeks per TASK_697 §"Implementation Estimate" line 242. Workstation owns the implementation; verify estimates against the Gate 1 review.
- **Phase A imprint** (server): 3 seeds × ~165 min = ~8-9 h serial, ~5 h wall at 2-wide on GPU 1+2.
- **Phase B paired Stage-B retrain** (server): 3 seeds × 2.5 h = 7.5 h serial, ~4-5 h wall at 2-wide.
- **Phase C** (out of scope for ship gate; only if B clears): 1-2 days additional wall.
- **Total time to first measurable Phase-B result**: ~2.5-3 weeks from code-budget greenlight (2 weeks code + ~3-4 days smoke + ~10 h Phase A wall + ~5 h Phase B wall).

The campaign budget for this candidate is freed by: (a) single-mechanism class CLOSED (`project_single_mechanism_exhausted.md`) — no more small-perturbation probes can ship; (b) TASK_768 Option A landed bit-identical without re-imprint, so the channel-restructure unblock prerequisite is satisfied; (c) Phase-1 re-imprints are now economical at 165 min/seed (TASK_764 k8_s7 actual per `lore/handoff_2026-05-24_764-launch.md` and `tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log").

## Risk + anti-correlation check

**Anti-correlation hypothesis**: UNKNOWN. See §"Hypothesis" above for the structural argument. The mechanism replaces the target shape (visit-distribution instead of soft-one-hot) rather than adding an auxiliary aux head (740/757) or removing signal (738). The honest unknown is that a richer policy target *should* produce a lower policy CE during Phase-1, and if that representation transfers to Stage-B as a lower BC loss, the anti-correlation pattern is triggered exactly as in 737/738/741.

**Mitigations**:
- Phase-A sanity gate #1 (per-asset Bal.Acc ≥ canon − 3pp) is the leading indicator. If the visit-distribution target is producing a *much sharper* classifier than canon during Phase-1, the bull-blindness pathway (`project_bull_blindness.md`) is reactivated even before Phase-B. Sanity-gate failure exits the experiment cheaply.
- During Phase-B Stage-B, read the per-epoch `policy_entropy` and `dir_acc` metrics from the CSV. If the bridge-trained Phase-B run shows `Δ-Bal.Acc > 0 AND Δα < 0` at 3-seed mean vs 760 a1b1, the mechanism did trigger anti-correlation through an unmodeled coupling. The result is then a **closure** (negative result with a strong structural signal: the visit-distribution target as a class is closed on this stack).

**Worst case**: TASK_780 produces a Phase-1 imprint that sharpens the classifier and collapses Alpha. The result is informative — it forecloses the "redesign the Phase-1 target shape" mechanism class, leaving only the runtime structural pivots (Phase 3 ExIt closed loop, self-play dojo, NeuralOGCM physics core) per `lore/ship_path_recommendations_2026-05-25.md` §"No-rewrite verdict".

**Best case**: the encoder learns representations that compose multi-step under dynamics (a property the current Foxglove imprint does not have), the Phase-B Stage-B retrain ships Δα ≥ +3pp, and the bridge trainer becomes canon for all subsequent Phase-1 imprints. The campaign's first promoted mechanism. The largest unexecuted design in the project converts to the first deployed structural rewrite.

**Special risk**: largest re-imprint in the project. **Breaks comparability with all 738-769 paired-Δα numbers** — every subsequent paired Δα from a bridge-trained imprint is vs a *new* canon. The campaign's accumulated comparator data is retired. Operator decision: accept the campaign reset as the cost of the structural pivot.

## Phase 3 follow-on (out of scope for ship gate)

Phase C (TASK_697 §C) is *standard ExIt*: MCTS generates improved policy targets, network trains on MCTS targets, better network → better priors → better targets → loop. The difference from canon is the starting point — the bridge-trained TRM starts the ExIt loop with representations *already shaped for search*, instead of being shaped for classification + scalar regression.

This is the path `docs/easy_mode.md` §"Phase 3: The Thought" originally specified (lines 65–127), particularly the autodidact mechanism via the oracle-mask head (lines 84–127). TASK_780 Phase A + TASK_775 (oracle-mask head) together form the prerequisite for Phase 3 to exist on this stack. If both ship, the project's original 3-phase design is finally complete.

## Why this is here

The largest unexecuted design in the project that is *not* already a roadmap entry. Per `lore/ship_path_recommendations_2026-05-25.md` §6 (lines 188–224):
- TASK_768 (channel restructure) is now done as Option A bit-identical (`tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log" 2026-05-23).
- TASK_697 has been "ON HOLD" since February 2026 (`tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md:6`) and was the most carefully designed Phase-1 rewrite.
- The campaign budget previously locked to small-mechanism probes is freed by the single-mechanism class closure (2026-05-20).
- The 165-min server imprint baseline (TASK_768 implementation log + TASK_764 actuals) makes Phase-1 retrains economical at scale — Phase-A's 3-seed × 165 min wall is comparable to a single canon Stage-B paired run.

The Feb 2026 "ON HOLD" rationale was *not* a closure — it was a deferral pending TASK_696 fixes (now landed) and pending Foxglove baseline recovery (now matured into the 760 a1b1 canon). Both prerequisites are satisfied.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §6 (primary spec source — lines 188-224).
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` (original Feb-2026 design; this TASK is its revival).
- `tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log" — 165-min server imprint baseline + Option A bit-identical channel-restructure landing.
- `tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md` — companion Phase-1 imprint design (oracle-mask head); pairs with TASK_780 for the Phase 3 prerequisite.
- `docs/easy_mode.md` §"Phase 2: The Dream" (lines 45-63) — the dynamics warmup that Phase A replaces.
- `docs/easy_mode.md` §"Phase 3: The Thought" (lines 65-127) — the autodidact / ExIt loop that Phase C completes.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — campaign-class-closure verdict that freed this candidate's budget.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — anti-correlation pattern + the planning-side pivot framing.
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — closest prior on encoder-target redesign (740/757/758 closed); structural distinction noted in §"Hypothesis" above.
- `lore/notes_2026-05-18.md` §138 — two-gate review protocol.
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server imprint actual for TASK_764 k8_s7.
- `mcts_trader/jax_agent/path_solver.py:222-339` — `multipass_oracle_policy` (single-episode multi-pass + soft-policy mean + clean-path return).
- `mcts_trader/jax_agent/path_solver.py:342-392` — `batched_multipass_oracle_policy` (vmapped multi-episode multi-pass).
- `easy_trainer.py:46-65, 131-302, 317-333, 428-447` — current Phase-1 trainer (loss shape, train_step, checkpoint format).
- `train_easy.py:376, 659-945, 1740, 1847, 2117, 2202-2256` — current Phase-1 entry (`OnTheFlyGenerator` class start, batch-episode assembly, `main()`, args parse, trainer construction, training loop).
- `train_phase2.py:96-100, 2261-2267, 2650` — current Phase-2 load contract (`load_phase1_checkpoint`, `main()` entry, `initialize_missing_aux_heads`).
- `train_phase2.py:277-359, 1613-1700` — current Stage-A dynamics warmup (`create_dynamics_optimizer`, `train_stage_a`); the loop TASK_780 Phase A replaces.

## Implementation log

(reserved — populate at execution time, mirroring `tasks/TASK_768_BASE_MODEL_REFRESH.md` §"Implementation log" 2026-05-23 / §"Implementation deep-dive (2026-05-22)")
