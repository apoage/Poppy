# TASK_780 — Implementation Plan (phased build)

**Author**: implementing sub-agent (Phase-1 dispatch).
**Worktree branch**: `worktree-agent-a079d2f24a2cc3d53` (off `master` tip `79430e3`).
**Companion docs**:
- `tasks/TASK_780_ORACLE_AS_MCTS_DISTILLATION.md` (spec)
- `runs/780_oracle_as_mcts/AUTHORIZATION_2026-05-26.md` (operator authorization + build outline)
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` (original Feb-2026 design)

**Scope of THIS dispatch**: Phase 1 only — skeleton, helper math, config + CLI wiring. ~2 weeks of phased build follow in subsequent dispatches.

**Universal invariants enforced by every phase below**:
- Canon (`a1b1 + optional rs050`) training must remain **byte-identical** when new flags absent. The bridge trainer is a *parallel path* gated by a single boolean.
- Action space stays at **320** (5 × 64) per AUTHORIZATION §"Pre-implementation checklist" — LimitSell/LimitBuy visit-counts are zero-by-construction (`path_solver.py:406, 473` — oracle never emits actions 3/4).
- Checkpoint pickle keys preserved: `{'params', 'opt_state', 'epoch', 'metrics', 'config', 'n_params'}` (matches `easy_trainer.save_checkpoint` at `easy_trainer.py:434-445`), so `train_phase2.load_phase1_checkpoint` (`train_phase2.py:96-100`) works unchanged.
- New code default OFF. Sub-agent never calls `deploy.sh` and never pushes to remote.

---

## Phase 1 — Skeleton + helper (PHASE 1 EXECUTING NOW)

**Objective**: Land the bridge trainer file skeleton, the load-bearing `oracle_to_mcts_targets()` helper, and config/CLI plumbing — with `NotImplementedError` in the training-loop body. Default OFF; canon paths untouched.

**Files created**:
- `bridge_trainer.py` (top-level, alongside `easy_trainer.py`) — ~220 LOC target.
  - `class BridgeTrainer` skeleton with `__init__` that constructs `MuZeroNetworks(use_prediction_head=True, ...)`, initializes optimizer (same `optax.chain(clip + adamw)` pattern as `easy_trainer.py:97-100`), and exposes `save_checkpoint` / `load_checkpoint` matching the EasyTrainer pickle contract.
  - `_train_step_impl()` method as a JIT-ready signature (params, opt_state, batch) → (new_params, new_opt_state, metrics) — raises `NotImplementedError("bridge_trainer._train_step_impl: Phase 2")` in the body.
  - `train_step()` and `evaluate_batch()` thin wrappers also raising `NotImplementedError` with explicit "Phase 2" tag.
  - `create_bridge_trainer(config, seed)` factory mirroring `easy_trainer.create_trainer` at `easy_trainer.py:570`.
- `mcts_trader/jax_agent/oracle_distillation.py` (new sibling module) — ~180 LOC target.
  - `oracle_to_mcts_targets(all_policies, clean_pnl, clean_selections, ...)` — the load-bearing math. JIT-compatible. Output dict: `{policy_target, value_target, q_values, step_policies, step_values, step_actions, visit_counts}`. Phase-1 deliverable is the **fully implemented helper**, not a stub — it is small enough (~80 LOC body) to land cleanly today and unblocks Phase 2.
  - `multipass_oracle_with_passes(...)` — wrapper around `multipass_oracle_policy` that keeps the un-collapsed `(n_passes, T, n_actions)` tensor in addition to the canon return tuple. Default config-gated; canon callers see no change.

**Files modified**:
- `easy_config.py` (current ~202 LOC) — add ~30 LOC for new bridge fields. All defaults preserve canon. Fields:
  - `use_oracle_mcts_distill: bool = False` — master gate.
  - `oracle_distill_n_passes: int = 32` — N passes for the bridge target generator (TASK_697 §A.3 initial value).
  - `oracle_distill_pass_schedule: List[List[int]] = field(default_factory=lambda: [[0, 32], [50, 16], [150, 8]])` — epoch → n_passes anneal table (TASK_697 §A.3 verbatim).
  - `oracle_distill_temperature: float = 1.0` — visit-distribution sharpening temperature.
  - `oracle_distill_consistency_weight: float = 1.0` — multi-step latent consistency loss weight (TASK_697 §A.2).
  - `oracle_distill_reward_weight: float = 1.0` — per-step reward CE weight (TASK_697 §A.2).
  - `oracle_distill_unroll_steps: int = 5` — multi-step unroll depth (matches Stage-A default).
  - `oracle_distill_q_value_weight: float = 0.0` — optional Q-value aux loss weight (defaults OFF, lets Phase 2 add it if needed).
- `train_easy.py` (current ~2400 LOC) — add ~15 LOC near the existing argparse block (`:1740-1847`) and at the trainer-construction site (`:2117`).
  - New CLI flag: `--oracle-mcts-distill` (default False) → sets `config.use_oracle_mcts_distill = True`.
  - New CLI flags: `--oracle-distill-passes <int>`, `--oracle-distill-unroll <int>` (override-only; default None = use config value).
  - At the `create_trainer(config, seed=config.seed)` call site (`:2117`), branch on `config.use_oracle_mcts_distill`. The bridge branch instantiates `BridgeTrainer` via `create_bridge_trainer`; the canon branch continues to use `create_trainer`. The branch is one `if/else` block; the rest of the train loop is unchanged for Phase 1 (the bridge path raises `NotImplementedError` once the loop tries to call `trainer.train_step`).
- `mcts_trader/jax_agent/path_solver.py` — **NO MODIFICATIONS in Phase 1**. The existing `multipass_oracle_policy` is unchanged. The wrapper in `oracle_distillation.py` invokes it via `single_pass` re-implementation (re-uses `greedy_lookahead_path` + `optimal_path_to_policy` directly) so we get the un-collapsed `(n_passes, T, n_actions)` tensor without mutating the canon function's signature.
  - **Rationale**: `multipass_oracle_policy` at `:222-339` is touched by 11 call sites across `train_easy.py`/`easy_trainer.py`/benchmark; mutating its return tuple in Phase 1 is high-risk. Phase 2 may reconsider if profiling shows duplicate work.

**Risk**: LOW.
- The new module is parallel to existing code; the new config fields default OFF; the new CLI flag defaults OFF.
- The bridge_trainer body raises `NotImplementedError` — anything that tries to run it fails loudly at line 0 of the training step, never silently.
- `oracle_to_mcts_targets()` is a pure JAX function with deterministic shape derivation; AST-parse-time correctness can be verified before any compute happens.

**Rollback strategy**:
- `git checkout master -- bridge_trainer.py mcts_trader/jax_agent/oracle_distillation.py easy_config.py train_easy.py` (the new files vanish; the modified files revert).
- Or simply delete the worktree branch — the new files have no canon importers.

**Smoke gate (Phase 1)**:
- `python -c "import ast; ast.parse(open('bridge_trainer.py').read()); ast.parse(open('mcts_trader/jax_agent/oracle_distillation.py').read())"` — both AST-parse without `SyntaxError`.
- `python -c "from easy_config import EasyConfig; c = EasyConfig(); assert c.use_oracle_mcts_distill is False, 'default must be OFF'; assert c.action_space_size == 320; c.__post_init__()"` — config loads, defaults OFF, validators pass.
- `python -c "from easy_config import EasyConfig; c = EasyConfig(use_oracle_mcts_distill=True, oracle_distill_n_passes=16); c.__post_init__()"` — config accepts the new fields, validators pass.
- `python -c "from bridge_trainer import BridgeTrainer, create_bridge_trainer; print('import OK')"` — class importable; no compile error.
- `python -c "from mcts_trader.jax_agent.oracle_distillation import oracle_to_mcts_targets, multipass_oracle_with_passes; print('import OK')"` — helper module imports cleanly.
- `python -c "import jax, jax.numpy as jnp; from mcts_trader.jax_agent.oracle_distillation import oracle_to_mcts_targets; \
    n_passes, T, n_assets, n_actions_per_asset = 4, 8, 64, 5; \
    all_policies = jnp.ones((n_passes, T, n_assets * n_actions_per_asset)) / n_actions_per_asset; \
    clean_pnl = jnp.zeros((T,)); clean_sel = jnp.zeros((T, n_assets)); \
    out = oracle_to_mcts_targets(all_policies, clean_pnl, clean_sel, n_assets=64, n_actions_per_asset=5); \
    assert out['policy_target'].shape == (n_assets * n_actions_per_asset,); \
    assert out['value_target'].shape == (); \
    assert out['step_policies'].shape == (T, n_assets * n_actions_per_asset); \
    assert out['step_values'].shape == (T,); \
    assert out['step_actions'].shape == (T, n_assets); \
    assert out['visit_counts'].shape == (n_assets * n_actions_per_asset,); \
    print('oracle_to_mcts_targets shape smoke PASS')"` — the helper produces shapes per the TASK_697 §A.1 mapping.
- Canon-path control: `python train_easy.py --help 2>&1 | grep -q 'oracle-mcts-distill'` — the new flag is visible in the canon entry's help text without breaking existing flags.

**Dependencies**: None. Phase 1 is the entry point.

**Estimated wall time**: 1 dispatch (~today). Phase 1 is small enough to land in one sitting.

---

## Phase 2 — Phase-A imprint loop (TODO; next dispatch)

**Objective**: Implement the bridge trainer's actual training step + buffer-format change. The loss is MuZero-shaped per TASK_697 §A.2.

**Files modified**:
- `bridge_trainer.py` — fill in `_train_step_impl()` body:
  - Loss structure (per TASK_697 §A.2):
    ```
    total = policy_weight * CE(policy_logits, visit_distribution)
          + value_weight * hybrid_value(value_logits, root_value)
          + reward_weight * CE(reward_logits, oracle_step_reward)
          + consistency_weight * ‖encode(s_t) − dynamics(h_{t−1}, a_{t−1})‖²
    ```
  - Multi-step unroll loop modeled on `train_phase2.py:1613-1700` `train_stage_a` + `:1235-1545` joint-MCTS loss site.
  - JIT compilation via `jax.jit(self._train_step_impl, static_argnames=['unroll_steps'])`.
  - Class-balance: mirror TASK_149's `MAX_TRADE_WEIGHT = 64.0` from `easy_trainer.py:174-194` on the visit-distribution target to avoid sharpening the classifier (anti-correlation indicator per AUTHORIZATION §"Risk").
- `train_easy.py` — extend `OnTheFlyGenerator._generate_batch_episodes` (`:659-945`) with a config-gated branch at the buffer assembly site (`:924-935`). New buffer format per TASK_697 §"Buffer-format change":
  ```
  {observations, actions [trajectory (T, n_assets)],
   rewards [per-step (T,)],
   visit_distributions [(T, n_actions)],
   values [per-step root (T,)],
   dones [(T,) bool],
   step_actions [(T,) int — argmax per step for dynamics target]}
  ```
- Likely add `_generate_batch_episodes_bridge` as a sibling method to keep canon path 100% untouched.

**Files created**: none.

**Risk**: MEDIUM. The buffer-format change is the largest single delta. Class-balance correctness is non-trivial (the visit-distribution target has continuous mass on each (asset, action) slot, so the `hold_prob` extraction in `easy_trainer.py:175` needs to be re-derived per the TASK_697 §A.1 mapping rather than reused as-is).

**Rollback strategy**: revert `bridge_trainer.py` + the `train_easy.py` branch. Canon path was never modified.

**Smoke gate**:
- 5-epoch run on a single segment with `--oracle-mcts-distill --epochs 5 --batch-size 16 --periods-file config/benchmark_periods.json --start-date 2024-01-01 --end-date 2024-02-01` (or equivalent small slice).
- All loss components finite, no NaN.
- `total_loss` decreases monotonically across the 5 epochs (loose check: epoch-5 < epoch-1).
- The bridge checkpoint loads under `train_phase2.load_phase1_checkpoint` — `python -c "from train_phase2 import load_phase1_checkpoint; p, c, e, m = load_phase1_checkpoint('runs/000_smoke/780_bridge/phase1_best.pkl'); assert 'dynamics_dense1' in str(jax.tree_util.tree_leaves_with_path(p)), 'dynamics params must be in checkpoint'"`.

**Dependencies**: Phase 1 lands first.

**Estimated wall time**: 3-5 days (TASK_697 §"Implementation Estimate" estimate for `bridge_trainer.py` plus buffer change).

---

## Phase 3 — Phase-B freeze bootstrap (DEFERRED; later dispatch)

**Objective**: Verify Phase-2 Stage-B loads the bridge checkpoint cleanly and the dynamics block is trained (not reset).

**Files modified**:
- `train_phase2.py` — at `load_phase1_checkpoint` (`:96-100`) site, detect a "bridge-trained" Phase-1 ckpt via a flag in the config dict (e.g., `config.get('use_oracle_mcts_distill', False)`) and skip or shorten Stage-A's consistency-only warmup at `:1613-1700`. Operator decision at smoke time: full skip vs short fine-tune.

**Risk**: MEDIUM. The Stage-A skip mechanism touches a hot path. Wrap it in an `if bridge_trained:` branch with the canon path under the `else:`.

**Rollback strategy**: revert the `train_phase2.py` branch; canon Stage-A path was never modified.

**Smoke gate**:
- Phase-A `phase1_best.pkl` (from Phase 2 dispatch) loaded via `train_phase2.main()` smoke command.
- Stage-A consistency_loss starts < 0.01 (the dynamics is already warm from the bridge), OR Stage-A is skipped per operator config.
- No `ValueError` from `initialize_missing_aux_heads` (`:2698`).

**Dependencies**: Phase 2 ckpt available.

**Estimated wall time**: 1 day.

---

## Phase 4 — Phase-C unfreeze fine-tune (DEFERRED; later dispatch)

**Objective**: Gradual unfreeze + ExIt loop per TASK_697 §C. **Out of scope for the ship gate** — only triggered if Phase-B clears the +3pp paired Δα gate.

**Files modified**:
- `train_phase2.py` — extend the existing `optax.multi_transform` infrastructure (`:919-921, 2217-2253`) with a 4-stage unfreeze schedule (dynamics → value+policy → TRM → encoder). Discriminative LR (10× reduction per newly-unfrozen group).

**Risk**: LOW (infrastructure exists; the change is a config-driven schedule).

**Estimated wall time**: 1-2 days.

---

## Phase 5 — End-to-end smoke + Phase-A sanity gate (DEFERRED; later dispatch)

**Objective**: Per AUTHORIZATION §"Phase-A sanity gate": bridge ckpt's per-asset Bal.Acc within 3pp of canon imprint Bal.Acc, dynamics consistency loss < 0.01.

**Files modified**:
- Likely add `tools/benchmark_bridge_sanity.py` (~50 LOC) reusing `tools/benchmark_easy.py`'s inference path but computing per-asset argmax of `visit_distribution[asset, :]` over the 5 actions (matches `easy_trainer.py:287` `oracle_agreement` decomposition).
- Write `runs/780_oracle_as_mcts/canon_imprint_baseline.json` with the frozen canon baseline metrics per AUTHORIZATION §"Pre-implementation checklist" item 4.

**Risk**: LOW.

**Smoke gate**: per-asset Bal.Acc ≥ canon − 3pp on ≥ 2/3 seeds; dynamics consistency_loss < 0.01.

**Dependencies**: Phase 2 + Phase 3.

**Estimated wall time**: 2 days.

---

## Two-gate review protocol (per `lore/notes_2026-05-18.md` §138)

- **Gate 1** (pre-implementation, read-only sub-agent): runs *between Phase 1 and Phase 2*. Reviews this plan + the Phase 1 skeleton against the cited code sites. Returns ≤ 1 page of risks, computation sites, out-of-scope items, effort refinement.
- **Gate 2** (post-implementation smoke): runs *between Phase 2 and Phase 3*. Full-shape smoke at production batch (64), window (64), unroll (5), n_passes_max (32). Verifies all losses finite, multi-step unroll works without VRAM OOM, checkpoint loads under `train_phase2.load_phase1_checkpoint`.

The implementing agent does NOT proceed past Phase 2 without Gate 1 sign-off.

---

## Surprises and amendments to the spec (Phase-1 reads)

1. **`multipass_oracle_policy` already throws away the per-pass detail at line 324.** The AUTHORIZATION §"Framing" notes this; the operator's hint is to "keep the un-collapsed `(n_passes, T, n_assets * 5)` tensor at `path_solver.py:320`". **Plan refinement**: do NOT mutate `multipass_oracle_policy`'s return tuple — instead, write a parallel `multipass_oracle_with_passes()` wrapper in the new `oracle_distillation.py` module that re-implements the per-pass loop and exposes the un-collapsed tensor. This keeps the canon return shape unchanged across all 11 existing call sites and avoids a high-risk signature mutation. Phase 2 may revisit if profiling shows duplicate oracle work.

2. **The TASK_697 §A.1 pseudocode uses `at[idx].add(1)` in a Python `for` loop over `oracle_passes` — that is NOT JIT-compatible.** The Phase-1 helper instead derives visit counts via a JAX-friendly path:
   - `per_pass_actions = jnp.argmax(all_policies.reshape(n_passes, T, n_assets, n_actions_per_asset), axis=-1)` → `(n_passes, T, n_assets)` integer action ids per (pass, step, asset).
   - `visit_one_hot = jax.nn.one_hot(per_pass_actions, n_actions_per_asset)` → `(n_passes, T, n_assets, n_actions_per_asset)`.
   - `step_visit_counts = jnp.sum(visit_one_hot, axis=0)` → `(T, n_assets, n_actions_per_asset)` per-step counts.
   - `step_policies = step_visit_counts / n_passes` → per-step visit distributions (normalized).
   - `policy_target = jnp.mean(step_policies, axis=0).reshape(n_assets * n_actions_per_asset)` → root visit distribution (mean over T).
   - `visit_counts = (step_policies * n_passes).reshape(T, n_assets * n_actions_per_asset)` → integer-valued visit counts in float (for downstream Q-value aggregation).

3. **The TASK_697 §A.1 Q-value pseudocode is also JIT-incompatible.** Phase-1 helper derives Q-values via:
   - Each pass contributes `clean_pnl[T-1]` (the full-episode PnL) to every action it visited. Since all passes share the same `clean_pnl` (passes vary in *which assets they select* with noisy returns, but the clean path provides the value-target trajectory per `multipass_oracle_policy:270-279`), this collapses to: Q-value at (asset, action) = mean PnL across passes that selected this (asset, action) slot.
   - JAX form: `weighted_pnl_per_slot = jnp.einsum('p,ptak->tak', per_pass_total_pnls, visit_one_hot)`; `q_values = weighted_pnl_per_slot / jnp.maximum(step_visit_counts, 1)` (per-step Q). Root Q = mean over T.
   - **Spec amendment**: in Phase 1 the per-pass total PnL is **identical across passes** (the clean path is shared), so per-slot Q-values reduce to `total_pnl * visit_indicator`. This is the trivial degenerate case noted in TASK_697's "Open Question 1" — the multi-pass oracle does NOT generate different per-pass returns (it generates different *selections* under noise). To get meaningful Q-values we would need to compute the **per-pass actual realized PnL** along each pass's selected path. **Phase 1 implements the simple case** (root_value = `clean_pnl[T-1]` for all visited slots) and **flags this** for Phase 2 refinement — the Q-value head is gated `oracle_distill_q_value_weight: float = 0.0` in the default config to keep this surface non-load-bearing until the math is upgraded.

4. **The "step actions" trajectory target (TASK_697 §A.1 row 6) is straightforward**: `step_actions = jnp.argmax(step_policies, axis=-1)` → `(T, n_assets)` per-step argmax actions for the dynamics unroll loop. This matches the format that `recurrent_inference` expects as input (`muzero_model.py:701-728`).

5. **MuZeroNetworks must be constructed with `use_prediction_head=True`** in Phase 1 of the *real* bridge build (Phase A imprint). The Phase-1 *skeleton dispatch* (today) constructs the model with `use_prediction_head=True` in `BridgeTrainer.__init__` so the parameter tree carries the dynamics + decode-latent-to-market subtree from day one — keeping `train_phase2.load_phase1_checkpoint` happy without `initialize_missing_aux_heads` having to fabricate fresh dynamics weights. Verified at `muzero_model.py:701` (recurrent_inference) and `:734` (decode_latent_to_market) — both require `use_prediction_head=True` to instantiate the price predictor subtree.

6. **The "Q-value aux head" mentioned in the parent TASK_780 spec is OPTIONAL and gated OFF by default in Phase 1.** Reason: there is no existing `q_value_head` in `MuZeroNetworks`; adding one is a parameter-tree change that Phase-2 Stage-B would have to absorb (or `initialize_missing_aux_heads` would have to fabricate). Phase 1 keeps the head OFF; if Phase 2 finds the Q-value signal load-bearing, we add the head then. The Phase-1 `oracle_to_mcts_targets()` does compute and expose `q_values` for diagnostic/visualization but the bridge loss does not use them.

---

## Acceptance criteria summary

| Phase | Criterion | When verified |
|---|---|---|
| 1 | AST parse OK; config defaults OFF; helper shapes correct on smoke input; canon training byte-identical | This dispatch |
| 2 | 5-epoch smoke loss decreases monotonically; checkpoint loads under `train_phase2.load_phase1_checkpoint` | Next dispatch |
| 3 | Stage-A loads bridge ckpt; consistency_loss starts < 0.01 OR Stage-A skipped | Later dispatch |
| 4 | Gradual unfreeze runs without errors; discriminative LR applied per group | Later dispatch (only if Phase-B clears +3pp) |
| 5 | Phase-A sanity gate: per-asset Bal.Acc ≥ canon − 3pp on ≥ 2/3 seeds | Later dispatch |
