# Codebase Modularity Audit — 2026-05-23

Read-only audit by Explore sub-agent (`ae619ab6…`). Goal: map files > 1k–2k LOC, propose splits to improve coding-agent effectiveness. NOT an architectural redesign — splits must preserve current behavior bit-identically.

## Scope notes

- `mcts_trader/agent/` and `mcts_trader/encoder/` are tagged LEGACY/DEPRECATED in `mcts_trader/CLAUDE.md`; excluded.
- `tools/orchestration/dojo_v2.py`, `dojo_manager.py`, `dojo_dynamic_curriculum.py`, `train_manager.py` are tagged "Deprecated — DO NOT EDIT" in `tools/orchestration/CLAUDE.md`; excluded.
- `mcts_trader/env/vectorbt_env.py` is tagged legacy in `mcts_trader/CLAUDE.md`. **Kept in inventory only** (no split proposed — legacy code shouldn't be churned).
- `.git-training-worktree/` is a duplicate of root files (worktree); excluded.

---

## Section 1 — File inventory (≥ 500 LOC, live code only)

| File | LOC | Top-level symbols (classes/functions) | Status |
|---|---:|---|---|
| `train_easy.py` | 3541 | `build_state_channels`, `corrupt_oracle_selections`, `compute_target_position`, `_build_model_state_trajectory`, `_build_model_state_trajectory_batched`, `SegmentPool` (924 LOC L453-1376), `OnTheFlyGenerator` (1255 LOC L1377-2632), `cleanup_tiered_checkpoints`, `CSVLogger`, `train_epoch`, `validate`, `_launch_benchmark`, `_check_benchmark_passed`, `main` | 🔴 |
| `train_phase2.py` | 3140 | `scalar_to_categorical`, `load_phase1_checkpoint`, `analyze_params`, `warm_policy_direction_refiner`, `initialize_missing_aux_heads`, `initialize_dynamics`, `create_dynamics_optimizer`, `make_dynamics_train_step`, `create_model`, `load_segment_pool`, `AsyncBatchPipeline` (L502-876), `make_stage_b_labels`, `create_stage_b_optimizer`, `make_stage_b_train_step` (L1021-1558 = 538 LOC monolithic JIT closure), `save_checkpoint`, `train_stage_a`, `train_stage_b` (L1670-2194 = 525 LOC), `create_stage_b_optimizer_with_encoder`, `parse_args` (L2233-2564 = 332 LOC), `main` (L2567-3140 = 573 LOC) | 🔴 |
| `tools/orchestration/e-dojo_v1.py` | 2199 | `cleanup_merged_checkpoints`, `DojoConfig`, `DojoState`, `HistoricalScheduleManager`, `EasyDojo` (L396-1219 = 823 LOC), `TrainingLauncher` (L1224-1909 = 685 LOC), `main` | 🔴 |
| `mcts_trader/jax_agent/trainer.py` | 1932 | `compute_large_margin_loss`, `compute_q_filtered_bc_loss`, `_compute_per_head_grad_norms`, `_clip_gradients_per_head`, `TrainState`, `ExpertBCConfig`, `PredictionLossConfig`, schedule helpers, `MuZeroTrainer` (L361-1932 = 1571 LOC; `_loss_fn` alone is L1065-1560 = 495 LOC, `train` L1614-1789, ckpt L1790-1914) | 🔴 |
| `mcts_trader/env/vectorbt_env.py` | 1698 | `LimitOrder`, `Position`, `PortfolioState`, `VectorBTTradingEnv` (L148-1623 = 1475 LOC monolithic gym env), `validate_against_vectorbt` | 🟡 (legacy — no-op) |
| `mcts_trader/env/jax_trading_env.py` | 1672 | `EnvConfig`, `TradingState`, helpers, TP/SL, margin call, limit orders (process/fill), execute_{hold,buy,sell,limit_buy,limit_sell}, `reset`, `step`, observation builders (27ch/31ch/router), action mask, batched/jit wrappers, info getters | 🟡 |
| `train_jax_fast.py` | 1620 | `apply_runtime_price_noise`, `main` (L110-1619 ≈ 1500-line entrypoint) | 🟡 |
| `tools/orchestration/dojo_v4_full_curriculum.py` | 1389 | `DojoV4` (L266-1316 = 1050 LOC), `main` | 🟡 |
| `mcts_trader/env/jax_encoder.py` | 1331 | `EncoderConfig`, `_compute_per_asset_channels` (tiled), `_compute_cross_sectional`, `_compute_all_channels`, temporal variants (`_compute_per_asset_channels_temporal`, `_compute_cross_sectional_temporal`, `_compute_all_channels_temporal`, `encode_observation_temporal`), 5 jit/batch/vmap wrappers, state-channel encoder, simple encoder, compressed-encoder block (L1076-end), `validate_encoder_output`, tile helpers | 🟡 |
| `tools/orchestration/dojo_v3_historical.py` | 1103 | `DojoV3Historical` (L192-1044 = 853 LOC), `main` | 🟡 (reference-only per CLAUDE.md — exclude) |
| `tools/orchestration/dual_dojo.py` | 1096 | `cleanup_merged_checkpoints`, `DualDojoConfig`, `DualDojoState`, `DualDojoLauncher` (L236-511 = 276 LOC), `DualDojo` (L516-965 = 450 LOC), `main` | 🟡 |
| `mcts_trader/jax_agent/muzero_model.py` | 1065 | `DirectionHead`, `RankHead`, `AuxPriceHead`, `AuxBellaHead`, `AuxBellaShapeHead`, `PerAssetPolicyHead`, `PolicyDirectionRefiner`, `MuZeroNetworks` (L243-1065 = 822 LOC; setup, representation, prediction, oracle_mask_prediction, dynamics, initial_inference, recurrent_inference, decode/predict helpers) | 🟡 |
| `mcts_trader/data/synthetic_market.py` | 1039 | `SyntheticMarket` — single class with 15 level generators `_level_0_*` … `_level_14_*` + noise helpers + GBM helper + OHLC helper | 🟡 |
| `diff_teacher/data_loader.py` | 1027 | `MarketWindow`, `MarketDataLoader` (L31-440 = 410 LOC), `RainbowTableLoader` (L442-1027 = 586 LOC) | 🟡 |
| `mcts_trader/training/self_play.py` | 929 | `denormalize_value`, `run_self_play` (L43-330 = 288 LOC), `run_self_play_batched` (L331-561 = 231 LOC), `run_self_play_jax` (L562-879 = 318 LOC), `reconstruct_observations` | 🟢/🟡 borderline |
| `mcts_trader/jax_agent/expert_buffer.py` | 894 | `UnifiedIndexBuffer` (L43-868 = 826 LOC), `create_expert_buffer` | 🟢/🟡 borderline |
| `diff_teacher/train.py` | 785 | `parse_args`, `create_config`, `train_encoder` (L274-392), `train_generator` (L393-521), `train_generator_direct` (L522-642), `main` | 🟢 |
| `mcts_trader/jax_agent/prediction/tree_search.py` | 760 | `TreeSearchConfig`, `TreeSearchResult`, `create_physics_aware_tree_search_fn`, `StructuralScenarioConfig`, `create_structural_scenario_search_fn`, `analyze_tree_results`, `create_risk_penalty_fn`, `create_mcts_reward_adjuster`, `create_batched_tree_search_fn`, `create_tree_search_fn`, `create_dynamic_tree_search_fn`, `create_dynamic_reward_adjuster` | 🟢 |
| `tools/orchestration/dojo_v2.py` | 720 | DEPRECATED — exclude | — |
| `mcts_trader/training/setup.py` | 704 | adversarial helpers, `setup_jax_environment`, `setup_environment` (L308-426), `setup_model_and_trainer` (L427-557), `setup_bella_generator`, `generate_bella_prices_on_device` | 🟢 |
| `diff_teacher/test_all.py` | 675 | 8 `test_*` functions + `main`. Stand-alone test script | 🟢 |
| `tools/orchestration/dojo_manager.py` | 670 | DEPRECATED — exclude | — |
| `mcts_trader/jax_agent/fast_generator.py` | 656 | 5 nn.Modules + `create_fast_generator`, `init_fast_generator`, `generate_adversarial_batch`, `generate_adversarial_prices`, `sample_regimes`, `load_trained_generator`, `load_or_init_generator` | 🟢 |
| `diff_teacher/teacher.py` | 637 | `GeneratedSequence`, `DiffusionTeacher` (50–637 = 587 LOC) | 🟢 |
| `mcts_trader/data/history.py` | 630 | `HistoricalMarket` (single cohesive class) | 🟢 |
| `diff_teacher/flow_matching.py` | 594 | 4 nn.Modules + `FlowMatchingGenerator`, `flow_matching_loss`, `create_train_step`, regime helpers | 🟢 |
| `diff_teacher/conv_flow_matching.py` | 593 | 4 nn.Modules + `ConvFlowMatchingGenerator`, `flow_matching_loss`, `create_conv_generator`, helpers | 🟢 |
| `diff_teacher/adversarial_training.py` | 583 | `AdversarialConfig`, `AdversarialMetrics`, `compute_realism_loss`, `compute_diversity_loss`, `AdversarialTrainer`, `train_adversarial_cli` | 🟢 |
| `diff_teacher/curriculum_bridge.py` | 582 | `CurriculumConfig`, `AdversarialDataGenerator`, `AdversarialDataLoader`, `CurriculumMonitor`, `run_curriculum_loop`, `curriculum_bridge_cli` | 🟢 |
| `diff_teacher/encoder.py` | 582 | 5 nn.Modules + `MarketVAE`, `vae_loss`, factories | 🟢 |
| `easy_trainer.py` | 573 | `EasyTrainer` (29-569 = 541 LOC) | 🟢 |
| `diff_teacher/agent_evaluator.py` | 560 | `EvaluationResult`, `SimpleStateEncoder`, `AgentEvaluator`, `test_agent_evaluator` | 🟢 |
| `mcts_trader/jax_agent/titans_memory.py` | 556 | (titans memory module) | 🟢 |

Files between 500 and ~700 LOC not listed above are 🟢 and stable — no split warranted.

---

## Section 2 — Per-file split proposals (🔴 + 🟡 only, excluding legacy/deprecated)

### 1. `train_easy.py` — 3541 LOC — 🔴

This file mixes (a) a 924-LOC `SegmentPool` historical-data manager, (b) a 1255-LOC `OnTheFlyGenerator` background prefetch pipeline, (c) a JIT-decorated trajectory builder, (d) plain training/validate loops, (e) a CSV logger, (f) a benchmark launcher, and (g) `main`. Three of these can move cleanly without touching numerics.

**Split 1.1** — New file `mcts_trader/data/segment_pool.py` (~924 LOC)
- Move: `SegmentPool` class (`train_easy.py:453-1376`).
- Imports needed by callers: `from mcts_trader.data.segment_pool import SegmentPool` in `train_easy.py` and `train_phase2.py:427-499` (`load_segment_pool` also constructs one).
- Risk: **MEDIUM** — class touches the oracle precomputation path which feeds training labels. No JIT closure lives in this class, but it owns numpy-side data shaping.
- Confidence: **HIGH** — `SegmentPool` is self-contained (no global state beyond stdlib + numpy + sqlite); moving it is a pure-import refactor.

**Split 1.2** — New file `train_easy_pipeline.py` (~1255 LOC) (live at repo root so it sits next to `train_easy.py`)
- Move: `OnTheFlyGenerator` (`train_easy.py:1377-2632`) + `CSVLogger` (`train_easy.py:2690-2729`) + `cleanup_tiered_checkpoints` (`train_easy.py:2633-2683`).
- Imports needed: caller uses `from train_easy_pipeline import OnTheFlyGenerator, CSVLogger, cleanup_tiered_checkpoints`. The pipeline itself imports `SegmentPool` from the new location.
- Risk: **HIGH** — `OnTheFlyGenerator.generate_batch_*` is the hot path that materializes JAX arrays per step. Closures captured by `@partial(jax.jit, ...)` in the file (e.g. `_build_model_state_trajectory*` at `train_easy.py:181-451`) are referenced from inside generator methods. Moving the generator without moving those JIT-traced functions risks an extra retrace per invocation if the module object identity changes (XLA cache is keyed by traced source).
- Confidence: **MEDIUM** — bit-identity depends on keeping the JIT-traced helpers (`_build_model_state_trajectory`, `_build_model_state_trajectory_batched`, `build_state_channels`) co-located with their callers, or moving them together. **Smoke-test prerequisite:** train 1 epoch pre/post split, byte-compare `metrics.json` and the resulting checkpoint params dict via `pickle`.

**Split 1.3** — New file `train_easy_loop.py` (~250 LOC)
- Move: `train_epoch` (`train_easy.py:2736-2805`), `validate` (`train_easy.py:2808-2840`), `_launch_benchmark`, `_check_benchmark_passed` (`train_easy.py:2841-2896`).
- Imports needed: re-export so callers do `from train_easy_loop import train_epoch, validate`.
- Risk: **LOW** — these are plain Python control flow over the trainer/generator/CSVLogger interface.
- Confidence: **HIGH** — no JIT, no flax, no captured arrays.

After splits 1.1+1.2+1.3, `train_easy.py` shrinks to ~150 LOC (JIT helpers + `main`).

---

### 2. `train_phase2.py` — 3140 LOC — 🔴

This file is structurally similar to `train_easy.py`: data-side (`AsyncBatchPipeline`), two big JIT train-step factories (`make_dynamics_train_step`, `make_stage_b_train_step`), and two stage runners.

**Split 2.1** — New file `train_phase2_pipeline.py` (~375 LOC)
- Move: `AsyncBatchPipeline` (`train_phase2.py:502-876`) + `load_segment_pool` (`train_phase2.py:427-499`).
- Imports: callers do `from train_phase2_pipeline import AsyncBatchPipeline, load_segment_pool`.
- Risk: **MEDIUM** — batch tensors built numpy-side feed JIT'd steps. Moving the class doesn't touch numerics, but it captures `from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_temporal` inside `__init__`; that lazy import must be preserved verbatim.
- Confidence: **HIGH** — pure refactor.

**Split 2.2** — New file `train_phase2_loss.py` (~975 LOC)
- Move: `make_stage_b_train_step` (`train_phase2.py:1021-1558`), `make_dynamics_train_step` (`train_phase2.py:311-372`), `scalar_to_categorical` (`train_phase2.py:78-89`), `make_stage_b_labels` (`train_phase2.py:896-925`), `create_stage_b_optimizer` (`train_phase2.py:926-1020`), `create_stage_b_optimizer_with_encoder` (`train_phase2.py:2195-2230`).
- Imports: callers do `from train_phase2_loss import make_stage_b_train_step, make_dynamics_train_step, ...`.
- Risk: **HIGH** — `make_stage_b_train_step` builds the 538-line `loss_fn` body that is JIT-traced. The closure captures `n_assets`, `n_actions_per_asset`, `value_clip_mode`, `entropy_coef`, `disable_limit_orders`, `shape_w_vec`, etc. from the enclosing factory. Any inadvertent rebinding (e.g., changing module-level constants the closure resolves to via `jnp.asarray` calls at trace time) breaks bit identity.
- Confidence: **MEDIUM** — the function is a pure builder taking `(model, config, optimizer)` and returning a closure; mechanically extractable. **Smoke-test prerequisite:** run 50 stage-B steps pre/post split, diff `metrics.json` row-for-row and SHA the post-step `params` pickle.

**Split 2.3** — New file `train_phase2_init.py` (~280 LOC)
- Move: `load_phase1_checkpoint`, `analyze_params`, `warm_policy_direction_refiner`, `initialize_missing_aux_heads`, `initialize_dynamics`, `make_dynamics_mask`, `create_dynamics_optimizer`, `_is_dynamics_key`, `_is_encoder_key`, `_is_titans_key`, `create_model`, `save_checkpoint` (`train_phase2.py:90-426` + `877-895` + `1560-1582`).
- Imports: caller does `from train_phase2_init import ...`.
- Risk: **MEDIUM** — `initialize_dynamics` and `initialize_missing_aux_heads` write the initial params PyTree. Reordering tree walks across module boundaries can change PRNG key consumption order if not careful. They take an explicit `rng_key` so a single move with no order changes is safe.
- Confidence: **HIGH** — straight extraction of helpers used once each from `main`/`train_stage_*`.

**Split 2.4** — New file `train_phase2_cli.py` (~333 LOC)
- Move: `parse_args` (`train_phase2.py:2233-2564`).
- Risk: **LOW** — argparse only.
- Confidence: **HIGH**.

After splits 2.1-2.4, `train_phase2.py` keeps `train_stage_a`, `train_stage_b`, `main`, ~1000 LOC.

---

### 3. `tools/orchestration/e-dojo_v1.py` — 2199 LOC — 🔴

`EasyDojo` (823 LOC) is mostly merge-decision logic; `TrainingLauncher` (685 LOC) is subprocess plumbing. Both could move out.

**Split 3.1** — New file `tools/orchestration/edojo_launcher.py` (~700 LOC)
- Move: `TrainingLauncher` class (`e-dojo_v1.py:1224-1909`), `cleanup_merged_checkpoints` (`e-dojo_v1.py:64-107`).
- Risk: **MEDIUM** — `TrainingLauncher.launch_branch` is the source of "operator argv is silently dropped" per `tools/orchestration/CLAUDE.md` (the `--oracle-k 8` hazard at L1407). The log-file-handle invariant (TASK_141) must move with the class.
- Confidence: **HIGH** — class is well-encapsulated.

**Split 3.2** — New file `tools/orchestration/edojo_schedule.py` (~180 LOC)
- Move: `HistoricalScheduleManager` (`e-dojo_v1.py:236-395`), `DojoConfig`, `DojoState` (`e-dojo_v1.py:112-227`).
- Risk: **LOW** — pure config/scheduling data classes.
- Confidence: **HIGH**.

After 3.1+3.2, `e-dojo_v1.py` is ~1100 LOC (the `EasyDojo` orchestration class + `main`), still 🟡 but no longer a 2k+ outlier. Further splitting `EasyDojo` into "merge-decision" vs "advancement-gate" methods is possible but cross-cutting (shared mutable state), so leave as-is.

---

### 4. `mcts_trader/jax_agent/trainer.py` — 1932 LOC — 🔴

`MuZeroTrainer` (1571 LOC) is mostly cohesive (shares optimizer state, PopArt state, params), but two pieces can split out cleanly.

**Split 4.1** — New file `mcts_trader/jax_agent/trainer_losses.py` (~150 LOC)
- Move: `compute_large_margin_loss` (L35-75), `compute_q_filtered_bc_loss` (L76-114), `_compute_per_head_grad_norms` (L115-157), `_clip_gradients_per_head` (L158-248), `compute_annealed_lambda`, `compute_bc_weight`, `compute_scheduled_sampling_prob`, `compute_neural_scale` (L320-358).
- Imports: `trainer.py` does `from .trainer_losses import compute_large_margin_loss, compute_q_filtered_bc_loss, _compute_per_head_grad_norms, _clip_gradients_per_head, ...`.
- Risk: **MEDIUM** — `_clip_gradients_per_head` uses `flax.core.FrozenDict` traversal and is called from `MuZeroTrainer._train_step`. Pure function move, no closure capture.
- Confidence: **HIGH**.

**Split 4.2** — New file `mcts_trader/jax_agent/trainer_checkpoint.py` (~250 LOC)
- Move: `MuZeroTrainer.save_checkpoint` (L1790-1833), `load_checkpoint` (L1834-1910), `restore_checkpoint` (L1911-1914), `reset_optimizer_state` (L1915-end) — extract as module-level functions that take `(trainer, path)` or move as a mixin.
- Risk: **HIGH** — these methods touch `self.state` (Flax TrainState) and `self.popart_state`. The simpler move is a `trainer_ckpt.py` mixin class that `MuZeroTrainer` inherits from. Mixin path keeps `self.*` semantics bit-identical.
- Confidence: **MEDIUM** — mixin pattern is mechanical but introduces MRO. **Smoke-test prerequisite:** save → load round-trip and SHA-compare params.

**Recommendation:** Do split 4.1, defer 4.2. The `_loss_fn` body (L1065-1560, 495 LOC) and `_train_step` (L1561-1613) are tightly coupled to `MuZeroTrainer` state — leave them in place. The model code in `muzero_model.py` (1065 LOC) is one cohesive flax Module per CLAUDE.md ("TRM + MuZero ... do not suggest transformer/attention changes") — **do not split**.

---

### 5. `mcts_trader/env/jax_trading_env.py` — 1672 LOC — 🟡

Already well-factored by concern (config / state / arithmetic helpers / TP-SL / margin / limit orders / execute_* / reset+step / observation / mask / batched / jit wrappers). One natural split:

**Split 5.1** — New file `mcts_trader/env/jax_trading_observations.py` (~140 LOC)
- Move: `compute_observation_27ch` (L1305-1337), `compute_observation_31ch` (L1338-1367), `compute_observation` (L1368-1437).
- Imports: caller `step` (L1113) calls `compute_observation` at the end.
- Risk: **HIGH** — JIT closures (`jit_step`, `jit_batch_step` at L1607-1651) bake in the observation function via `step` → `compute_observation`. Bit-identity hinges on the function object being identically traced.
- Confidence: **LOW** — moving across modules invalidates the XLA persistent cache for `jit_step`. Numerically identical but slower first run. **Smoke-test prerequisite:** numerical diff on a 100-step rollout pre/post.

**Verdict:** This file is "merely big but cohesive" — the env logic is a single contract (`reset`/`step` over `TradingState`), and the JIT decorators key on the module's traced source. **Recommendation: leave as-is.**

---

### 6. `train_jax_fast.py` — 1620 LOC — 🟡

`main()` is 1500 lines of orchestration over `mcts_trader/training/{cli,setup,self_play,evaluator,...}`. Already well-decomposed at the dependency level (most logic is in `mcts_trader/training/`), so the single big `main()` is a sequence of phase blocks (data load → env build → trainer init → outer epoch loop → callbacks).

**Split 6.1** — New file `train_jax_fast_epoch.py` (~400 LOC)
- Move: the per-epoch body inside `main` (epoch-loop interior; needs careful line extraction). Inline closures `regenerate_level_async` (L167), `on_episode_complete_callback` (L557), `get_last` (L1077) live in `main` — leave them.
- Risk: **HIGH** — `main`'s body shares ~30 local variables (`config`, `env`, `trainer`, `replay_buffer`, `csv_logger`, ...) with the would-be extracted epoch function.
- Confidence: **LOW** — extracting requires inventing a function signature that's brittle to add/remove args.

**Verdict:** This is "merely big but cohesive". The 1500-line `main` is glue, not numerics. **Recommendation: leave as-is** — splitting would force a parameter explosion.

---

### 7. `tools/orchestration/dojo_v4_full_curriculum.py` — 1389 LOC — 🟡

`DojoV4` (L266-1316 = 1050 LOC) has three phase-runners (`_run_synthetic_training`, `_run_speed_training`, `_run_historical_training`) plus several training-driver methods (`_train_synthetic_advanced`, `_train_synthetic_adaptive`, `_train_synthetic`) plus benchmark plumbing (`_run_benchmark`, `_parse_benchmark_output`, `_calculate_avg_benchmark_reward`, `_save_benchmark_to_csv`).

**Split 7.1** — New file `tools/orchestration/dojo_v4_benchmark.py` (~250 LOC)
- Move: `_run_benchmark` (L1127), `_parse_benchmark_output` (L1217), `_calculate_avg_benchmark_reward` (L1259), `_save_benchmark_to_csv` (L1270) — as module-level helpers taking `(checkpoint, phase_name)` and a config dict.
- Risk: **LOW** — subprocess + CSV plumbing, no JAX.
- Confidence: **HIGH**.

After 7.1, `dojo_v4_full_curriculum.py` is ~1140 LOC, still 🟡 but borderline acceptable for a single orchestrator class.

---

### 8. `mcts_trader/env/jax_encoder.py` — 1331 LOC — 🟡

This file already had a major split for TASK_768 (the per-asset / cross-sectional decomposition). Per the audit's explicit caution ("just split for TASK_768; further splits could break bit-identity"), and because the entire file's surface is `@partial(jax.jit, static_argnames=['config'])` wrappers around the channel builders that share static tracing assumptions, **do not split further**. The 32-channel encoder is canonical and reproducibility-critical.

**Recommendation: no action.** Confidence: **HIGH** that further splits would break checkpoint bit-identity by changing trace-source identity.

---

### 9. `tools/orchestration/dual_dojo.py` — 1096 LOC — 🟡

Has clean `DualDojoLauncher` (276 LOC) + `DualDojo` (450 LOC) split internally. Mirrors `e-dojo_v1.py` structure but smaller.

**Split 9.1** — New file `tools/orchestration/dual_dojo_launcher.py` (~310 LOC)
- Move: `DualDojoLauncher` class (L236-511), `cleanup_merged_checkpoints` (L66-98).
- Risk: **MEDIUM** — subprocess launching with hard-coded `/mnt/scratch/trader_checkpoints` paths (per CLAUDE.md). Pure refactor.
- Confidence: **HIGH**.

After 9.1, `dual_dojo.py` ~780 LOC — back into 🟢 range.

---

### 10. `mcts_trader/data/synthetic_market.py` — 1039 LOC — 🟡

`SyntheticMarket` has 15 level generators (`_level_0_sine_waves` through `_level_14_volatility_clustering`) plus shared noise/OHLC helpers. Levels are independent.

**Split 10.1** — New file `mcts_trader/data/synthetic_levels.py` (~700 LOC)
- Move: `_level_0_sine_waves` through `_level_14_volatility_clustering` (L325-end). Recompose as module-level functions taking `(self_state)` — or move them to a `SyntheticLevels` mixin class that `SyntheticMarket` inherits from.
- Risk: **MEDIUM** — these methods read `self.n_assets`, `self.window_size`, `self.rng`, `self.current_step`, etc. Mixin pattern preserves `self.*` semantics.
- Confidence: **MEDIUM** — needs careful inspection of which `self.*` attributes each level uses. **Smoke-test prerequisite:** generate L0-L14 fixed-seed price arrays pre/post split; SHA-compare. Note: SyntheticMarket is used by both `train_easy.py` and `train_jax_fast.py`, so this change has wide blast radius.

**Verdict:** Worth doing — 15 levels in one class is the textbook case for split, and they don't share inter-level state.

---

### 11. `diff_teacher/data_loader.py` — 1027 LOC — 🟡

Two unrelated loaders in one file: `MarketDataLoader` (L31-440, 410 LOC) and `RainbowTableLoader` (L442-1027, 586 LOC). They share only the `MarketWindow` dataclass.

**Split 11.1** — New file `diff_teacher/rainbow_loader.py` (~595 LOC)
- Move: `RainbowTableLoader` class (L442-1027).
- Imports: callers do `from diff_teacher.rainbow_loader import RainbowTableLoader`. The shared `MarketWindow` dataclass can move to a third file or stay in `data_loader.py` and be re-exported.
- Risk: **LOW** — class is self-contained (sqlite + numpy + a private `_get_jit_encoder`).
- Confidence: **HIGH**.

After 11.1, `data_loader.py` is ~440 LOC.

---

### Files explicitly NOT proposed for split (cohesive)

- `mcts_trader/jax_agent/muzero_model.py` (1065 LOC) — single `MuZeroNetworks` flax Module with weight-shape contracts to checkpoints (per `mcts_trader/CLAUDE.md`: "Encoder v2 vs v3 — Check which the active checkpoint uses before edits"). Splitting heads into separate files would change Flax variable scopes ⇒ checkpoint incompatibility.
- `mcts_trader/env/jax_encoder.py` — already split for TASK_768; further splits risk bit-identity per user's stated caution.
- `mcts_trader/training/self_play.py` (929 LOC) — three top-level `run_self_play{,_batched,_jax}` are alternate implementations of the same contract; cohesive even though long.
- `mcts_trader/jax_agent/expert_buffer.py` (894 LOC) — single `UnifiedIndexBuffer` class with shared per-instance state and an inner JIT helper at L645.
- `mcts_trader/data/history.py` (630 LOC) — single `HistoricalMarket` class, cohesive.
- `mcts_trader/jax_agent/prediction/tree_search.py` (760 LOC) — explicitly flagged in the user's audit guidance as "search code with JIT" — leave alone.
- `mcts_trader/jax_agent/path_solver.py` (943 LOC) — 9 `@partial(jax.jit, static_argnames=...)`-decorated functions sharing static argument names and types; bit-identity of trained models depends on stable trace identity. Leave alone.
- `mcts_trader/env/vectorbt_env.py` (1698 LOC) — legacy per `mcts_trader/CLAUDE.md`; do not churn.
- `tools/orchestration/dojo_v3_historical.py` (1103 LOC) — "Reference only (read, don't edit)" per `tools/orchestration/CLAUDE.md`.

---

## Section 3 — Priority ordering (top 5)

| Rank | Split | Effort | Risk | Recommended? |
|---|---|---|---|---|
| 1 | **Split 1.1** — extract `SegmentPool` from `train_easy.py` → `mcts_trader/data/segment_pool.py` (~924 LOC) | LOW | MEDIUM | YES — large agent-context win, used by both train_easy and train_phase2 |
| 2 | **Split 1.3** — extract `train_epoch`/`validate`/`_launch_benchmark` from `train_easy.py` → `train_easy_loop.py` (~250 LOC) | LOW | LOW | YES — pure plumbing move, big readability win |
| 3 | **Split 2.4** — extract `parse_args` from `train_phase2.py` → `train_phase2_cli.py` (~333 LOC) | LOW | LOW | YES — argparse only, free agent-context savings |
| 4 | **Split 4.1** — extract loss/grad helpers from `mcts_trader/jax_agent/trainer.py` → `trainer_losses.py` (~150 LOC) | LOW | MEDIUM | YES — pure function move, decouples MuZeroTrainer class from utilities |
| 5 | **Split 11.1** — extract `RainbowTableLoader` from `diff_teacher/data_loader.py` → `rainbow_loader.py` (~595 LOC) | LOW | LOW | YES — two unrelated loaders, trivial separation |

Deliberately NOT in top 5: Split 2.2 (`make_stage_b_train_step` → `train_phase2_loss.py`) — high value but **HIGH risk + smoke-test prerequisite** due to the 538-line JIT closure. Promote only after a numerical smoke test is wired into CI. Split 10.1 (synthetic levels) is good but second-tier — wide blast radius across two train scripts.

---

## Final summary

**Top 5 files by LOC:** (1) `train_easy.py` (3541), (2) `train_phase2.py` (3140), (3) `tools/orchestration/e-dojo_v1.py` (2199), (4) `mcts_trader/jax_agent/trainer.py` (1932), (5) `mcts_trader/env/vectorbt_env.py` (1698 — legacy, skip).

**Top 3 recommended splits:** (1) carve `SegmentPool` out of `train_easy.py` into `mcts_trader/data/segment_pool.py` (it's already imported by `train_phase2.py`'s pool loader, so the dependency is begging for this); (2) move `train_epoch`/`validate`/benchmark helpers out of `train_easy.py` into a `train_easy_loop.py` (lowest-risk 250 LOC reduction); (3) extract `parse_args` from `train_phase2.py` into `train_phase2_cli.py` — free 333-LOC win with zero numerical risk.

**Surprises:**
- (a) `tools/orchestration/e-dojo_v1.py` at 2199 LOC is bigger than `mcts_trader/jax_agent/trainer.py` — the dojo orchestrator outweighs the actual JAX trainer, and CLAUDE.md flags a real foot-gun in there (silently-dropped operator argv at L1407).
- (b) `mcts_trader/env/vectorbt_env.py` (1698 LOC) is the second-largest non-train file but explicitly marked legacy in `mcts_trader/CLAUDE.md` — it should probably move under an `_old/` sentinel or get a deprecation comment so audit-bot agents don't waste cycles on it.
- (c) The `make_stage_b_train_step` body in `train_phase2.py` (`L1021-1558`, 538 LOC) is the largest single JIT closure in the codebase — it carries the entire Stage-B loss surface (policy CE, value CE, dynamics consistency, BC, direction head, rank head, aux price head, Bella distill, Bella shape, flow matching, entropy reg, TASK_767 limit-action masking) — splitting it is high-value but the *only* split here that genuinely warrants a smoke-test gate.
- (d) `train_jax_fast.py`'s `main()` at ~1500 LOC is the longest single function I saw; resists clean extraction because of dense local-variable sharing.
- (e) `mcts_trader/env/jax_encoder.py` (1331 LOC) was already split once for TASK_768, the comment infrastructure preserves bit-identity, and further splits would gain very little — listed as 🟡 but proposal is **no action**.
