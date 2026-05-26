# TASK_775 — Oracle-Mask Aux Head, Phase-1 Re-Imprint

**Status**: DRAFT — code already wired in tree at workstation tip `79430e3`; experiment unrun on every recorded Phase-1 / Phase-2 ckpt to date.
**Created**: 2026-05-25
**Ship-path candidate**: #1 of 10 per `lore/ship_path_recommendations_2026-05-25.md`.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Hypothesis

Activate the planning-side `oracle_mask` auxiliary head at Phase-1 imprint, so the encoder learns a per-asset binary selection-mask predictor in parallel with the policy/value targets. The head implements the autodidact pattern from Klissarov 2025 (`docs/arXiv-2602.16066v1/`) and is the load-bearing gate for Phase 3 "The Thought" in `docs/easy_mode.md:96-127` — without it the autodidact / inference-time MCTS prior cannot exist on this stack.

The choice of mechanism class follows the standing finding `project_classifier_alpha_anti_correlation.md` line 27 — "The bottleneck is **NOT** the encoder representation, **NOT** the cross-sectional feature stack, **NOT** the per-asset signal density. It is somewhere in: value/policy head aggressiveness scaling, BC loss weighting in Stage-B, trade gating / softmax temperature at inference, the planning-side use of the model." The mask head is planning-side: it produces a separate gradient signal that shapes the encoder toward "which assets are worth attending to," distinct from the per-action softmax CE that the anti-correlation pattern implicates (see §"Risk + anti-correlation check").

**Falsifiable in two stages**:
- Phase-1 sanity gate: oracle-mask head Bal.Acc on held-out > 60% (head learned a signal at all) AND bull-regime classifier Bal.Acc must not regress > 3pp vs the canon `jax_ckpt_easy_phase1_xs32/easy_medium/` imprint at matched epoch budget.
- Phase-2 promotion gate (canon): paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Verified code state (workstation tip `79430e3`)

Head is fully wired with `use_oracle_mask_head: bool = False` default. Verified by reading the cited file:line ranges directly:

**Model graph** (`mcts_trader/jax_agent/muzero_model.py`):
- `:286` — `use_oracle_mask_head: bool = False`, `:287` `oracle_mask_hidden_dim: int = 256` config fields on `MuZeroNetworks` (TASK_695 block).
- `:417-421` — Setup branch: instantiates `oracle_mask_dense1 = nn.Dense(hidden_dim)` + `oracle_mask_dense2 = nn.Dense(n_assets)` when flag is True.
- `:564-568` — `oracle_mask_prediction(latent_state)` method: returns `(batch, n_assets)` logits via `Dense → relu → Dense` on the shared latent.
- `:661-663` — Initial-inference branch: when flag is True, attaches `result['oracle_mask']` to the model output dict.
- `:1003-1004` — Parameter-init branch: when flag is True, runs `oracle_mask_prediction(...)` on dummy latent so the head's params get materialized inside the `initialize_parameters` call.
- `:1051-1053` — Recurrent-inference branch: same as initial-inference, attaches mask to dict.

**Phase-1 trainer loss term** (`easy_trainer.py`):
- `:60-61` — `MuZeroNetworks` constructor forwards `use_oracle_mask_head` and `oracle_mask_hidden_dim` from `config`.
- `:215-225` — Loss term computed under `if self.config.use_oracle_mask_head and 'oracle_mask' in batch:` — uses `optax.sigmoid_binary_cross_entropy(logits, targets)` against the `(B, 64)` binary mask in the batch dict; accuracy metric is `mean((sigmoid(logits) > 0.5) == targets)`.
- `:228-232` — `total_loss = policy_weight * policy_loss + value_weight * value_loss + oracle_mask_weight * oracle_mask_loss`.
- `:298-299` — Metrics dict exports `oracle_mask_loss` + `oracle_mask_accuracy` per epoch.

**Phase-1 CLI flag** (`train_easy.py`):
- `:1885-1886` — `if args.oracle_mask_head: config.use_oracle_mask_head = True`. Flag is `--oracle-mask-head`. (Optional flag `--oracle-mask-weight` is **not** currently exposed; weight is taken from `easy_config.oracle_mask_weight = 0.1` default — see "Code-change required" below.)

**Config defaults** (`easy_config.py`):
- `:86-88` — `use_oracle_mask_head: bool = False`, `oracle_mask_hidden_dim: int = 256`, `oracle_mask_weight: float = 0.1` on `EasyConfig`.

**Phase-2 inheritance + freeze** (`train_phase2.py`):
- `:392-393` — `MuZeroNetworks` factory forwards `use_oracle_mask_head` + `oracle_mask_hidden_dim` from the inherited `Phase2Config` (which extends `EasyConfig`).
- `:919-920` — Optimizer label_fn: any param-tree path containing the substring `oracle_mask` is labelled `'freeze'` (zero-grad through `optax.set_to_zero()`).
- `:2217-2226` — Same logic in the Stage-A optimizer: `should_train(path, _) = 'oracle_mask' not in str(path[0])` → split into `train` / `freeze` groups via `optax.multi_transform`.
- `:2650` — `initialize_missing_aux_heads(...)` is called when `use_oracle_mask_head` (among others) is True in the Stage-B config, so a base ckpt that lacks the mask params is auto-extended at load time. Confirms a head-ON Phase-2 can also load a head-OFF base ckpt (not the experiment path, but useful for downstream ablations).

**Current base imprint config** (`jax_ckpt_easy_phase1_xs32/easy_medium/config.json:44`): `"use_oracle_mask_head": false`. Spot-checked recent Stage-B runs:
- `runs/762_reward_shaping/rs050_seed_7/stage_b_config.json:44` — `false`.
- `runs/749_titans_mag/*/stage_b_config.json:44` — `false` (3 cells sampled).

Every Phase-2 run in the campaign inherits this flag-off state. **The mechanism has never been activated in any campaign run.**

## Code-change required: NONE for the load-bearing experiment

The `--oracle-mask-head` flag flips `config.use_oracle_mask_head = True`, the head is constructed, the loss term fires, the metrics export, Phase-2 inherits + freezes. Confirmed against current source — every wire is in place.

**Minor convenience flag missing** (not load-bearing for this TASK; mentioned for transparency):
- `--oracle-mask-weight FLOAT` is **not** exposed as a CLI override at the `train_easy.py` argparse layer. The cell matrix below sweeps `oracle_mask_weight ∈ {0.05, 0.1, 0.2}`, and `0.1` is the config default. Two of the three weight cells require setting the value some other way:
  - (a) launch with a `--config <path>` override file that pins the weight, or
  - (b) launch via the dojo with a per-cell override, or
  - (c) add the 2-line argparse flag (`--oracle-mask-weight`) + 2-line override block in `train_easy.py` next to `:1885-1886`, identical pattern to `--oracle-k`.
- Operator may choose option (c) before launch as a 4-line workstation patch. This task does **not** include that patch in the experiment-blocking critical path — if operator skips it, the SPEC will be edited to drop the weight sweep down to a single cell at the default `oracle_mask_weight=0.1` × 3 seeds (3 cells instead of 9). The 3-cell variant still tests the load-bearing hypothesis (head ON vs head OFF); the 9-cell variant additionally calibrates the weight coefficient. Default for this TASK is the 3-cell minimal version.

## Variants

### Phase-1 re-imprint matrix — 3-cell minimal (default, no code change)

| cell | use_oracle_mask_head | oracle_mask_weight | seed | wall (server, est.) |
|---|---|---:|---:|---:|
| 775P1-w01_s7  | True | 0.1 (config default) | 7  | ~165 min |
| 775P1-w01_s13 | True | 0.1 (config default) | 13 | ~165 min |
| 775P1-w01_s23 | True | 0.1 (config default) | 23 | ~165 min |

Wall: 3 cells × ~165 min server (TASK_764 k8_s7 actual, `lore/handoff_2026-05-24_764-launch.md`) ≈ ~8 h serial, ~3 h wall at 2-wide.

### Phase-1 re-imprint matrix — 9-cell weight sweep (only if 4-line patch lands)

| cell | use_oracle_mask_head | oracle_mask_weight | seed |
|---|---|---:|---:|
| 775P1-w005_s7  | True | 0.05 | 7  |
| 775P1-w005_s13 | True | 0.05 | 13 |
| 775P1-w005_s23 | True | 0.05 | 23 |
| 775P1-w01_s7   | True | 0.1  | 7  |
| 775P1-w01_s13  | True | 0.1  | 13 |
| 775P1-w01_s23  | True | 0.1  | 23 |
| 775P1-w02_s7   | True | 0.2  | 7  |
| 775P1-w02_s13  | True | 0.2  | 13 |
| 775P1-w02_s23  | True | 0.2  | 23 |

Wall: 9 cells × ~165 min ≈ ~25 h server serial, ~12 h wall at 2-wide.

### Phase-2 Stage-B retrain — paired 3-seed at best Phase-1 weight

Only triggered after Phase-1 sanity gate passes on at least one weight setting (or unconditionally on `oracle_mask_weight=0.1` for the 3-cell variant). Stage-B uses the canon a1b1 config — `use_oracle_mask_head=True` carries through via `Phase2Config(EasyConfig)` inheritance; head is frozen during Stage-B per `train_phase2.py:919, 2217`.

| cell | base ckpt | seed | wall (server, est.) |
|---|---|---:|---:|
| 775P2-best_s7  | best Phase-1 weight, seed 7  | 7  | ~2.5 h |
| 775P2-best_s13 | best Phase-1 weight, seed 13 | 13 | ~2.5 h |
| 775P2-best_s23 | best Phase-1 weight, seed 23 | 23 | ~2.5 h |

Wall: 3 cells × 2.5 h = 7.5 h serial, ~5 h wall at 2-wide.

## Phase-1 sanity gate (early exit)

Two metrics, both required for promotion to Phase-2 retrain:

1. **Oracle-mask Bal.Acc on held-out > 60%**. Test that the head learned anything at all — random would be ~50%. Read from the per-epoch `oracle_mask_accuracy` metric exported at `easy_trainer.py:299`. Use the held-out validation split via the existing `eval_every` cadence (no new code).
2. **Bull-regime classifier Bal.Acc must not regress > 3pp vs canon imprint at matched epoch**. Compare against `jax_ckpt_easy_phase1_xs32/easy_medium/` final-epoch bull Bal.Acc on the same validation split. The mask head's gradient should compete with (not destroy) the policy head's classifier training; > 3pp regression means the mask head is dominating the encoder budget and the experiment is dead before Phase-2.

If either gate fails on all 3 seeds at a given weight, drop that weight (or in the 3-cell variant, fail the TASK). If at least one weight passes both gates on ≥ 2/3 seeds, proceed to Phase-2 on that weight × {7, 13, 23}.

## Phase-2 promotion gate (canon)

- paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint.
- No seed < −5pp.
- Eval command per the canon contract: `python tools/benchmark_easy.py --checkpoint <p2_ckpt> --periods-file config/benchmark_periods.json --eval-mcts-sims 16 --eval-mcts-decode joint --output runs/775_oracle_mask/eval_p2_<cell>/benchmark.json`.

5-seed extension (seeds 31, 42) only if 3-seed clears.

## Cost estimate

- **3-cell minimal Phase-1**: ~165 min/seed × 3 = ~8 h server serial; ~3-4 h wall at 2-wide on GPU 1+2.
- **9-cell weight sweep Phase-1** (if patch lands): ~25 h server serial; ~12-15 h wall at 2-wide.
- **Phase-2 paired 3-seed**: ~2.5 h/seed × 3 = 7.5 h serial; ~4-5 h wall at 2-wide.
- **Total (default 3-cell path)**: ~8 h wall, ~13-15 h GPU-hours.
- **Total (9-cell path)**: ~17-20 h wall, ~32 h GPU-hours.

The user's brief estimates ~30 h server total in the 9-cell + paired-Phase-2 case; the 3-cell + paired-Phase-2 default is materially cheaper because the load-bearing question (mask head ON vs OFF) is the same whether or not the weight is swept.

## Risk + anti-correlation check

The campaign's standing finding `project_classifier_alpha_anti_correlation.md` documents three independent failures (737, 738, 741) where a *sharper* classifier in Stage-B reliably collapsed Alpha via trade-rate explosion. §1 of `ship_path_recommendations_2026-05-25.md` (this TASK's spec section) addresses the question of whether the mask head triggers the same pathway:

**Mechanism distinction**:
- The pattern that anti-correlates with Alpha is **per-action softmax cross-entropy** at the policy head's 5-class-per-asset output. "Sharper" means lower CE on per-asset action classes → more confident wrong trades.
- The oracle-mask head produces **per-asset binary selection-set BCE** against the oracle's top-k mask. "Sharper" here means better-predicting *which* assets are profitable to attend to — not which *action* to take on a given asset.

These are distinct gradient surfaces. The mask head shapes the encoder toward a **selection-set** representation; the policy head shapes it toward an **action-class** representation. The closest prior on this anti-correlation surface is 740_bella_distill (encoder regression aux, paired Δα −1.92pp — `project_encoder_bella_class_closed.md`). 740 distilled *price predictions*; 775 distills *selection-set membership*. Selection-set membership is what MCTS searches over at inference time; price predictions are what the dynamics MLP simulates internally. Different surfaces.

**Anti-correlation diagnostic during Phase-2**: read the per-epoch `policy_entropy` and `dir_acc` metrics from the Stage-B CSV. If the mask-head Phase-2 run shows **Δ-Bal.Acc > 0 AND Δα < 0** at 3-seed mean vs canon, the head DID trigger anti-correlation through an unmodeled coupling pathway, and the result is a closure (negative result, planning-side mechanism class dies). If Δ-Bal.Acc ≤ 0 (or flat) AND Δα > 0, the mechanism distinction held.

**Worst-case**: same outcome as 740 — encoder learns a representation that helps the auxiliary head but hurts downstream Alpha via subtle anti-correlation. Cost-of-falsification is the ~13-15 h GPU-hour budget.

**Best-case**: head learns useful selection-set features, encoder generalizes them into policy/value representations, Stage-B Δα clears +3pp — the planning-side ship.

## Phase 3 follow-on (out of scope for ship gate)

If Phase-2 ships, the head's outputs are also a per-asset prior usable at inference-time MCTS to gate which assets the search expands. Wiring `oracle_mask_prediction(latent)` into `factored_search.py:_q_to_policy` or `mctx_search.py:_recurrent_fn` is ~30 LOC. That work is the Phase 3 ExIt loop (`docs/easy_mode.md`), tracked separately. The ship gate for TASK_775 is **training-time gradient signal only**.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §1 — primary spec source for this TASK.
- `tasks/TASK_695_ORACLE_MASK_PREDICTION_HEAD.md` — original Klissarov-inspired design (Feb 2026).
- `docs/easy_mode.md:96-127` — Phase 3 "The Thought" autodidact design + the head's role as the prerequisite gate.
- `docs/arXiv-2602.16066v1/` — Klissarov 2025 RL²F.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` (line 27 = encoder NOT the bottleneck → planning-side).
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` (closest prior on encoder aux surface — 740 −1.92pp, 757 −5.80pp, 758 −16.54pp).
- `tasks/TASK_764_ORACLE_K_REFORMULATION.md` — Phase-1 imprint baseline + ~165-min server wall reference.
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server actual for k8_s7.
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` — style reference for AUTHORIZATION format (and the eval-MCTS sims=16 joint contract).
- `mcts_trader/jax_agent/muzero_model.py:286, 417-421, 564-568, 661-663, 1003-1004, 1051-1053` — head wiring.
- `easy_trainer.py:60-61, 215-225, 228-232, 298-299` — loss term + metrics.
- `train_easy.py:1885-1886` — `--oracle-mask-head` CLI flag.
- `easy_config.py:86-88` — config defaults (`use_oracle_mask_head=False`, `oracle_mask_weight=0.1`).
- `train_phase2.py:392-393, 919-920, 2217-2226, 2650` — Phase-2 inheritance, freeze, missing-head init.
- `jax_ckpt_easy_phase1_xs32/easy_medium/config.json:44` — base imprint has `use_oracle_mask_head: false`.

## Implementation log

(reserved — populate at execution time, mirroring `tasks/TASK_764_ORACLE_K_REFORMULATION.md` §"Implementation deep-dive (2026-05-22)")
