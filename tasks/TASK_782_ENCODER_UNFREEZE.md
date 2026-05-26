# TASK_782 — Encoder Unfreeze Schedule at Canon Stage-B

**Status**: DRAFT — wiring is 100% in tree at workstation tip `79430e3`; experiment is a flag-turn only (one missing 2-line CLI override is documented below). No new code in the load-bearing variant; one minor patch optional for the second schedule.
**Created**: 2026-05-25
**Ship-path candidate**: #8 of 10 per `lore/ship_path_recommendations_2026-05-25.md`.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Hypothesis

The default Stage-B canon (`phase2_config.py:247` `encoder_lr: float = 0.0`) freezes the encoder. The Phase-1 imprint shaped it under a one-hot oracle classification objective; Stage-B's MCTS-derived targets (visit distributions, factored Q-values, joint BC + value loss) may want different features. Unfreezing the encoder at a very low LR — and varying when in training it unfreezes — lets Stage-B adapt the representation toward the planning-time gradient without destroying the Phase-1 imprint.

The candidate is bidirectional: gradient flows into the encoder both from policy/value heads (classifier-side) AND from MCTS rollouts feeding back through the dynamics path (planning-side). Whether the net effect is anti-correlation (per `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md`) or planning-side gain depends on which gradient surface dominates and at which phase of training it engages.

**Important context discovered during verification**: the **canon a1b1 in 760 baselines is NOT frozen-encoder.** `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` shows `"encoder_lr": 3e-05, "encoder_unfreeze_epoch": null`, and the training log line `Encoder UNFROZEN at lr=3e-05 (TASK_728)` confirms the encoder has been unfrozen from epoch 0 in 760 (and 762 — both rs050 and the baseline). The dataclass default of `encoder_lr=0.0` is the *config-file default*, not the *run-time canon*. Therefore Schedule A (`--encoder-lr 3e-5` from epoch 0) is **byte-identical to canon a1b1**. This makes Schedule A the null-control: it MUST reproduce 760 to within seed noise. Schedule B (delayed-unfreeze at epoch 100) is the actual lever in this TASK.

**Falsifiable on canon scorecard**: yes — eval-MCTS sims=16 joint vs 760 a1b1 baselines is the deploy-truthful operator.

## Verified code state (workstation tip `79430e3`)

### `phase2_config.py:247-248` — encoder LR and unfreeze-epoch fields (canonical lines)

```
247    encoder_lr: float = 0.0        # Frozen initially, optionally 3e-5 after 200 epochs
248    encoder_unfreeze_epoch: Optional[int] = None  # Epoch to unfreeze encoder (None = never)
```

(The user's brief cited `phase2_config.py:225-226`; on current tip those lines fall in the `value_target_mode` block. The actual fields live at `247-248`. The brief's logical content matches.)

`phase2_config.py:306` confirms the Stage-B `get_stage_defaults()` sets `freeze_encoder = True` — with the comment `"Unfrozen later via encoder_unfreeze_epoch"`.

### `train_phase2.py:957-972` — encoder unfreeze logic in `create_stage_b_optimizer` (canonical lines)

```
957    encoder_trainable = getattr(config, 'encoder_lr', 0.0) > 0.0
958    if encoder_trainable:
959        enc_schedule = optax.warmup_cosine_decay_schedule(
960            init_value=0.0,
961            peak_value=config.encoder_lr,
962            warmup_steps=warmup_steps,
963            decay_steps=max(total_steps, warmup_steps + 1),
964            end_value=config.encoder_lr * 0.1,
965        )
966        encoder_opt = optax.chain(
967            optax.clip_by_global_norm(0.5),
968            optax.adamw(enc_schedule, weight_decay=1e-4),
969        )
970        logger.info(f"  Encoder UNFROZEN at lr={config.encoder_lr} (TASK_728)")
971    else:
972        encoder_opt = optax.set_to_zero()
```

This branch fires at optimizer-creation time (epoch 0). When `encoder_lr > 0`, the encoder is unfrozen *from the start* with its own warmup-cosine schedule.

### `train_phase2.py:2213-2220` — dynamic unfreeze check inside the epoch loop (canonical lines)

```
2213        # Encoder unfreeze check
2214        if (config.encoder_unfreeze_epoch is not None and
2215                epoch == config.encoder_unfreeze_epoch):
2216            logger.info(f"  Unfreezing encoder at epoch {epoch} (lr={config.encoder_lr})")
2217            # Rebuild optimizer with encoder included
2218            # For simplicity, recreate with full mask (no encoder exclusion)
2219            # This is safe because encoder_lr is very low (3e-5)
2220            optimizer, opt_state = create_stage_b_optimizer_with_encoder(config, params)
```

(The user's brief cited `train_phase2.py:2173`; that line on current tip is inside the Titans CSV metrics block. The actual dynamic-unfreeze handler is at `2213-2220`. The brief's logical content matches.)

This branch fires at the start of each epoch when `epoch == encoder_unfreeze_epoch`. The optimizer is rebuilt via `create_stage_b_optimizer_with_encoder(config, params)`, swapping in a mask that *no longer excludes* the encoder param tree.

### `train_phase2.py:2538-2539` — `--encoder-lr` CLI override

```
2538    if args.encoder_lr is not None:
2539        config.encoder_lr = args.encoder_lr
```

The CLI argument itself lives at `train_phase2_cli.py:120-121`:
```
120    parser.add_argument('--encoder-lr', type=float, default=None,
121                        help='Encoder learning rate (0=frozen, default: config)')
```

### Missing CLI flag — `--encoder-unfreeze-epoch` is **NOT** wired

Confirmed: `grep -n "encoder.unfreeze\|encoder_unfreeze" /home/lukas/Trader/train_phase2_cli.py /home/lukas/Trader/train_phase2.py` returns:
- `phase2_config.py:248` — the dataclass field (✓)
- `phase2_config.py:306` — Stage-B default comment (✓)
- `train_phase2.py:2213-2220` — the dynamic check (✓)
- `train_phase2_cli.py:117` — TASK_728 comment (just the section header, no argparse line)

**No `parser.add_argument('--encoder-unfreeze-epoch', ...)`** anywhere. TASK_725 §C explicitly proposed adding it (`tasks/TASK_725_BULL_EXPERIMENTS.md:152-156`) — the proposal never landed. This is the same finding the brief flagged ("just turn on") with a corner-case correction: the *load-bearing default* (`encoder_lr` schedule via the static branch at `957-970`) is exposed via `--encoder-lr`, but the *delayed-unfreeze schedule* (the dynamic branch at `2213-2220`) is **not** reachable through current CLI.

## Code-change required

- **Schedule A (`encoder_lr=3e-5` unfrozen from epoch 0)**: NONE. Already canon — every 760+ Stage-B run uses this. This TASK's Schedule A is the no-op control (used as the *seed-noise null* for Schedule B's paired comparison).
- **Schedule B (`encoder_unfreeze_epoch=100`)**: ONE optional 4-line CLI patch:
  ```python
  # train_phase2_cli.py — add next to the --encoder-lr add_argument at line 120-121:
  parser.add_argument('--encoder-unfreeze-epoch', type=int, default=None,
                      help='Epoch to unfreeze encoder (None = never, 0 = canon, '
                           'use 100 for mid-training unfreeze)')
  # train_phase2.py — add next to the args.encoder_lr override at 2538-2539:
  if args.encoder_unfreeze_epoch is not None:
      config.encoder_unfreeze_epoch = args.encoder_unfreeze_epoch
  ```
  This is **out-of-scope for this TASK as a sub-agent** per the deploy contract ("DO NOT edit existing source files"). The operator may either:
  - (a) land the 4-line patch as part of TASK_782 pre-flight, OR
  - (b) launch Schedule B by writing a one-off `stage_b_config.json` override that pins `encoder_unfreeze_epoch: 100` and using the existing config-file load path (`get_phase2_config(phase1_config_path=…)` at `phase2_config.py:312-333`).
- **Schedule A on its own** (Schedule A alone, no Schedule B) reduces to "rerun a1b1 canon under a new run-id" — that's not a useful experiment by itself.

The load-bearing experiment is therefore **Schedule B vs canon at 3 seeds**, paired against 760. Schedule A is included as a sanity-control: confirm Schedule A reproduces 760 within seed noise (Δα within ±2pp at the same seed). If it doesn't, the test harness is broken and Schedule B's signal is unreadable.

## Variants — 2 schedules × 3 seeds = 6 cells

Per the user's brief specification, with the canon-discovery correction:

| cell | schedule | encoder_lr | encoder_unfreeze_epoch | seed | wall (server, est.) |
|---|---|---:|---:|---:|---:|
| 782A_s7  | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 7  | ~2.5 h |
| 782A_s13 | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 13 | ~2.5 h |
| 782A_s23 | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 23 | ~2.5 h |
| 782B_s7  | B (delayed unfreeze)  | 3e-5 | 100                  | 7  | ~2.5 h |
| 782B_s13 | B (delayed unfreeze)  | 3e-5 | 100                  | 13 | ~2.5 h |
| 782B_s23 | B (delayed unfreeze)  | 3e-5 | 100                  | 23 | ~2.5 h |

Wall: 6 cells × ~2.5 h = 15 h serial, **~5 h wall at 2-wide** on GPU 1+2.

**Comparison contract**:
- **Schedule A vs 760 a1b1 sims=32** at matched seed = sanity null. Expected paired Δα ≈ 0 ± seed noise. Failure mode: if Δα differs from 0 by > 2pp at any seed, treat as harness drift / git-commit divergence and STOP.
- **Schedule B vs 760 a1b1 sims=32** at matched seed = the falsifiable test. The mechanism question: does delayed-unfreeze at epoch 100 outperform from-epoch-0 unfreeze.
- **Schedule B vs Schedule A at matched seed** = the additional paired comparison; isolates the "schedule" lever from the rest of the harness.

## Anti-correlation diagnostic — encoder gradient norm logging

Per the user's brief, the experiment must instrument encoder gradient norm at convergence and abort cells where it diverges.

**Hard rules**:
- Encoder gradient norm at end-of-Schedule-B (epoch 200) must stay **< 2× canon** (where "canon" = average end-of-training encoder grad norm across `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` Stage-B metrics).
- **Abort the cell** if encoder gradient norm > **5× canon** at any point between epoch 50 and epoch 100 (or, for Schedule B, between epoch 100 and 150 — the first 50 epochs immediately after the dynamic unfreeze).
- Encoder gradient norm is reported via the existing per-group `grad_norm` aggregation that `make_stage_b_train_step` emits in the metrics dict (Titans group + encoder group + reasoning group). The encoder group's grad-norm key (if not already present) must be readable from the per-epoch CSV at `runs/782_encoder_unfreeze/<cell>/stage_b_metrics.csv`.

**Soft rules (log-only, no abort)**:
- Δ-Bal.Acc vs canon — if Schedule B shows Δ-Bal.Acc > 0 AND Δα < 0 at 3-seed mean, the candidate triggered the anti-correlation pattern despite the planning-side framing. Mark as TIER-3 closure of the encoder-LR-scheduling surface.
- Trade-rate at convergence — if Schedule B trade-rate at convergence > 2× canon, flag in result.md (this is the classic anti-correlation signature).

## Promotion gate (canon)

- paired Δα ≥ +3pp vs 760 a1b1 sims=32 baselines under eval-MCTS sims=16 joint.
- No seed < −5pp.
- Eval command per the canon contract: `python tools/benchmark_easy.py --checkpoint <p2_ckpt> --periods-file config/benchmark_periods.json --eval-mcts-sims 16 --eval-mcts-decode joint --output runs/782_encoder_unfreeze/eval_<cell>/benchmark.json`.

5-seed extension (seeds 31, 42) only if 3-seed clears.

## Risk + anti-correlation check — MEDIUM

`~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` documents three encoder-side mechanism failures (737 no_aux_solo, 741 mint_xs, 738 legacy_27ch) where "sharper classifier" → Alpha collapse via trade-rate explosion. Line 27: *"The bottleneck is NOT the encoder representation, NOT the cross-sectional feature stack, NOT the per-asset signal density."*

`~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` reinforces: three Bella-encoder attempts (740 distill, 757 shape, 758 augment) all paired-negative; "encoder-side improvements (better representation per Bella's target) are confirmed anti-correlated with Alpha on this stack."

**Justification for testing despite this**:

1. **Surface distinction**: Prior nulls were on encoder *design changes*:
   - 737/738/741 modified the *architecture* / *channel composition* (`cross-sectional features`, `solo aux`, `mint per-asset injection`)
   - 740/757/758 modified the *training signal* feeding the encoder (`distillation target = Bella prediction`, `shape descriptor`, `synthetic data augmentation`)
   - TASK_782 modifies neither architecture nor target. It modifies *when in training* the encoder receives gradient — a scheduling lever, not a design lever.

2. **Bidirectional gradient flow**: Once unfrozen, the encoder receives gradient from *both* the classifier-side losses (policy CE, BC) AND the planning-side losses (value MSE through the dynamics path, factored Q-value backprop). The first three failure cases all reinforced classifier-side gradient pressure unilaterally. A delayed-unfreeze schedule allows the encoder to absorb planning-side gradient after the policy/value heads have already stabilized — different dynamic.

3. **Cost-of-falsification is the lowest in the ranked queue**: 0 LOC for Schedule A; 4 LOC optional patch for Schedule B. 5 h wall at 2-wide. If null, closes the encoder-LR-scheduling surface and frees the operator from re-considering this lever later.

4. **Discovery during verification**: canon 760 is *already* `encoder_lr=3e-5` from epoch 0 (verified above). This means the campaign's null findings did NOT control for encoder LR — every Phase-2 result in the 738–771 sweep was measured with the encoder actively training at 3e-5. The "encoder unfreeze" mechanism as historically framed (from-frozen-canon to unfrozen-variant) does not exist as a clean A/B in any prior run. TASK_782 Schedule B is the first comparison of *delayed-unfreeze* against *from-epoch-0 unfreeze*, holding everything else fixed.

**Worst case**: Schedule B shows Δ-Bal.Acc > 0 AND Δα < 0 at 3-seed mean. Encoder-LR-scheduling surface joins the closed encoder mechanism class. Result is informative — closes the surface.

**Best case**: Schedule B clears +3pp paired Δα. Lever ships, encoder-LR scheduling moves from "untouched" to "load-bearing."

**Null case (most likely prior)**: Schedule B and Schedule A are within seed noise of canon. Confirms encoder-LR scheduling is not the lever; closes the surface cheaply. Schedule A independently confirms harness reproducibility for the entire 782+ campaign queue.

## Cost estimate

- 0 LOC for Schedule A (already canon).
- 4 LOC optional patch for Schedule B (out-of-scope for sub-agent; operator-decision).
- 6 cells × ~2.5 h Stage-B = 15 h serial.
- ~5 h wall at 2-wide on server GPU 1+2.
- Eval: 6 × ~30 min on GPU 3, overlapped with subsequent training.
- **Total**: 0 LOC + ~5 h wall + ~15 GPU-hours.

This is the cheapest variant-vs-canon probe in the ranked ship-path queue.

## References

- `lore/ship_path_recommendations_2026-05-25.md` §8 — primary spec source for this TASK (Encoder unfreeze schedule).
- `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment C: Unfreeze Encoder (Learn New Features)" (lines 100-198) — original design proposal and the implementation plan for the missing `--encoder-unfreeze-epoch` CLI flag.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — MEDIUM risk: encoder-side changes anti-correlate with Alpha on this stack.
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — three additional encoder-aux failures (740/757/758).
- `lore/notes_2026-05-19.md` §149 — Bella mechanism class closed; encoder design changes confirmed anti-correlated. (Note: §149 does not explicitly name "LR scheduling" as untouched; that's an inference from the surface taxonomy.)
- `phase2_config.py:247-248` — `encoder_lr`, `encoder_unfreeze_epoch` field definitions (verified `79430e3`).
- `phase2_config.py:306` — Stage-B default comment (verified `79430e3`).
- `train_phase2.py:957-972` — encoder unfreeze logic in `create_stage_b_optimizer` (verified `79430e3`).
- `train_phase2.py:2213-2220` — dynamic unfreeze handler inside epoch loop (verified `79430e3`).
- `train_phase2.py:2538-2539` — `--encoder-lr` CLI override (verified `79430e3`).
- `train_phase2_cli.py:120-121` — `--encoder-lr` `add_argument` (verified `79430e3`).
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}/stage_b_config.json` — canon a1b1 sims=32 baselines (showing `encoder_lr: 3e-05`, `encoder_unfreeze_epoch: null`).
- `runs/760_train_mcts_sims/baseline_paired_seed_7/train.log` — confirms `Encoder UNFROZEN at lr=3e-05 (TASK_728)` line emits in canon runs.
- `experiments/782_encoder_unfreeze/SPEC.md` — execution SPEC (sibling file).

## Implementation log

(reserved — populate at execution time)
