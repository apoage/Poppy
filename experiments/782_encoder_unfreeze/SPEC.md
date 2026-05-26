# 782_encoder_unfreeze — Encoder Unfreeze Schedule at Canon Stage-B

**Status**: SPEC — DRAFT. Wiring 100% in tree at workstation tip `79430e3`. Load-bearing variant (Schedule A) requires 0 code; second variant (Schedule B) requires a 4-line optional patch out-of-scope for sub-agent (see TASK_782 §"Code-change required"). The execution under this SPEC's default plan assumes the operator has either landed the patch or will pin `encoder_unfreeze_epoch: 100` via a stage_b_config.json override.
**Owner**: server agent executes; workstation operator reviews and decides on the 4-line patch.
**Reference**: `tasks/TASK_782_ENCODER_UNFREEZE.md`, `lore/ship_path_recommendations_2026-05-25.md` §8.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Goal

Quantify whether *when* the encoder unfreezes during Stage-B affects deployed Alpha under the canon scorecard. Two schedules × three seeds = 6 cells, paired against 760 a1b1 sims=32 baselines.

**Key context discovered during TASK_782 verification**: the canon 760 a1b1 baselines are already `encoder_lr=3e-05` with the encoder unfrozen from epoch 0 (`runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` + `train.log`). The dataclass default `encoder_lr=0.0` is the config-file default, not the runtime canon. Schedule A in this SPEC is therefore the null-control (byte-equivalent to canon, sanity-check that the harness reproduces 760); Schedule B is the actual experimental lever.

## Cells matrix

| cell | schedule | encoder_lr | encoder_unfreeze_epoch | seed | output dir | wall (server, est.) |
|---|---|---:|---:|---:|---|---:|
| sched_A_seed_7  | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 7  | `runs/782_encoder_unfreeze/sched_A_seed_7`  | ~2.5 h |
| sched_A_seed_13 | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 13 | `runs/782_encoder_unfreeze/sched_A_seed_13` | ~2.5 h |
| sched_A_seed_23 | A (canon-equivalent) | 3e-5 | None (= 0 effective) | 23 | `runs/782_encoder_unfreeze/sched_A_seed_23` | ~2.5 h |
| sched_B_seed_7  | B (delayed unfreeze)  | 3e-5 | 100                  | 7  | `runs/782_encoder_unfreeze/sched_B_seed_7`  | ~2.5 h |
| sched_B_seed_13 | B (delayed unfreeze)  | 3e-5 | 100                  | 13 | `runs/782_encoder_unfreeze/sched_B_seed_13` | ~2.5 h |
| sched_B_seed_23 | B (delayed unfreeze)  | 3e-5 | 100                  | 23 | `runs/782_encoder_unfreeze/sched_B_seed_23` | ~2.5 h |

Wall: 6 cells × ~2.5 h = 15 h serial, **~5 h wall at 2-wide** on GPU 1+2.

## Launch command per cell

The canon a1b1 sims=32 flag set is the same as `runs/760_train_mcts_sims/baseline_paired_seed_*` was launched with. Each cell below is canon plus the schedule-specific encoder flags.

### Schedule A (canon-equivalent, 0 LOC)

```bash
python -u train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B \
  --seed 7 \
  --epochs 200 \
  --batch-size 256 \
  --steps-per-epoch 200 \
  --use-factored-search \
  --factored-temperature 1.0 \
  --reward-scale 100 \
  --value-target-mode nstep \
  --nstep-horizon 20 \
  --td-discount 0.95 \
  --mcts-sims 32 \
  --bc-class-balanced \
  --bc-max-trade-weight 4.0 \
  --encoder-lr 3e-5 \
  --output-dir runs/782_encoder_unfreeze/sched_A_seed_7 \
  --name 782_sched_A_s7
```

Substitute `--seed 13` / `--output-dir .../sched_A_seed_13` / `--name 782_sched_A_s13` for `_s13`; analogue for `_s23`.

### Schedule B (delayed unfreeze, requires 4-line patch OR config-override path)

If the operator landed the optional patch (see TASK_782 §"Code-change required"):

```bash
python -u train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --stage B \
  --seed 7 \
  --epochs 200 \
  --batch-size 256 \
  --steps-per-epoch 200 \
  --use-factored-search \
  --factored-temperature 1.0 \
  --reward-scale 100 \
  --value-target-mode nstep \
  --nstep-horizon 20 \
  --td-discount 0.95 \
  --mcts-sims 32 \
  --bc-class-balanced \
  --bc-max-trade-weight 4.0 \
  --encoder-lr 3e-5 \
  --encoder-unfreeze-epoch 100 \
  --output-dir runs/782_encoder_unfreeze/sched_B_seed_7 \
  --name 782_sched_B_s7
```

If the operator deferred the patch, prepare a one-off `stage_b_config_override.json` containing `{"encoder_unfreeze_epoch": 100}` and load via the existing `get_phase2_config(phase1_config_path=...)` path in `phase2_config.py:312-333`. Document the path in result.md.

**Important**: confirm against `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` that the only flags differing between this SPEC and canon are `--encoder-lr` and `--encoder-unfreeze-epoch`. If any other canon flag is missing from the above launch, ADD it before launching — do not deviate from canon in any other respect.

## GPU placement

- **GPU 0**: FORBIDDEN — VirtualBox-reserved (workstation memory standing rule).
- **GPU 1**: cell N (Stage-B training).
- **GPU 2**: cell N+1 (Stage-B training, paired-concurrent).
- **GPU 3**: rolling eval on completed cells (`tools/benchmark_easy.py --eval-mcts-sims 16 --eval-mcts-decode joint`), plus optional encoder-grad-norm CSV post-process (see §"Anti-correlation diagnostic logging" below).

TASK_742 cap (≤ 2 concurrent `train_phase2.py` per host) is respected at 2-wide on GPU 1+2.

Server-stays-under-load priority (per `feedback_server_stays_under_load.md`): the moment a GPU frees, launch the next cell from the queue. Standing 15-min-idle-slot rule per `lore/handoff_2026-05-24_764-launch.md` applies to GPU 1+2+3.

**Suggested ordering** (gives the most informative paired diff first):
1. GPU 1: sched_B_seed_7; GPU 2: sched_A_seed_7. Paired same-seed comparison ready as soon as both finish; GPU 3 evals both immediately.
2. GPU 1: sched_B_seed_13; GPU 2: sched_A_seed_13.
3. GPU 1: sched_B_seed_23; GPU 2: sched_A_seed_23.

This sequencing means after ~10 h wall the operator already has the s7 paired comparison and can early-exit if it fails the abort criteria below.

## Anti-correlation diagnostic logging — encoder gradient norm CSV

The existing Stage-B training already emits a per-epoch metrics CSV at `runs/<output_dir>/stage_b_metrics.csv` via the `csv_file.flush()` block in `train_phase2.py` around `:2174-2175` (within the Titans metrics block — the file includes `grad_norm` and `total_loss` columns).

**Diagnostic protocol**:
- Verify the CSV at `runs/782_encoder_unfreeze/<cell>/stage_b_metrics.csv` exposes a per-group grad-norm field. The current `make_stage_b_train_step` emits a global `grad_norm` (top of the optax chain). If a **per-encoder-group grad norm** is not exposed in the CSV header on first cell, log this in result.md and fall back to the global `grad_norm` — the Schedule-B-vs-Schedule-A delta in *global* grad-norm is still load-bearing at the unfreeze epoch (epoch 100, where Schedule B's grad-norm should spike upward as the encoder optax group switches from `set_to_zero` to `clip_by_global_norm + adamw`).
- **Baseline value** for "canon grad-norm": pick the median over the final 50 epochs (epochs 150-199) of `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}/stage_b_metrics.csv` global `grad_norm` column. Call this `gn_canon`.
- **Hard abort**: if at any epoch ∈ [50, 100] (Schedule A) OR [100, 150] (Schedule B) the cell's `grad_norm` > 5 × `gn_canon`, kill the cell and document the abort in `runs/782_encoder_unfreeze/<cell>/abort.md`. Move to the next cell in queue.
- **Soft flag**: if at epoch 199 the cell's median final-50-epoch `grad_norm` > 2 × `gn_canon`, write a NOTE in `result.md` ("encoder destabilizing — Δα result may be obscured by gradient pathology") but DO NOT treat the cell as failed — the deploy-truthful Δα measurement is the ship gate, grad-norm is a secondary diagnostic.

**Post-experiment CSV product**: after all 6 cells complete, the server-agent should produce `runs/782_encoder_unfreeze/grad_norm_summary.csv` with columns:
```
cell, mean_grad_norm_epochs_0_50, mean_grad_norm_epochs_50_100,
mean_grad_norm_epochs_100_150, mean_grad_norm_epochs_150_200,
gn_canon_ratio_final, gn_canon (constant per row, the reference median),
abort_epoch (NULL if cell completed)
```

This summary is the deliverable that lets the post-experiment analysis decide whether the surface is closed (Schedule B null AND grad-norm bounded) vs open (Schedule B null BUT grad-norm pathological — would warrant a follow-on at a different `encoder_lr`).

## Eval protocol per cell (on completion)

```bash
python tools/benchmark_easy.py \
  --checkpoint runs/782_encoder_unfreeze/sched_B_seed_7/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 \
  --eval-mcts-decode joint \
  --output runs/782_encoder_unfreeze/eval_sched_B_seed_7/benchmark.json
```

The eval runs on GPU 3 in parallel with subsequent training cells. Once both `sched_A_seed_7` and `sched_B_seed_7` evals complete, the paired-Δα comparison (Schedule B vs Schedule A at matched seed AND Schedule B vs 760 baseline at matched seed) is the first signal point. Operator may early-exit the queue at this point if abort criteria met.

## Early-exit protocol

Stop the queue and write a partial `result.md` if ANY of:

1. **Sanity-control failure**: Schedule A at any single seed shows |Δα vs 760 baseline at matched seed| > 2pp. This means the harness has drifted from 760 (different git commit, different env, different data file). Stop and investigate — do NOT report Schedule B numbers, as they are not comparable to the canon scorecard.
2. **Hard grad-norm abort**: any cell hits the `5 × gn_canon` ceiling between the protected epoch range above. Kill that cell, document in `abort.md`, attempt the next cell. If 2/3 Schedule B cells hit the hard abort, stop the queue entirely (the schedule is destabilizing).
3. **Schedule B catastrophic Alpha collapse on first seed**: if `sched_B_seed_7` eval shows paired Δα ≤ −10pp vs 760 baseline, stop the queue. This is the 758-style catastrophe signal; further seeds are wasted GPU.
4. **Schedule A AND Schedule B both showing >2× canon trade-rate at end-of-training on seed 7**: write a note and proceed cautiously; do not stop on this signal alone but do downgrade the result to "expected null" framing in result.md.

Continue queue otherwise. The full 6-cell queue is the smallest paired comparison that lets the operator draw a 3-seed conclusion on both Schedule A vs canon (sanity null) AND Schedule B vs canon (the falsifiable test).

## On completion — write `result.md`

Standard format mirroring `runs/760_train_mcts_sims/result.md`:

- 6-cell table of seed, Schedule, paired Δα vs canon, paired Δα vs Schedule A (matched seed), mean/STD over 3 seeds for each schedule.
- 3-seed paired-Δα summary (Schedule A vs 760, Schedule B vs 760, Schedule B vs Schedule A).
- Final 50-epoch median grad-norm per cell + the `gn_canon_ratio_final` for each.
- Anti-correlation diagnostic: per-cell trade-rate, Bal.Acc Δ vs canon.
- Verdict: PASS / NULL / REGRESSION / TIER-3-CLOSURE-OF-ENCODER-LR-SCHEDULING-SURFACE.
- If PASS (3-seed Schedule B mean ≥ +3pp vs canon, no seed < −5pp): launch 5-seed extension at seeds 31, 42. After 5-seed confirmation, ship.
- If NULL or REGRESSION: server pivots to next ship-path candidate per `lore/ship_path_recommendations_2026-05-25.md` ranked queue (next candidate after #8: #9 = class-balance + entropy joint sweep, see TASK_783 once that SPEC is drafted; or operator picks).

## Cost summary

| stage | cells | wall (serial) | wall (2-wide) |
|---|---:|---:|---:|
| Stage-B training (3 Schedule A + 3 Schedule B) | 6 | ~15 h | ~5 h |
| Eval (per cell on GPU 3, parallel) | 6 | ~3 h | (overlapped with training) |
| **Total** | 6 | ~18 h GPU-hours | ~5 h wall |

The lowest-cost variant-vs-canon probe in the ranked ship-path queue.

## References

- `tasks/TASK_782_ENCODER_UNFREEZE.md` — full task spec (sibling file).
- `lore/ship_path_recommendations_2026-05-25.md` §8 — primary spec source.
- `tasks/TASK_725_BULL_EXPERIMENTS.md` §"Experiment C: Unfreeze Encoder (Learn New Features)" (lines 100-198) — original mechanism design.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — MEDIUM risk; mitigated by surface-distinction argument in TASK_782 §"Risk".
- `~/.claude/projects/-home-lukas-Trader/memory/project_encoder_bella_class_closed.md` — also relevant; three encoder-aux failures.
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` — eval contract + GPU-scheduling pattern (style reference).
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server Phase-1 imprint baseline (informs Phase-2 wall estimates).
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` — paired-Δα reference baselines (canon a1b1 sims=32, eval-MCTS sims=16 joint).
- `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` + `train.log` — confirms canon uses `encoder_lr: 3e-05` + `encoder_unfreeze_epoch: null`, log line `Encoder UNFROZEN at lr=3e-05 (TASK_728)`.
- `phase2_config.py:247-248` — `encoder_lr`, `encoder_unfreeze_epoch` fields (verified `79430e3`).
- `train_phase2.py:957-972` — static encoder-unfreeze branch in `create_stage_b_optimizer` (verified `79430e3`).
- `train_phase2.py:2213-2220` — dynamic encoder-unfreeze branch inside epoch loop (verified `79430e3`).
- `train_phase2.py:2538-2539` — `--encoder-lr` CLI override (verified `79430e3`).
- `train_phase2_cli.py:120-121` — `--encoder-lr` `add_argument` (verified `79430e3`).
