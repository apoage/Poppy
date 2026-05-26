# 775_oracle_mask — Oracle-Mask Aux Head, Phase-1 Re-Imprint

**Status**: SPEC — DRAFT. Code already wired in tree at `79430e3`; `--oracle-mask-head` flag is the load-bearing activator (see `tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md` §"Verified code state").
**Owner**: server agent executes; workstation operator reviews.
**Reference**: `tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md`, `lore/ship_path_recommendations_2026-05-25.md` §1.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Goal

Activate the `oracle_mask` planning-side auxiliary head at Phase-1 imprint and measure whether a Stage-B retrain on top clears the canon ship gate. The mechanism class is **planning-side aux** (not classifier sharpening) — see TASK_775 §"Risk + anti-correlation check".

## Cells matrix

### Default (3-cell minimal — no code change needed)

Sweeps seed only at the config-default `oracle_mask_weight = 0.1`. Cheapest probe of the load-bearing hypothesis (head ON vs head OFF).

| cell | use_oracle_mask_head | oracle_mask_weight | seed | output dir | wall (server) |
|---|---|---:|---:|---|---:|
| w01_seed_7  | True | 0.1 (default) | 7  | `runs/775_oracle_mask/w01_seed_7`  | ~165 min |
| w01_seed_13 | True | 0.1 (default) | 13 | `runs/775_oracle_mask/w01_seed_13` | ~165 min |
| w01_seed_23 | True | 0.1 (default) | 23 | `runs/775_oracle_mask/w01_seed_23` | ~165 min |

Total wall: ~8 h serial, ~3-4 h at 2-wide on GPU 1+2.

### Extended (9-cell weight sweep — requires 4-line `--oracle-mask-weight` patch in `train_easy.py`)

Only activate if operator lands the optional `--oracle-mask-weight FLOAT` argparse flag (see TASK_775 §"Code-change required"). 3 weights × 3 seeds.

| cell | oracle_mask_weight | seed | output dir |
|---|---:|---:|---|
| w005_seed_7  | 0.05 | 7  | `runs/775_oracle_mask/w005_seed_7`  |
| w005_seed_13 | 0.05 | 13 | `runs/775_oracle_mask/w005_seed_13` |
| w005_seed_23 | 0.05 | 23 | `runs/775_oracle_mask/w005_seed_23` |
| w01_seed_7   | 0.1  | 7  | `runs/775_oracle_mask/w01_seed_7`   |
| w01_seed_13  | 0.1  | 13 | `runs/775_oracle_mask/w01_seed_13`  |
| w01_seed_23  | 0.1  | 23 | `runs/775_oracle_mask/w01_seed_23`  |
| w02_seed_7   | 0.2  | 7  | `runs/775_oracle_mask/w02_seed_7`   |
| w02_seed_13  | 0.2  | 13 | `runs/775_oracle_mask/w02_seed_13`  |
| w02_seed_23  | 0.2  | 23 | `runs/775_oracle_mask/w02_seed_23`  |

Total wall: ~25 h serial, ~12-15 h at 2-wide.

## Launch command per cell (Phase 1)

Default 3-cell variant — no `--oracle-mask-weight` flag, uses config default `0.1`:

```
python -u train_easy.py \
  --preset small \
  --bella \
  --oracle-mask-head \
  --seed 7 \
  --output-dir runs/775_oracle_mask/w01_seed_7 \
  --name 775_w01_s7 \
  --encoder-version 3 \
  --embed-dim 64 \
  --encoder-heads 8 \
  --epochs 200 \
  --batch-size 256 \
  --steps-per-epoch 200
```

Substitute `--seed 13` / `--output-dir runs/775_oracle_mask/w01_seed_13` / `--name 775_w01_s13` for the second cell, and the obvious analogue for `_s23`.

### 9-cell variant — only after `--oracle-mask-weight` patch lands

```
python -u train_easy.py \
  --preset small \
  --bella \
  --oracle-mask-head \
  --oracle-mask-weight 0.1 \
  --seed 7 \
  --output-dir runs/775_oracle_mask/w01_seed_7 \
  --name 775_w01_s7 \
  --encoder-version 3 \
  --embed-dim 64 \
  --encoder-heads 8 \
  --epochs 200 \
  --batch-size 256 \
  --steps-per-epoch 200
```

(Substitute the weight in `--oracle-mask-weight` and the output-dir / name prefix for the `w005_*` and `w02_*` cells.)

## GPU placement

- **GPU 0**: FORBIDDEN — VirtualBox-reserved (workstation memory standing rule).
- **GPU 1 + GPU 2**: 2-wide concurrent `train_easy.py`. TASK_742 cap (≤2 concurrent `train_phase2.py`) does not bind on Phase-1 (`train_easy.py`), but the operator prefers consistency.
- **GPU 3**: parallel — runs the **Phase-1 sanity-eval probe** (`tools/benchmark_easy.py` or equivalent against `phase1_best.pkl` on a smoke window) and, on Phase-2 follow-on, the eval-MCTS sims=16 joint benchmark.

Server-stays-under-load priority (per `feedback_server_stays_under_load.md`): the moment a GPU frees, launch the next cell from the queue. Standing 15-min-idle-slot rule per `lore/handoff_2026-05-24_764-launch.md` applies to GPU 1+2+3.

## Phase-1 sanity diagnostics to log

Read from the per-epoch metrics CSV the trainer already writes (no new instrumentation):

| metric | source | gate |
|---|---|---|
| `oracle_mask_accuracy` | `easy_trainer.py:299` (held-out validation slice) | > 60% (head learned anything) |
| Bull-regime classifier Bal.Acc | per-regime `dir_acc` / `policy_acc` on the validation split | Δ vs canon imprint ≥ −3pp (not worse than 3pp below `jax_ckpt_easy_phase1_xs32/easy_medium/` at the matched epoch) |
| Trade-rate (Phase-1 oracle agreement) | `oracle_hold_ratio` / `pred_hold_ratio` at `easy_trainer.py:294-295` | log only, no gate (the bonafide trade-rate gate is Phase-2) |

If `oracle_mask_accuracy ≤ 60%` across all 3 seeds at the chosen weight → head failed to learn the signal at this capacity. STOP, write `runs/775_oracle_mask/result.md` with the negative-but-informative verdict (head wired correctly but didn't train), do NOT launch Phase-2.

If bull-regime classifier Bal.Acc regresses > 3pp on all 3 seeds → mask head is dominating encoder budget. STOP, same as above.

Pass criterion: at least one weight setting (in the 3-cell variant, that's just `0.1`) shows `oracle_mask_accuracy > 60%` AND bull Bal.Acc Δ ≥ −3pp on ≥ 2/3 seeds.

## Phase-2 follow-on (only on Phase-1 sanity pass)

Picks the best-Phase-1 weight (in 3-cell variant: `0.1`) and retrains Stage-B at 3 seeds against the canon `runs/760_train_mcts_sims/` a1b1 sims=32 baselines.

Stage-B inherits `use_oracle_mask_head=True` from the Phase-1 ckpt's config via `Phase2Config(EasyConfig)`. The head's params are loaded from the Phase-1 ckpt and **frozen** during Stage-B (per `train_phase2.py:919-920, 2217-2226`). No new code; standard a1b1 Stage-B args.

Launch (per cell, illustrative — substitute seed + paths):

```
python -u train_phase2.py \
  --checkpoint runs/775_oracle_mask/w01_seed_7/phase1_best.pkl \
  --stage B \
  --seed 7 \
  --output-dir runs/775_oracle_mask/p2_w01_seed_7 \
  --name 775_p2_w01_s7 \
  --reward-scale 100 \
  --use-factored-search \
  [... other canon a1b1 flags from `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` ...]
```

Operator should diff the launching config against `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` before each cell to confirm nothing but `--checkpoint`, `--seed`, `--output-dir`, `--name` differs (and the inherited `use_oracle_mask_head: true` line).

Eval (per cell, on completion):

```
python tools/benchmark_easy.py \
  --checkpoint runs/775_oracle_mask/p2_w01_seed_7/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 \
  --eval-mcts-decode joint \
  --output runs/775_oracle_mask/eval_p2_w01_seed_7/benchmark.json
```

Phase-2 wall: 3 seeds × ~2.5 h = 7.5 h serial, ~4-5 h at 2-wide.

## Promotion gate

- **Phase-1 sanity gate** (above): mask Bal.Acc > 60% AND bull Bal.Acc Δ ≥ −3pp on ≥ 2/3 seeds, else STOP at Phase-1.
- **Phase-2 standard gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension (seeds 31, 42) → if confirmed, ship.
- **Anti-correlation diagnostic**: if Phase-2 shows Δ-Bal.Acc > 0 AND Δα < 0, the head did trigger the anti-correlation pattern via some unmodeled coupling pathway (despite the mechanism distinction). Mark as TIER-3 closure of the planning-side aux class on this stack.

## On Phase-1 completion — write result.md

Standard format:
- 3-seed table of `oracle_mask_accuracy`, per-regime Bal.Acc Δ vs canon, trade-rate.
- Sanity gate verdict (PASS / FAIL).
- If PASS: launch Phase-2 immediately (server-stays-under-load).
- If FAIL: write the negative verdict, server pivots to next ship-path candidate per `lore/ship_path_recommendations_2026-05-25.md` ranked queue.

## Cost summary

| stage | cells | wall (serial) | wall (2-wide) |
|---|---:|---:|---:|
| Phase-1, 3-cell minimal | 3 | ~8 h | ~3-4 h |
| Phase-1, 9-cell weight sweep (extended) | 9 | ~25 h | ~12-15 h |
| Phase-2 paired 3-seed | 3 | ~7.5 h | ~4-5 h |
| Eval (per Phase-2 cell on GPU 3, parallel) | 3 | ~30 min/cell | (overlapped with training) |

Default (3-cell + Phase-2): ~13-15 h GPU-hours, ~8 h wall.
Extended (9-cell + Phase-2): ~32 h GPU-hours, ~17-20 h wall.

## References

- `tasks/TASK_775_ORACLE_MASK_PHASE1_REIMPRINT.md` — full task spec.
- `lore/ship_path_recommendations_2026-05-25.md` §1 — primary spec source.
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` — eval contract + GPU scheduling pattern (style reference).
- `lore/handoff_2026-05-24_764-launch.md` — 165-min server Phase-1 imprint baseline (TASK_764 k8_s7 actual).
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` — paired-Δα reference baselines (canon a1b1 sims=32, eval-MCTS sims=16 joint).
- `mcts_trader/jax_agent/muzero_model.py:286, 417-421, 564-568, 661-663, 1003-1004, 1051-1053` — head wiring (verified `79430e3`).
- `easy_trainer.py:60-61, 215-232, 298-299` — Phase-1 loss term + metrics (verified `79430e3`).
- `train_easy.py:1885-1886` — `--oracle-mask-head` CLI activator (verified `79430e3`).
- `easy_config.py:86-88` — config defaults `use_oracle_mask_head=False`, `oracle_mask_weight=0.1` (verified `79430e3`).
- `train_phase2.py:392-393, 919-920, 2217-2226, 2650` — Phase-2 inheritance + freeze (verified `79430e3`).
- `jax_ckpt_easy_phase1_xs32/easy_medium/config.json:44` — base imprint `use_oracle_mask_head: false` (verified).
