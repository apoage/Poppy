# 784_smoothed_oracle — Smoothed-Oracle Value Target (paired 3-seed × 3-λ)

**Status**: SPEC — DRAFT. Code already wired in this worktree
(`worktree-agent-ad1e54ad52f9aeb0f`); `--oracle-smoothing` is the
load-bearing activator.
**Owner**: server agent executes; workstation operator reviews.
**Reference**: `tasks/TASK_784_SMOOTHED_ORACLE_VALUE_TARGET.md`,
`lore/ship_path_recommendations_2026-05-25.md` §10,
`tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.3.
**Promotion gate (canon)**: paired Δα ≥ +3pp vs `runs/760_train_mcts_sims/`
a1b1 sims=32 baselines under eval-MCTS sims=16 joint, no seed < −5pp.
**Mechanism diagnostic gate**: STD across seeds at best-λ cell <
746's 5-seed STD 2.28pp.

## Goal

Activate EWMA smoothing on `oracle_cum_pnl` per-step deltas (TASK_784) and
sweep λ ∈ {0.1, 0.3, 0.5} × seeds {7, 13, 23}. Measure whether smoothing
the per-window peakiness of the raw oracle PnL signal reduces the seed-
variance signature of 746 (STD 2.28pp) and lifts paired Δα above the +3pp
ship gate. Likely-null per ship-path memo "Low" EV ranking; falsified
cheaply.

## Cells matrix

| cell | oracle_smoothing | seed | output dir | wall (server, single GPU) |
|---|---:|---:|---|---:|
| l01_seed_7  | 0.1 | 7  | `runs/784_smoothed_oracle/l01_seed_7`  | ~2.5 h |
| l01_seed_13 | 0.1 | 13 | `runs/784_smoothed_oracle/l01_seed_13` | ~2.5 h |
| l01_seed_23 | 0.1 | 23 | `runs/784_smoothed_oracle/l01_seed_23` | ~2.5 h |
| l03_seed_7  | 0.3 | 7  | `runs/784_smoothed_oracle/l03_seed_7`  | ~2.5 h |
| l03_seed_13 | 0.3 | 13 | `runs/784_smoothed_oracle/l03_seed_13` | ~2.5 h |
| l03_seed_23 | 0.3 | 23 | `runs/784_smoothed_oracle/l03_seed_23` | ~2.5 h |
| l05_seed_7  | 0.5 | 7  | `runs/784_smoothed_oracle/l05_seed_7`  | ~2.5 h |
| l05_seed_13 | 0.5 | 13 | `runs/784_smoothed_oracle/l05_seed_13` | ~2.5 h |
| l05_seed_23 | 0.5 | 23 | `runs/784_smoothed_oracle/l05_seed_23` | ~2.5 h |

Total: 9 Stage-B runs, ~22.5 h serial, **~12-14 h wall at 2-wide** on
GPU 1+2.

**No baseline runs needed** — the comparison baseline is
`runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` (canon a1b1
sims=32, already executed). The smoothing=0 reference is
`runs/746_value_oracle/oracle_seed_{7,13,23}` (also already executed).

## Launch command per cell

Mirrors the 746 launch (`experiments/746_value_oracle/SPEC.md`) with the
EWMA flag added. Example for `l01_seed_7`:

```bash
cd ~/Trader && source trader_env/bin/activate

COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head \
  --value-target-mode oracle \
  --oracle-smoothing 0.1 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1"

mkdir -p runs/784_smoothed_oracle/l01_seed_7
screen -d -m -S so_l01_s7 bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -u train_phase2.py $COMMON \
  --seed 7 \
  --output-dir runs/784_smoothed_oracle/l01_seed_7 \
  --name 784_l01_s7 2>&1 \
  | tee runs/784_smoothed_oracle/l01_seed_7/train.log; \
  echo SO_L01_S7_DONE > /tmp/so_l01_s7_done.flag"
```

Substitute `--oracle-smoothing` ∈ {0.1, 0.3, 0.5}, `--seed` ∈ {7, 13, 23},
the GPU index, output dir, and screen-tag for each cell.

## GPU placement

- **GPU 0**: FORBIDDEN — VirtualBox-reserved.
- **GPU 1 + GPU 2**: 2-wide concurrent `train_phase2.py`. TASK_742 rule
  enforced (≤2 per host).
- **GPU 3**: parallel eval. Each completed training kicks an
  eval-MCTS sims=16 joint benchmark on GPU 3 — overlaps with the next
  training wave.

Server-stays-under-load priority: launch the next cell from the queue the
instant a GPU frees. The 15-min idle-slot rule from
`lore/handoff_2026-05-24_764-launch.md` applies.

### Wave schedule (illustrative 2-wide)

```
Wave 1 (T=0:00, ~2.5h):
  GPU 1: l01_seed_7        GPU 2: l01_seed_13
Wave 2 (T=2:30, ~2.5h):
  GPU 1: l01_seed_23       GPU 2: l03_seed_7
Wave 3 (T=5:00, ~2.5h):
  GPU 1: l03_seed_13       GPU 2: l03_seed_23
Wave 4 (T=7:30, ~2.5h):
  GPU 1: l05_seed_7        GPU 2: l05_seed_13
Wave 5 (T=10:00, ~2.5h):
  GPU 1: l05_seed_23       GPU 2: idle / picks up next ship-path candidate

Eval on GPU 3, parallel with training, ~30 min per cell.
```

## Variance diagnostic logging (load-bearing)

The mechanism diagnostic gate requires per-cell STD across seeds. At eval
time, capture:

1. **Per-seed Alpha** (the canonical scorecard column) per cell — write to
   `runs/784_smoothed_oracle/eval_l{λ}_seed_{s}/benchmark.json`.
2. **Per-seed paired Δα** vs the matched canon baseline
   (`runs/760_train_mcts_sims/baseline_paired_seed_{s}`).
3. **Across-seed STD of paired Δα** at each λ cell — must be tracked
   explicitly in the result.md.

The result.md `Verdict` section must include:

| λ | mean Δα | STD | seed range | 746 STD comparison |
|---|--------:|----:|-----------:|-------------------:|
| 0.1 | … | … | … vs … | … vs 2.28pp |
| 0.3 | … | … | … vs … | … vs 2.28pp |
| 0.5 | … | … | … vs … | … vs 2.28pp |

If any λ cell shows STD < 2.28pp the **variance hypothesis is partially
confirmed**, even if the mean fails the +3pp gate. Document this as a
mechanism-class-narrowing finding — it tells us the next planning-side
candidate (e.g., TASK_777 mixture) should focus on the lift side.

## Eval protocol

Per completed training, on GPU 3:

```bash
python tools/benchmark_easy.py \
  --checkpoint runs/784_smoothed_oracle/l01_seed_7/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 \
  --eval-mcts-decode joint \
  --output runs/784_smoothed_oracle/eval_l01_seed_7/benchmark.json
```

Comparison: paired against `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}/`
under the same eval-MCTS settings.

## Promotion gate

- **Pass primary**: at least one λ cell shows paired mean Δα ≥ +3pp AND no
  seed < −5pp. → 5-seed extension (seeds 31, 42 paired against
  `runs/752_value_oracle_baseline_pair/seed_{31,42}` baselines or canon
  equivalents); if confirmed, ship.
- **Pass mechanism diagnostic** (independent): best-λ STD < 2.28pp. Even
  on primary-gate-fail this is publishable as a narrowing finding.
- **Fail both**: write the negative verdict with per-λ table; close the
  "oracle-side variance reduction" sub-class together with TASK_777's
  verdict if both end null. Pivot to next ship-path candidate.

## On completion — write `runs/784_smoothed_oracle/result.md`

Standard format:
- Per-cell table: Bal.Acc, Alpha, ModelPnL, beats, pos α.
- Paired Δα vs `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}`.
- Per-λ STD across seeds (the load-bearing diagnostic).
- Verdict (PROMOTE / 5-SEED / HOLD / REJECT) per the promotion gate.
- Anti-correlation diagnostic: if any cell shows ΔBal.Acc > 0 AND Δα < 0,
  flag it explicitly — the smoothing was supposed to be planning-side
  only; unexpected encoder/classifier interaction is reportable.

## Cost summary

| stage | cells | wall (serial) | wall (2-wide) |
|---|---:|---:|---:|
| Stage-B paired 3-seed × 3-λ | 9 | ~22.5 h | ~12-14 h |
| Eval (per cell on GPU 3, parallel) | 9 | ~30 min/cell | (overlapped) |

Total: ~22.5 GPU-h, ~12-14 h wall.

## References

- `tasks/TASK_784_SMOOTHED_ORACLE_VALUE_TARGET.md` — full task spec + code
  state verification.
- `lore/ship_path_recommendations_2026-05-25.md` §10 — primary spec source.
- `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.3 — original
  smoothed-oracle design (TASK_756c stub).
- `runs/746_value_oracle/result.md` — STD 2.28pp baseline + per-seed
  paired Δα reference (s7 −3.99pp, s23 +8.69pp).
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` — paired-Δα
  reference baselines (canon a1b1 sims=32, eval-MCTS sims=16 joint).
- `experiments/746_value_oracle/SPEC.md` — launch-command template.
- `train_phase2.py:640-665` (post-edit) — EWMA precompute.
- `train_phase2_cli.py:252-272` (post-edit) — `--oracle-smoothing` flag.
- `phase2_config.py:206-225` (post-edit) — `oracle_smoothing` field.
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` —
  campaign context (single-mechanism class closed).
- `lore/handoff_2026-05-24_764-launch.md` — 2-wide GPU scheduling pattern.
