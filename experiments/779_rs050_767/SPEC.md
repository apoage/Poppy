# 779_rs050_767 — rs050 + 767 Combination (Reward-Scale × Action-Space)

**Status**: SPEC — DRAFT — zero new code, ready to launch. Both flags shipped + production-deployed at workstation tip `79430e3`; the combination has never been jointly tested.
**Owner**: server agent executes; workstation operator reviews.
**Reference**: `tasks/TASK_779_RS050_767_COMBO.md`, `lore/ship_path_recommendations_2026-05-25.md` §5 (ship-path candidate #5 — lowest-cost on the list).
**Promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint, no seed < −5pp.

## Goal

Test whether **rs050 (`reward_scale=50`)** stacked with **767 (`disable_limit_orders=True`)** clears the canon ship gate. The two levers attack different bottlenecks of the `[[classifier-alpha-anticorrelation]]` pathway: rs050 attenuates reward magnitude (a *gradient-magnitude lever*), 767 prunes the policy + MCTS action space from 320 → effective 192 (a *search-branching-factor lever*). Both legs are CLI-orthogonal and combining them in one cell costs the same wall as either alone.

This is **Phase-2-only** — the Phase-1 imprint (`jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`) is unchanged; the comparability contract with the 760 a1b1 baselines and the 762 rs050 cells is preserved. See `tasks/TASK_779_RS050_767_COMBO.md` for the full hypothesis + verified code state + mechanism interaction analysis.

## Cells matrix

Single experimental cell, 3 seeds. Paired against three baselines (all already on disk except possibly the 767-only leg).

| cell | reward_scale | disable_limit_orders | seed | output dir | wall (server) |
|---|---:|---|---:|---|---:|
| 779-rs050_767_s7  | 50 | True | 7  | `runs/779_rs050_767/seed_7`  | ~2.5 h |
| 779-rs050_767_s13 | 50 | True | 13 | `runs/779_rs050_767/seed_13` | ~2.5 h |
| 779-rs050_767_s23 | 50 | True | 23 | `runs/779_rs050_767/seed_23` | ~2.5 h |

**Baselines (already trained + eval'd, reused for paired-Δα):**

| baseline | reward_scale | disable_limit_orders | source path |
|---|---:|---|---|
| 760-a1b1 (canon absolute) | 100 | False | `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` + `eval_baseline_paired_seed_*` |
| 762-rs050 (rs050 isolated leg) | 50 | False | `runs/762_reward_shaping/rs050_seed_{7,13,23}` + `eval_rs050_seed_*` |
| 767-3action (767 isolated leg) | 100 | True | reuse 767 standalone result whenever it lands per `experiments/767_action_space/SPEC.md` (status `READY-TO-LAUNCH`; in-flight or queued) |

If the 767-3action standalone is not yet on disk at TASK_779 launch time, the mechanism-interaction gate (combo vs best of {rs050-only, 767-only}) falls back to comparing against **rs050-only** alone until 767-3action lands.

Total wall: ~7.5 h serial, ~4-5 h at 2-wide on GPU 1 + GPU 2. **Lowest-cost candidate on the ship-path list.**

## Launch command per cell

The arg layout below is verified against `train_phase2_cli.py` at workstation tip `79430e3`. Every flag is canon a1b1 (matched to `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` field-by-field) except the two intervention flags (`--reward-scale 50` and `--disable-limit-orders`) and the per-seed/output overrides.

```
CUDA_VISIBLE_DEVICES=$G python -u train_phase2.py \
  --stage B \
  --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 \
  --max-runs-per-group 10 \
  --factored \
  --factored-temperature 0.1 \
  --no-reset-value-head \
  --value-target-mode nstep \
  --td-discount 0.95 \
  --value-weight 0.5 \
  --bc-weight-min 0.05 \
  --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 \
  --encoder-lr 3e-5 \
  --encoder-version 3 \
  --embed-dim 64 \
  --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse \
  --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --reward-scale 50 \
  --disable-limit-orders \
  --seed $S \
  --output-dir runs/779_rs050_767/seed_$S \
  --name 779_rs050_767_s$S
```

Substitute `$G` ∈ {1, 2, 3} (NOT 0 — VirtualBox-reserved) and `$S` ∈ {7, 13, 23} per cell.

**Flag-name verification (against `train_phase2_cli.py` at `79430e3`):**

| flag | argparse line | verified |
|---|---:|:---:|
| `--stage` | `:24` | ✓ |
| `--epochs` | `:26` | ✓ |
| `--batch-size` | `:28` (omitted — config default 256, matches 760) | ✓ |
| `--steps-per-epoch` | `:30` (omitted — config default 200, matches 760) | ✓ |
| `--seed` | `:32` | ✓ |
| `--periods-file` | `:36` | ✓ |
| `--factored` | `:53` | ✓ |
| `--factored-temperature` | `:55` | ✓ |
| `--no-reset-value-head` | `:59` | ✓ |
| `--bc-weight-min` | `:61` | ✓ |
| `--bc-max-trade-weight` | `:68` | ✓ |
| `--reward-scale` | `:71-73` | ✓ |
| `--disable-limit-orders` | `:89-98` (`BooleanOptionalAction`) | ✓ |
| `--direction-weight` | `:113` | ✓ |
| `--encoder-lr` | `:120` | ✓ |
| `--rank-weight` | `:133` | ✓ |
| `--rank-loss` | `:135` | ✓ |
| `--aux-price-weight` | `:147` | ✓ |
| `--value-target-mode` | `:253-256` | ✓ |
| `--td-discount` | `:259` | ✓ |
| `--n-slots` | `:275` | ✓ |
| `--max-runs-per-group` | `:277` | ✓ |
| `--output-dir` | `:281` | ✓ |
| `--name` | `:283` | ✓ |
| `--embed-dim` | `:301` | ✓ |
| `--encoder-heads` | `:303` | ✓ |
| `--encoder-version` | `:306` | ✓ |
| `--value-weight` | `:240` | ✓ |
| `--checkpoint` | `:20` | ✓ |

Pre-launch sanity: diff the resulting `stage_b_config.json` against `runs/760_train_mcts_sims/baseline_paired_seed_7/stage_b_config.json` — only `reward_scale=50` (vs 100), `disable_limit_orders=true` (vs false), and `seed` should differ.

## GPU placement

- **GPU 0**: FORBIDDEN — VirtualBox-reserved (workstation memory standing rule, `MEMORY.md`).
- **GPU 1 + GPU 2**: 2-wide concurrent `train_phase2.py`. TASK_742 cap (≤2 concurrent `train_phase2.py` per host) bound.
- **GPU 3**: parallel — eval `tools/benchmark_easy.py` on each cell as soon as `phase2_stageB_best.pkl` is written. `benchmark_easy.py` is inference, not `train_phase2.py`, so it does not count against the TASK_742 cap.

Server-stays-under-load priority per `feedback_server_stays_under_load.md` — as each training cell finishes, immediately launch the next + its eval. Standing 15-min-idle-slot rule per `lore/handoff_2026-05-24_764-launch.md`.

## Launch order (keep all 3 usable GPUs under load)

1. **Training** (GPU 1 + GPU 2, 2-wide): `s7` first for an early 1-seed read of the combo's trade-rate behavior, then `s13`, then `s23`.
   - GPU 1: `s7` → `s23` (or whichever is free first)
   - GPU 2: `s13`
2. **Eval** (GPU 3, parallel): as each training cell writes `phase2_stageB_best.pkl`, launch its eval on GPU 3 immediately. Do not batch to the end.

## Eval command per cell

Canon eval-MCTS sims=16 joint contract — bit-identical to the 760 + 762 + 767 launches:

```
python tools/benchmark_easy.py \
  --checkpoint runs/779_rs050_767/seed_$S/phase2_stageB_best.pkl \
  --periods-file config/benchmark_periods.json \
  --eval-mcts-sims 16 \
  --eval-mcts-decode joint \
  --output runs/779_rs050_767/eval_seed_$S/benchmark.json
```

**Flag-name verification (against `tools/benchmark_easy.py` at `79430e3`):**

| flag | argparse line | verified |
|---|---:|:---:|
| `--checkpoint` | (consumed via `--checkpoint` arg-tree, present in main entry) | ✓ |
| `--periods-file` | | ✓ |
| `--eval-mcts-sims` | `:1213` | ✓ |
| `--eval-mcts-decode` | `:1215` | ✓ |
| `--output` | | ✓ |

767's `disable_limit_orders` mask is **auto-detected** from the checkpoint config by `benchmark_easy.py:931-942` and threaded through eval-MCTS at `:544-548` + `:633-638` — no extra CLI needed at the benchmark step. Eval-time mask consistency is guaranteed by checkpoint config inheritance.

## Diagnostics to log

Read from the per-epoch `stage_b_metrics.csv` (no new instrumentation; everything is already wired):

| metric | source | gate / interpretation |
|---|---|---|
| `mean_trade_rate` | `train_phase2.py:1543-1544` (in CSV per TASK_767 logging) | Compare vs canon a1b1 (~0.40 limit channel + market channel) — combo expected closer to ~0.20-0.25 (rs050 reduces over-trade + 767 caps the limit channel at 0.0). |
| `limit_action_fraction` | same | Must be 0.000 (sanity that 767 mask is active). Canon ~0.398. |
| `policy_entropy` | per-epoch policy entropy (joint or factored) | Anti-correlation diagnostic — combo should NOT show entropy collapse alongside Bal.Acc rise. |
| `dir_acc` | per-epoch direction-head accuracy | Bal.Acc proxy — sharp rise vs canon + Δα < 0 → anti-correlation triggered. |
| eval-MCTS sims=16 joint α per period | `runs/779_rs050_767/eval_seed_*/benchmark.json` | per-period Δα table (vs 760 a1b1) + 3-seed mean. |
| paired Δα reduction CSV | derived from the 3-cell eval JSON × baseline JSON | Final table — one row per period × seed, columns: `α_combo`, `α_760`, `Δα = α_combo − α_760`, `Δ-Bal.Acc`. |

**The `result.md` (written on Wave completion)** must report:
1. 3-seed table of per-period α (combo, 760 canon, rs050-only, 767-only if available).
2. Paired Δα table: combo vs 760 canon (the standard +3pp gate read), combo vs rs050-only (mechanism interaction read), combo vs 767-only (mechanism interaction read).
3. Anti-correlation diagnostic — Δ-Bal.Acc × Δα plot or table at 3-seed mean.
4. Mechanism interaction verdict: super-additive / additive / sub-additive / null.
5. Verdict: PASS (5-seed extension authorized) / FAIL.

## Promotion gate (three, in order, early-exit on FAIL)

1. **Anti-correlation diagnostic (early-exit)**: combo must NOT show Δ-Bal.Acc > 0 AND Δα < 0 at 3-seed mean. If triggered → TIER-3 closure of joint reward+search mechanism class on this stack. Stop.

2. **Mechanism-interaction gate**: Δα_combo > max(Δα_rs050, Δα_767). If FAIL, combination is redundant with one of the legs alone — the campaign learns the over-trade pathway has one degree of freedom on this stack. Stop the campaign branch, log result.md, server pivots per ranked queue.

3. **Standard promotion gate (canon)**: paired Δα ≥ +3pp vs 760 a1b1 sims=32 under eval-MCTS sims=16 joint AND no seed < −5pp → 5-seed extension (seeds 31, 42).

5-seed extension only if 3-seed clears gate #3. If 5-seed confirms (mean Δα ≥ +3pp, no seed < −5pp), **ship**.

## On 3-seed completion — write result.md

Standard format:
- 3-seed table of `mean_trade_rate`, `limit_action_fraction`, eval-MCTS sims=16 joint α per period, Δα paired vs each baseline (760, rs050, 767).
- Mechanism interaction verdict (additive / sub-additive / super-additive / null).
- Anti-correlation diagnostic (Δ-Bal.Acc × Δα).
- Standard gate verdict + next-step instruction (5-seed extension OR pivot per ranked queue).
- If FAIL: 1-sentence summary of *which* gate failed and *what* the closure implies (joint mechanism class status).

## Cost summary

| stage | cells | wall (serial) | wall (2-wide) |
|---|---:|---:|---:|
| Training | 3 | ~7.5 h | ~4-5 h |
| Eval (per cell on GPU 3, parallel) | 3 | ~30 min/cell | (overlapped with training) |
| **Total** | | **~5 h wall, ~8-9 GPU-hours** | |

Lowest-cost candidate on the ship-path list per `lore/ship_path_recommendations_2026-05-25.md` Recommended execution order (the recommended first launch).

## Contract reminders

- ≤2 concurrent `train_phase2.py` per host (TASK_742).
- GPU 0 forbidden (VirtualBox-reserved). GPU 1/2/3 only.
- No commits to master from server; no force-push; server pulls, never pushes back. Sub-agent deploy contract: DO NOT call `deploy.sh`. Workstation operator masters server-side writes.
- This SPEC is zero-code — `train_phase2.py` + `train_phase2_cli.py` + `tools/benchmark_easy.py` are unchanged from workstation tip `79430e3`.

## References

- `tasks/TASK_779_RS050_767_COMBO.md` — full task spec, verified code state, mechanism interaction analysis.
- `lore/ship_path_recommendations_2026-05-25.md` §5 — primary spec source for this TASK (the cheapest candidate on the list, with the "rs050 + 767" combination call-out + LOW anti-correlation risk assessment).
- `tasks/TASK_767_ACTION_SPACE_RESTRUCTURE.md` — original 767 design.
- `experiments/767_action_space/SPEC.md` — 767 standalone SPEC, includes oracle-audit (0% limit usage) + bug-fix history (TASK_771/F1 P0).
- `experiments/762_reward_shaping/SPEC.md` — 762 rs050 wave-1 SPEC.
- `runs/762_reward_shaping/AUTHORIZATION_2026-05-21.md` + `runs/762_reward_shaping/AUTHORIZATION_WAVE2_2026-05-22.md` — style references for AUTHORIZATION/SPEC format.
- `runs/762_reward_shaping/wave1_status.md` + `lore/notes_2026-05-23.md` §167 — rs050 W1 +0.27pp result.
- `runs/762_reward_shaping/rs050_seed_{7,13,23}` (+ `eval_rs050_seed_*`) — rs050-only baseline checkpoints + eval-MCTS sims=16 joint benchmarks.
- `runs/760_train_mcts_sims/baseline_paired_seed_{7,13,23}` (+ `eval_baseline_paired_seed_*`) — canon a1b1 sims=32 absolute baseline.
- `runs/760_train_mcts_sims/result.md` — 760 closure verdict (train-time sims=64 NULL → structural pivot activated).
- `phase2_config.py:30, 45` — `reward_scale=100.0` + `disable_limit_orders=False` config fields (verified at `79430e3`).
- `train_phase2.py:637, 679` — rs050 reward-scale multiplication (verified at `79430e3`).
- `train_phase2.py:1060-1090, 1742-1762` — 767 mask build + MCTS expansion guard (verified at `79430e3`).
- `train_phase2.py:2348-2349, 2355-2356` — CLI override plumbing (verified at `79430e3`).
- `train_phase2_cli.py:71-73, 89-98` — `--reward-scale` + `--disable-limit-orders` argparse (verified at `79430e3`).
- `mcts_trader/jax_agent/factored_search.py:190-197` — factored-search Q-value mask (TASK_771/F1 mean-shift fix applied).
- `mcts_trader/jax_agent/path_solver.py:406, 473` — oracle 0% limit-action usage.
- `tools/benchmark_easy.py:931-942, 544-548, 633-638, 1213-1215` — eval-time mask auto-detection + eval-MCTS CLI flags.
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — over-trade pathway evidence (the anti-correlation pattern this TASK avoids triggering).
- `~/.claude/projects/-home-lukas-Trader/memory/project_single_mechanism_exhausted.md` — single-mechanism class closed verdict (the rationale for testing 2-mechanism combinations).
- `lore/notes_2026-05-23.md` §165 — TASK_771 multi-agent QA pass (the 767 P0 catch that makes the 767 leg trustable).
