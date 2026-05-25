# 777_mixture_value — α·oracle + (1−α)·nstep Value Target (paired 3-seed × 3-α)

**Status**: READY-TO-LAUNCH-AFTER-SMOKE. Code shipped in workstation worktree `worktree-agent-ad35b376f93d08f62` on 2026-05-25 (see TASK_777 implementation log). Flag-gated `--value-target-mode mixture`, default OFF; canon path bit-identical. Deploys safe mid-campaign.
**Owner**: workstation implements + smoke-tests + deploys; server agent executes; workstation reviews.
**Reference**: `tasks/TASK_777_MIXTURE_VALUE_TARGET.md`, `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` §4.1, `lore/ship_path_recommendations_2026-05-25.md` §3.

---

## Goal

Test whether a convex blend of the oracle value target (746, paired Δα +0.72pp, 5-seed STD 2.28pp) and the canon nstep value target (a1b1) lifts paired Δα beyond 746's +0.72pp. The hypothesis is that 746's +0.72pp signal is **variance-limited**, not bias-limited: mixing with the lower-variance nstep leg should reduce per-seed STD and raise the mean.

This is the **only positive paired-Δα signal** that survives proper 5-seed pairing on this stack (per `runs/746_value_oracle/result.md:160-168` + `project_single_mechanism_exhausted.md`); the mixture is the cheapest extension of that signal — same mechanism class, no encoder/classifier touch, ~50 LOC.

---

## Hypothesis (variance-limited oracle)

746 STD across 5 seeds: 2.28pp. Per-seed paired Δα spread: `(−3.99, +8.69)pp` — high variance vs marginal mean. If the bias is correct (oracle V* is the right signal in trending regimes — bear/ftx_crash gains in 746's per-period breakdown) but the variance is the cap, an interior α should produce **lower STD AND higher mean** than 746. Falsifiable predictions in TASK_777 §Hypothesis.

If sweep is monotone-in-α with peak at α=1.0 → 746 was best-in-family (variance is bias-driven, mixing dilutes signal); informative non-promote. If sweep is flat → value-target mechanism class closed.

---

## Prerequisites

- [x] Phase-1 ckpt: `jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl`
- [x] `--value-target-mode mixture` wired in `train_phase2.py:1292-1332` (TASK_777, this worktree)
- [x] `oracle_values` + `nstep_rewards` already populated in `SegmentBatchGenerator` batches (`train_phase2.py:618-633`); no batch-builder change needed
- [x] `value_mixture_alpha` field + `[0, 1]` validator in `Phase2Config` (this worktree)
- [x] 760 a1b1 sims=32 baselines (seeds 7, 13, 23) at `runs/760_train_mcts_sims/a1b1_sims32_seed_*/`
- [x] 746 oracle (seeds 7, 13, 23, 31, 42) at `runs/746_value_oracle/oracle_seed_*/` — variance reference, NOT the paired baseline
- [ ] Bit-identity smoke-gate verification on a GPU (gates 1+2; see §Smoke below)

---

## Smoke gates (run BEFORE production cells)

Two gates per TASK_756 §7a + the 769 protocol. Both must PASS or the patch is rolled back.

### Gate 1 — `value_target_mode=oracle` bit-identical post-patch

Pre-patch baseline (`master` branch) vs post-patch worktree, identical args, same seed:

```bash
cd ~/Trader && source trader_env/bin/activate
# Run BOTH: from worktree (post-patch) and from master (pre-patch).
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --stage B --epochs 1 --steps-per-epoch 10 --batch-size 64 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode oracle \
  --td-discount 0.95 --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --seed 7 --output-dir runs/000_smoke/777_gate1_<branch> \
  --name 000_777_gate1_<branch> 2>&1 | tee runs/000_smoke/777_gate1_<branch>/train.log
```

**Pass**: max abs diff ≤ 1e-6 on `value_loss / td_target_mean / td_target_std / grad_norm / pa_mean / pa_std` across all 10 step rows.

### Gate 2 — `value_target_mode=mixture --value-mixture-alpha 1.0` reproduces oracle bit-identical (CRITICAL)

Same args as Gate 1 but swap `--value-target-mode oracle` → `--value-target-mode mixture --value-mixture-alpha 1.0`. Compare against the post-patch oracle run from Gate 1.

**Pass**: max abs diff ≤ 1e-6 on the same six columns. By IEEE 754, `1.0 * x + 0.0 * y == x` exactly for finite `x, y` — the gate verifies the JIT and PopArt-statistics paths preserve this.

**Why this is critical**: a sign or clip-order bug in the mixture branch's oracle leg would surface as a non-zero diff here. TASK_756 §7a calls this out explicitly — proceed to production only after this gate is green.

### Gate 3 (bonus) — `--value-mixture-alpha 0.0` reproduces nstep

Same protocol, comparing against a `--value-target-mode nstep` post-patch run. PASS = same six-column diff ≤ 1e-6.

**Static-trace evidence (worktree)**: All three gates PASS by static-trace + IEEE 754 argument. The mixture branch's oracle leg is a line-for-line mirror of the dedicated oracle branch; the nstep leg mirrors the dedicated nstep branch. GPU rerun deferred to server smoke per the 769 protocol.

---

## Resources

- Workstation 3090: smoke gates (~30 min wall total). Server 3× 3090 production.
- Server TASK_742 rule: ≤2 concurrent `train_phase2.py` per host. GPU 0 = VirtualBox-reserved.
- GPU 1+2: training; GPU 3: eval (concurrent with training per ship-path standing priority on filling GPU 3 with evals).
- Per-cell training wall: ~2.5h × 3 seeds × 3 α = 22.5h serial → **~12-14h at 2-wide on GPUs 1+2**.
- Eval wall: 9 variant evals + reuse of 760 baseline evals + reuse of 746 evals (already cached) → 9 × ~30 min = ~4.5h at single-stream, or **~2h** at 2-wide on GPU 3.

**Total**: ~14-16h server wall (training + eval).

---

## Plan

3 α × 3 seeds = 9 variant cells. Two reference baselines reused — no recompute.

| variant | value_target_mode | α | seeds | source |
|---|---|---:|---|---|
| **a1b1 baseline** | nstep | n/a | 7, 13, 23 | reuse `runs/760_train_mcts_sims/a1b1_sims32_seed_*/` |
| **746 reference** | oracle | n/a | 7, 13, 23 (+ 31, 42 for STD) | reuse `runs/746_value_oracle/oracle_seed_*/` |
| 777-α0.25 | mixture | 0.25 | 7, 13, 23 | new |
| 777-α0.50 | mixture | 0.50 | 7, 13, 23 | new |
| 777-α0.75 | mixture | 0.75 | 7, 13, 23 | new |

α∈{0.0, 1.0} explicitly NOT in production sweep (smoke-only — they reproduce a1b1 / 746 already cached).

---

## Wave plan (2-wide, GPUs 1 + 2, GPU 3 eval-concurrent)

```
Wave 1 (T=0:00, ~2.5h):
  GPU 1: 777-α0.25_s7
  GPU 2: 777-α0.50_s7
  GPU 3: idle (or pick up backlog eval — per "server stays under load" feedback)

Wave 2 (T=2:30, ~2.5h):
  GPU 1: 777-α0.75_s7
  GPU 2: 777-α0.25_s13
  GPU 3: eval 777-α0.25_s7 + 777-α0.50_s7

Wave 3 (T=5:00, ~2.5h):
  GPU 1: 777-α0.50_s13
  GPU 2: 777-α0.75_s13
  GPU 3: eval 777-α0.75_s7 + 777-α0.25_s13

Wave 4 (T=7:30, ~2.5h):
  GPU 1: 777-α0.25_s23
  GPU 2: 777-α0.50_s23
  GPU 3: eval 777-α0.50_s13 + 777-α0.75_s13

Wave 5 (T=10:00, ~2.5h):
  GPU 1: 777-α0.75_s23
  GPU 2: idle (or pick up a different campaign cell)
  GPU 3: eval 777-α0.25_s23 + 777-α0.50_s23

T=12:30 — final eval pass:
  GPU 3: eval 777-α0.75_s23 (~30 min) → result.md
```

Net: ~13h training + ~30 min for final eval ≈ **~13.5h server wall**.

---

## Commands

```bash
cd ~/Trader && source trader_env/bin/activate

# COMMON args (mirror of `tools/run_a1b1_stacked.sh` canon + flag swap):
COMMON="--stage B --epochs 200 \
  --periods-file config/foxglove_periods.json \
  --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode mixture \
  --value-weight 0.5 \
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1 \
  --td-discount 0.95"
# NOTE: nstep leg uses --td-discount (canon 0.95). Oracle leg ignores it.
# Both legs are consumed in the mixture branch — pass it.

# For ALPHA in {0.25, 0.50, 0.75} and SEED in {7, 13, 23}, GPU in {1, 2} (2-wide):
ALPHA_TAG=$(echo $ALPHA | tr -d '.')  # 0.25 → 025
RUN_DIR=runs/777_mixture_value/mixture_a${ALPHA_TAG}_seed_${SEED}
mkdir -p ${RUN_DIR}
screen -d -m -S vm_a${ALPHA_TAG}_s${SEED} bash -c "source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=<GPU> python -u train_phase2.py $COMMON \
  --value-mixture-alpha ${ALPHA} \
  --seed ${SEED} \
  --output-dir ${RUN_DIR} \
  --name 777_vm_a${ALPHA_TAG}_s${SEED} 2>&1 \
  | tee ${RUN_DIR}/train.log; \
  echo VM_A${ALPHA_TAG}_S${SEED}_DONE > /tmp/vm_a${ALPHA_TAG}_s${SEED}_done.flag"
```

Eval pattern matches 746's: `tools/benchmark_easy.py --checkpoint <run_dir>/phase2_stageB_best.pkl --eval-mcts-sims 16 --json <eval_dir>/benchmark.json` against the standard 6-period benchmark set. **eval-MCTS sims=16 joint** is the canon scorecard per `project_single_mechanism_exhausted.md`.

---

## Success criteria

- [ ] Gate 1 + Gate 2 PASS on GPU (max abs diff ≤ 1e-6 vs pre-patch oracle).
- [ ] All 9 training runs complete 200 epochs cleanly. No NaN, no rank-head deadlocks.
- [ ] **Paired Δα** vs 760 a1b1 sims=32 baseline, eval-MCTS sims=16 joint, per α:
  - `mean(Δα) ≥ +3pp AND no seed < −5pp` at best α → **CLEAR WIN**. 5-seed extension (s31, s42). Promote if extension also clears.
  - `mean(Δα) ∈ (−5pp, +3pp)` → **inconclusive / informative**. Document per-α split. If sweep is monotone in α with peak at α=0.75, hypothesis is the bias-limited interpretation → close mixture roadmap.
  - `mean(Δα) < −5pp for any α` → **early-exit** the cell (one seed below the floor closes that α; cell-level abort if pattern repeats).
- [ ] **Mechanism diagnostic (not gate)**: STD across seeds at best-α cell vs 746's STD 2.28pp.
  - STD < 2.28pp AND mean > +0.72pp → **variance-limited hypothesis confirmed**, document as the cleanest variance-of-variance evidence on this stack.
  - STD ≥ 2.28pp → mixing didn't reduce variance; α slides between two known points. Sub-hypothesis closed.
- [ ] **Anti-correlation early-exit (per cell)**: if E50 of seed 7 shows `policy_loss < 0.7 × baseline_E50` AND `trade_rate > 2 × canon`, flag + abort cell. Risk LOW (value-target class is the existence-proof counter-example), but the protocol stays. Document any cell that trips this in `result.md`.

---

## Output expectations

```
runs/777_mixture_value/
├── mixture_a025_seed_{7,13,23}/        train.log, phase2_stageB_best.pkl, stage_b_metrics.csv, stage_b_config.json
├── mixture_a050_seed_{7,13,23}/        (same)
├── mixture_a075_seed_{7,13,23}/        (same)
├── eval_mixture_a025_seed_{7,13,23}/   benchmark.json, benchmark.log (eval-MCTS sims=16 joint)
├── eval_mixture_a050_seed_{7,13,23}/   (same)
├── eval_mixture_a075_seed_{7,13,23}/   (same)
└── result.md                           paired Δα × per-α table + per-period split + trade-rate + STD vs 746 + per-cell anti-correlation status
```

`result.md` template should mirror `runs/746_value_oracle/result.md` — paired Δα table (per seed, per α), per-period breakdown averaged across seeds, trade-rate signature on `bull` period (the canonical over-trade-collapse diagnostic), and the explicit variance-vs-746 comparison.

---

## Diagnostics required in `result.md`

1. Paired Δα table: 3 seeds × 3 α + AVG + STD (per α).
2. Per-period Δα table averaged across 3 seeds (6 regimes: bull / bear / sideways / ftx_crash / etf_approval / boring_week).
3. Trade-rate per period (compare against canon trade-rate and 746's per-period trade-rate signature).
4. STD-across-seeds vs 746's STD 2.28pp at best-α cell — the variance-limited-hypothesis diagnostic.
5. Training-loss + Bal.Acc deltas (anti-correlation check; ΔBal.Acc should NOT be strongly positive at best-α — value-target class precedent is ΔBal.Acc ≤ 0 with ΔAlpha > 0).
6. If sweep is monotone or flat in α, the verdict explicitly notes which sub-hypothesis is closed (bias-limited 746-near-optimal vs flat / mechanism dead).

---

## Notes

- This SPEC is independent of TASK_756b (longer oracle horizon — `--oracle-horizon` flag added in the same patch but not consumed in 777 production cells; available for a follow-up).
- This SPEC is independent of TASK_784 (TASK_756c smoothed oracle — separate task, separate ~15 LOC batch-builder change, not in 777's patch).
- Stacking with TASK_734 flow_matching is the natural Tier-3 follow-up if 777 promotes. Out of scope here.
- The `oracle_cum_pnl` precompute is unchanged; both legs read from the same per-slot RAM buffer (`train_easy.py:762-766`). Zero new disk I/O, zero new VRAM (the `out_tN` apply in the mixture branch is shared with the nstep branch's compute footprint; mixture allocates the same activations).
- Sub-agent deploy contract honored: this SPEC + the code are produced in worktree only; main agent owns `deploy.sh` and server-side writes.
