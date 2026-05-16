# Experiment Queue

Master ordering of upcoming experiment SPECs, batched for the
`gpu-server` 4× RTX 3090 host under the **TASK_742 operational rule**
(≤2 concurrent `train_phase2.py` per host).

## Status legend

- ✅ executable (SPEC ready, code wired, prerequisites met)
- 🟡 executable with caveat (e.g. needs precompute step first)
- 🔴 requires code implementation before becoming executable
- ⛔ blocked (missing artifact / data)

## Queue

### Priority 1 — ~~confidence in today's TASK_737 result~~ DONE 2026-05-14

| ID | SPEC | Status | Result |
|---|---|---|---|
| 743 | `experiments/743_no_aux_multiseed/SPEC.md` | ✅ DONE | **AVG Alpha +3.61%, range -4.4 to +7.8 (12.2pp spread).** `no_aux_solo` NOT reliably better than a1b1 baseline. Single-seed +6.21% was upward outlier. See `lore/notes_2026-05-13.md` §99. |

**Decision (2026-05-14):** `no_aux_solo` is **deprecated** as "new
canonical Stage-B best". Revert canonical baseline to **a1b1**
(rank+aux, TASK_736 ref). Subsequent stacked-mechanism experiments
(739, 734, 741) pair each mechanism with **a1b1**, not no_aux_solo.

Single-seed comparisons on this setup are **untrustworthy** for
≤ ~5pp Alpha gaps. Any new claim of improvement needs ≥3-seed
validation.

### Priority 2 — break the 0.574 dir_acc / +6% Alpha ceiling

These stack new mechanisms ON TOP of the **a1b1** baseline (rank+aux).
Each is independent — no interdependencies. Each variant should be
multi-seed (≥3 seeds) to filter the ~5pp single-seed noise.

All Tier-2 SPECs **share the a1b1 baseline runs at seeds {7,13,23}**
produced by 741_mint Wave 1-3 (2026-05-14 08:00→15:30). Each mechanism
SPEC adds 3 variant runs (~3.75h wall with 2-wide) and pairs against
the shared baseline for delta analysis.

| ID | SPEC | Status | Wall time | Notes |
|---|---|---|---|---|
| 741 | `experiments/741_mint/SPEC.md` | 🔴 DONE — REGRESSION | 6 runs × 2.5h = ~7.5h (2-wide) | Mint xs paired 3-seed: **mean ΔAlpha −6.98pp, all 3 negative**. Better classifier (+3.29pp Bal.Acc) but worse trader. **Mint xs abandoned.** See `runs/741_mint/result.md` + `lore/notes_2026-05-13.md` §106. Baseline at seeds {7,13,23} still shared by Tier-2 SPECs. |
| 739 | `experiments/739_dense_sign/SPEC.md` | 🔴 DONE — INCONCLUSIVE | 3 runs × 2.5h = ~3.75h | **mean Δα −2.63pp**, 2/3 positive but s7 outlier (−11.46pp). dir_acc 0.569 — **TASK_733 ceiling hypothesis closed regardless**. See `runs/739_dense_sign/result.md` + `lore/notes_2026-05-15.md` §108. Optional 5-seed extension low priority. |
| 734 | `experiments/734_flow_matching/SPEC.md` | 🔴 DONE — 5-seed REVERSAL | 3+2 runs × 2.5h | 3-seed mean Δα +3.38pp looked promising, but **5-seed extension (s31 +3.56, s42 −12.71pp) drops mean to +0.20pp**. Pattern matches TASK_743 reversal of TASK_737: positive 3-seed → outlier reveals on 5-seed. **NOT PROMOTED.** Canonical Stage-B stays a1b1. See `lore/notes_2026-05-16.md` §115. |
| 738 | `experiments/738_legacy_27ch/SPEC.md` | 🔴 DONE — REGRESSION | 3 runs × 2.5h = ~3.75h | **mean Δα −18.44pp**, all 3 seeds, all 6 periods negative. Same over-trading collapse signature as 741_mint. **TASK_732 cross-sectional channels confirmed load-bearing.** See `lore/notes_2026-05-15.md` §107. |
| 740 | `experiments/740_bella_distill/SPEC.md` | 🟢 READY (SPEC refreshed 2026-05-16) | 3 runs × 3h = ~4.5h (2-wide) | Bella precompute COMPLETE (270/270 shards, server). SPEC refreshed to paired 3-seed with screening + 5-seed promotion protocol. **Top server priority.** |

### Priority 2b — Tier-3 planning-side wave (queued, 2026-05-15)

Three new SPECs added this evening to use idle GPUs 1+2 after 734-x5 finishes (~22:00 UTC). Same paired-3-seed protocol against 741_mint baseline, same TASK_742 ≤2-concurrent rule. All three target the **planning side** (value or dynamics), the only mechanism class that has shown positive Alpha on this stack so far (per 734 result + `project_classifier_alpha_anti_correlation.md`).

| ID | SPEC | Status | Wall time | Notes |
|---|---|---|---|---|
| 746 | `experiments/746_value_oracle/SPEC.md` | 🟡 DONE — INCONCLUSIVE | 3 runs × 3h | **mean Δα +2.54pp**, range −3.99 to +8.69 (12.7pp spread). Only Tier-3 result with non-zero mean. s23 +8.69pp is strongest signal seen on this stack. **5-seed extension (s31, s42) is the highest-value next probe** to settle whether it promotes. See `lore/notes_2026-05-16.md` §114. |
| 747 | `experiments/747_value_td0/SPEC.md` | 🔴 DONE — FLAT | 3 runs × 3h | **mean Δα +0.16pp**, no signal. Value-target roadmap CLOSED (oracle inconclusive, td0 flat → nstep stays canonical). See `lore/notes_2026-05-16.md` §114. |
| 748 | `experiments/748_flow_velsweep/SPEC.md` | 🔴 DONE — vw=0.8 ROBUST | 2 runs × 3h | vw=0.5 Δα −7.84pp, vw=1.0 Δα −2.47pp (both single-seed vs vw=0.8 reference). 0.8 default is robust to ±0.3 perturbation, both directions worse. Now academic since 734 didn't promote. See `lore/notes_2026-05-16.md` §114. |
| 746-x5 | (extension) | 🟢 READY | 2 runs × 3h = ~3h (2-wide) | Variant-only at seeds 31, 42 (same pattern as 734-x5). Settles 746 inconclusive verdict. |
| 749 | `experiments/749_titans_mag/SPEC.md` | 🟢 READY | 3 runs × 3h = ~4.5h (2-wide) | Titans-MAG implementation COMPLETE. Smoke PASSED 2026-05-16 with `--titans-lr 3e-3 --titans-gate-init -3`: gate Δ +3.4e-3 (3.4× threshold), mem_norm 0.05→0.92, grad finite, loss monotonic. 5 fix commits ended `04b8632`. Required flags: `--titans-memory --titans-variant mag --titans-memory-depth 2 --titans-memory-hidden 32 --titans-persistent-tokens 4 --titans-gate-init -3 --titans-lr 3e-3 --reset-titans-memory` (first run only). |

### Priority 2c — Overnight followup wave (auto-fire after Priority 2/2b complete, ~22:00 UTC 2026-05-16)

Three new SPECs to fill the ~10h overnight window + tomorrow daylight. Designed for GPU 1 (hero, long) + GPU 2 (short queue, sequential). GPU 3 reserved for eval workloads and result.md generation. TASK_742 ≤2-concurrent train_phase2 held strict.

| ID | SPEC | Status | Wall | Notes |
|---|---|---|---|---|
| 750 | `experiments/750_titans_hero/SPEC.md` | 🟢 READY | ~15h (1000 epochs, single seed, GPU 1) | **Hero calibration.** Does Titans-MAG have learnable headroom past E200? Single seed s7 @ 1000 epochs, intermediate checkpoints at E100/200/.../1000. Eval at E200/E500/E1000 vs `baseline_seed_7`. NOT a promotion experiment. |
| 751 | `experiments/751_titans_lr_sweep/SPEC.md` | 🟢 READY | 2 runs × 3h = ~6h sequential (GPU 2) | Bracket probes at `--titans-lr 1e-3` and `1e-2`. Confirms 3e-3 is near-optimal or finds better LR. Pairs vs `runs/749_titans_mag/titans_seed_7/`. Same pattern as 748 was for 734. |
| 752 | `experiments/752_value_oracle_baseline_pair/SPEC.md` | 🟢 READY | 2 runs × 3h = ~6h sequential (GPU 2, after 751) | a1b1 baselines at s31, s42. Completes properly-paired 5-seed for 746 (combined with 746-x5 variant runs). Baselines reusable for any future 5-seed at those seeds. |

### Priority 3 — followups

| ID | Description | Status | Action |
|---|---|---|---|
| 740 SPEC refresh | Update 740 to paired 3-seed using 741_mint baseline | ✅ | Done 2026-05-16. |
| 27-ch ckpt rsync | Recover ckpt + push to server | ✅ | Done 2026-05-14 19:30. |
| 734 baseline pair (s31, s42) | Run a1b1 baseline at the 5-seed extension's missing seeds to validate the variant-only reversal claim | ⏸ | LOW priority — 734-x5 verdict is "no promotion regardless of pairing" since s42's −9.91% absolute Alpha is bad in any interpretation. **Side effect: SPEC 752 produces these baselines as a byproduct (they're indistinguishable a1b1 runs). After 752 completes, 734 5-seed becomes properly paired post-hoc — no separate work needed.** |
| Multi-mechanism stacks | Combine winners | ⏸ | Moot — no individual mechanism has promoted 5-seed as of 2026-05-16. |
| 739 dense_sign 5-seed | Settle whether s7 outlier is unlucky or representative | ⏸ | Low priority — dir_acc hypothesis already closed. |

---

## Recommended overnight batching (4× RTX 3090, ≤2-concurrent rule)

Total budget: ~12-15 hours.

```
T=0:00   Wave A (training, 2 concurrent train_phase2):
  GPU 0:  743 seed 7
  GPU 1:  743 seed 13
  GPU 2:  IDLE (or eval workloads only — no train_phase2)
  GPU 3:  IDLE

T=2:30   Wave A flags drop. Launch Wave B:
  GPU 0:  743 seed 23
  GPU 1:  743 seed 31

T=5:00   Wave B flags drop. Launch Wave C:
  GPU 0:  739 dense_sign + no_aux
  GPU 1:  734 flow_matching + no_aux

T=7:30   Wave C flags drop. Launch Wave D (optional, if Bella precompute done):
  GPU 0:  740 bella_distill + no_aux       (training step)
  GPU 1:  739 dense_sign + a1b1            (companion variant)

T=10:00  Wave D flags drop. Final wave:
  GPU 0:  740 bella_distill + a1b1         (companion variant)
  GPU 1:  734 flow_matching + a1b1         (companion variant)

Total wall: ~12.5h overnight. Output: 4 multi-seeds + 6 feature variants = 10 new checkpoints.
```

Run benchmark eval as each variant finishes (eval is GPU-light and won't
trigger contention; can run on any free GPU including GPU 2 or 3).

### Pre-night step (manual)

Before Wave A starts, run Bella precompute serially on GPU 2 or 3:

```bash
python tools/precompute_bella_predictions.py --dry-run    # verify
python tools/precompute_bella_predictions.py \
  --bella-checkpoint <path> \
  --periods-file config/foxglove_periods.json \
  --output-manifest config/bella_preds_manifest.json
```

~45 min. Gate Wave D on its completion.

## Tasks needing implementation (NOT executable yet)

These need code changes before they can become experiment SPECs:

| Task | What's needed | Effort |
|---|---|---|
| `TASK_741_PER_ASSET_INJECTION_FIX.md` (Mint) | New module/branch in `mcts_trader/jax_agent/trm_encoder_v3.py` to inject per-asset scalars *after* temporal pool (not before). Spec is detailed; implementation is ~1-2 days. | Medium |
| Regenerate 27-ch Phase 1 ckpt | Need a workstation Foxglove run with `n_channels=27` (TASK_732 disabled). Phase 1 is ~12h on workstation. Then can run TASK_738. | Long |
| Multi-seed `a1b1` companion | Add 4 paired-seed runs of `a1b1` config (rank+aux) at the same seeds as `no_aux_solo` for paired t-test. Trivial — just 4 more SPEC entries; could be folded into TASK_743. | Trivial |

After any of these are implemented and tested, promote to executable
SPECs under `experiments/<id>/`.

## Rules going forward

1. **TASK_742 rule (CRITICAL):** ≤2 concurrent `train_phase2.py` per
   host. The original 3-parallel run deadlocked at E18→E19. Verified
   2-parallel works clean overnight (2026-05-12/13).
2. **Stagger benchmark eval** — also `train_phase2.py`? No, `benchmark_easy.py`
   is a separate path and doesn't seem to trigger the deadlock. Safe to
   run alongside training.
3. **Each new experiment** stacks on top of the **current best** Stage-B
   config. Today (2026-05-13) the current best is `no_aux_solo` (rank
   ON, aux OFF). Update this comment when a new best emerges.
4. **Always write `result.md`** at the end of an experiment. The
   experiment isn't done until result.md exists.
5. **Flag drops are the source of truth** for completion. The waiters
   (`/tmp/*_done.flag`) survive workstation sleep. The morning resume
   is `ssh gpu-server 'ls /tmp/*_done.flag; screen -ls'`.

## Related references

- `runs/737_eval/result.md` — completed TASK_737 4-cell analysis
- `tools/deploy/README.md` — workstation→server deploy contract
- `SERVER.md` — server agent operational contract
- `lore/notes_2026-05-12.md` §87-93 — recent session notes
