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
| 734 | `experiments/734_flow_matching/SPEC.md` | 🟡 DONE — PROMISING | 3 runs × 2.5h = ~3.75h | **mean Δα +3.38pp, all 6 periods positive**, STD 3.95pp. s7 = −0.54pp (flat). **First credible Tier-2 candidate.** Queue 5-seed extension (seeds 31, 42) to settle. See `runs/734_flow_matching/result.md` + `lore/notes_2026-05-15.md` §109. |
| 738 | `experiments/738_legacy_27ch/SPEC.md` | 🔴 DONE — REGRESSION | 3 runs × 2.5h = ~3.75h | **mean Δα −18.44pp**, all 3 seeds, all 6 periods negative. Same over-trading collapse signature as 741_mint. **TASK_732 cross-sectional channels confirmed load-bearing.** See `runs/738_legacy_27ch/result.md` + `lore/notes_2026-05-15.md` §107. |
| 734-x5 | (extension) | 🟡 RUNNING (server agent, 2026-05-15 19:32) | 2 runs × 2.5h + eval = ~3h (2-wide) | Seeds 31+42 on GPUs 1+2. Promote --flow-matching to canonical if mean(Δα) ≥ +3pp holds with no seed < −3pp. |
| 740 | `experiments/740_bella_distill/SPEC.md` | 🟡 PRECOMPUTING (205/270) | precompute remaining ~8h + 3 runs × 2.5h | At 2026-05-15 19:08 server time, 205/270 shards complete at 8.8 w/s. ETA Bella ready 2026-05-16 early morning. SPEC refresh + queue 740 after that. |

### Priority 2b — Tier-3 planning-side wave (queued, 2026-05-15)

Three new SPECs added this evening to use idle GPUs 1+2 after 734-x5 finishes (~22:00 UTC). Same paired-3-seed protocol against 741_mint baseline, same TASK_742 ≤2-concurrent rule. All three target the **planning side** (value or dynamics), the only mechanism class that has shown positive Alpha on this stack so far (per 734 result + `project_classifier_alpha_anti_correlation.md`).

| ID | SPEC | Status | Wall time | Notes |
|---|---|---|---|---|
| 746 | `experiments/746_value_oracle/SPEC.md` | 🟢 READY | 3 runs × 2.5h = ~3.75h | `--value-target-mode oracle` — re-test of TASK_719a under paired-Δα. April rejection was on value_loss only (now known unreliable). Pure value-side, no encoder/classifier/dynamics change. |
| 747 | `experiments/747_value_td0/SPEC.md` | 🟢 READY | 3 runs × 2.5h = ~3.75h | `--value-target-mode td0` — completes the value-target ablation matrix (oracle / nstep / td0). April characterization was "weak signal"; re-test on Δα protocol. |
| 748 | `experiments/748_flow_velsweep/SPEC.md` | 🟢 READY | 2 runs × 2.5h = ~2.5h (2-wide) | Velocity-weight sweep around 734's default 0.8. Two single-seed probes at 0.5 and 1.0 paired vs `flow_seed_7`. If |Δα| < 3pp, 0.8 is robust. |

### Priority 3 — followups

| ID | Description | Status | Action |
|---|---|---|---|
| 740 SPEC refresh | Update 740 to paired 3-seed using 741_mint baseline | 🟡 | Do AFTER Bella precompute finishes (~2026-05-16 morning). |
| 27-ch ckpt rsync | Recover ckpt + push to server | ✅ | Done 2026-05-14 19:30. |
| Multi-mechanism stacks | Combine winners (e.g. flow_matching + oracle, flow_matching + bella_distill) | ⏸ | Open after Tier-2 + Tier-3 winners land independently. |
| 739 dense_sign 5-seed | Settle whether s7 outlier is unlucky or representative | ⏸ | Low priority — dir_acc hypothesis already closed. Run if idle slots. |

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
