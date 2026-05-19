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
| 746 | `experiments/746_value_oracle/SPEC.md` | 🟡 DONE — 5-seed all positive, NOT promoted | 5 runs × 3h | **3-seed mean Δα +2.54pp** → **5-seed mean Δα +2.23pp, STD 2.28, ALL 5 seeds positive**. First all-positive 5-seed seen on this stack. Below +3pp promotion bar but qualitatively the strongest non-anti-correlated signal. See `runs/746_value_oracle/result.md`. |
| 747 | `experiments/747_value_td0/SPEC.md` | 🔴 DONE — FLAT | 3 runs × 3h | **mean Δα +0.16pp**, no signal. Value-target roadmap CLOSED (oracle inconclusive, td0 flat → nstep stays canonical). See `lore/notes_2026-05-16.md` §114. |
| 748 | `experiments/748_flow_velsweep/SPEC.md` | 🔴 DONE — vw=0.8 ROBUST | 2 runs × 3h | vw=0.5 Δα −7.84pp, vw=1.0 Δα −2.47pp (both single-seed vs vw=0.8 reference). 0.8 default is robust to ±0.3 perturbation, both directions worse. Now academic since 734 didn't promote. See `lore/notes_2026-05-16.md` §114. |
| 749 | `experiments/749_titans_mag/SPEC.md` | 🔴 DONE — SECOND OOM, DEPRIORITIZED | 6 runs × ~1.5h = ~9h (2-wide) | **Two OOMs.** First-pass default-batch: 42.66 GB request, all 3 seeds OOM. b128 relaunch (2026-05-17): 21.33 GB request, all 3 variant seeds OOM (baselines trained fine). Structural memory pressure from truncated-BPTT scan retaining inner-MLP activations. **Titans-MAG deprioritized at this stack's production batch.** See `runs/749_titans_mag/result.md` (b128 verdict section). 750/751/749-x5 CANCELLED. |

### Priority 2c — Overnight followup wave (RE-PLANNED 2026-05-17 after 749 OOM)

**Status: original auto-fire chain (2c after 2/2b) did NOT trigger overnight.** Server agent instead detected the 749 OOM and relaunched 749 at `--batch-size 128` paired (`/tmp/749_b128_orch.sh`, started 2026-05-17 17:10 UTC, ETA early 2026-05-18 UTC). This consumes GPUs 1+2 for ~8–10h.

| ID | SPEC | Status | Plan |
|---|---|---|---|
| 750 | `experiments/750_titans_hero/SPEC.md` | ❌ CANCELLED 2026-05-18 | Titans-MAG deprioritized post-OOM. SPEC retained for reference if Titans ever revisited at smaller stack. |
| 751 | `experiments/751_titans_lr_sweep/SPEC.md` | ❌ CANCELLED 2026-05-18 | Same reason. |
| 752 | `experiments/752_value_oracle_baseline_pair/SPEC.md` | 🟢 READY — TOP PRIORITY 2026-05-18 | a1b1 baselines at s31, s42 to convert 746 5-seed proxy Δα (+2.23pp) to true paired Δα. Sequential on one GPU, ~6h. **Highest-value next action** — 746 is now the only campaign result with positive shape and 752 firms it up. |

### Priority 2d — Resolved 2026-05-18: Titans OOM #2 → pivot to 746/Bella v2

749-b128 resolved as **second OOM**. Decision-tree branch (2) taken: Titans-MAG deprioritized. 750/751/749-x5 cancelled.

**Wave status as of 2026-05-19 ~15:00 UTC:**

| Order | ID | Action | Wall | Outcome |
|---:|---|---|---:|---|
| 1 | 752 | a1b1 baselines at s31, s42 | ~6h | ✅ DONE. s31 α=+6.20, s42 α=+6.92. Used to revise 746 5-seed Δα to +0.72pp paired (was +2.23pp proxy). |
| 1b | 749-bf16 | Titans-MAG @ b=128 paired 3-seed (post bf16 fix) | ~13h paired | ✅ DONE. **REGRESSION mean Δα −3.11pp (1/3+, range −8.33 to +7.28)**. bf16 fix structurally correct (gates engaged) but mechanism doesn't promote. Titans-MAG closed. |
| 2 | 757 | Bella shape-distill 3-seed (TASK_754, b=128) | ~9h paired | ✅ DONE. **REGRESSION mean Δα −5.80pp (0/3+)**. Encoder-side Bella supervision 0-for-2 (740 + 757). |
| 3 | 758 | Bella augmentation 3-seed (TASK_753, b=128) | ~10h paired | ✅ DONE 2026-05-19 17:19 UTC. **CATASTROPHIC REGRESSION mean Δα −16.54pp (0/3+, s13 −37.93pp)**. Synthetic-data augmentation destabilizes training. Entire Bella mechanism class closed. |
| 4 | 756 | Value-target deeper probes | — | ❌ DEPRIORITIZED → reopens as TASK_759 P2 at reduced scope. |

### Priority 2e — Planning-side pivot (2026-05-19 EOD)

Bella mechanism class CLOSED 2026-05-19 (740 + 757 + 758 all negative at paired 3-seed). Pivot to planning-side mechanisms — the only class with positive paired Δα signals (746 +0.72pp, 734 +0.20pp at 5-seed).

Spec: `tasks/TASK_759_PLANNING_PIVOT.md`.

| Order | ID | Action | Wall | Why |
|---:|---|---|---:|---|
| 1 | 759_P1 | MCTS rollouts sweep: sims ∈ {64, 128} paired 3-seed | ~22h paired (sims=64), conditional ~45h for sims=128 | Untouched surface, highest EV. `mcts_sims=32` unchanged since inception. |
| 2 | 759_P2 | Value-target small probe: 0.7·nstep + 0.3·MC mixture, paired 3-seed | ~12h paired | Reopens TASK_756 at reduced scope; builds on 746's +0.72pp direction. |
| 3 | TASK_760 (conditional) | Policy-head architecture sweep | TBD | Only spec if both P1 and P2 close negative. |

**Promotion gate**: paired mean Δα ≥ +1pp AND no seed < −3pp at 3-seed → extend to 5-seed. 5-seed mean ≥ +3pp with no seed < −5pp → **first campaign promotion**, Bloom-trigger.

### Priority 3 — followups

| ID | Description | Status | Action |
|---|---|---|---|
| 740 SPEC refresh | Update 740 to paired 3-seed using 741_mint baseline | ✅ | Done 2026-05-16. |
| 740 provisional retired | TASK_745 spot-check PASS → 740 verdict no longer provisional | ✅ | Done 2026-05-17 (commit `ceea0e8`). |
| 741 Mint reconciliation | Audit COMPLETE/INCOMPLETE; remove stale QUEUE row | ✅ | Done 2026-05-18 (commit `cc0b864`, audit at `runs/741_audit/status.md`). |
| 27-ch ckpt rsync | Recover ckpt + push to server | ✅ | Done 2026-05-14 19:30. |
| 734 baseline pair (s31, s42) | Run a1b1 baseline at the 5-seed extension's missing seeds | ⏸ | LOW priority. **Side effect: SPEC 752 produces these baselines as a byproduct. After 752 completes, 734 5-seed becomes properly paired post-hoc.** |
| Multi-mechanism stacks | Combine winners | ⏸ | Moot — no individual mechanism has promoted 5-seed as of 2026-05-18. |
| 739 dense_sign 5-seed | Settle whether s7 outlier is unlucky or representative | ⏸ | Low priority — dir_acc hypothesis already closed. |
| b128 baseline drift investigation | Why does b128 baseline (+0.84% mean) underperform default-batch baseline (+2.80% mean)? | ⏳ | Low priority — n=3 may just be variance. Cross-check if relevant after 757/758 results land. |

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
| Regenerate 27-ch Phase 1 ckpt | Need a workstation Foxglove run with `n_channels=27` (TASK_732 disabled). Phase 1 is ~12h on workstation. Then can run TASK_738. | Long |
| Multi-seed `a1b1` companion | Add 4 paired-seed runs of `a1b1` config (rank+aux) at the same seeds as `no_aux_solo` for paired t-test. Trivial — just 4 more SPEC entries; could be folded into TASK_743. | Trivial |

**Removed 2026-05-18:** TASK_741 Mint row — Mint shipped in commit `1957a6c` and was tested at 3-seed (regressed, mean Δα −6.98pp). See `runs/741_audit/status.md` (commit `cc0b864`). Code remains in tree as default-OFF, zero-cost surface.

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
