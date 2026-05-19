# Reboot Resume Marker — 2026-05-19 ~06:00 UTC

Workstation reboot in progress. This file captures everything needed to
pick up cleanly post-reboot. Read this first when you return.

## Last workstation commit

`92a788b` — Notes §139-145: 746 paired-5-seed REVISED to +0.72pp + Titans bf16 validated

## Last server commit (training branch)

`39cfe04` (deployed 2026-05-18 11:08 CEST). Server has:
- TASK_749 bf16 carry fix in `mcts_trader/jax_agent/titans_memory.py`
- TASK_754 Bella shape-distill (757_bella_shape/SPEC.md)
- TASK_753 Bella augment (758_bella_augment/SPEC.md)
- New `mcts_trader/data/bella_market.py`

## Server work currently in flight (as of 2026-05-19 05:47 UTC)

### Wave 1 (752 baselines @ s31/s42) — DONE 2026-05-18 16:11 UTC

- s31 baseline α = **+6.20%**
- s42 baseline α = **+6.92%**
- Done flags: `/tmp/eval_vobl_s{31,42}_done.flag`
- Eval JSONs: `runs/752_value_oracle_baseline_pair/eval_baseline_seed_{31,42}/benchmark.json`

### 746 paired 5-seed Δα (CORRECTED) — DONE in lore + result.md

**The headline revision of the session.** Proper paired against 752 baselines:

| seed | baseline α | variant α | Δα (paired) |
|---:|---:|---:|---:|
| 7 | +7.15 | +3.16 | −3.99 |
| 13 | +2.28 | +5.21 | +2.93 |
| 23 | −1.04 | +7.65 | +8.69 |
| 31 | +6.20 | +2.33 | −3.87 |
| 42 | +6.92 | +6.77 | −0.15 |
| **mean** | **+4.30** | **+5.02** | **+0.72** |
| sign | | | **2/5 +** |

NOT PROMOTED. Value-target roadmap effectively closed. TASK_756 deprioritized.

### Wave 2 (749 Titans bf16 retry @ b=128) — IN PROGRESS

- s7: 200ep DONE 2026-05-19 00:53 UTC. mem_norm 0.231 → 0.687 (3×). NO OOM.
- s13: 200ep DONE 2026-05-19 00:11 UTC. mem_norm 1.122 → 1.159. NO OOM.
- s23: at E140/200 at 05:47 UTC, running on GPU 1 (21 GB VRAM). Pace ~3.9 min/epoch.
- s23 ETA done: **~09:30 UTC 2026-05-19**, evals add ~30min → fully closed **~10:00 UTC**

**Titans bf16 fix VALIDATED on real data.** Gate engagement preserved.

### Wave 3 (757 Bella shape-distill paired 3-seed) — QUEUED

- Shape precompute DONE 2026-05-18 12:20 UTC
- Launches when s23 evals start (GPU 2 freed)
- ETA start ~10:00 UTC, ~5h paired wall → done ~15:00 UTC

### Wave 4 (758 Bella augment paired 3-seed) — QUEUED

- ETA start ~15:00 UTC, ~6h paired wall → done ~21:00 UTC
- C3 caveat: watch oracle-PnL-ratio at training time; if >2× tag as `too_easy`

## Post-reboot recovery sequence

When workstation comes back:

1. **Verify mount**: workstation `.git/objects` symlinks to `/dev/sdb1`
   (per MEMORY.md `workstation_storage_mount`). If not mounted:
   `udisksctl mount -b /dev/sdb1`
   Test: `mount | grep storage disk` should show it.

2. **Verify git HEAD**: `git log --oneline -5` should show `92a788b` at top.

3. **Server status poll** (single command, fast):
   ```bash
   ssh -p 22 operator@<redacted-ip> \
     'date -u; ls -lt /tmp/*_done.flag 2>/dev/null | head; \
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; \
      ps -ef | grep train_phase2 | grep -v grep | awk "{for(i=8;i<=NF;i++) if(\$i~/(--name|--seed)/) printf \"%s %s \", \$i, \$(i+1); print \"\"}"'
   ```

4. **Check flag for Wave 2 completion**:
   - `/tmp/tt128_s23_done.flag` — training done
   - `/tmp/eval_tt128_s{7,13,23}_done.flag` — all 3 evals done
   - If both exist: Wave 2 closed, pull paired Δα.

5. **Pull Titans paired Δα** (after Wave 2 closed):
   ```bash
   ssh -p 22 operator@<redacted-ip> 'cd /home/operator/Trader && python3 << "PYEOF"
import json
print("=== 749 Titans b128 paired Δα ===")
for s in (7, 13, 23):
    try:
        v = json.load(open(f"runs/749_titans_mag/eval_titans_b128_seed_{s}/benchmark.json"))["summary"]["avg_alpha"]
        b = json.load(open(f"runs/749_titans_mag/eval_baseline_b128_seed_{s}/benchmark.json"))["summary"]["avg_alpha"]
        print(f"  s{s}: variant={v:+.4f} baseline={b:+.4f} Δα={v-b:+.4f}")
    except Exception as e:
        print(f"  s{s}: ERR {e}")
PYEOF'
   ```

6. **Write 749 result.md b128 final verdict** (under `runs/749_titans_mag/result.md`,
   append section "## 749 b128 paired Δα — bf16 fix verdict (date)").
   Promotion bar: mean Δα ≥ +3pp AND no seed < −5pp. If clears: this is
   the first campaign promotion (Bloom-trigger).

## Current state of all in-flight files

### Committed (in git, survives anything)

- `lore/notes_2026-05-19.md` — full day notes §139-145
- `lore/notes_2026-05-18.md` — yesterday §132-138
- `lore/notes_2026-05-17.md` — 2 days back §123-131
- `experiments/QUEUE.md` — Priority 2d table
- `experiments/757_bella_shape/SPEC.md`
- `experiments/758_bella_augment/SPEC.md`
- `tasks/TASK_753_BELLA_AS_DATA_AUGMENTATION.md` (stub)
- `tasks/TASK_753_REVIEW.md` (Phase 1)
- `tasks/TASK_754_BELLA_TRAJECTORY_SHAPE_DISTILL.md` (stub)
- `tasks/TASK_754_REVIEW.md` (Phase 1)
- `tasks/TASK_755_BELLA_REGIME_CONDITIONAL.md`
- `tasks/TASK_756_VALUE_TARGET_DEEPER_PROBES.md` (stub — deprioritized today)
- `tasks/TASK_749_OOM_FIX_REVIEW.md`
- All implementation commits (TASK_754, TASK_753, TASK_749 fix)

### On disk but gitignored (in runs/, survives reboot but not pushed via git)

- `runs/746_value_oracle/result.md` — REVISED today with paired Δα section
- `runs/749_titans_mag/result.md` — has b128 verdict + bf16 fix log
  (will need final paired Δα verdict appended post-Wave-2)
- `runs/745_spot_check/verdict.md` — column alignment PASS
- `runs/741_audit/status.md` — Mint reconciliation
- `runs/740_bella_distill/result.md` — soft regression (confirmed)
- `runs/752_value_oracle_baseline_pair/` — baselines on server, pulled to workstation
- `runs/749_titans_mag/baseline_b128_seed_{7,13,23}/` — clean baselines for pairing

### Workstation has-only (not pushed/synced)

These survive workstation reboot because they're on the workstation disk:

- All `lore/notes_*.md` files (workstation-only per `.training-exclude`)
- All `tasks/TASK_*.md` files (workstation-only)
- All commits since deploy `39cfe04` if any were made post-deploy
  (only `92a788b` so far — lore + 746 result.md update — `runs/`
  doesn't get committed)

## Campaign state at moment of reboot

11 mechanism evaluations done, **zero promoted under paired Δα with current
threshold**:

| ID | Status | Mean Δα (paired) | Notes |
|---|---|---:|---|
| 737 no_aux_solo | DONE — REGRESSION (5-seed) | +3.61 | Single-seed +6.21 was outlier |
| 738 legacy_27ch | DONE — REGRESSION | −18.44 | All 3 negative, all 6 periods negative |
| 739 dense_sign | DONE — INCONCLUSIVE | −2.63 | s7 outlier −11.46 |
| 740 bella_distill | DONE — REGRESSION | −1.92 | 0/3 positive, confirmed |
| 741 mint_xs | DONE — REGRESSION | −6.98 | All 3 negative |
| 734 flow_matching | DONE — REVERSAL | +0.20 (5-seed) | 3-seed +3.38 reversed |
| 746 value_oracle | DONE — NOT PROMOTED | **+0.72** | Today's revision; 2/5 positive |
| 747 value_td0 | DONE — FLAT | +0.16 | Roadmap closed |
| 748 flow_velsweep | DONE — vw=0.8 ROBUST | mixed | Academic |
| 749 titans_mag | OOM → bf16 fix → Wave 2 in flight | TBD | s7+s13 trained clean |
| **752** | DONE — baselines for 746 pairing | (baseline data) | s31=+6.20, s42=+6.92 |
| **757** | QUEUED Wave 3 | TBD | Bella shape-distill |
| **758** | QUEUED Wave 4 | TBD | Bella augment (C3 caveat) |

**Next promotion-eligible verdicts**:
- 749 Titans bf16 paired 3-seed @ b128 (~10:00 UTC today)
- 757 paired 3-seed (~15:00 UTC today)
- 758 paired 3-seed (~21:00 UTC today)

If ANY of these clears +3pp mean Δα with all 3 positive, it would be
**the first campaign promotion** under the new methodology. Bloom-trigger.

If all three close negative: pivot. Untouched surfaces remaining:
**policy head** and **MCTS search**.

## Methodology updates codified today (load-bearing)

From lore §141:
- "All-positive variant α" ≠ "all-positive paired Δα"
- Variant-only extension is now considered **preliminary 5-seed,
  NOT promotion-eligible**
- Proxy-paired (variant-only against baseline mean) produces direction
  signal only; cannot be promotion-eligible
- Always commission paired baselines as part of the same wave, not as
  a follow-up

## Memory files to remember are still fresh

These under `~/.claude/projects/-home-lukas-Trader/memory/`:

- `feedback_lore_dual_role.md` — lore = failsafe + publication
- `feedback_memory_layer_model.md` — L0-L4 model
- `feedback_jax_vram_preallocate.md` — JAX env vars
- `feedback_read_current_state.md` — always check lore/tasks at session start
- `project_classifier_alpha_anti_correlation.md` — 3-way confirmed pattern
- `project_flow_matching_promising.md` — HISTORICAL (5-seed reversed)
- `project_task741_result.md`, `project_task741_staged.md`
- `project_bull_blindness.md`
- `project_bella_column_mismatch.md` — RESOLVED (TASK_745 spot-check PASS)
- `project_benchmark_gap.md`
- `workstation_storage_mount.md` — CRITICAL for boot recovery
- `infrastructure_hosts.md`

## Tasks list state (in this session)

Tasks 1-40 across sessions. Most recent:
- #32 P1 TASK_745 spot-check ✓
- #33 P2 TASK_754 trajectory-shape impl ✓
- #34 P3 TASK_756 spec stub ✓
- #35 P4 TASK_753 Bella-as-data impl ✓
- #36 P5 TASK_741 audit ✓
- #37 Finalize 749 b128 verdict ✓
- #38 Fix Titans VRAM (bf16 fix) ✓
- #39 Pull server reports ✓
- #40 Status check + 2026-05-19 lore ✓

## When you return

1. Verify `mount | grep storage disk` shows the drive
2. `git log --oneline -3` should show `92a788b` Notes §139-145
3. Run the server status poll command (above)
4. If Wave 2 is done: pull Titans paired Δα and write final verdict
5. If still running: wait for next ping

---

*This file is committed to git. If git is broken post-reboot, it's
still readable at `lore/REBOOT_RESUME_2026-05-19.md` on the
workstation filesystem regardless of git state.*
