> **STATUS (2026-07-07): PARKED per DECISIONS D1/D4 — "interesting outview", non-binding; 828 is the focus. OHLC-only constraint stands (D4).**

# TASK_824 — Server honest re-measurement queue (use the back-online GPUs on truth, not more model)

**Status**: SPEC'D 2026-07-06. gpu-server is back online (4×3090, GPU 1/2/3 usable). The campaign
verdict is that MORE MODEL does not help on the existing data ([[project_single_mechanism_exhausted]],
[[project_planner_closed_clean_cell]]) and the data fork (Iris-DEX) is parked for storage. So the
honest, high-value use of the returned GPUs is **re-measurement with the instruments we fixed**,
not new training. Two of these could still change a conclusion; all three close open audit loops.

**Read first:** [[project_verdict_contamination_audit]], [[project_stageb_reward_head_diseased]].
Flags now exist that didn't when these verdicts were set: `--eval-mcts-decode per_asset` and
`--q-no-reward` (commit `88b2f02`), `--seed` (`687c146`).

## Pre-req: re-sync master to the server

Server was offline during commits `d110118…4b7171c`. Before running anything: `deploy.sh` current
master to gpu-server (it carries the eval flags + self-contained-HEAD fixes). Verify the server
tree py_compiles and `benchmark_easy.py --help` shows `--eval-mcts-decode` and `--q-no-reward`.

## Exp 1 — close the 739/741 audit loose ends (cheap, completes the record)

The 2026-07-05 reconciliation re-scored 740/743/758 on the canonical operator (all → null; 758's
"catastrophe" was an argmax artifact). **739 (−2.63pp) and 741 (−6.98pp) were never re-scored**
(no local checkpoints; the reconciliation agent flagged them). If their checkpoints survive
server-side (`runs/739_*`, `runs/741_*`), re-eval them under the canonical eval-MCTS sims=16 joint
operator, paired vs the a1b1 baselines (`runs/759_mcts_eval/seed_*`). Expected: null (consistent
with everything). Append dated CORRECTION to each `result.md`; update
[[project_verdict_contamination_audit]] to "fully reconciled". Low surprise, closes the matrix.

## Exp 2 — the un-blunted re-scorecard (HIGHEST value; could un-close a mechanism)

The audit proved the canonical operator (`decode='joint'`) structurally caps trade rate at 1/64,
so every Stage-B paired null was measured through a blunted instrument (biased toward null,
under-powered — [[project_stageb_reward_head_diseased]]). Now that `--eval-mcts-decode per_asset`
removes that cap (demonstrated: same 743/seed7 checkpoint 1.49%→69.2% trade rate), **re-run the
key 738–784 paired comparisons under per_asset decode** and ask: does any mechanism that was
"null under joint" show a real paired Δα when the operator can actually express multi-asset trades?
- Scope: the mechanisms with surviving server checkpoints — prioritise the ones whose joint-decode
  Δ was least-negative / borderline (734 flow_matching was the only ever-"promising" one; 763
  entropy +0.55; 762 reward_shaping). Run each variant AND its matched baseline under per_asset,
  same seeds, full 6-period scorecard.
- **Discipline (do not repeat the campaign's mistakes):** paired Δ (variant−baseline) only;
  per_asset ABSOLUTE levels are still not "skill" (they're a different operator artifact); ≥3
  seeds before ANY promotion claim; a mechanism only "un-closes" if paired Δα ≥ +3pp across ≥3
  seeds under the un-blunted operator. Report per-mechanism Δ table.

## Exp 3 — q-no-reward ranking (bypass the diseased reward head)

The Stage-B reward head is measured action-blind (60× variance gap, `runs/stageb_reward_probe_*`).
`--q-no-reward` zeros the reward term in the MCTS Q-backup so ranking uses value+policy only.
Re-score a1b1 baseline + 1–2 top variants with `--q-no-reward` (both joint and per_asset). Question:
does removing the blind reward term change trade behaviour / any paired verdict? This is diagnostic
— it tells us whether the reward head was actively degrading eval or just inert.

## Gates / outcome
- Any mechanism clearing paired Δα ≥ +3pp / ≥3 seeds under per_asset → **spawn a real promotion
  task** (the first genuine positive of the campaign; verify adversarially before belief).
- Otherwise → the Stage-B mechanism class is formally closed WITH the fixed instrument (not just
  the blunted one), and [[project_stageb_reward_head_diseased]] + the audit are settled. Either
  way the record becomes honest and complete.

## Server contract
`safe_run` + tmux, GPU 1/2/3 (never 0 = VirtualBox), VRAM envelope per [[feedback_server_vram_envelope]]
(single-GPU per job, MEM_FRACTION-aware). These are EVALS (inference on existing checkpoints), not
training — light and fast; fill GPU 3 with them while any train runs elsewhere. Standing priority
[[feedback_server_stays_under_load]]: keep it busy, but on THIS (honest re-measurement), not on
resurrecting closed model surfaces.
