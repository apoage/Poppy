# Handoff 2026-07-08 (#5) — Factory run 4: the three instrument modules (834 / 835 / 832)

Previous: `lore/handoff_2026-07-08_stage3-SHIPPED.md`
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR (parallel implementer
fan-out; panels + fix passes + release ritual owned here; no hand-implementation)

## What shipped (three parallel lanes, all zero-canon-diff `experiments/` modules)

| Tag | Task | Panel outcome |
|-----|------|---------------|
| `task834-e3-runner` (`e28e62c`) | **TASK_834** checkpoint E3/DEC-1 attribution runner (`experiments/834_e3_probe/`) | 2 MAJORs crossval-CONFIRMED → FIXED + adversarially re-verified: (1) legality mask omitted from proposal sampling (93% of probe vectors carried a training-illegal action) → per-state canon mask on BOTH legs; (2) flax `restore_checkpoint` silent no-op → a **reproduced FALSE DEC-1 PASS on a fresh-init model** → fail-loud guards (attacked 3 ways, all refuse). +2 MINORs fixed. |
| `task835-stage4-eval` (`35abe2a`) | **TASK_835** faithful Stage-4 vector eval harness (`experiments/835_stage4_eval/`) | 0 BLOCKER/MAJOR — fidelity verified line-level vs the self-play vector branch; determinism bit-identical across re-runs. 3 MINORs fixed (LEDGER pointer follows real out_dir — kills the 734-819 stale-pointer hazard; partial-rollout guard withholds corridor booleans; provenance/docs) with EXACT smoke reproduction. |
| `task832-linear-imprint` (`5014620`) | **TASK_832** E5 linear-imprint runner, factory half (`experiments/832_linear_imprint/`) | 1 MAJOR crossval-CONFIRMED → FIXED + re-verified: imprint `entropy_coef=0.001` leaked into the ExIt half via `trainer_metadata.json` (unconditional restore at `trainer.py:1719`, CLI override ineffective) → `--ckpt-entropy-coef` default canon 0.01, proven with an adversarial-constructor restore probe. |

Also committed: the planner's battery pre-flight log on `tasks/TASK_828_VECTOR_ACTION_MUZERO.md`
(shared lane; committed by wrap-up to leave a clean tree).

## Verification discipline record
- Builds: 3 implementers in PARALLEL (disjoint modules). Panels: medium tier ×3 (9 lens agents),
  adversarial crossval of every MAJOR (all CONFIRMED — zero false alarms), fix passes routed back
  to the original implementers, targeted fix re-verify panel (3 agents; attacks, not re-reads).
- Orchestrator independently re-ran every smoke + selftest before shipping. All green.
- Workflows: panels `wf_a3affed3-ac0` (834+832), `wf_49a4844e-6dc` (835), fix re-verify `wf_355a67be-ee3`.

## The state of the line (load-bearing)
- **The Stage-3 battery is NOT launched.** Planner pre-flight found the 821 collapse signature at
  hold-bias 3.0 under an untrained value (pct_hold 94.7→99.97 by ep40). A pre-registered 3-arm
  diagnostic (A hb0.0/d1.0 · B hb1.0/d1.0 · C hb3.0/d2.0, 80 ep, GPUs 1/2/3) decides the operator;
  all-fail ⇒ STOP (structural finding, escalate to pathway). See the TASK_828 banner log.
- **Instrument coupling to the diagnostic outcome:** 834 (E3 probe) and 835 (Stage-4 eval) both
  default to the ORIGINAL manifest operator (hold-bias 3.0/dither 1.0/K16/sims32). The scheduled
  runs MUST pass the flags of the battery actually launched. 832's scheduled run is likewise gated
  on a working operator existing.
- Queue Ready = EMPTY. Everything next is planner/scheduled: diagnostic verdict → battery relaunch
  → snapshots → 834 probe (DEC-1) + 835 eval (corridor) → 832 imprint/ExIt run (DEC-2 ε).

## Git state
```
git log -6 --oneline:
<wrap-up commit>  Factory run 4 wrap-up: queue regroom + handoff
1e6a0d4-style     TASK_828: planner scheduled-run log — battery pre-flight (committed by wrap-up)
5014620  TASK_832 factory half: E5 linear-imprint runner (buffer-injection BC), zero canon diffs
35abe2a  TASK_835: Stage-4 faithful vector eval harness (checkpoint -> corridor scorecard), zero canon diffs
e28e62c  TASK_834: E3/DEC-1 attribution runner (checkpoint R_model vs R_true), zero canon diffs
516f2d9  Handoff 2026-07-08 (#3 interrupted + #4 SHIPPED): TASK_828 Stage 3 vector-action build
```
Tags this session: `task828-stage3`, `task834-e3-runner`, `task835-stage4-eval`, `task832-linear-imprint`.
Nothing pushed (owner pushes after review).

## Next action line
Planner: run the 3-arm hold-bias diagnostic on gpu-server (GPUs 1/2/3, run_queue.sh); on an arm
PASS, relaunch the battery with the selected operator and point the shipped instruments at its
snapshots (834 → DEC-1, 835 → corridor grading, then the 832 scheduled run). Factory: idle until
the planner seeds new rows.
