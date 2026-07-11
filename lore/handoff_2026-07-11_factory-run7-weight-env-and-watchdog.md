# Handoff 2026-07-11 (#9) — Factory run 7: TASK_840 (weight-rebalance env) + TASK_841 (wedge watchdog)

Previous: `lore/handoff_2026-07-10_chandoff-compaction.md` (#8 → its Previous #7 is the substantive env-wall-fix doc)
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR (implementer fan-out + panels + fix passes + release ritual)
Nothing pushed (owner pushes after review).

## What shipped

| Tag | Commit | Task | Panel |
|-----|--------|------|-------|
| `task840-weight-rebalance-step` | `31a545f` | **TASK_840** target-weight rebalance step (H6 env, D8 opt (c)) — canon `step_vector`, flag `vector_weight_rebalance` default OFF | FULL HIGH TIER. security+regression CLEAN; **1 BLOCKER (crossval-CONFIRMED) + 1 flow-verifier-CONFIRMED 2nd mechanism, BOTH FIXED**; 2 MINOR → D9/D10 |
| `task841-checkpoint-wedge` | `52b155f` | **TASK_841** post-checkpoint-save wedge watchdog in `run_queue.sh` (opt-in) + root-cause characterization | MEDIUM. **1 MAJOR double-CONFIRMED (correctness+regression) → FIXED**; 3 MINOR fixed; root-cause STOP confirmed sound |

## TASK_840 — the substance
Original 828 thesis env: per-asset action → TARGET signed weight; every position re-trimmed toward its target fraction of CURRENT PV each step; bounded allocation (Σ|w|≤1, no leverage) so at fee=0 bankruptcy is impossible. Fixes the residual kill mechanism after 839 (LEDGER 835/S4-p1b: budget-step-trained policies still bankrupted; nets learn — 834 DEC-1 PASS ×2, 836 value>momentum ×2 — the env killed).

**Two parity pins, captured by the orchestrator on the clean pre-edit tree (55a1d6a, JAX 0.6.2) BEFORE any edit:** canon `48e1f467…d25b` (both flags OFF) and 839-budget `4f8b692e…19d3` (new gate `tools/tests/test_vector_weight_parity.py`, budget-ON/weight-OFF). Regression judge independently re-derived BOTH on clean worktrees + proved OFF jaxpr byte-identical for both flag combos.

**The BLOCKER (and the fix).** The panel found the ON path did NOT deliver its defining guarantee — two mechanisms, one root cause (only actioned `do_reb` legs were re-trimmed/budgeted):
1. Held WINNING SHORT under-reservation: budget reserved held legs by signed market notional, but a winning short's true capital footprint `posval = 2·size·entry − size·price` exceeds it → overdraft → all-short `process_margin_call` wipes PV→0 at fee=0 (crossval-CONFIRMED with a legal-action repro).
2. Held positions (Hold / limit-fills) never re-trimmed → `|w|` drifts past the cap to 1.768, unbounded bleed (flow-verifier-CONFIRMED).
FIXED to the spec's actual item-2 semantics: the ladder/cap CLIPS every asset (drifted held legs get pulled back → re-trim), held-leg cash reserved by true `posval/PV`, whole-book budget `1 − Σ_held posval/PV` → `weight_cash ≥ 0` and Σ|w|≤1 by induction → `process_margin_call` a genuine no-op ON. My pre-panel concern (A) "limit-fill discard under ON" was REFUTED (the preamble reassigns `state` before the override, so fills are post-fill). Both fee=0 repros now show cash≥0 / bounded |w| / done=False; ON path re-verified by me.

**Acceptance (fixture, 279-asset panel, orchestrator-reproduced):** (i) survives full 1697-step walk, 2/2, zero bankruptcies, min_final_pv 63834, NEW hard invariant gate (min cash 38642≥0 / max Σ|w| 0.53≤1 / margin no-op) PASS; (ii) sticky-short PASS; (iii) E1 net@21 edge **+0.235** (flipped from −0.23 pre-fix; below the +0.436 bar) → recorded FINDING, NON-gating per SPEC item 6(iii). Fee under-charge ~16bps/re-trim vs canon (by-design) and Σ|w|=1.02 fee-wedge overshoot on adversarial sticky books both noted.

## TASK_841 — the substance
`run_queue.sh --stall-timeout` watchdog: a training job whose CSV freezes while its process lives → kill process-group, archive `_wedged_ep<N>`, relaunch ONCE (WEDGED-RETRIED); second wedge → wedged-failed, fail-loud. **The MAJOR (double-CONFIRMED):** the watchdog was ARMED BY DEFAULT (2700) and its CSV-name regex only matched `--name <space>` (not `--name=` nor the no-`--name`→`training_default.csv` case) → false-killed healthy CSV-less/`--name=` jobs at ~55min (HEAD exit 0 → NEW wedged-failed). FIXED: **default STALL_TIMEOUT=0 (opt-in)** — orchestrator-authorized deviation from spec line 16 per invariant #1; unarmed summary byte-identical to HEAD; when armed, only CSV-declaring jobs are watched (evals immune); `--name(=|space)`/`default` resolution. 3 MINORs fixed. Root cause characterized (orbax post-save GC vs training-loop RLock) but STOPPED at out-of-scope trainer surgery; `--debug-stall-dump` wired to capture the next real wedge's stack. Smoke 18/18, scheduler 16/16.

## Decisions seeded (planner walls — OPEN)
- **D9** — TASK_840 weight-path fee undercharge (~16bps/re-trim vs canon). Rec: ship for phase-1c (fee0); add canon-equivalent close-fee as a new task before fee16 phase-2.
- **D10** — TASK_840 (iii) expressibility FINDING (E1 edge +0.235 < +0.436). Rec: PROCEED to phase-1c (edge now positive; trained policy is the real test); widen granularity only if phase-1c underperforms.
- **D11** — TASK_841 watchdog default. Rec: RATIFY default-OFF; arm `--stall-timeout 2700` explicitly on the phase-1c/phase-2 training launches only.

## Orchestrator discipline this run
Captured BOTH parity pins on the clean tree before any canon edit; ran two high/medium panels (8 members) + 1 crossval; every objective gate independently re-run by me (parity ×2, selftest, 40-test battery, both acceptance probes, both adversarial repros, watchdog+scheduler smokes); fix passes routed to the ORIGINAL implementers via SendMessage (context intact); BLOCKER/MAJOR resolved by fixing to the ratified spec (not by weakening a decision) + seeding planner rows for the genuinely-planner calls. NOTE: the original two background implementers actually COMPLETED (late notifications) — a concurrent "completion" agent overlapped harmlessly; ground truth was re-established from disk, not prose.

## Git state
```
52b155f TASK_841: post-checkpoint-save wedge watchdog (opt-in) + root-cause characterization   [tag: task841-checkpoint-wedge]
31a545f TASK_840: target-weight rebalance step (H6 env, D8 opt c), flag-gated default-OFF        [tag: task840-weight-rebalance-step]
55a1d6a p1b verdicts: nets learn / env kills — DEC-1 PASS x2, value>momentum x2, D8 escalation fired; TASK_840+841 seeded   [planner]
```
Working tree: clean after this handoff commit. Factory: DISARMED. Nothing pushed.

## Next action
Factory idle — Ready queue is now EMPTY (840+841 shipped). Planner: ratify D9/D10/D11; launch the SCHEDULED phase-1c re-train under the weight env (arm `--stall-timeout 2700` on that launch — D11), then phase-2 fee16 (after D9's close-fee task if fees bind) + 834/836 re-runs. Watchdog root-cause follow-up: apply a flag-gated trainer save fix once `--debug-stall-dump` captures the wedge stack on gpu-server. Resume the factory with "armed".
