# Handoff 2026-07-16 (#13) — Factory run 10: TASK_843 (weight netting) + TASK_845 (imprint weight-compat)

Previous: `lore/handoff_2026-07-14_chandoff-compaction.md` (#12 — carries the run 7–9 chain #9/#10/#11)
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR
Nothing pushed. Factory DISARMED at wrap-up.

## What shipped

| Tag | Commit | Task | Panel |
|-----|--------|------|-------|
| `task843-weight-incremental-netting` | `ed27849` | **TASK_843** — weight-path incremental netting + snap-to-flat (D12 cleanup); fee0 pin re-based | Escalated 4-member (correctness/regression/fail-closed/flow-verifier). **1 BLOCKER FIXED** + D14 seeded |
| `task845-imprint-weight-compat` | `4ac7721` | **TASK_845** — 832 imprint runner threaded through the weight env + teacher-expressibility gate; zero canon diffs | Medium + fresh fix-judge. **1 MAJOR + 2 MINOR FIXED** + D15 seeded |

This was the D12-cleanup → E5-compat chain: 843 first (env reaches its FINAL fee0 form), then 845 (measures against that form). Strictly sequential — 845's expressibility gate must see the final env.

## TASK_843 — the substance
Replaced the TASK_840 full-close+reopen re-trim with **incremental netting** (delta-only trades, entry preserved so PnL realizes on the closed portion) + **snap-to-flat** (kills slippage micro-short dust). Fixes D12: short round-trip 185.2 → ≈canon, per-step cost now constant not escalating.
- **Pins**: canon-OFF `48e1f467…` + budget `4f8b692e…` UNCHANGED; weight-ON/fee0 RE-BASED `1c50d58c…`(842) → `6293668f034ef42f7fdd71005d625c4958e189705256463aae6eb49d4c3d90bf`. Orchestrator captured the clean-tree baseline (all three) before any edit.
- **BLOCKER (a genuine 843 regression, orchestrator-crossval-CONFIRMED vs the pre-edit tree)**: winning-short survivor under-reservation. Entry-preserving netting reserved re-trim legs by MARKET notional while PV charges a winning short's larger `size·(2·entry−price)` footprint → phantom budget funded fresh shorts → all-short `process_margin_call` full wipe **at fee0 from a valid state** (minimal repro: cash 10000/PV 100000/one winning short → PV 0; 842 preserved it). FIXED: reserve staying-short legs at footprint fraction `(2E−P)/P` (clamped ≥1 so losing shorts aren't under-reserved); entry-preservation intact. Re-run fail-closed judge attacked the fix → fee0 hole genuinely closed (3,840-case fuzz + repro-sweep 0 hits, was ~110).
- The panel SPLIT is the lesson: correctness + flow-verifier missed the BLOCKER (their action sequences never reduced a deep winning short *while* opening fresh shorts); fail-closed caught it. Parallel adversarial panel earns its keep.
- **DECISIONS D14 (pre-existing, NOT a 843 regression)**: a fresh near-fully-invested all-short book (Σ|w|→1) overdraws by ~unreserved turnover fees at fee>0 → all-short margin wipe. `git stash`-confirmed 842 wipes identically (fee16 cash −160.0 vs 843's −149.99). The known `project_vector_env_cash_wall`; out of 843's fee0 scope; affects phase-2 fee16 (recommend: reserve fee headroom in `budget`). Overclaiming ON-invariant comments (843's + 840's) scoped to fee0 + D14.
- Acceptance (i)–(iv) PASS: short RT rel 0.0/2.47e-4; zero residual dust; 840 walks survival 2/2 + cash≥0/Σ|w|≤1/margin-noop; E1 fee16 edge **+2.1667** (from 842's honest −0.134). selftest 7/7; parity+rebalance 30 passed; 2 new adversarial regression tests.

## TASK_845 — the substance
Threaded the TASK_840/843 weight env config through the 832 imprint runner's BOTH in-env walks (teacher trajectory + sanity replay); default OFF = order-semantics byte-identical. Added the pre-registered **teacher-expressibility gate** (order-code teacher through the WEIGHT env vs the fixed order-env corridor floor, ≤10% rel; weight-native ladder re-gate on miss). BC targets = MC returns over the teacher's WEIGHT-env rewards. **Zero canon diffs.**
- **MAJOR (crossval-CONFIRMED)**: the weight-native remediation mapping was DIAGNOSTIC-ONLY — `inject_teacher` always consumed the order-code teacher, so a real-panel order-code-FAIL/weight-native-PASS would still imprint the failed teacher (spec item-2 defeated). FIXED: `--imprint-mapping {order-code,weight-native}` selector (default order-code = byte-identical) rebinding teacher/floor/agreement/values/ledger. A fresh judge on the new path confirmed it solid (all rebinds complete, deterministic) + found **2 MINOR guards → FIXED**: (1) fail-closed — weight-native asserts `--vector-weight-rebalance` (else it imprinted the ladder through the ORDER env, floor +8.454, while the gate certified the weight-env teacher −4.593); (2) None-guard the gate-MISS WARN vs a fee16 bankrupt-teacher `TypeError`.
- **DECISIONS D15 (on D13's critical path)**: the fixture expressibility gate is REPORTED, PASS/FAIL INCONCLUSIVE (≈50-decision losing slice, relative threshold ill-defined on negative Sharpes), BUT the ~1.25 net@21 gap is a real STRUCTURAL sizing divergence under default w_unit/w_max (order-code ~2×, weight-native up to ~8× the order env's per-asset exposure; ±0.10-absolute would also fail). Binding read = planner-owned real-panel run (DEC-2); D15 pre-registers the response (run both mappings; retune w_unit/w_max if both miss). Un-blocks the scheduled E5 imprint-sanity→imprint→ExIt.
- Orchestrator re-verified: default order-code core BYTE-IDENTICAL to the committed artifact; weight-native genuinely imprints (floor −4.593, CE 2.03→0.13, no bankruptcy); assert fails fast; selftest 7/7.

## Operational note — session-token-limit incident (recurrence)
Both 845 fix passes (the MAJOR wiring, then the two MINOR guards) hit the nightly session-token limit (resets 22:20 Prague / 20:20Z) — the FIRST died mid-final-report with its edits already complete. Recovered per the run-8 precedent: assessed the on-disk state, confirmed the edits were complete + consistent (compiles, both smokes pass, default byte-identical), and — once the session reset (verdict timestamps crossed 20:20Z) — ran a FRESH adversarial judge on the new path so there is NO self-review gap. Lesson stands: on a session-limit death, check the tree state before re-dispatching; the edits are often done.

## Git state
```
4ac7721 TASK_845: thread 832 imprint runner through the weight env + teacher-expressibility gate   [tag: task845-imprint-weight-compat]
ed27849 TASK_843: weight-path incremental netting + snap-to-flat (D12 cleanup)                      [tag: task843-weight-incremental-netting]
8f76a71 D13 RULED: redirect (operator) — E5 imprint→ExIt critical path; 845 seeded, 843 un-gated    [planner]
```
Working tree after the QUEUE + this handoff commit: clean. Factory: DISARMED. Nothing pushed.

## Verbatim anchors (must survive)
- Parity pins (843): canon-OFF `48e1f467c4e7e37316e3555015a89ec8e266297c8515e55c66ea4b54a536d25b` + budget `4f8b692ead92248f16850f4874d9b9fca7d85fb9447fe26721442d9f164c19d3` UNCHANGED; weight-ON/fee0 RE-BASED `1c50d58c…` → `6293668f034ef42f7fdd71005d625c4958e189705256463aae6eb49d4c3d90bf`.
- Tags@commits: `task843-weight-incremental-netting@ed27849`, `task845-imprint-weight-compat@4ac7721`.
- OPEN decisions: **D14** (fee16 all-short overdraft), **D15** (E5 teacher-expressibility divergence). Prior ratified: D9/D10/D11/D12/D13.

## Next action
Factory idle — Ready EMPTY, no D-independent row. **Next moves are PLANNER-owned**: (1) rule **D14** before phase-2 fee16, (2) rule **D15** (mapping/params) before the scheduled E5 real-panel runs, (3) launch the scheduled E5 imprint-sanity→imprint→ExIt (un-blocked by 845, run both mappings via `--imprint-mapping`). Resume the factory with "armed" once a Ready row lands or a decision spawns factory work.
