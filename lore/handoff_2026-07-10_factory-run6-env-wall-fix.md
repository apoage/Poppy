# Handoff 2026-07-10 (#7) — Factory run 6: TASK_839 (the env-wall fix)

Previous: `lore/handoff_2026-07-09_factory-run5-instruments2.md`
Branch: `master` · Working dir: `/home/lukas/Trader` · Role: ORCHESTRATOR (implementer fan-out; panel + fix pass + release ritual owned here)
Commit: `6ca2a95` · Tag: `task839-budget-vector-step` · Nothing pushed (owner pushes after review).

## What shipped

| Tag | Task | Panel outcome |
|-----|------|---------------|
| `task839-budget-vector-step` (`6ca2a95`) | **TASK_839** budget-constrained vector step — the env-wall fix (canon `mcts_trader/env/jax_trading_env.py` `step_vector`, flag default OFF) | FULL HIGH TIER (`wf_9bf385cf-124`). OFF bit-identical: pre-edit pin `48e1f467…d25b` captured by the orchestrator on the clean tree BEFORE any edit (commit 39ae0bf, JAX 0.6.2), MATCH after; a panel judge reproduced the pin from the pre-edit commit in a detached worktree AND proved the OFF **jaxpr byte-identical** (850 lines) on a config outside the pin harness. **1 MAJOR crossval-CONFIRMED → RESOLVED**; 4 MINOR + 4 NIT (2 NITs pre-existing-not-ours). |

## The change (RATIFIED D7 option (a))
`step_vector` ON path sizes each simultaneous OPENING trade at `PV × min(position_size_pct, budget_pct/n_open_trades)`, where `n_open_trades` counts SPEND legs only (opening+adding longs, opening shorts; closing legs are proceeds → excluded). One new `ideal_spend_value` block gated on the static config bool; the only two consumers changed are `actual_cost`/`actual_short_cost` (limit legs keep the original `ideal_trade_value`). Config fields `vector_budget_step: bool=False` + `vector_budget_pct: float=0.99`. Threaded default-OFF through training `config`/`cli`/`env_factory`/`setup`/`train_jax_fast.py:260` and (eval-operator symmetry, the 838 precedent) `experiments/835_stage4_eval/eval_vector.py` (env site + self-describing result JSON) + the 834/836 walks.

## The MAJOR of the run (and why it did NOT block the ship)
Flow-verifier raised, crossval CONFIRMED + reproduced: the "aggregate spend ≤ cash **by construction**" claim holds only at PV≈cash (episode start). On an invested book (PV>cash) a breadth of opens still overdraws `PV·budget_pct > cash` → `process_margin_call` fires — never WORSE than OFF (per-leg spend elementwise ≤ OFF; long-opens self-heal via new-position collateral, done=False; short-opens can still bleed). This is **not an implementation bug** — the implementer built the ratified PV-relative D7 formula faithfully (correctness lens confirmed). Resolved at the factory level (the 838 pattern):
1. Corrected the overstated claims in code comments + test docstrings to the honest scoped wording (comment-only → parity hash unchanged).
2. Added `test_budget_step.py::test_on_pv_gt_cash_overdraws_and_degrades_to_off` — PINS the PV>cash behavior (ON overdraws like OFF; post-margin book byte-identical).
3. Seeded **DECISIONS D8** — the planner rules on PV-relative vs cash-relative / D7 opt(b)/(c) before flipping the flag ON. The ratified formula is untouched.

The step-0 env-wall (the lethal symptom) IS eliminated. The "reintroduces the bankruptcy cascade" escalation is crossval-labeled *plausible-but-not-demonstrated* (short-opening variant only; the masked 70-step self-play walk showed 0 overdrafts) → the decisive measurement is the scheduled trained-policy ON run.

## Also fixed (panel MINORs, orchestrator-re-verified)
- **Security**: `experiments/839_budget_step/acceptance_probe.py::_guard_out` used `os.path.abspath` → a symlink named innocuously but targeting `trading_data.db` bypassed the protected-DB write guard. Now `os.path.realpath`-resolves + checks all protected roots (incl. `~/trader-live/`). Orchestrator independently confirmed a symlink→`trading_data.db` is now BLOCKED (called `_guard_out` directly — it only checks/raises, never writes).
- **Fail-closed**: the probe exited 0 even on a genuine gate FAIL → now `SystemExit(1)` on `survival_pass`/E1 FAIL (JSON still written); `--force-fail-gate` self-test demonstrates exit 1.
- **NIT**: committed `results_fixture_smoke.json` `db` field now repo-relative (host-portable).
- **NOT ours**: `tools/tests/test_jax_env.py::TestLimitOrders` ×2 fail identically on pre-edit HEAD 39ae0bf (scalar limit-order path) — left untouched, noted.

## Acceptance (fixture, 279-asset daily panel, orchestrator-reproduced bit-for-bit twice)
- (i) survival walk (4 envs × 70 steps, near-uniform hb0.0, masked): **OFF all 4 bankrupt at step 0** (env-wall reproduced, ~40% breadth, 4 overdrafts); **ON 4/4 survive ≥64 steps, ZERO overdrafts**. PASS.
- (ii) E1 perfect-model probe K16: net@21 **OFF == ON == +7.6826630293827725** (delta 0.0). Narrow-regime cap non-binding (`0.99/n ≥ pct = 0.5/A`) → byte-identical. PASS (read: ON ≥ OFF − 0.10).

## State of the line
- **Phase-1 cost0 battery is DONE 3/3** (planner note in QUEUE: s1/s2 first pass + s0 solo retry clean; the ep1581 wedge did NOT reproduce — non-deterministic, not data-positional).
- With 839 shipped, the **phase-1 RE-RUN under the fixed step** is unblocked (scheduled, gpu-server GPUs 1/2/3 — planner/main session, NOT a factory-loop job). Then 834/836 re-runs on its snapshots, then phase-2 fee16 continuation + 832 imprint-sanity (all gated on the re-run).
- **Before flipping `--vector-budget-step` ON in that run**: resolve **DECISIONS D8** (PV-relative sufficiency). Recommendation seeded: ship opt(a) as-is + measure; layer opt(c) Σw≤1 only if mid-episode cascades appear.
- Instrument coupling reminder (unchanged): 834/835/836 take the operator as CLI flags — pass the flags of the battery actually launched, and now also thread `--vector-budget-step` when the re-run turns it ON (835/834/836 all already accept it, default OFF).
- Queue Ready = EMPTY. All next work = planner (scheduled runs / grooming / D8).

## Git state
```
6ca2a95 TASK_839: budget-constrained vector step (env-wall fix), flag-gated default-OFF
39ae0bf D7 ratified option (a): budget-constrained vector sizing — TASK_839 unblocked, factory green
c1247a8 Phase-1 verdicts: env-wall structural finding (828s3/env-wall), D7 seeded, TASK_839 queued
```
Tag this run: `task839-budget-vector-step`. Nothing pushed.

## Ops note (carried)
Parity pin for the vector env step: `tools/tests/test_vector_step_parity.py` (pin `48e1f467…d25b`, commit 39ae0bf, JAX 0.6.2). Joins the Stage-3 scalar pin (`a0627ce…`) and the 838 vector-search pin (`22b0c7f5…`) as the third canon parity gate.

## Next action line
Planner: resolve D8, launch the phase-1 re-run under the fixed step (`--vector-budget-step` ON), then 834/836 re-runs + phase-2. Factory: idle until new Ready rows.
