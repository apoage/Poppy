# TASK_890 — repair `distill.py`'s dead CLI, and make it apply 868's open gate

> **STATUS (2026-08-16): SHIPPED as `task890-distill-open-gate` — with acceptance (i) EXPLICITLY UNMET.**
> 2 cycles; verified by a 3-lens panel (correctness · regression · flow-verifier). Gates **(ii)–(v) MET**.
>
> **The CLI works and the gate is applied.** Five-value unpack repaired; `frame_hold_mask(prices_j,
> start_row, N, open_gate=open_gate)` — the same call `imprint.py:2231` makes — applied to every
> counterfactual candidate AND the teacher's walked action, across all four passes. The in-process path is
> byte-identical when the gate is inert, proven by loading both versions in one process and comparing every
> returned array.
>
> **ACCEPTANCE (i) IS NOT MET.** The documented bare invocation (`SPEC.md:74`) now runs the whole pipeline
> and then raises `AssertionError` on its **own** `--smoke-assert` quality gate before writing artifacts
> (v̂-alone calibration `+0.1251 → −0.3953` at seed 0). An unpack-and-discard arm fails a *different* check;
> at **seed 1 both arms fail the same one**. Thresholds were **deliberately not re-tuned** and `SPEC.md` was
> not touched. **This is not a failure of the row** — the CLI had been dead since 868/874, so it had never
> once executed past the arity bug and this gate's state was unknowable until now. **Seeded as D43** (OPEN):
> re-validate the smoke-assert at the program's own ≥3-seed bar and establish whether it was ever green
> post-868. **Until then, no 857 number should be quoted from this path.**
>
> **Two defects the panel caught that are worth carrying forward.** (1) The first implementation computed the
> `force-HOLD` log line and the artifact counts **before** forwarding the mask, so reverting one kwarg still
> printed "8 of 240 … applied" and wrote `n_force_hold_decision_cells: 8` while zero candidates were gated —
> D40's own "absence read as a value", one level below where it ruled. Counts now accumulate **on the
> force-HOLD branch itself**. (2) The first test never called `run()`; two reviewers independently rebuilt
> the real regression and it stayed **11/11 green**. It now runs the real CLI in a subprocess and grades the
> artifact, and the same mutations produce 15 FAIL / 1 FAIL.
>
> **`8 of 240` understates the effect by ~15×, and the real number is measured.** Gating a WALKED action
> diverges the portfolio trajectory from that decision onward: `decisions_with_ANY_changed_G = 29/30` per
> teacher pass, `0/30` per search pass — **58 of 120** decisions, against 8/240 raw cells. Both the floor and
> the blast radius are now reported.
>
> **Remainder → planner:** D43, plus `SPEC.md:74` still advertising the unreproducible pre-890 numbers
> (`v̂-alone +0.2331 → +0.3416`), and a gitignored `experiments/857_vhat_distill/ckpt_smoke/` written by an
> earlier verification run — the silently-rebasable shared-artifact class from the run-25 fixture incident.

**Status**: READY 2026-08-12 (planner; created by the D40 ruling out of factory run 33).
**Size: XS/S. Risk: low to build, MEDIUM in meaning** — it decides whether the v̂-leaf distiller
trains on fills that could not happen.
**Sequenced AFTER TASK_886, BEFORE TASK_878.** 878 re-specifies exactly this cell and should read
this row first.
Surface: `experiments/857_vhat_distill/distill.py` (the `run()` entry path and its `--help` text).

## The defect

Reproduced first-hand, planner, 2026-08-12:

```
$ distill.py --iters 1 --max-decisions 4 --tag planner_probe --out …
  File "experiments/857_vhat_distill/distill.py", line 710, in run
    prices_j, logc, env_cfg = imp.load_training_panel(args)
ValueError: too many values to unpack (expected 3)
```

`imprint.load_training_panel` returns **five** values —
`(prices_j, logc, env_cfg, open_gate, asset_aliases)` — since TASK_868 (D31 a2) added the causal
OPEN-GATE and TASK_874 added the alias record. Distill still unpacks three, so **its CLI has been
dead since those rows shipped** and only the in-process import path was exercised.

## The ruling (D40) — apply the gate, do not unpack-and-discard

`imprint.py:2213` unpacks all five and feeds `open_gate` straight into `frame_hold_mask`
(`:2231`), force-HOLDing OPENING on any panel row where the slot printed no real close. Distill
builds counterfactual v̂ targets from walks over the same panel and is the warm-start path, so an
ungated distill propagates targets that encode entries on bars that never printed.

1. **It is the same physical fact the operator already ruled on.** A no-print bar means the order
   was PLACED and UNFILLED (the ruling behind TASK_878). 868's gate is the conservative reading —
   it refuses the entry rather than coin-flipping it — so applying it is consistent with the
   standing ruling, and 878 later refines it in one place instead of two.
2. **Uniformity is the point of 868.** Distill would otherwise be the only surviving path training
   on impossible fills.
3. **No live number breaks.** TASK_868's banner already declares **837/844/857 KNOWN-STALE**.
4. **The gate is fail-closed** — it can only remove entries, never add them.

## Scope

1. Unpack all five values at `distill.py:710` and route `open_gate` through `frame_hold_mask`
   exactly as `imprint.py:2231` does.
2. **Do not unpack-and-discard.** Taking five values and using three is the same defect wearing a
   working signature — this program's signature failure, absence read as a value. If the gate is
   ever deliberately not applied, that must be an explicit flag that says so **in the artifact**.
3. **Correct the `--codistill-reward` help text.** It claims OFF *"reproduces the shipped 857
   deliverables byte-for-byte"*. Those deliverables are already KNOWN-STALE and this row changes
   what the gated path produces; leave no assertion of a byte-identity that no longer holds.
4. Record `n_open_gated_decision_cells` in distill's own artifact, as `imprint.py:2753` does — the
   count is the evidence that the gate is doing something.

## Acceptance (gates)

- (i) `distill.py`'s documented invocation runs to completion. **Run the real CLI** — run 33's own
  finding was that both shipped BLOCKERs lived outside the module under review and were found by
  running the documented invocation rather than the test.
- (ii) The gate is applied: with `open_gate` present, the decision-cell mask is strictly larger
  than without, and the artifact reports the count.
- (iii) A test that would pass if the gate were dropped goes red when it is dropped — rebuild the
  mutation independently.
- (iv) No test in this row monkeypatches the function under test. (Run 33: the retro-stamp's own
  test stubbed `infer_vector_action`, the function that was failing; a test that stubs the thing
  under test cannot fail.)
- (v) `857`'s suites stay green under their KNOWN-STALE status; the change in outputs is expected
  and must be stated, not hidden.

## Out of scope

TASK_878's failed-fill/coinflip semantics — this row applies the CONSERVATIVE gate that exists
today; 878 replaces it with the coinflip in one place. Re-grading any 857 LEDGER row.
