# TASK_890 — repair `distill.py`'s dead CLI, and make it apply 868's open gate

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
