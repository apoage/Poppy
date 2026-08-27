# TASK_881E — authority binding for the manifest coverage predicate

**Status**: READY 2026-08-24 (planner; seeded by D64's ruling, option (a) — after the run-42
security lens measured that a grammatically perfect `covers:` line in an UNRELATED
`experiments/<id>/PROTECTED.txt` unlocks score-based deletion of a run dir nobody with standing
reviewed; orchestrator-reproduced on an isolated tree).
**Size: S/M. Risk: LOW to run, HIGH in consequence** — this is the row that gates lifting the
`--apply` freeze (with 880's D35).

## The defect (D64, measured)

The 881I/881G ingestion grammar is a flat namespace: "either name, **anywhere under a scanned
directory**, is read as a manifest" (`tools/ckpt_gc.py` docstring). That is settled and NOT
reopened (D63). What is new is the consequence: 881D makes coverage gate an irreversible
deleter, and coverage currently has no authority binding — a manifest anywhere can cover
anything. The no-manifest refusal HOLDS (recorded precisely in D64); what is missing is the
converse: "covered" must mean "someone with standing over THIS run dir declared it".

## Scope

4. **The in-run deleter's freeze relationship is decided here** (run-43 open item 5): 881D's
   in-run pruner is deliberately NOT behind the D64 code gate today (a naive freeze would
   convert "do not prune" into "kill the training run" — prune failure propagates to
   `sys.exit(1)`). That is disclosed-only; this row owns giving it a real design: the in-run
   deleter must respect the authority/freeze story WITHOUT being able to kill its parent
   training process (e.g. refuse-and-skip pruning with a prominent durable log line, never an
   exit-code change on the training path).

1. **Authority binding on the coverage predicate only.** A `covers:`/`protect:` entry grants
   coverage only where the manifest's own `experiments/<id>/` has a declared relationship to
   the named run dir — id-correlation (the manifest's id matches the run dir's task/experiment
   segment) or an explicit per-experiment allow-list. **Additive and fail-closed by
   construction: it can only ever REMOVE coverage, never add it** — so no existing protection
   weakens and the failure direction is "refuses more". The parse/strip grammar itself is
   untouched.
2. **The refusal text stops teaching the wrong thing.** The current refusal instructs the
   operator to add the line to "an `experiments/<id>/PROTECTED.txt`" — any id. Post-binding it
   must name the manifest that WOULD have standing.
3. **Mutation proof at the D64 repro:** the unrelated-`hostile_unrelated` manifest no longer
   covers; the same content under the CORRELATED manifest still covers; a manifest with no
   declared relationship refuses.

## Acceptance (gates)

- (i) The D64 repro flips from "covers, deletes" to "refuses, deletes nothing" — on the real
  tree, orchestrator-reproduced.
- (ii) Every protection the CURRENT corpus grants is re-derived under the binding (a coverage
  diff: anything the binding removes is NAMED in the artifact, and the corpus is amended
  deliberately if a removal was unintended).
- (iii) The 881D/881I/881G suites stay green; the canon pins are re-pinned ONLY where the
  binding legitimately changes bytes (enumerated).

## Out of scope

The parse/strip grammar (881I, settled). 880's D35 gate mechanics. Any change to what the
deleter deletes. **Note the sequencing: this row ships BEFORE any `--apply` freeze lift — D64's
code gate (shipped with 881D) is what enforces that.**
