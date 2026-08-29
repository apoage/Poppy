# TASK_881E — authority binding for the manifest coverage predicate

**Status**: RE-CUT 2026-08-29 (planner; **D72 RULED by operator** after run-46's provable wall:
name-derived ownership is dead — `runs/875_wf/f1_s0` and `runs/000_smoke/adhoc_debug_run`
decompose identically under every name rule, one must cover, one must refuse). Earlier history:
READY 2026-08-24 (D64(a)); halted run 45 (work preserved); run 46 cycles 1–2 broke every
name-derived candidate, cycle 3 deliberately not spent.
**Size: S. Risk: LOW to run, HIGH in consequence.**
**D72's binding constraint (operator, verbatim intent): SIMPLE. The run knows what config
created it — ownership flows from the run's own record, not from path names, not from a new
auth system. The check lives in the deleter's module. Nothing else grows authority logic.**

## D72 ruling, as spec'd

1. **Authority = the run dir's own creation record.** A `covers:`/`protect:` entry grants
   coverage over a run dir only where **that run dir's own manifest (881i-ingested) names the
   requesting experiment as its creator/owner.** The manifest the run already writes IS the
   declaration — no central registry, no `owns:` blocks to maintain, no marker files, no
   promote-pin ceremony.
2. **No name-derived fallback.** The id-correlation fallback from the original (a) is REMOVED
   by D72's simplicity constraint — it was precisely the unprovable part. A run dir with no
   manifest (legacy dirs) is owned by nobody: the deleter refuses, and it stays
   operator-manual-delete. That is the status quo under the freeze — refusing more, never less.
3. **Fail-closed default, unchanged direction.** Additive only: the binding can only REMOVE
   coverage, never add it. Undeclared = refuse.
4. **The `--apply` freeze lift is DECOUPLED from this row** (D72, explicit). This row can no
   longer deliver "someone with standing" over legacy dirs; the lift — if it ever happens — is
   its own future decision with its own evidence, still behind 880's D35 gate. Remove every
   "this row gates the freeze lift" framing; the D64 code gate stays regardless.

## The defect (D64, unchanged)

The 881I/881G ingestion grammar is a flat namespace: "either name, anywhere under a scanned
directory, is read as a manifest" (`tools/ckpt_gc.py` docstring). Settled, NOT reopened (D63).
881D makes coverage gate an irreversible deleter, and coverage currently has no authority
binding — a manifest anywhere can cover anything. The no-manifest refusal HOLDS; the missing
converse is: "covered" must mean "the run's own record names this experiment."

## Scope

1. **The binding on the coverage predicate**, per D72 above. The parse/strip grammar itself is
   untouched. The authority check lives in the deleter's module (`ckpt_gc`), NOT in the
   checkpoint core — the core training path stays clean of authority logic (modularity, per
   operator).
2. **The refusal text stops teaching the wrong thing.** Post-binding it must name what WOULD
   confer standing ("this run dir's manifest names experiment X") — never "add a line under any
   `experiments/<id>/`".
3. **Mutation proof at the D64 repro:** the unrelated-`hostile_unrelated` manifest no longer
   covers; the same content under the manifest the run's own record names still covers; a run
   dir with no manifest refuses.
4. **The in-run deleter's freeze relationship** (run-43 open item 5, unchanged): the in-run
   pruner must respect the authority story WITHOUT being able to kill its parent training
   process (refuse-and-skip with a durable log line, never an exit-code change).
5. **D70's missing measurement survives the re-cut:** binding-ON breakage split (real vs
   placeholder fixtures) and a coverage diff against the real corpus — anything the binding
   removes is NAMED in the artifact. Expect the diff to be large-by-construction (legacy
   unmanifested dirs lose standing entirely); that is intended, but it must be enumerated, not
   discovered later.

## Resume note

Re-apply `lore/handoff_2026-08-28_factory_run_46_881E_wall.patch` FIRST (sha256 `d8d95bd9…`,
`git apply --check` verified 2026-08-29): cycle 2 deleted `_path_task_numbers` (closing
ancestor-path pollution, inner-segment forgery, numeric-container-with-numeric-child) and
removes NO real coverage — a strict improvement. Then STRIP the name-derived standing rules it
added (the "unambiguous-union" rule is exactly what D72 kills: it grants
`experiments/000_ci` standing over `runs/000_smoke/adhoc_debug_run`) and put manifest-authority
in their place. The patch is the base, not the design.

## Acceptance (gates)

- (i) The D64 repro flips from "covers, deletes" to "refuses, deletes nothing" — on the real
  tree, orchestrator-reproduced.
- (ii) Every protection the CURRENT corpus grants is re-derived under the binding (coverage
  diff, NAMED in the artifact, per scope item 5).
- (iii) The 881D/881I/881G suites stay green; canon pins re-pinned ONLY where the binding
  legitimately changes bytes (enumerated).
- (iv) The D72 proof pair is codified as a test: `runs/875_wf/f1_s0` (manifest names 875 →
  covers) vs `runs/000_smoke/adhoc_debug_run` (no such manifest → refuses) — the wall becomes
  the fixture.

## Out of scope

The parse/strip grammar (881I, settled). 880's D35 gate mechanics. Any change to what the
deleter deletes. Any central registry, marker-file scheme, or name-derived rule — all excluded
by D72.
