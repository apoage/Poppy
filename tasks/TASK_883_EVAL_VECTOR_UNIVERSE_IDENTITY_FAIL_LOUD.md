# TASK_883 — `eval_vector.py` fails OPEN on universe identity; make it refuse like `imprint.py` does

**Status**: READY 2026-08-08 (planner; proposed by the run-28 orchestrator, adopted verbatim).
**Size: XS/S. Risk: low to build, HIGH in value.** Surface:
`experiments/835_stage4_eval/eval_vector.py` (arg validation + the checkpoint-grading entry path).
**DEPENDS ON TASK_874** — see *Sequencing* below. Do not schedule this before 874 ships.

## The defect

`--panel-frame-from-checkpoint` defaults to `None`, documented as *"Default OFF = the minute/daily
paths are byte-identical to HEAD"* (`eval_vector.py:3775-3783`). An unpinned run therefore builds
the panel from **today's coverage-ranked universe**. The encoder is **positionally
universe-locked** — slot *k* IS asset *k* — so grading a frame-locked checkpoint without the pin
feeds the model a scrambled slot→asset mapping and then prints a paste-ready LEDGER row for it,
with no warning anywhere in the output.

TASK_882 measured what that costs: **63/64 positions differ, 24 symbols absent entirely**, first
divergence at **decision 0** — the very first observation window is already wrong, not accumulated
drift. It produced `−0.6145 / PV 41,919` against the same checkpoint's `+0.8394 / PV 299,547`, and
the program spent a full planner window treating that gap as a crisis about which *instrument* was
real. It was never an instrument disagreement. It was a missing flag that the tool accepted in
silence.

`imprint.py` does not have this hole: when the frame is unverifiable it **REFUSES**
(`imprint.py:1234`, and the provenance refusal at `:1487`). Two harnesses, one contract, opposite
default behaviour — and the permissive one is the one the LEDGER quotes most.

**This is the reader-defect class again, in its purest form: absence read as a value.** A missing
pin is not "no pin requested", it is "the universe is unknown", and an unknown universe cannot
produce a comparable number.

## Scope

1. **Refuse an unpinned checkpoint-grading run**, by name, naming the checkpoint and the flag.
   Scope the refusal precisely to runs that load a `--checkpoint` and emit a scored row; `--smoke`,
   corridor/floor construction, and any path that grades no learned weights are NOT affected and
   must stay byte-identical.
2. **Refuse a pin/checkpoint MISMATCH too, not just an absent pin.** If the checkpoint dir carries
   its own provenance (which is what TASK_874 makes true), a `--panel-frame-from-checkpoint` that
   disagrees with it is worse than no pin at all — it looks correct in the artifact. Compare
   ORDER-EXACT, refuse naming the first differing slot index and both symbols.
3. **One explicit, loud escape hatch** — e.g. `--allow-unpinned-universe` — because a legitimately
   frameless checkpoint exists (anything trained before 853). It must: require the operator to
   type it, print a multi-line warning, and **stamp the emitted JSON with
   `"universe_pinned": false`** so the artifact carries the caveat forever. A number without a
   universe is not a result; if it is going to be produced anyway it must be self-labelling.
4. **Record the pin in the artifact unconditionally** — path, sha256, `n_frame_symbols`, and the
   order-exactness verdict — so a future reader never has to infer it from the command line. The
   882 investigation only closed because `results_s*.json` happened to embed `panel_frame`;
   make that a guarantee rather than a lucky habit.

## Sequencing — the dependency, stated explicitly

**874 first, then this row.** `imprint.py` consumes frame provenance but never emits it for the
checkpoint it writes (TASK_874). If 883 lands first, a freshly-imprinted checkpoint has no
provenance file to pin against, and the new refusal blocks grading it — the tool would be correct
and useless at the same time. 874 supplies the artifact; 883 makes it mandatory.

*Recorded because this is the fifth time a queue row's real dependency was on an artifact another
row produces, and the previous four were only found by a factory cycle dying on them (D33/D34/D35).*

## Acceptance (gates)

- (i) An unpinned `--checkpoint` grading run REFUSES before any panel is built, naming the
  checkpoint, the missing flag, and the escape hatch.
- (ii) A pinned run whose frame disagrees with the checkpoint's own provenance REFUSES, naming the
  first differing slot index and both symbols.
- (iii) `--allow-unpinned-universe` runs to completion, prints the warning, and the emitted JSON
  carries `"universe_pinned": false`.
- (iv) Every non-refused path is byte-identical to HEAD when the pin is supplied — proven, not
  asserted (the 882 decisive artifacts are a ready-made control).
- (v) A regression test covers all four of the above against the fixture.

## Out of scope

Changing what a pinned run measures. Re-grading any historical LEDGER row (882 already withdrew
the unpinned numbers). Touching `imprint.py`'s refusal, which is the model being copied.
