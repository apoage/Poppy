# TASK_874 — `imprint.py` must emit frame provenance for the checkpoints it produces

**Status**: READY 2026-08-05 (seeded from the 861 chain: the gap cost a REFUSE round-trip).
**Size: XS. Risk: low.** Surface: `experiments/832_linear_imprint/imprint.py` (+ its smoke).

## Why (measured 2026-08-03, during the 861 leg-0/OOS battery)

`imprint.py` CONSUMES a frame provenance (`--panel-frame-from-checkpoint`, TASK_863) but never
EMITS one for the checkpoint it writes. The 863/865 provenance gate is keyed on the recorded
`checkpoint` field, so grading a freshly imprinted checkpoint through the teacher's provenance
file is correctly REFUSED:

> frame/provenance mismatch: provenance `runs/832_e5/frame_provenance_imprint.json` was recorded
> for checkpoint `runs/832_e5/ckpt_imprint_full` but the manifest grades
> `runs/861_ref/ckpt_ref_s0`. Grading a checkpoint through ANOTHER checkpoint's slot frame
> scrambles asset identity — refusing.

The gate is right and must not be weakened. The defect is the missing emission: every scorecard
run over a new imprint needs a provenance file hand-derived first. On 2026-08-03 the planner
derived three by hand from the training artifacts (`runs/861_ref/frame_provenance_ref_s{0,1,2}.json`,
order-exact `frame_symbols` match ASSERTED at derivation) — those exist and are correct, but
hand-derivation on the read-of-record path is exactly the kind of step that eventually gets done
wrong.

## Scope

1. **Emit on write**: whenever `imprint.py` writes a checkpoint, write
   `frame_provenance_<ckpt>.json` alongside it (same schema as
   `runs/832_e5/frame_provenance_imprint.json`: `frame_symbols`, `n_assets`, `source`,
   `train_window`, `db`, `checkpoint`), with `checkpoint` set to the OUTPUT path and `source`
   naming how the frame was obtained: pinned (record the input provenance path it was pinned to,
   plus an order-exact match assert) or derived (coverage-ranked at this run's window/source).
   `panel_frame_source` recorded too — a minute-framed run must never be gradable as daily.
2. **Fail loud on inconsistency**: if the run was pinned and the loaded frame does not match the
   pin order-exactly, refuse to write provenance (the run itself already refuses — this must not
   become the weaker of the two checks).
3. **Backfill the three existing files**: verify the planner-derived
   `runs/861_ref/frame_provenance_ref_s{0,1,2}.json` match what the new emitter would produce,
   byte-for-byte modulo the `source` note; report any difference rather than overwriting silently.
4. Smoke: a fixture imprint emits provenance; the 862 scorecard grades that checkpoint through
   its OWN emitted provenance with no hand-editing; a cross-checkpoint provenance is still
   REFUSED (the gate stays live).

## Acceptance (gates)

- (i) Fixture imprint → provenance emitted; scorecard over that ckpt runs 7/7 with zero manual
  steps.
- (ii) Cross-checkpoint grading still REFUSED with the existing message (gate not weakened).
- (iii) Pinned-run mismatch ⇒ refuse to emit (demonstrated with a scrambled pin).
- (iv) Scope-3 comparison reported; existing suites green; no committed baseline moves.

## Out of scope

Changing the provenance schema or the 863/865 gate semantics; `eval_vector.py`'s emission (it
already records what it loads); re-grading anything (the 861 cards are ledgered — LEDGER
861/ref-oos-rebase).
