# TASK_874 — `imprint.py` must emit frame provenance for the checkpoints it produces

> **STATUS (2026-08-08): SHIPPED as `task874-imprint-frame-provenance`.** All four acceptance
> gates (i)-(iv) met and independently re-verified by a flow-verifier on a fixture copy. 3 cycles,
> 4-lens panel, **4 BLOCKERs + 3 MAJORs found and closed**, 0 outstanding. Emission is now built
> pre-training and written LAST, after every refusal the run can raise: a self-refused (863
> degenerate-replay) run leaves NO provenance and its checkpoint is ungradable; a derived frame
> carrying duplicate symbols refuses pre-training; an existing DIFFERING provenance refuses without
> clobbering (`--overwrite-frame-provenance` replaces and stamps `provenance_replaced` +
> `replaced_provenance_sha256` into the document). Scope 3 landed as a read-only tool: all three
> `runs/861_ref/frame_provenance_ref_s{0,1,2}.json` agree with the emitter on every substantive
> field, frame order-exact; only the `source` note differs and `panel_frame_source` is added —
> reported, not repaired. Canon byte-identical (953/953 leaves), selftest 12/12, 874 smoke 8/8.
>
> **Intended behavioral break:** an unpinned run with `--n-assets` above the in-window universe now
> exits 1 before training (it previously trained on a column-duplicated panel and exited 0).
>
> **Incident disclosed:** the first pass wrote-then-deleted the real
> `runs/861_ref/frame_provenance_ref_s0.json` and reconstructed it; restoration verified
> independently (order-exact vs the pin and vs s1/s2, same sha256, 1524 B).
>
> **Remainders → planner:** (a) provenance is bound to a checkpoint by a PATH STRING only, no
> content hash — a stale/foreign file at the derived path grades through; ruled out of scope here
> (863/865 gate semantics), same fail-open class as TASK_883. (b) `runs/861_ref/*` still has no
> committed copy or checksum sidecar. (c) `env_factory.py:434-444` pad-by-modulo duplicates columns
> in the TRAINING PANEL itself, not just in provenance. (d) `--write-parity-baseline`'s dirty gate
> checks only the baseline file, not the tree (pre-existing 864 behavior).

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
