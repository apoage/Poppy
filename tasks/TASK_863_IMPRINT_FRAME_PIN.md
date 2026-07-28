# TASK_863 — Teach `imprint.py` the 853 frame pin (D28 ruled opt (a): pin the frame on both 861 legs)

**Status**: READY 2026-07-28 (D28 RULED — planner, opt (a); operator may override).
**Size: S. Risk: low** (one shipped mechanism ported to one module; default bit-identical; the
853 semantics are already proven in `eval_vector.py` and pinned by 862's battery).

## Why (DECISIONS D28 — measured, not suspected)

TASK_861's real legs are BLOCKED: the 858 backfill moved the coverage ranking, so the unpinned
`ORDER BY COUNT(*) DESC LIMIT 64` universe over the leg-1 window differs from
`runs/832_e5/frame_provenance_imprint.json` in **59/64 positional slots** (membership ≥10 symbols
each way; the frame no longer reproduces even from its own recorded `train_window`). Slot k IS
asset k positionally, so leg 3's ">+0.839" bar would compare a net trained on universe A against
a reference measured on universe B — confounding "better teacher" (the task's only question) with
"different assets". `eval_vector.py` already has the shipped 853 fix
(`--panel-frame-from-checkpoint`); `imprint.py` does not — it only WARNs (`imprint.py:620`).
This task ports the pin, which unblocks the whole 861 real-leg chain AND keeps the +0.839 /
+0.898 / OOS / walk-forward references directly comparable (leg 4 reuses
`frame_provenance_imprint.json` as-is).

## Scope (one file: `experiments/832_linear_imprint/imprint.py`)

1. **`--panel-frame-from-checkpoint <provenance.json>`** with the SHIPPED 853 semantics (mirror
   `eval_vector.py`'s implementation — reuse/share code where clean rather than re-deriving):
   fixed 64-slot panel in the frame's positional order; zero-coverage and pre-listing slots
   INACTIVE and dual-disclosed (flat-zero features + price-validity force-HOLD, exactly the 853
   mechanics); fail-loud on identity scrambling. The imprint's teacher-target stream and BC loss
   must see the pinned slot order everywhere (any internal coverage-ranked re-derivation is a
   defect).
2. **Upgrade the frame WARN to a fail-closed gate** in the 861 provenance check: if the
   collection header records a frame pin, the imprint MUST be pinned to the identical frame
   (path may differ; the frame identity — symbol list + order — must match exactly) or it
   REFUSES; if the imprint is pinned and the collection was not (or vice versa) → REFUSE. Both
   directions named in the refusal message. Unpinned-both stays allowed (legacy path) but keeps
   the existing WARN.
3. **Fixture smoke**: collect with a frame pin → imprint with the matching pin passes end-to-end
   (through sanity replay); mismatched frame refused; pinned-collection + unpinned-imprint
   refused; inactive-slot disclosure appears in the imprint artifact.

## Acceptance (gates)

- (i) No new flag + unpinned collection = bit-identical to `task861-search-teacher-imprint`
  (parity vs the shipped module; selftests green). The ONLY intended behavior change on legacy
  inputs is scope-2's pinned↔unpinned refusal, which cannot trigger without a pinned artifact.
- (ii) Fixture smoke per scope 3 (pass leg + both refusal legs + disclosure present).
- (iii) Positional honesty demonstrated: a deliberately reordered (same-membership) frame file is
  REFUSED — the order is load-bearing, membership-only matching is the named failure mode.

## Real leg (planner post-ship — this re-registers TASK_861 legs 1–4; see the D28 addendum in
`tasks/TASK_861_SEARCH_TEACHER_IMPRINT.md`)

Both 861 legs run `--panel-frame-from-checkpoint runs/832_e5/frame_provenance_imprint.json`;
leg 2/3 becomes a **3-imprint-seed battery** (recorded sanity-replay seed noise −8.93…+12.74
across 10 seeds makes a single-seed ">+0.839" read uninterpretable; standing ≥3-seed
discipline). Leg 4 reuses the SAME frame provenance, so the 858-OOS + walk-forward battery and
862's manifest remain valid unchanged.

## Out of scope

Any change to `eval_vector.py`'s frame machinery (shipped, proven); grading bars (861's file owns
them); the tracked parity-baseline hygiene issue (TASK_864 — same file, run AFTER this row, not
in parallel).
