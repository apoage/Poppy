# TASK_863 — Teach `imprint.py` the 853 frame pin (D28 ruled opt (a): pin the frame on both 861 legs)

> **STATUS (2026-07-29): SHIPPED as `task863-imprint-frame-pin`.** Acceptance (i)/(ii)/(iii) all
> PASS: OFF bit-identical proven five independent ways (two orchestrator re-proofs post-fix at
> matched `_REPO` with identical `--out`/`--ckpt-out` → stdout byte-identical, 170 leaves / 0
> differing; the fixer's mirrored tree 0-of-759 and 0-of-545; the regression judge's 30-leg legacy
> battery with ZERO rc/artifact divergence; the fresh judge's mirrored-tree control); fixture smoke
> **13/13 legs** (8 → 11 → 13 across two fix rounds); reordered same-membership frame REFUSED with
> the rule named. selftest 12/12, 861 e2e 9/9 incl. `i_canon_832`, `eval_vector.py` byte-unmodified.
> 3-member panel + a FRESH fail-closed judge over the fix pass; **1 MAJOR + 6 MINOR + 3 NIT closed
> over 2 fix rounds**, every new guard demonstrated to fail when disabled.
>
> **REMAINDERS, named:**
> 1. **The real leg must NOT launch on this task's own pre-registration — DECISIONS D29 (OPEN).**
>    The pin is only available on a DAILY loader (no minute-framed loader exists; `eval_vector`'s
>    frame machinery is out of scope here). Measured read-only: `ohlcv_daily` covers **33/64** frame
>    slots on the leg-1 window (31 forced HOLD all window) while 858 brought MINUTE to **62/64**, and
>    the `+0.839` bar is minute-panel-native with a 64-asset book (29L/32S). D28 opt (a) as written
>    therefore trades a 59/64 universe confound for a source confound (D20: |Δ|=0.0555, cross-source
>    comparisons ruled forbidden) PLUS a **64→33 breadth confound** of unknown sign — first-order
>    (ST3: 2.76×). Recommended: teach the pin to the MINUTE path.
> 2. **This task file's scope-2 wording was factually wrong** and is corrected below: `imprint.py:620`
>    did NOT "only WARN" — shipped 627–631 is a hard `RuntimeError` (verified against the tag). The
>    hard refusal was preserved; adding a WARN would have broken byte-identical stdout.
> 3. Pinned-path behavior rests on the smoke legs and guard teeth, **not** on byte-identity — no
>    pre-fix-vs-post-fix byte diff of a PINNED artifact is possible (pre-fix pinned code existed only
>    in the edited tree; the pinned artifact intentionally gains keys).
> 4. Other named remainders (ExIt continuation re-derives a coverage-ranked universe; partial-slot
>    return-clip saturation now entering gradients; the UNPINNED silent-collapse path left unchanged;
>    `degenerate_replay` advisory/unwired; `--p` abbreviation now ambiguous; `is_fixture`-on-a-copy
>    trap) are in CHANGELOG 2026-07-29 and `.dark-factory/verdicts/`.

**Status**: SHIPPED 2026-07-29 (was READY 2026-07-28; D28 RULED — planner, opt (a); operator may override).
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
(`--panel-frame-from-checkpoint`); `imprint.py` does not. **CORRECTION (2026-07-29, verified against
tag `task861-search-teacher-imprint`): the claim that it "only WARNs (`imprint.py:620`)" is WRONG —
shipped lines 627–631 are a hard `mism.append` → `RuntimeError`.** DECISIONS D28 and TASK_861 line
179 carry the same error and should be corrected where authored, since the 861 pre-registration
leans on it. Consequence: scope 2's "keeps the existing WARN" was unimplementable as written, and
adding a WARN would itself have broken byte-identical stdout on the unpinned path.
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
