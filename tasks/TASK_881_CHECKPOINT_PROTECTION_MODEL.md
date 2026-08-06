# TASK_881 — The checkpoint protection model: explicit, tracked, fail-loud

**Status**: GATED 2026-08-06 — do not start before TASK_876 (rescoped emission half) and
TASK_880 land. Created by the D32 ruling, opt (a) as the model + opt (d) as the split.
**Size: M. Risk: HIGH** (it decides what an irreversible deleter may touch).
Surface: `tools/ckpt_gc.py` (`scan_referenced_paths`, `_is_protected`), the in-run retention
path from the TASK_876 stash, and a new per-run manifest convention.

## Why this is its own row

The factory built TASK_876's mechanism and PROVED the case it exists for (on a curve rising to
+9 then crashing to −9.5, old `keep=3` retains the three MOST collapsed checkpoints; the score
policy retains the pre-collapse peak plus the resume point). It then halted, correctly, because
scope 4 — "a checkpoint referenced by an `experiments/` artifact is never garbage" — has no
mechanical definition in this repo. Measured: only 3 of 7 run dirs get any protection, and all
four under `jax_ckpt_fast/` get zero, including `e5exit2_pq_s{0,1,2}`, which
`experiments/828_stage3_train/SPEC.md:211` cites **by run name** ("Battery e5exit2_pq_s{0,1,2}")
and never by path; `experiments/LEDGER.md` contains zero `jax_ckpt_fast/` references at all.

Inferring protection by grepping prose is the wrong instrument. Prose citation style is a
writing habit, and here it varies run to run — so the protection set silently depends on how a
SPEC happened to be worded. **The planner's decomposition error is the root cause**: emission
(non-destructive, needed now) and deletion policy (irreversible, needs a protection model) were
written as one "S" row. They are separated here.

## The model (ruled)

1. **Protection is EXPLICIT and TRACKED.** A run directory declares what must survive in a
   manifest committed to the repo — a `protected:` key in `experiments/<id>/SPEC.md` or a
   `PROTECTED.txt` beside it — naming checkpoint paths, not prose.
2. **No manifest ⇒ FAIL LOUD.** The GC refuses to prune a run dir it cannot resolve a manifest
   for, and says which one. It never guesses, and it never treats "no manifest" as "nothing to
   protect" — that is the exact inversion that makes a deleter dangerous.
3. **Prose citations are ADVISORY.** `--protect-from-ledger` stays as a safety net that can only
   ADD protection, never subtract it, and its hits are logged as advisory rescues.
4. **Descendant-aware.** A manifest entry naming a path INSIDE a checkpoint (e.g.
   `<run>/snapshot_ep10/checkpoint_500`, the exact form `restore_checkpoint` accepts) protects
   the enclosing checkpoint too. Today `_is_protected` walks ancestors only, so the enclosing
   dir stays deletable and the delete takes the protected path with it (reproduced, real
   deletion) — that fix belongs to the 876 half, and this row must not regress it.
5. **Re-scanned, not snapshotted.** `SERVER.md` documents mid-run SPEC drops as normal workflow,
   so the protect set is re-read on a TTL rather than frozen at launch — otherwise the in-run
   deleter and the offline GC disagree about the same checkpoint.

## Scope

1. Implement the manifest convention + resolver, with the fail-loud refusal and a clear message
   naming the missing manifest path.
2. Backfill manifests for the run dirs that currently exist, from the LEDGER/SPEC citations —
   this is the one place prose is read, ONCE, by a human-reviewed change rather than by the
   deleter at runtime.
3. Only then flip in-run score retention to default-ON, and only with `overwrite=False`. Until
   this row ships, `--ckpt-retention` stays `recency` (D32).
4. Audit-index honesty: a failed `rmtree` must record a FAILED prune (today it leaks the bytes
   under `*.gc-trash-<pid>` forever while the index records a normal success). If the 876 half
   already landed this, assert it here rather than re-doing it.
5. The 4th deletion site (`--safe-checkpoints`, default OFF, bounded by its own `keep_best_n`
   rmtree) either comes under the same protection model or is documented as deliberately outside
   it, with the reason.

## Acceptance (gates)

- (i) A run dir with no manifest is REFUSED, loudly, and nothing under it is deleted — including
  under `--apply`.
- (ii) A manifest naming an inner checkpoint path protects the enclosing checkpoint (the
  reproduced deletion no longer reproduces).
- (iii) A manifest added DURING a run is honoured within the TTL, demonstrated with both
  deleters (in-run and offline) agreeing on the same tree.
- (iv) Advisory ledger rescues can only add: a test where the ledger cites a path the manifest
  omits keeps it; a test where the manifest cites a path the ledger omits also keeps it.
- (v) A failed `rmtree` produces a FAILED audit row and a non-zero exit, never a silent success.
- (vi) Only after (i)-(v): the default flip, with the collapse case re-proven end-to-end on real
  flax saves (the 876 verification, re-run, not assumed).

## Out of scope

Score EMISSION (TASK_876); the remote archive tool's discovery (TASK_880); any change to the
`--unscored drop` ruling.
