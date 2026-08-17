# TASK_881 — The checkpoint protection model: explicit, tracked, fail-loud

> **STATUS (2026-08-17): HALTED on the 3-cycle rule (factory run 35) → `DECISIONS` D45 (OPEN).**
> Fourth halt on this lane, and the same shape D33/D36/D38/D39 each named: every cycle's FIXING
> CODE contained the next defect, and every defect lived in ONE layer the row never scoped —
> **manifest INGESTION** (what counts as a manifest, what the scan can see, what a line MEANS).
>
> **BUILT AND VERIFIED across three cycles and six opus judges, every kill re-run from scratch:**
> all six prior kills are DEAD — symlinked manifest FILE (exit 2), symlinked manifest DIRECTORY
> (refused even when `--experiments-dir` points AT the symlink), torn `scores.jsonl` (per-run-dir
> refusal, peak survives), **the cycle-2 killer's full four-step sequence** (refuse → follow the
> refusal's own remediation → re-run under the floor → **the 42.0 peak SURVIVES**), and
> missing/symlinked/directory `--protect-file` (exit 2). `write_run_index` atomicity holds under
> SIGKILL mid-`json.dump`. **`root_resolution()` closes the D38-c3 asymmetry — measured under
> `--apply` on BOTH deleters with a simulated mounted scratch disk, survivor sets BYTE-IDENTICAL.**
> `plan()`'s 4th value is honoured by both production callers. MAJOR E's dangling-entry refusal is
> correctly SCOPED (real tree: three unresolvable `828_stage3_train` entries print loudly, refuse
> **0** run dirs, all **7** checkpoints kept even at `--min-free-gb 100000 --keep-unscored 0`).
> **The disputed D33 `_order_newest_first` bug is SETTLED BY EXECUTION** (cross-family:
> `checkpoint_100`, the true newest on disk, was deleted while a 999 s-stale `checkpoint_200`
> survived as "newest (resume)") and fixed. Suites 62/62 + 20/20 + selftest 12/12.
>
> **WHAT HALTED IT — cycle 3 shipped a NEW silent deletion, caused by cycle 3's own fix.**
> `_parse_spec_block` (`ckpt_gc.py:618`) consumes only the FIRST `protected:`/`covers:` block
> (`re.search`); `_strip_spec_manifest_blocks` (`:646`, added in cycle 3) strips EVERY block
> (`while True`). The strip removes strictly more than the parse consumes, so every block after the
> first is neither manifest-protected nor advisory-rescued — a `SPEC.md` naming a checkpoint under
> a second `protected:` block has it **DELETED at exit 0, zero diagnostic, by BOTH deleters**.
> Proven a regression by monkeypatching the strip to identity (0 deletions, 3 kept). It inverts
> ruled model item 3 using the code added to honour it, and it is **D38-c3's shape at the TEXT
> level: two functions disagreeing about what the same bytes mean.** Plus two MAJORs — a malformed
> `protect:` line is reported then IGNORED (an inline trailing comment gets the named checkpoint
> deleted at exit 0), and `_root_escape` gets an empty allowed-list when `was_resolved` is False so
> the in-run pruner deletes outside every configured root for an out-of-tree non-symlinked run dir,
> contradicting its own docstring.
>
> **Nothing is at risk today:** the `--apply` freeze (D32/D34) STANDS, `--ckpt-retention` stays
> `recency`, unreadable ⇒ UNSCORED ⇒ never score-pruned, and the real-tree dry run keeps all 7.
> **Resume per D45(a): re-cut by capacity — `881I` (ingestion only: ONE function owns both parse
> and strip, returning `(entries, spans)`; may PARSE, CLASSIFY and REFUSE, deletes nothing) and
> `881D` (the protection model + deleter, resumed from the verified work above).**

> **SCOPE ADDITION 2026-08-08 (factory run 28, TASK_876 shipped as `task876-score-emission`).
> This row now also owns the GC's SCORE READER, and it is a hard gate on TASK_880 that D34
> did not encode.**
>
> TASK_876 now emits `scores.jsonl` at every save — but **the shipped reader cannot resolve
> those rows for 2 of the 3 checkpoint naming schemes.** Reproduced independently by THREE
> agents during run 28:
>
> * `tools/ckpt_gc.py::_load_scores` builds its dict keyed by **`epoch`**, while `plan()` looks
>   up `scores.get(_epoch_of(p))` and `_epoch_of` parses the number out of the **directory
>   basename**. For flax-native `checkpoint_<step_count>` dirs — every periodic and final save
>   `train_jax_fast.py` writes — that number is the **STEP**, not the epoch, so the lookup
>   misses and the checkpoint reads UNSCORED forever.
> * For `SafeCheckpointManager`'s `checkpoint_epoch_<N>` dirs, `_epoch_of` returns `None`
>   outright (the regex needs digits immediately after `checkpoint_`), and `_is_ckpt_dir` does
>   not match the name at all.
> * **Only `snapshot_ep<N>` round-trips correctly.**
> * Separately, `_is_ckpt_dir`'s fallback treats any directory merely CONTAINING a
>   `checkpoint_*` entry as one opaque leaf checkpoint. Since `train_jax_fast.py` writes
>   `checkpoint_<step>` directly under the run dir, pointing the tool at its own default root
>   collapses the **entire run** into a single unit and never reads its `scores.jsonl` at all —
>   reproduced live: `1 ckpts, 0 scored` against a run whose `scores.jsonl` held 3 valid rows.
>
> **The fix:** resolve by the `path` / `step` fields TASK_876 already writes, not by
> re-deriving a number from the directory's own name; and stop the fallback matching a
> directory whose children are themselves real checkpoints.
>
> **Failure direction is fail-SAFE today** (unreadable ⇒ UNSCORED ⇒ never score-pruned) and
> `--apply` remains under the D32/D34 freeze, so nothing is at risk right now. But it is
> load-bearing for sequencing:
>
> **D34's gate for TASK_880 is NECESSARY BUT NOT SUFFICIENT.** D34 un-gates 880 on "876
> deployed to `gpu-server` AND a run has written real `scores.jsonl` there". That condition
> can now be satisfied in full while `--unscored drop` *still* reads essentially every
> checkpoint as unscored — which is precisely the state D34 exists to eliminate. 880
> additionally requires this row's path-first reader. Seeded as **D35 (OPEN)**.
>
> Also inherited from run 28: `stash@{0}` ("TASK_876 rescoped half …") is **deliberately
> retained** — it still holds the protect-scan hardening this row wants (`_is_protected`
> descendants, TTL re-scan, failed-`rmtree` audit row, `scan_referenced_paths`
> `followlinks=False`) plus the disputed `_order_newest_first` finding. Do not drop it.

**Status**: GATED 2026-08-06 — do not start before TASK_876 (rescoped emission half) and
TASK_880 land. **TASK_876 SHIPPED 2026-08-08.** Created by the D32 ruling, opt (a) as the
model + opt (d) as the split.
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

## Absorbed from TASK_876 by the D33 ruling (2026-08-07) — these are THIS row's items now

Run 27 halted 876 twice in this surface. All of it lands here, and `stash@{0}` already carries
verified implementations of the first three:

- **`_is_protected` descendant-awareness** and the **TTL re-scan** — D32 mis-filed these as
  "mechanical"; they are the protection model itself (planner error, owned in D33).
- **Failed `rmtree` → FAILED audit row + non-zero exit** (never a silent successful prune).
- **`scan_referenced_paths` walks `followlinks=False` — BLOCKER, reproduced under `--apply`:** a
  `SPEC.md` behind a symlink cited `checkpoint_100` and the tool deleted exactly it, silently.
  `onerror` never fires, `ProtectScanIncomplete` never raises, `stale` stays `False`. Not live
  today (`experiments/` has no symlinks) but **`jax_ckpt_fast` and `jax_ckpt_easy` ARE symlinks**
  to `/mnt/scratch`, so the same class is one directory move away. A protect scan that
  cannot see a path must FAIL, never silently return "nothing to protect".
- **DISPUTED: `_order_newest_first` deleting the freshest dir on disk** through `plan()` under
  `under_floor=True`. Correctness reproduced it by execution; fail-closed hand-traced it and could
  not break it. It is a **pre-existing HEAD bug**, not from any 876 cycle. Resolve it with a THIRD
  independent measurement — an executed repro outranks a trace, but two disagreeing reads deserve
  a tiebreak rather than a winner by seniority.

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
