> **STATUS (2026-08-27, factory run 45): SHIPPED.** Panel fix pass closed 1 MAJOR / 1 MINOR / 1 NIT.
>
> **Scope item 1 decision — option (a), resolve against `--fold-root`.** Option (b) ("keep `_REPO`
> resolution, refuse when `--fold-root` disagrees with the manifest's own tree") closes the false
> PASS equally well but leaves the tool useless for the case `--fold-root` exists to serve: a
> relocated tree is *by definition* not the tree the manifest describes, so (b)'s refusal would fire
> on every legitimate audit. Materialize always writes `<out_root>/<fold_id>/<basename>` (and the two
> job manifests at `<out_root>/<basename>`), so the recorded path's prefix is discarded and rebuilt
> from `fold_root` — correct under any relocation, in-repo or not, without knowing the tree's origin.
>
> **Scope item 2:** a path resolving outside `--fold-root` is refused by `_require_within_fold_root`
> **before any `open()`** — containment checked on `realpath` of BOTH sides (the TASK_886/888 lesson;
> a `startswith` on unnormalized strings catches `..` but not a planted symlink). Re-proved by running
> the escape case with a REAL, fully readable target directory present and again with it deleted:
> byte-identical stdout+stderr, exit 1 both times — so no `open()` of the out-of-root path occurs.
>
> **Panel FINDING 1 (MAJOR, fixed).** The guard raised immediately from inside the fold loop, breaking
> `validate_folds`' documented "raises LeakError listing EVERY violation" contract: violations from
> earlier folds were discarded from the raised message (the only thing on stderr, the only thing CI
> greps), and folds iterated AFTER the escape were never evaluated at all — an order-dependent,
> silently incomplete audit, in exactly the scenario `--fold-root` exists for (a partially restored
> backup with more than one problem). The refusal is now recorded via `rec()` like every other check
> and the single terminal raise carries all of them. **Not softened** — still exit 1, still no
> out-of-root read; the guard still raises, the CALLER catches, one line above the first read. Covered
> by a both-iteration-orders test that was RED before the fix (post-fix stderr carries 8 violations in
> one message where HEAD emitted 1).
>
> **Panel FINDING 3 (NIT, fixed):** the read-only sqlite handle now closes via `try/finally` on every
> exit path (a pre-existing leak reachable via the unguarded corridor-JSON parse).
>
> **DISCLOSURE — a SECOND, unrequested bug was fixed alongside the named one.** Replacing
> `os.path.relpath(files["train_pit"], _REPO)` with the raw recorded string in `fold_train_files` is a
> real behaviour change, not a refactor. Both `fold_manifest.json` and the job manifests store
> `_rel(p)`, but the old code re-derived one side with `relpath(p, _REPO)`; the two spellings agree
> **only in-repo** (`_rel()` falls back to an absolute path outside the repo, `relpath` does not).
> Measured: materialize + validate against a genuinely out-of-repo directory — HEAD **falsely REJECTS**
> (`membership in the fold's train files=False` on every train job, though the files are correct); the
> new code gives `ALL 34 checks PASS`. A false-refusal closure, not an under-detection, violating no
> gate since no current caller materializes out-of-repo — but it was fixed silently, so it is recorded
> here rather than only in a code comment. `run_smoke`'s NOTE 1, which documented that divergence as
> "out of scope to fix", is rewritten accordingly.
>
> **Known reading caveat (disclosed, not a defect):** an escaping fold now `continue`s, so its
> `fold_train_files`/`fold_uni_syms` entries are never populated and the later manifest cross-check
> emits ADDITIONAL failures for it. That is more noise, never less — an escaping fold is genuinely
> unvalidated, so reporting its jobs as unverified is honest — but a reader should not count those as
> separate independent defects.
>
> **Gates:** (i) tampered relocated copy → exit 1 naming `smokefold1`/`period_train.json` (HEAD on the
> SAME input: `ALL 34 checks PASS`, exit 0 — the false PASS reproduced first-hand, twice, by two
> independent parties). (ii) pristine relocated copy → exit 0, `ALL 34 checks PASS`. (iii) in-place
> invocation byte-identical to HEAD, stdout md5 `5df05024fef118fcc2d3a8211a72ce76`, stderr empty,
> and unchanged across the fix pass. (iv) `runs/875_wf/grade_correct.py` md5
> `e87b080d4ea092e85b98deef1af7805a`, 269 files, zero diff vs HEAD; grader re-run reproduces
> `positive: 4 | above own floor: 1`. Suite **66 → 69 passed**; `trader selftest` 12/12; full `smoke`
> PASS. Panel: correctness + regression + flow-verifier; fix pass at opus.
>
> **Remainder → planner:** `experiments/837_walkforward/folds/smoke/` is a gitignored leftover tree
> from an earlier smoke run sitting in the repo; and the `except Exception` around `_read_periods` in
> the fold loop is broad enough to swallow a `LeakError` from the parse path as a generic
> "artifacts-parse" failure (pre-existing; the resolution guard sits outside it, so the escape path is
> not reachable through it).

# TASK_905 — `validate --fold-root` audits the ORIGINAL tree, not the one you pointed it at

**Status**: READY 2026-08-23 (seeded by factory run 41 from TASK_904's flow-verifier — a
pre-existing defect found while rebuilding 904's decoy demonstration, NOT introduced by 904 and
deliberately left out of that row's scope so a nearly-green row would not widen).
**Size: S. Risk: MEDIUM in consequence** — this is an AUDIT tool that returns a false PASS.

## The defect

`validate_folds` resolves every periods-file path from the manifest's own recorded repo-relative
path rather than from the `--fold-root` the operator named:

```python
# experiments/837_walkforward/walkforward.py:731
files = {k: os.path.join(_REPO, v) for k, v in f["files"].items()}
```

So `validate --fold-root <copy>` re-reads the files under `_REPO` — the ORIGINAL tree — no matter
what directory it was pointed at.

**Measured, first-hand (factory run 41, flow-verifier, on a copy):** `cp -r` a fold tree, tamper
one `period_*.json` in the copy, run `validate --fold-root <copy>` → **`ALL 36 checks PASS`,
exit 0**, because the tampered file was never opened; the untampered original was validated
instead. A tool whose entire job is to detect fold corruption reports clean on a corrupted tree.

**The harness already knows about this failure mode in one place and fixed it only there.**
Smoke's own internal helper `_clone_fold_tree` does a text `old_rel`→`new_rel` replace across
`fold_manifest.json` and both job manifests, with a comment naming the exact hazard ("job
manifests were previously left pointing at the original tree, so even a PRISTINE copy was
rejected on paths, and the negative tests had zero discriminating power"). That fix is internal
to smoke's test helper; the public `validate` CLI never got it.

## Why it matters beyond a copied directory

`--fold-root` is the natural audit action: point the validator at a relocated tree, a restored
backup, a staged copy on `gpu-server`, or a candidate re-materialization, and ask whether it is
intact. Every one of those returns a verdict about the WRONG FILES today. This is the same class
the program keeps paying for — a guard that reasons about a different object than the one it
opens (D59's `//localhost` URI bypass; TASK_886/888's realpath-vs-abspath divergence).

## Scope

1. Resolve fold-file paths against the tree actually being validated. Decide deliberately
   between (a) resolving relative to `--fold-root` / the manifest's own directory, and (b)
   keeping `_REPO` resolution but REFUSING when `--fold-root` is not the tree the manifest
   describes. Either closes the false PASS; (a) makes the tool useful on copies, (b) makes it
   honest but still unusable for the audit case — say which and why in the task banner.
2. Whatever is chosen, a validated path that resolves OUTSIDE the named `--fold-root` must be a
   loud refusal, never a silent redirect.
3. Fold `_clone_fold_tree`'s knowledge into the shared path so the CLI and the smoke helper stop
   diverging — "one implementation" per the 870 remainder, scoped to this surface.
4. Keep TASK_904's shipped guarantees intact: `materialize`'s `--overwrite-existing` gate, the
   provenance disclosure block, the fixture-over-real refusal, and the scratch-dir sweep.

## Acceptance (gates)

- (i) The measured repro goes RED: tamper a `period_*.json` in a COPY, `validate --fold-root
  <copy>` must FAIL and name the tampered file. Prove the pre-fix code passes the same input.
- (ii) A pristine copy at a different path still PASSES (no false refusal — the `_clone_tree`
  comment records that over-strict path handling once made the negative tests vacuous, which is
  the opposite failure and just as bad).
- (iii) The in-place, non-relocated invocation is byte-identical in behavior to HEAD — this is
  the path every existing caller uses and no existing verdict may move.
- (iv) TASK_875's 903-verified numbers do not move (`folds_positive=4`,
  `folds_above_own_floor=1`); `runs/875_wf/grade_correct.py` md5 `e87b080d4ea092e85b98deef1af7805a`
  and the 269 raw per-seed JSONs stay byte-identical.

## Out of scope

TASK_904's own scope (shipped separately). The `SPEC.md:60-72` real-db re-materialization. The
`orch_wf875.sh` dead inline summary tail (documented as historical record in
`runs/875_wf/NOTE_summary_key_mismatch.md`).
