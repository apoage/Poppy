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
