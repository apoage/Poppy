# TASK_881D cycle 2 — two fixes REVERTED at the ingestion scope wall (planner decision needed)

Both are blocked ONLY by `tools/tests/test_ckpt_881i_manifest_ingestion.py::INGESTION_NAMES`,
which pins `load_manifests` and `_run_manifest_check` byte-identical to HEAD. Neither has any
other drop site; implementing either outside its drop site would require a SECOND reader of the
same inputs — the exact anti-pattern this cycle's MAJOR just removed from `--protect-file`.

## A. MAJOR — `--verify-manifests` can never exit 0 on the real tree
Drop site: `_run_manifest_check` (PINNED). Shape that was built and measured working:
  * new CLI flag `--known-dangling PATH [PATH ...]` (named `known-` not `expect-` because
    `--expect-dangling` collides with `--experiments-dir` on the prefix `--exp`, which would
    break `test_abbreviated_flag_spelling_is_detected_as_explicit_too`);
  * new module-level `_spellings_of(path)` + `class _AcknowledgedIndex` (NOT pinned, fine);
  * inside `_run_manifest_check`: the dangling loop and `_root_error` consult the index; an
    acknowledged condition still PRINTS in full but does not set `ok = False`; a summary line
    plus a loud `STALE ACKNOWLEDGEMENT` for an acknowledgement that matched nothing.
  Measured before revert: real tree `--verify-manifests --known-dangling jax_ckpt_fast
  jax_ckpt_fast/e5exit2_pq_s{0,1,2}` -> exit 0, all 4 conditions printed; a NEW dangling entry
  beside them -> exit 2; a malformed manifest at an acknowledged path -> exit 2.

## B. MINOR — a manifest under `.git`/`__pycache__` is a SILENT withdrawal
Drop site: `load_manifests` (PINNED). Shape that was built and measured working:
  * two locals `read_paths: set[str]` / `pruned_manifests: list[str]`;
  * in the walk loop, for each name in `names` that is in `_WALK_PRUNED_DIRNAMES` and is a real
    non-symlink dir, `os.walk` it for `MANIFEST_BASENAMES` and append;
  * `read_paths.add(os.path.realpath(fp))` on every manifest actually read;
  * after the loop, print `MANIFEST NOT SCANNED: <path> ...` on stderr for every pruned-dir
    manifest whose realpath is NOT in `read_paths` (the suppression keeps the D53
    `--experiments-dir experiments experiments/__pycache__` shape green), and add
    `manifest_files_unscanned_pruned` to `stats`.
  Deliberately NOT a refusal: `test_control_the_ancestor_alone_really_does_not_reach_it`
  asserts exit 0 on exactly this tree.
