# TASK_880 — `ckpt_archive.py` carries `ckpt_gc.py`'s discovery defect, on the tool that deletes REMOTELY

**Status**: READY 2026-08-06 (planner, from the factory's run-26 recommendation; D32 ruling).
**Size: S. Risk: HIGH by blast radius** (this tool deletes checkpoints on `gpu-server` after a
local verify — the failure mode is a whole multi-day run's weights, not a dry-run line).
Surface: `tools/ckpt_archive.py`, the embedded `REMOTE_PROBE` (~lines 38-80). Author of the
defect: the planner (this tool was planner-shipped in `3e74a1e`, not a factory row).

**OPERATIONAL FREEZE until this lands: `tools/ckpt_archive.py` must not be run with `--apply`.**
Dry-run enumeration is fine and is how you demonstrate the fix.

## The defect (factory-found run 26; planner-verified in source before scheduling)

`REMOTE_PROBE` defines `CK = (^snapshot_ep\d+$, ^ckpt(_.*)?$)` and then:

```python
def is_ck(p):
    b = os.path.basename(p.rstrip("/"))
    if any(r.match(b) for r in CK): return True
    try: return any(n.startswith("checkpoint_") for n in os.listdir(p))   # <-- the defect
    except OSError: return False
```

Three consequences, compounding in the same direction (toward deletion):

1. **A run directory that merely CONTAINS `checkpoint_<step>` subdirs is itself classified as a
   checkpoint**, so the walk is pruned there and the run's individual checkpoints are never
   enumerated. The run becomes one opaque leaf.
2. **`ep_of()` then finds no epoch** in that leaf's basename (a run dir is named e.g.
   `e5exit2_pq_s0`), and `scores()` is read from the leaf's own directory — so the run's
   `scores.jsonl` / `checkpoints_index.json`, which live at exactly that level, are looked up
   one level too deep and never found.
3. The pattern set also omits `checkpoint_\d+`, `checkpoint_epoch_\d+` and `checkpoint_golden`,
   and scores are epoch-keyed only, so flax dirs named by `step_count` cannot resolve a score
   even when discovery is correct.

**Net failure mode:** an entire SCORED run classifies as UNSCORED → under the `--unscored drop`
default (correct, operator-ruled — "no reason to store weights that i dont have info what they
are") it is judged uninteresting → and this tool then DELETES IT ON THE TRAINING HOST. The
default is right; the discovery feeding it is wrong. That is the whole task.

## Scope

1. Port the three landed `ckpt_gc.py` fixes into `REMOTE_PROBE`: a `_has_ckpt_subdirs()` test
   BEFORE the listdir fallback (a dir containing checkpoints is a RUN, not a checkpoint), the
   full pattern set (`checkpoint_\d+`, `checkpoint_epoch_\d+`, `checkpoint_golden`, plus the
   two existing), and **path-keyed score reading with the epoch key as fallback** (never the
   reverse — epoch-keyed-only is how one checkpoint's score gets lent to another).
2. Keep the probe a single self-contained string: it is executed on the remote host through the
   venv interpreter and must not import from the repo.
3. **Fail-closed on ambiguity**: if a path is classifiable both ways, it is a RUN (enumerate),
   never a leaf (delete). State this as a comment at the decision point.
4. Re-verify the LEDGER-derived protection (`--protect-from-ledger`) still rescues cited
   `runs/...` paths after the walk changes shape — it currently matches leaves, and the leaves
   move.

## Acceptance (gates)

- (i) On a synthetic remote tree containing (a) a run dir with `checkpoint_{1000,2000,3000}`
  subdirs + a `scores.jsonl`, (b) a `snapshot_ep\d+` leaf, (c) a bare `ckpt` leaf, (d) a run dir
  with NO score file: enumeration returns the three individual checkpoints of (a) each carrying
  its own score, not one opaque blob — demonstrated by the dry-run listing.
- (ii) A regression test pinning the exact pre-fix behaviour as WRONG (run dir → single leaf,
  score None) so it cannot silently return.
- (iii) `--apply` still refuses to delete anything not verified present locally (unchanged
  invariant, re-asserted after the walk change).
- (iv) Dry-run against the REAL `gpu-server` tree, output attached to the task: every run dir
  enumerated, and the scored/unscored split reported per run. No `--apply` in this task.
- (v) The freeze above is lifted in the task banner only when (i)-(iv) are green.

## Out of scope

The retention POLICY (that is TASK_876 / TASK_881); anything that uploads; changing the
`--unscored drop` default (operator-ruled, stays).
