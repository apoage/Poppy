# TASK_880 — `ckpt_archive.py` carries `ckpt_gc.py`'s discovery defect, on the tool that deletes REMOTELY

> **STATUS (2026-08-07): HALTED, NOT SHIPPED — stopped on the 3-cycle rule, new wall D34.**
> **THE `--apply` OPERATIONAL FREEZE STANDS AND IS NOT LIFTED.** Gate (v) is explicitly NOT met.
> Cycle 1 → 4 BLOCKERs + 4 MAJOR (two judges, independently). Cycle 2 (opus) → 1 BLOCKER.
> Cycle 3 (opus) → 1 BLOCKER. Code preserved in `git stash` ("TASK_880 ckpt_archive discovery fix —
> cycles 1-3, HALTED …"); nothing committed, tree clean at HEAD, selftest 12/12, `trading_data.db`
> and fixture byte-untouched, `--apply` never once invoked against the real host.
>
> **THE MEASUREMENT THAT JUSTIFIES THE FREEZE RETROACTIVELY.** Real-host dry-run, `--no-keep-newest`:
> BEFORE this work `ARCHIVE 1 ckpt / 0.09 GB · DROP-WITHOUT-ARCHIVE 27 ckpts / 2.51 GB` — one
> operator command from deleting 27 of 28 checkpoints on `gpu-server`, including
> `runs/861_ref/ckpt_ref_s0/checkpoint_2000` at net@21 **0.9914**, the clean-recipe anchor.
> AFTER: `ARCHIVE 28 / DROP 0`. The planner's freeze was load-bearing, not precautionary.
>
> **VERIFIED AND EARNED (in the stash):** 33 tests, all real subprocess/`ca.main()` runs, 20/20
> mutation red-then-green, re-derived independently by two fresh judges — containment tested BEFORE
> the basename pattern, `NEVER_DELETE_BASENAMES` ported and enforced, symlinked checkpoints never
> collapsing a run nor becoming a deletion target, per-line `scores.jsonl` isolation biasing KEEP,
> non-finite scores rejected as UNSCORED, `_run` fallback gated, ambiguity flagged not silent.
> Outcome-neutrality on the real host was PROVEN, not asserted (fixed vs reverted probe: DIFF NONE,
> 21/28 scored both ways).
>
> **WHY IT STOPPED:** `sym_below_root()` silently no-ops when a configured root is itself a symlink.
> `ROOT_ABS` uses `realpath()` (resolves symlinks); every path compared against it comes from
> `os.walk` on the UNRESOLVED root and only passes `abspath()` (which does not), so nothing
> prefix-matches and every row under that root goes unguarded. Reproduced: swapping
> `latest -> checkpoint_2000` moved a 0.9 score across checkpoints and made the genuinely-scored one
> an unflagged drop candidate, with a completely clean dry-run. The suite tests both halves
> separately and never in combination.
>
> **THE DEEPER FINDING → D34, and it is a QUEUE dependency, not a code defect.** MEASURED: **no
> `scores.jsonl` or `checkpoints_index.json` exists anywhere on `gpu-server`.** The emitter is real
> in code (TASK_876) but undeployed, so 100% of this tool's score signal is inference over
> `results*.json` sanity replays — ad-hoc eval artifacts with no contract on the `checkpoint` field.
> `--unscored drop` (correctly operator-ruled) then reads "nobody wrote the file" as "measured and
> worthless". Live instance already present: `runs/832_e5/imprint_full.json` carries net@21
> **0.8394** but sits outside the probe's filename glob, so `ckpt_imprint_full/checkpoint_2000`
> reads UNSCORED today and survives only because one LEDGER line cites it. **880 cannot be safe
> until 876's emission is deployed and running on the host** — the queue lists these as adjacent
> rows with no such gate. Sanctioned interim (D34 rec (a)): `--unscored archive` only.

**Status**: HALTED 2026-08-07 (was READY 2026-08-06 — planner, from the factory's run-26 recommendation; D32 ruling).
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
