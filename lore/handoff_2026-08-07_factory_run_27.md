# Handoff — dark factory run 27 (2026-08-07)

Implementer lane. Two rows taken, **both HALTED on the 3-cycle rule, nothing shipped, nothing
committed beyond this documentation**. Two new OPEN walls: **D33** (TASK_876) and **D34** (TASK_880).

Tree is clean at HEAD (`702b2e8`). `python -m trader selftest` 12/12. `trading_data.db`
mtime `1785091652` / size `26245337088` unchanged throughout. `fixtures/trading_fixture.db`
sha256 `b0f8de8b…` unchanged, never regenerated. Nothing pushed. `tools/ckpt_archive.py --apply`
never invoked.

## Stashes (all work preserved, re-apply verified clean on HEAD)

| Stash | Contents |
|---|---|
| `stash@{0}` | TASK_876 rescoped half, cycles 1-3 |
| `stash@{1}` | TASK_880 ckpt_archive discovery fix, cycles 1-3 |
| `stash@{2}` | TASK_876 original (run 26, pre-rescope) — untouched, still the D32 artifact |

## Outcome

| Row | Cycles | Result |
|---|---|---|
| TASK_876 (rescoped, emission half) | 3/3 failed | HALTED → D33 |
| TASK_880 (ckpt_archive discovery) | 3/3 failed | HALTED → D34. **`--apply` freeze STANDS** |

Verification spend: 11 subagents — 6 verification lenses on 876 (correctness ×2, fail-closed ×2,
regression ×2, flow-verifier), 3 on 880, plus 5 implementer passes (2 sonnet, 3 opus). Every
BLOCKER below was reproduced by execution, not argued.

## The measurement that matters most this run

`tools/ckpt_archive.py` dry-run against the real `gpu-server` with `--no-keep-newest`, **before**
any of run 27's work:

```
ARCHIVE  1 ckpt  / 0.09 GB
DROP-WITHOUT-ARCHIVE 27 ckpts / 2.51 GB
```

One operator command from deleting 27 of 28 checkpoints on the training host, including
`runs/861_ref/ckpt_ref_s0/checkpoint_2000` at net@21 **0.9914** — the clean-recipe anchor the
program's in-window result rests on. After the discovery fix: `ARCHIVE 28 / DROP 0`.

The planner's operational freeze was load-bearing, not precautionary. It stays shut.

## Root cause behind D34: a missing queue dependency, not a bug

MEASURED on the host: **no `scores.jsonl` and no `checkpoints_index.json` exists anywhere on
`gpu-server`.** The emitter is real in code (TASK_876) but is not deployed. So 100% of
`ckpt_archive.py`'s score signal is inference over `results*.json` sanity replays — ad-hoc
downstream eval artifacts with no contract on the `checkpoint` field.

Consequence: `--unscored drop` (correctly operator-ruled) reads "nobody wrote the file" as
"measured and found worthless". Those are different facts. A live instance is already on the tree —
`runs/832_e5/imprint_full.json` carries net@21 **0.8394** but sits outside the probe's filename
glob, so `ckpt_imprint_full/checkpoint_2000` reads UNSCORED today and survives only because one
LEDGER line cites it.

**880 cannot be made safe by fixing discovery.** It needs 876's emission deployed and running on
the host first. The queue currently lists 876 → 880 as adjacent rows with no such gate; that gate
should be explicit.

## What is verified and waiting in the stashes

**TASK_876 (`stash@{0}`)** — all four rescoped items, across six lenses:
- `scores.jsonl` emission is unconditional at every save (it sits before the retention gate, not
  behind it). Measured live at the `recency` default; appended across a simulated restart; a
  pre-epoch crash records `score: null`, not a fabricated `0.0`.
- The three mechanical fixes, mutation-checked red-then-green in BOTH deleters.
- `flax_keep = 3` unconditional + `overwrite=False`. Measured on flax 0.10.7/Orbax:
  `overwrite=True` saving step 25 over 10/20/30/40 deletes 30 AND 40; `overwrite=False` leaves both.
- `--ckpt-retention` defaults to `recency` at all three leak points, and `_checkpoint_space_gate`
  makes that default **pre-876-equivalent, verified line-by-line against a HEAD control worktree**.
- Collapse case re-derived on real flax saves via the offline tool (keeps `snapshot_ep3000/4000/5000`
  + newest; legacy `keep=3` keeps the three most collapsed).
- Canon proven safe: the five `experiments/` sites that gained `overwrite=True` are numerics-neutral
  — the 835 smoke run twice moved only the embedded date string; Sharpe gross `-14.848`, net@21
  `-46.432`, finalPV `96836` byte-identical.

**TASK_880 (`stash@{1}`)** — 33 tests, all real subprocess/`ca.main()` runs, 20/20 mutation
red-then-green, re-derived by two fresh judges. Outcome-neutrality on the real host was **proven**
(fixed vs semantically-reverted probe: DIFF NONE, 21/28 scored both ways).

## Why each row stopped

**TASK_876 — both cycle-3 BLOCKERs are in the protect-scan, i.e. TASK_881's surface.**
`scan_referenced_paths` walks `followlinks=False`, so a symlinked directory under `experiments/` is
skipped with no error: `onerror` never fires, `ProtectScanIncomplete` never raises, `stale` stays
`False`. Reproduced under `--apply` — a `SPEC.md` behind a symlink cited `checkpoint_100` and the
tool deleted exactly that checkpoint, with no CRITICAL and no warning. Not live today
(`experiments/` has no symlinks), but this repo already uses that topology: `jax_ckpt_fast` and
`jax_ckpt_easy` are themselves symlinks to `/mnt/scratch`.

**TASK_880 —** `sym_below_root()` silently no-ops when a configured root is itself a symlink.
`ROOT_ABS` is built with `realpath()` (resolves); every path compared against it comes from
`os.walk` on the unresolved root and only passes `abspath()` (does not). Nothing prefix-matches, so
every row under that root is unguarded. Reproduced: swapping `latest -> checkpoint_2000` moved a 0.9
score across checkpoints and made the genuinely-scored one an unflagged drop candidate, with a
completely clean dry-run.

## Unresolved disagreement, recorded rather than adjudicated

On `_order_newest_first`, the correctness judge reproduced a wrong deletion through `plan()` under
`under_floor=True` (deleted `checkpoint_200`, the freshest dir on disk, kept a 300s-staler
`snapshot_ep10`); the fail-closed judge hand-traced the same function and could not break it,
arguing the per-family newest-keep defends independently. Executed reproduction is weighted above
trace, but no third measurement was taken. D33 asks whether this is settled by a third measurement
or reverted to HEAD behavior and left to 881.

## Orchestrator error, owned

The disputed `_order_newest_first` finding lives in a change that was **not in the rescoped scope**.
The mtime-tie bug is pre-existing at HEAD; the orchestrator added it to the cycle-3 brief after a
judge flagged it. D32 split TASK_876 precisely because emission and deletion policy had been welded
into one row — cycle 3 then welded a fourth concern onto the row's last cycle. The rescoped four
items were never the problem; the protect-scan hardening thread was pulled in cycle by cycle as each
judge finding pointed at TASK_881's surface.

The generalisable form: **when successive verification findings all point at another row's surface,
that is the signal to stop and re-split, not to keep fixing.**

## Process findings for the planner

1. **Verification agents write committed baselines in the shared working tree.** The regression
   judge ran the real 835 smoke, which rewrote `experiments/835_stage4_eval/results_fixture_smoke.json`;
   it noticed and restored with `git checkout --`, and disclosed it. Had it died mid-smoke, a tracked
   baseline would have been silently re-based. Same class as the run-25 fixture-regen incident, with
   a different file, and the lane guard does not cover it because `experiments/` baselines are not
   planner-owned.
2. **Green acceptance gates were not evidence of a safe deleter.** All four of TASK_880's cycle-1
   BLOCKERs are cases gates (i)–(iv) do not exercise — a basename collision, a symlinked tree, a
   corrupt `scores.jsonl` line, a NaN score. The gates were too narrow; that is a spec finding, not
   an implementer failure.
3. **Source-text assertions passed while the behavior was broken, twice.** `test_trainer_overwrite_false`
   was an AST check that "would not and did not" catch the `overwrite=False` regression. Same class
   as run 26's green-test-asserting-a-wrong-value on 872. Executed behavior tests plus a mutation
   check should be the default for any assertion about a deleter or a save path.
4. **Pre-existing red, confirmed not ours:** `tools/tests/test_jax_env.py` (2) and
   `test_physics_core.py` (1) fail identically at a clean HEAD worktree (`NameError`). Verified in a
   throwaway `git worktree`, not accepted on report.

## Recommended next moves

- Rule **D33** and **D34**. Both carry a recommendation the factory can act on without further
  discovery.
- The highest-value single action is **D34 rec (b)**: gate 880 behind deployed emission, and record
  that dependency in `QUEUE.md`.
- If the long base-training program is near launch, note D33 rec (a)/(b): emission is verified and
  independent of the protection model, so it can ship without the protect-scan once that work moves
  to TASK_881.
