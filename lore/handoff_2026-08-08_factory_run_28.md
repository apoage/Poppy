# Factory run 28 — 2026-08-08

Three rows shipped, one new human wall (D35). Factory GREEN, tree clean, nothing pushed.

| version | item | verification |
|---|---|---|
| `task876-score-emission` (`ae127b4`) | TASK_876 — checkpoint score emission + `overwrite=False` + `recency` default | 4-lens panel + 2-lens re-verify · 2 cycles · 1 MAJOR closed · 0 outstanding |
| `task882-instrument-reconciliation` (`bbb2d38`) | TASK_882 — the two argmax instruments reconciled | 3-lens panel + fresh correctness re-verify · 2 cycles · 1 BLOCKER + 1 MAJOR closed · 0 outstanding |
| `task873-fixture-identity-pin` (`261077b`) | TASK_873 — tracked fixture identity pin, fail-loud | 3-lens panel + fresh fail-closed re-verify · 2 cycles · 1 BLOCKER + 1 MAJOR closed · 0 outstanding |

Plus `86a462f` — D35 seeded, TASK_881 scope addition, queue grooming.

## TASK_876 — the third attempt, and D33's re-cut is what made it work

Two prior halts (6 cycles across runs 26–27) both died in the checkpoint protection surface.
D33 re-cut the row to three items with **zero deleter surface**; the implementer resumed
`stash@{0}` as a **subtraction** — 1,864 → 611 insertions, `tools/ckpt_gc.py` restored
byte-identical to HEAD. With no deleter surface left, no protect-scan finding had anywhere to
land. That is the whole story of why it shipped this time.

Shipped: unconditional `scores.jsonl` emission at every save · `flax_keep=3` + `overwrite=False`
(measured on flax 0.10.7/Orbax: `overwrite=True` saving step 25 over `checkpoint_{10,20,30,40}`
deletes 30 AND 40) · `--ckpt-retention` defaults to `recency`, `score` refuses naming TASK_881.

**The MAJOR was a good catch by the regression lens.** `_attempt_checkpoint_save`'s blanket
`except Exception` turned disk-full/permission failures into `TRAINING COMPLETE!` + exit 0, while
`dojo_manager.py:315` gates purely on returncode — so a curriculum level would advance having
written nothing. Now only the three designed outcomes log-and-continue. The inverted risk was
checked explicitly: a round-epoch run whose periodic and final saves collide at the same step
still exits 0 with the normal banner.

## THE WALL — D35 OPEN. D34's gate for TASK_880 is necessary but NOT sufficient.

Shipping 876 revealed that **emitting scores does not make the GC's score signal real.**
`tools/ckpt_gc.py::_load_scores` keys by `epoch`, while `plan()` looks up
`scores.get(_epoch_of(p))` and `_epoch_of` parses the number out of the **directory basename** —
which for flax-native `checkpoint_<step_count>` dirs is the **STEP**. `checkpoint_epoch_<N>`
matches nothing at all. **Only `snapshot_ep<N>` round-trips.** And `_is_ckpt_dir`'s fallback
collapses a whole run into one opaque leaf, so the tool pointed at its own default root never
reads that run's `scores.jsonl` (`1 ckpts, 0 scored` against a file with 3 valid rows).

**Reproduced independently by three agents.** Fail-safe today (unreadable ⇒ unscored ⇒ never
score-pruned) and the `--apply` freeze STANDS — so nothing is at risk. But D34 un-gates 880 on
"876 deployed AND real `scores.jsonl` on the host", and **that condition can now be met in full
while `--unscored drop` still reads everything as unscored** — the exact state the gate exists to
prevent. D35 recommends amending the gate, and asks whether **TASK_881 should move ahead of
TASK_880** (881 is currently gated *behind* 880, which is backwards under this finding).

`stash@{0}` is deliberately RETAINED — it still holds 881's protect-scan work.

## TASK_882 — the ruling got narrowed, and that was the point

Mechanism confirmed: a missing `--panel-frame-from-checkpoint` silently routes eval_vector onto a
coverage-ranked default universe. The encoder is **positionally universe-locked** (slot k IS asset
k), so the model gets a scrambled mapping — 63/64 positions differ, 24 symbols absent, because the
858 backfill moved the ranking. **First divergence at decision 0**: the very first observation
window is already wrong, not accumulated drift. Sizing eliminated with a working control.

**The first-round ruling was REJECTED by the panel** for claiming "imprint.py correct, +0.9914
restored" while ignoring a *correctly frame-pinned* eval_vector read of the same checkpoint
(`861/ref-oos-rebase`, `old_ckpt_clean`) reading −0.8805/−1.3883/−0.4102/−0.1697 with
weight-rebalance ON. The fix pass resolved it: **the counter-example is an EPOCH pair, not an
INSTRUMENT pair** — the same pinned scorecard reads **12/12 positive** on the clean-trained
reference seeds, so the instruments agree whenever checkpoint, pin and epoch match. The old
checkpoint's own training run was itself unpinned and predates 853/858/865/868/869.

What ships: the unpinned −0.6145 is **withdrawn**, and every unpinned eval_vector number with it;
imprint.py is **NOT** vindicated in general; **+0.8394 NOT restored**; **+0.9914 CORROBORATED, not
restored**; **the old checkpoint's disagreement is UNRESOLVED** and this row does **not** close
`861/ref-oos-rebase`. 875 and 879 unaffected.

**Planner action:** the decisive same-window pinned measurement is pre-registered and NOT run
(scheduled real-data work). **Blocker to scheduling it:** `runs/861_ref/frame_provenance_ref_s{0,1,2}.json`
are cited by shipped artifacts but are **absent locally** — recover them first.

**Program-level determinism hazard:** the same probe reads net@21 **−4.9678 on GPU** vs **−4.9876
on CPU** — 0.4%, enough to break the suite's own 1e-4 tolerance. Production paths self-guard
`JAX_PLATFORMS`; an in-process import of `eval_vector` does not.

## TASK_873 — the guard against the accident nearly repeated the accident

Tracked `fixtures/FIXTURE_IDENTITY.json` pins sha256 `b0f8de8b108b…` (79310848 B) — the generation
every committed baseline ran against. Fail-loud verification at the start of all four consuming
suites; atomic `--promote-fixture-identity`; a `make_fixture.py` overwrite guard.

**The BLOCKER is the instructive part.** The first write guard **failed OPEN** — falling through to
`os.remove` + rebuild, silently, exit 0 — when the pin was missing, corrupt, **or the fixture
already mismatched it**. That third case is the post-incident recovery state, so once mismatched
every later bare invocation would silently rebuild forever. All three lenses reproduced it; one
did so by deleting a single file. It inverted the 866 rule on the one side where damage is
irreversible. Fixed by deleting the flag: **if the fixture exists and promote was not passed, the
run refuses unconditionally.** Over-refusal was then checked in the other direction — a fresh
clone still generates normally.

**Intended behavioral break:** a bare `python fixtures/make_fixture.py` against an existing fixture
now always refuses. No caller exists in the repo today, only documentation references.

## Where it stands

**Ready:** 1 TASK_877 (XBT alias fold) → 2 TASK_874 (imprint emits frame provenance) → 3 TASK_878
(failed-fill semantics, last and alone — it rewrites eval-core execution semantics).
**GATED:** TASK_880 (D34 + now D35) · TASK_881 (and D35 asks whether it should jump ahead of 880).

## Process notes, both mine

1. **I started the 876 fix pass while the flow-verifier was still running**, so it observed the tree
   changing under it. It re-read the source and re-confirmed its findings, but the overlap was
   avoidable — stop the verifier first. I did not repeat it on 882 or 873.
2. **A subagent ran one bare GPU job** (~2 min, no `MEM_FRACTION`/`PREALLOCATE` guard) during the
   882 fix pass — an invariant violation, self-disclosed, mitigated in the script. It is also how
   the CPU/GPU determinism gap above was found.

## Recommended new rows (none implemented — all out of scope where found)

- **`eval_vector.py` fails OPEN on universe identity** — runs unpinned against a frame-locked
  checkpoint and emits a clean banner, a paste-ready LEDGER row and a `results_*.json` with no
  warning and no `panel_frame_pinned` marker, while `imprint.py` hard-`RuntimeError`s (TASK_863).
  A fail-open asymmetry between the two harnesses. **Highest-value row on this list.**
- **`--safe-checkpoints` is entirely broken at HEAD** — `_save_checkpoint_data` `json.dump`s
  `np.float32` → `TypeError`, crashing on the first golden save. Reproduced on clean HEAD by three
  agents. One-line `default=float`.
- The LEDGER's new INSTRUMENT DISCIPLINE rule is prose with no mechanical enforcement.
- Three pre-existing `tools/tests` failures at HEAD; malformed `experiments/LEDGER.md` table rows;
  three `experiments/` callers of `save_checkpoint(tmpdir)` without `overwrite=True` (safe today
  only because each writes a fresh `tempfile` dir).
- `tools/deploy/run_queue.sh` still does not activate the venv (bare `python` → python2 on
  gpu-server). Already recommended in the queue; it has now cost two launches.
