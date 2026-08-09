# Factory run 29 — 2026-08-08

One row shipped, one row halted. Both outcomes came from the same place: the verification panel
refusing to accept a guard's own account of itself.

## Shipped

| Version | Item | Verification outcome |
|---|---|---|
| `task874-imprint-frame-provenance` (`cecfcfd`) | TASK_874 — `imprint.py` emits frame provenance for the checkpoints it writes | 3 cycles · 4-lens panel · **4 BLOCKERs + 3 MAJORs found and closed**, 0 outstanding · selftest 12/12 · 874 smoke 8/8 · canon byte-identical (953/953 leaves) |

## Halted

| Row | Rule | Outcome |
|---|---|---|
| TASK_883 — `eval_vector.py` universe-identity fail-loud | 3-cycle halt rule | **D36 seeded (OPEN).** Work retained in `git stash`. 5 BLOCKERs total; three closed, and the last two were *introduced by the fixes*. |

## The decision-changing finding

**A guard that reasons about the command line is not a guard about the run.**

TASK_883 exists because `eval_vector.py` accepted a missing universe pin in silence and printed a
paste-ready LEDGER row for a scrambled slot→asset mapping — the defect that manufactured the 882
crisis and cost a full planner window. The row was declared XS/S, "arg validation". Three cycles
later the guard refuses everything the task file named, byte-identity holds on both pin paths, and
it **still ships a green label on a wrong panel** by two independent routes:

1. Provenance is located as a path-*sibling* of the `--checkpoint` string. Point at the orbax step
   subdir — a spelling that appears verbatim in committed manifests — and the guard reports the
   checkpoint as provenance-less, so a **fully reversed** pin grades to exit 0 with
   `universe_pinned: true`.
2. The `explicit-symbols` state stamps its label from argv, never from what `build_panel` returned.
   A DB-absent symbol is silently dropped, every later slot shifts, and the artifact reads
   `universe_pinned: true, padded: true` with 6 of 8 slots wrong — net@21 −3.953, paste-ready.

Both have one root: **argv is not evidence.** The only thing that can certify a run's universe is
the checkpoint that was actually loaded and the panel that was actually built. Patching the two
symptoms separately is what consumed cycles 2 and 3.

This is the same family as the run-26 finding the queue already records — *a guard that normalizes
looks identical to a guard that corrects* — with a third member now named: **a guard that asserts
looks identical to a guard that verifies.** TASK_874 hit it too, from the other side, and that is
why its emission had to move behind every refusal the run can raise: a provenance file written
before the run disowns itself is an assertion, not a verification, and a judge drove a
self-refused checkpoint through `scorecard.py` to exit 0 to prove it.

## An orchestrator call the planner should review

Cycle 2's guard was correct by the task file and **wrong for the program**: it refused the active
828/837 build's entire eval path, because `train_jax_fast.py` emits no frame provenance and those
jobs pin their universe through `--periods-file` instead. The only way through would have stamped
`universe_pinned: false` on a run whose universe *is* correctly pinned — a lie in the opposite
direction. I ruled that an explicitly-specified universe must not be refused, and cycle 3
implemented it.

The ruling was right that those runs must not be refused. It was wrong to let them claim the same
green label as a checkpoint-verified run. That is hole 2, and it is mine. D36's recommendation
(demote the weak state rather than patch it) is written to close it at the root.

## Why 874 took three cycles for an "XS" row

Worth recording, because the pattern is now three-for-three on this surface. Every cycle's panel
found a real BLOCKER in the *previous* cycle's fixing code:

- The first implementation emitted provenance ~380 lines *before* the run's own 863
  degenerate-replay refusal — a run that exits 1 saying "this artifact is NOT a result" still
  certified it, and the task's own "zero manual steps" goal removed the friction that used to make
  that visible.
- Closing that by refusing to overwrite an existing file then meant a *differing* file is preserved
  rather than corrected — which is right, but it had to be paired with the pre-training check to
  stay safe.
- Closing *that* left the refusal firing after `promote_parity_baseline` had already replaced the
  git-tracked `results_fixture_smoke.json` — a baseline moved by a run that then exits 1, the 864
  class exactly.

None of these were visible from the writing context. All three were found by a lens that had not
written the code.

## Incident — disclosed

During TASK_874's first pass the implementer's own scope-3 check wrote and then deleted the real
`runs/861_ref/frame_provenance_ref_s0.json` — gitignored, no git copy, load-bearing for
`LEDGER 861/ref-oos-rebase`. It reconstructed the file and self-reported.

The restoration was then verified **independently rather than on trust**: `frame_symbols` is
order-exact identical to the pin `runs/832_e5/frame_provenance_imprint.json` and to s1/s2 (same
sha256), 1524 B like its siblings, every other field differing only in the seed index. That list is
the file's entire load-bearing content and is fully re-derivable from the pin, so the file is
correct. A judge independently caught the same anomaly from mtimes alone.

**This is the second incident in three runs caused by an irreplaceable gitignored artifact.** The
class TASK_873 just closed for the fixture is still open for `runs/`. Mitigation applied this run:
`runs/861_ref/frame_provenance_ref.sha256` and `runs/832_e5/frame_provenance_imprint.sha256`
sidecars now exist (`sha256sum -c` verified). That is a floor, not a fix — they live on the same
disk as what they check.

## Where it stands

- **Ready, untouched:** TASK_877 (XBT alias fold), TASK_878 (failed-fill semantics), TASK_881
  (checkpoint protection model).
- **TASK_883 is blocked on D36**, not on effort. Do not resume it from the stash without a ruling
  on what `universe_pinned` is allowed to assert — a fourth cycle against the current contract
  would produce a fourth variant of the same lie.
- **Stash indices shifted this run.** `git stash list` now reads: `{0}` = TASK_883 (this run),
  `{1}` = TASK_876's retained half (D35 says TASK_881 resumes from it, calling it `stash@{0}`),
  `{2}` = TASK_880. **Resume by message, never by index.**
- `trading_data.db` never opened. Fixture never regenerated — `b0f8de8b…` verified at the start and
  end of every suite. No committed baseline moved (`results_fixture_smoke.json` still `ede04cec…`).
  Nothing pushed.

## The two questions that unblock the most

1. **D36 — what may `universe_pinned: true` assert?** My recommendation is (a)+(c): bind the guard
   to the loaded checkpoint and the realized panel, and ship the verified `checkpoint-provenance`
   half now behind an explicit "walkforward lane not yet covered" banner rather than letting it sit
   in a stash for three runs the way 876/880/881 did. The narrower question underneath it, which is
   really a program call: **should `--periods-file`-pinned walkforward runs be comparable to
   checkpoint-provenance-pinned runs at all?** Today's measurement says no — the same checkpoint on
   the same 8-symbol order reads −1.796 one way and −1.866 the other, and the gap is the frozen-cell
   mask, not slot identity.
2. **Does `ledger_row` need the caveat?** Scope 3 said "stamp the emitted JSON" and that is
   literally satisfied — but `ledger_row` is the field humans actually copy into
   `experiments/LEDGER.md`, and an unpinned run's row is still byte-shaped exactly like a pinned
   one's. The 882 postmortem's complaint was precisely "prints a paste-ready LEDGER row with no
   warning". It still does.
