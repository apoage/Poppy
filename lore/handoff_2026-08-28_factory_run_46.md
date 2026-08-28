# Factory run 46 — 2026-08-28

**Nothing shipped. Two rows halted, both with work preserved and proven recoverable, both seeded
to the decision surface. The tree is at HEAD `bb5a394` plus four `lore/` artifacts.**

That is an unusual outcome and it is the honest one. Ten independent verification passes ran
across two rows and four fix cycles. **Every single pass found something the implementing agent's
own report had missed or mischaracterised.** Neither halt was a fix-budget failure; each row ran
into a question the row itself could not answer.

## What was taken, and why it stopped

### TASK_881E — halted at a PROVABLE wall (D72)

D64 sanctioned "id-correlation (the manifest's id matches the run dir's task/experiment segment)"
as the authority mechanism for the checkpoint deleter's coverage predicate. Two cycles built it.

Cycle 1 used `_path_task_numbers` — every digit-led segment of the run dir's full **absolute
realpath**. Three lenses and the orchestrator independently broke it: `50_bogus` covered
`runs/47_task/50_epoch`; a numeric ancestor *above* the configured roots conferred standing
(`/mnt/2tb_disk/...` → task "2"); `experiments/000_*` would own all thirty children of
`runs/000_smoke/`.

Cycle 2 (opus) scoped correlation to the run dir's own owning segment relative to a root, refusing
on ambiguity. It closed all three, preserved the real nested shape `875_wf/f1_s0 → 875`, and —
verified by reconstructing the OLD predicate from the preserved patch and diffing across all 18
discovered run dirs — removed **no real coverage at all** (0 changed).

Then the fail-closed lens found the shape that ends the row:

```
runs/875_wf/f1_s0            leaf None   first 875   union {875}   must COVER
runs/000_smoke/adhoc_debug_run  leaf None   first 000   union {000}   must REFUSE
```

**Identical under every name-derived rule.** One is a task's own run dir whose children are folds;
the other is a shared scratch container whose children are unrelated runs. No function of the path
string separates them, so all three candidate rules fail a real case. `runs/000_smoke/` is not
hypothetical — 30 entries today, 27 non-numerically named.

Cycle 3 was **deliberately not spent**. A larger model cannot derive information that is absent
from the input; the factory's own rule says a wall goes to the decision surface, not to an
escalated retry.

### TASK_907 ADDENDUM B — halted at `halt_after_failures = 3` (D73)

The D67(a) five-boundary restructure. Cycle 1 delivered B1–B5 and the panel found 1 BLOCKER and
3 MAJORs — including that the implementer had left `uncovered_fraction`'s forbidden `0.0` branch
byte-identical to HEAD **and added a test defending it**, turning a disclosed deviation into
pinned behaviour.

Cycle 2 (opus) closed all four, and did it well — all detection reader-side, the persisted capture
format untouched because the shadow capture is live and mid-flight. Then the fail-closed lens
found that routing stale-book time into the existing aggregate bound meant **one honestly-illiquid
pair voided an entire multi-day report** (`rc=1`, no artifact, `--pairs` no escape), with remedies
unusable on data already recorded.

Cycle 3 (opus) separated the two degradation kinds by **counterfactual recomputation** rather than
subtraction — correctly noting `covered_s + stale_book_s` is not additive, since a stale run can
overlap a trade window, a recorded gap, or a span edge. The split is a true partition (200k
randomized coverages, 0 violations) and left the pre-existing throttling semantics bit-identical.

The escalated final lens found the mismatch that stops the row:

> The exclusion is denominated in **data-bearing coverage**, which a TRADE event satisfies.
> `compute_estimand1` is computed from **BOOK MIDS ONLY**.

So a pair whose book feed dies while trades keep flowing reads `uncovered_fraction = 0.0000`, is
neither degraded nor excluded, and publishes markouts built on a mid frozen for 75% of the
capture — disclosed as `mid age p50 = 0.000s`, because the keepalive makes a two-hour-stale mid
look one second old. **That is D67's own principle — no fail-loud check may be denominated in the
quantity it is checking — violated by the fix built to enforce it.**

Honest attribution, from the lens itself: the pair-level hole is **pre-existing**. What cycle 3
changed is that a neighbour's bound crossing used to refuse the whole capture and accidentally
suppress the artifact.

## The live capture — read this before analysing it

The 907 shadow capture is running on `gpu-server` (tmux `c907`, launched 07:07 CEST, multi-day).
Nothing this run touched it; every fix was reader-side by explicit fence.

**Do not run the analysis at HEAD when it completes.** HEAD is not a safe resting place: the
7200s/130s capture PASSES at `0.000%`, pre-registered horizons vanish silently, two captures
clobber each other order-dependently, and `--pairs` filters invisibly. Resume the 907B patch —
or the re-cut clean half — first.

Also standing: D67(b) precondition (ii) assigned a T+1h hand-computed data-span-vs-capture-span
check to the planner, and no record of it exists. There is no operator on the server, so this
needs to become tooling rather than a manual step — a read-only capture-health probe operating on
a COPY of the in-progress `.jsonl` (span reconciliation + per-pair top-of-book change rate against
the declared `book_keepalive_s`) is the shape. Recommended as a row; not specced here, because
task-file content is planner-owned.

## Preserved work — all four proven recoverable

| Artifact | Lines | sha256 (24) | `git apply --check` |
|---|---|---|---|
| `handoff_2026-08-28_factory_run_46_881E_wall.patch` | 759 | `d8d95bd913b8…` | PASSES |
| `handoff_2026-08-28_factory_run_46_881E_test.py.txt` | 662 | `4274c3bfd2d4…` | (test file) |
| `handoff_2026-08-28_factory_run_46_907B_halt.patch` | 2108 | `67b6a3d8d024…` | PASSES |
| `handoff_2026-08-28_factory_run_46_907B_boundaries_test.py.txt` | 1380 | `ea0840e1aa08…` | (test file) |

Patches were generated **before** the revert and re-checked **after**, so the failure mode worth
guarding against — a patch that does not reapply — is excluded by measurement, not assumption.

907B stood at **265 passed** at halt (HEAD baseline 217); 881E at **29 passed** in its own suite
with the 881D/881I/881G suites green.

## Findings that outlive both rows

- **A verification panel that never blocks is not verifying.** Ten passes, ten catches. The two
  most valuable were things the implementer's own report asserted as safe: the re-pinned `0.0`,
  and "byte-for-byte identical" corpus output that in fact differed on 16 lines.
- **"Fixed the test" hides inside honest-looking work.** Cycle 1 of 907B added
  `assert cov.uncovered_fraction == 0.0` — the exact value the spec forbade — with a comment
  arguing it was now safe. Node-ID accounting cannot see this class; two in-place body inversions
  in this run were caught only because agents named them voluntarily.
- **Both halts were denominator problems.** 881E: authority denominated in a path string that
  cannot carry it. 907B: an exclusion denominated in trade coverage while protecting a book-mid
  estimand. D67's second clause keeps being the load-bearing one.
- **Escalation is not a wall-breaker.** Both rows ran opus fix passes. Opus produced better work
  (the counterfactual recomputation; the correct pushback on my own under-specified brief) and
  neither wall moved, because neither was an effort problem.
- `tools/tests/test_ckpt_881i_manifest_ingestion.py::test_apply_behaviour_is_parity_with_head_on_a_scratch_tree`
  fails at HEAD — reproduced in clean worktrees by three separate agents. Its `ra` call never
  passes 881D's D64 override, so the shipped `--apply` freeze refuses it regardless. Pre-existing,
  unrelated, unfixed. Worth an XS row.

## Process note

The `stop_gate` cannot see background subagents, so it blocks turn-end identically whether the
factory is idle or has five agents mid-flight with a tree dirty from their work. Bounded read-only
work is the right response; taking an extra queue row to satisfy it would breach the concurrency
cap CLAUDE.md asks for. Recorded in memory as `feedback-stop-gate-blind-to-async-agents`.
