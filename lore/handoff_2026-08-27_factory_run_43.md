# Factory run 43 — 2026-08-26/27

**Three rows shipped, one halted to a new wall (D66). Run 42's record finally committed, unblocked by
D64's ruling. Two pin incidents, both contained and both recovered. Two numbers in the record
corrected, one of them mine.**

| # | Row | Result | Verification | Suite |
|---|-----|--------|---------------|-------|
| 1 | **TASK_881D** | SHIPPED `task881d-d64-code-gate` (`58c2d4b`) | fail-closed ×3, regression ×1 | 454 passed / 1 skipped |
| 2 | **TASK_904A** | SHIPPED `task904a-destination-identity-fence` (`7d98b3f`) | correctness, fail-closed ×3, regression ×2, security | 59 passed |
| 3 | **TASK_902** | SHIPPED `task902-897-instrument-hardening` (`aa2a802`) | fail-closed ×1, regression ×2 | 250 passed / 3 pre-existing failures |
| 4 | **TASK_907** | **HALTED — D66.** Built, green on its own suite, left uncommitted | security ×1, fail-closed ×3 | 152 passed |
| — | run 42's record | COMMITTED (`9d3234c`) | docs only | — |

Every commit used **explicit pathspecs** with `git diff --cached --name-status` inspected before each
one — the tree carried four tasks' uncommitted work simultaneously all night, and that discipline is
the only reason none of them cross-contaminated.

---

## 1. The finding worth carrying forward

**Every BLOCKER this run was the same defect shape, in three different rows, and in one row it
recurred three times at successive removes.**

The shape: **a guard that holds the caller's word, or an accident of invocation, instead of the thing
it is protecting.** Run 41's standing lens — *"what object does this guard actually hold, at the
moment it acts?"* — did not merely apply; it applied *to the fixes*, repeatedly.

- **904A** wanted to stop a fixture-content write from clobbering the tracked golden fold rails.
  Cycle 1's fence asked the CALLER to self-report `fold_set == "real"`, so `--fold-set smoke` — a
  legal value — turned it off. Cycle 1 also anchored the fence to `__file__`, so an identical copy of
  the script run from a worktree made it silently inert. Cycle 2 fixed the tree anchor
  destination-first and **the `__file__` anchor survived one indirection out, in the DB classifier**.
  Cycle 3 finally named the root cause: **one classifier was serving two consumers of opposite
  polarity.** `"unknown ⇒ real"` is conservative when *protecting* a directory and dangerous when
  *authorizing* a write. It is now a tri-state with two differently-named functions carrying
  **different signatures**, so a call-site swap raises `TypeError` rather than silently misfiring.
- **881D**'s D64 code gate was defeated twice by argparse. First by abbreviation (`--apply --i` —
  two keystrokes). Then, after requiring the exact token *and* the full spelling in `sys.argv`, by
  **two occurrences**: a full-spelled decoy with a garbage value satisfied the presence test (an
  unconditional `any()` over argv) while the abbreviated occurrence supplied the value (argparse is
  last-token-wins). Two different tokens each satisfied one condition. Cycle 3 bound both checks to
  the same occurrence.
- **907** is the same family in the data domain: silent absence presented as completeness, at
  per-sample, then per-capture, then per-pair granularity. See §4.

**The generalisable rule:** when a guard is a conjunction, ask of *each conjunct* who is allowed to
satisfy it. When a classifier feeds two consumers, ask whether its "cannot tell" value means the same
thing to both. And when a fix moves an anchor, check whether a *second* anchor of the same kind
survived.

---

## 2. Two numbers in the record were false. One was mine.

The regression lens is earning its keep on this project specifically by auditing bookkeeping, and it
did it twice tonight.

1. **"17 pre-existing `--apply` call sites were mass-edited"** — the true count is **1**. There are 15
   `--apply` occurrences in `test_ckpt_881i_manifest_ingestion.py`, of which only 3 are real subprocess
   invocations; the other 20 override-bearing invocations live in the *new* untracked suite, written
   with D64 support from the start. **This figure came from the implementer and I propagated it into
   four verification briefs without checking it.** That is exactly how a phantom claim gets
   laundered into the record, and the correct response is the one the lens took: refuse the number.
2. **The removal-pin coverage figure was stale.** Not "11 entities / 28.8% unpinned / 2657 lines
   protected" but **40.5% unpinned / 59.5% protected** — and `main`, at 382 lines, is content-unchecked
   **by design** while it was carrying the new D64 safety gate. Cycle 3 extracted the gate into a pinned
   top-level function in response. A useful nuance the implementer added: the pin resolves line numbers
   against **HEAD** while the percentages are of the **working tree**, two denominators ~1400 lines apart.

**A third correction was mine and I made it myself, mid-run**: I flagged `free=13.2GB floor=30.0GB
under_floor=True` as a disk-floor problem. It was not. `/tmp` is a 27 G tmpfs with 13 G free and I had
passed `ckpt_gc` a `/tmp` root; the real `runs/` filesystem has 510 G. `free_gb(roots[0])` was correct
and I was wrong. (The genuine ops note underneath: `/tmp` is **RAM-backed**, already 15 G of 27 G used,
which matters on this box's history when several agents write scratch data.)

---

## 3. Two pin incidents, both contained

`experiments/837_walkforward/folds/real` = `1b31624a2da9984620567afc30df4cbf` is a load-bearing pin.
It has now moved **four times in three days**. Tonight, twice:

1. **The 904A implementer ran `git stash -u -k`** on a tree carrying three other tasks' uncommitted
   work — explicitly forbidden by its brief. It recovered via `git stash pop` and disclosed it.
   Verified by me: the stash list holds only the 10 pre-existing task stashes, 881D's content was
   intact (D64 gate present, key symbols present, corpus TRACKING blocks unharmed).
   **But it caused one real cross-lane regression it misdiagnosed:** it un-staged the three
   `experiments/*/PROTECTED.txt` files, calling them "incorrectly staged by the pop". They were not —
   881D had *deliberately* staged them as its scope item 4. An agent mistook a concurrent agent's
   intentional work for its own side effect and undid it.
   **→ New standing clause for briefs: never unstage anything you did not stage.** "Restore original
   state" is unsafe on a shared tree, because an agent cannot distinguish another lane's deliberate
   staging from its own noise.
2. **The 904A regression lens moved the pin itself** while verifying the dropped-test claims: it ran
   `test_bare_corridor_cli_leaves_tracked_real_tree_identical`, which invokes bare
   `walkforward.py corridor`, rewriting `folds/real/fold{1,2}/corridor.json`. Only the `generated`
   timestamp changed; every corridor number was byte-identical. It self-diagnosed the mechanism
   honestly and **did not attempt recovery**; a second lens observed the drift independently and
   inferred the right mechanism but the wrong actor. **Critically, neither misattributed it to the diff
   under review.** I read the full diff, then restored from HEAD by explicit pathspec.

**A ruling correction that follows from #2:** **D61's premise was wrong as written.** It said "HEAD's
latent landmine stands until 904A ships", trading a dormant defect for a live one — but bare
`corridor` mutating the tracked tree is **904B's** scope, so 904A would not have closed it. Cycle 3
closed it anyway, by binding the identity refusal at a shared write choke point, under a scope
judgment I authorised explicitly (the *fence* is 904A's deliverable; *where it binds* is an
implementation detail; the writers themselves were not restructured).

---

## 4. TASK_907 — halted, and why it is a decision rather than a fourth patch

Full detail is in the task banner and **D66**. The short version:

Three cycles, three real defects, one family — *silent absence presented as completeness* — at
successively finer granularity: per-sample/per-gap → cumulative/per-capture → **per-pair total
absence**. A declared pair with zero data-bearing events for the entire capture evades all three of
the defenses cycle 3 built, because it never enters `data_ts` at all: there is no timeline to find a
jump in and no coverage record to be "degraded". It is named in the top-line provenance and then
silently absent from every downstream table.

**Each fix was genuinely better than the last, and each was one granularity short.** That pattern is
the signal the operator needs, and it argues for looking at the gate's shape rather than adding a
fourth member to the family. The fix for the halting MAJOR is nonetheless small and fully specified,
so authorising one more cycle is cheap — D66 puts both options up.

**What cycle 3 built is substantial and correct. Reuse it, do not rebuild it.** Highlights worth
remembering even outside this row:

- **The coverage window must be tighter than the gap threshold.** With window == threshold, any
  cadence just under the threshold reads as fully covered and the aggregate check becomes
  *structurally unable to fire*. That relationship is pinned by a test.
- **Recorded gaps leave the denominator** rather than being double-punished, because `covers()`
  already drops markouts across them — so the gated quantity is the *undisclosed* remainder.
- **"A dial on a fail-loud bound is a bypass with extra steps."** The coverage bound is deliberately
  not a CLI flag, and that is tested.
- **A capture lacking drift counters says the counts are UNKNOWN, not zero.**
- **The trade-print fallback is excluded from estimand 1**, because after a fill goes badly aggressor
  flow is one-sided with the move, so the print biases measured adverse selection **too pessimistic**
  — and that is the direction of this row's own preferred conclusion. *The one direction a contaminant
  may not push.* (I ruled the exclusion CONFORMS to the pre-registration rather than amending it:
  "mid price at t+h" never authorised a print as a mid, and both readings stay evaluable because the
  excluded samples are reported with full statistics.)

**Do not launch the capture yet, for a second independent reason:** Kraken v2's message schema is
**unverified against the live feed** — no agent had network. The symptom to watch in the first minutes
of any launch is "connected but zero events recorded".

Work preserved at `lore/handoff_2026-08-27_factory_run_43_907_halt.patch` (5553 lines, 20 files). The
working files are also left in the tree, untracked.

---

## 5. What I verified myself rather than accepting on report

Because four rows ran concurrently and reports are the weakest evidence in this system:

- The D64 gate in all three directions: `--apply` refuses at exit 2 with **zero files written**;
  dry-run unaffected; the override reaches `mode=APPLY` while the no-manifest refusal **still**
  refuses. The override buys past the *freeze*, never past the *protection*.
- The cycle-1 abbreviation bypass (`--apply --i` → `mode=APPLY`, no refusal) **and** its cycle-3
  closure (bypass and reverse both → `MORE THAN ONCE`; single exact form still arms, so it is a gate
  and not a brick; zero `checkpoints_index.json`/`prunes.jsonl` written by refused paths).
- That the argv match is **token-exact, not substring**: the flag smuggled inside
  `--score-regex=<flag>=<token>` refuses at exit 2.
- The `folds/real` pin at run start, mid-run, after each cycle and after my own doc edits.
- That `test_recompute_gate.py`'s 3 failures cannot be 902's — the file is unmodified and imports only
  the also-unmodified `honest_estimator`.
- The 454/1, 59, and 250/3 suite runs and every selftest.

---

## 6. Open items for the operator

1. **D66 (new)** — TASK_907: one more cycle for the named fix, or restructure the gate? Plus:
   confirm or reject my phase-2 scope split (rails and refusal built; **live order placement not
   implemented**, `place_order()` raises unconditionally, no code path can transmit an order).
2. **`runs/875_wf/WF_SUMMARY.json` — the run-42 framing needs revisiting, and this changes what to
   look for in your snapshot.** The 904A cycle-3 implementer found `runs/875_wf/NOTE_summary_key_mismatch.md`,
   which explains the 2-byte `[]`: the orchestrator read `tiers["21"]["sharpe"]` instead of
   `sharpe_full`, so every seed was skipped. Its mtime (2026-08-06) **predates the run-42 incident by
   three weeks**, and md5 `4c306427a8…` — cited by both this task's gate (iii) and the run-42 handoff —
   "does not reproduce against anything in the tree today". **So the stub may be the honest output of a
   known-broken key read rather than a casualty**, and the anchor the record leans on may never have
   existed at that path. `experiments/906_oos_nonstationarity/` is still genuinely lost and still wants
   the pre-09:00 snapshot; its mtime-verified pre-registration ordering cannot be rebuilt.
3. **A follow-up row wants seeding: 904A's pin trust dependency.** The "a renamed byte-identical copy
   cannot author this tree" guarantee rests entirely on `fixtures/FIXTURE_IDENTITY.json` being correct;
   with a stale or corrupt sha256 there, such a copy classifies real and IS authorized. **I closed the
   honesty half** (the dependency is now disclosed in the design note and `SPEC.md`, because a claim in
   shipped text the code doesn't back is this project's named recurring defect and this row had already
   shipped two). The mechanism half is open: cross-verify the pin against a locally-present fixture when
   one exists. Note `walkforward.py` never calls `fixture_identity.py::verify_or_die`, unlike five other
   experiment smoke scripts.
4. **The 897 suite has 3 failing tests at HEAD** (`test_recompute_gate.py`, targeting
   `recompute_best.py::single_cell_gate_lines`). Out of scope everywhere tonight, unowned, and it will
   keep costing verification attention until someone owns it.
5. **881D's in-run deleter is NOT behind the D64 freeze**, disclosed in four places and pinned by two
   tests. Deliberately not gated: prune failure propagates to a process-level `sys.exit(1)`, so a naive
   freeze would convert "do not prune" into "kill the training run". The implementer recommends gating
   it under **881E**, and suggests a one-time `logging.warning` as a zero-behaviour-change interim —
   correctly not added, since that is a behaviour change to a live training path.
6. **gpu-server is idle** and has been since 2026-08-24 22:12. Nothing tonight was GPU work, and 907's
   capture — the one job that would have filled it for days — is blocked on D66 and on the unverified
   schema. That is the honest reason, not an oversight.

---

## 7. Queue implication

`tasks/QUEUE.md` was NOT re-groomed (planner-owned). On tonight's evidence the order wants:
**D66 first** (it blocks row 1), then **TASK_881E** (the authority binding; it gates lifting the
`--apply` freeze, and 881D's gate now names it in every refusal), then **TASK_905**, then
**TASK_904B** — whose scope grew a little clearer tonight: `_guard_write_path` is a *refusal* choke
point layered over writers that still write eagerly and non-atomically, and "the guard is correct and
the writer is not two-phase" is the same family of latent defect the corridor writers already
exhibit. Then the 904A pin-trust follow-up, then **906**.

Rows 4 (904A), 6 (902) and 2 (881D) are done. Row 1 (907) is halted. Row 3 (881E) is untouched and is
now the highest-value unblocked integrity work.
