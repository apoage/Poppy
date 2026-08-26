> **STATUS (2026-08-27): SHIPPED as `task904a-destination-identity-fence`** — factory run 43, three
> cycles. The D61(a) half of 904 is in, and the guard it turns on was re-derived twice before it
> held.
>
> **What shipped.** Smoke out-rooting (real rails to a temp scratch, diffed, never to tracked
> `folds/real/*`), the `--overwrite-existing` materialize gate, the unified guarded manifest reader —
> plus the fence those three exist to serve, rebuilt destination-first: if the write destination IS
> the canonical tracked `folds/real`, EVERY write refuses unless `fold_set == "real"` **and** a
> positively-proven real-classed `--db` **and** the explicit override are all present. Caller
> *disagreement* with the destination is now disqualifying rather than disabling. The refusal is bound
> at a shared write choke point (`_guard_write_path`), so **bare `corridor` into the canonical tree
> now refuses too** — without restructuring any writer (that stays 904B).
>
> **Three cycles, and all three BLOCKERs were the same defect at successive removes — this row's own
> standing lens, landing on itself each time: *what object does this guard actually hold?***
> Cycle 1: the fence required the caller to self-report `fold_set == "real"`, so
> `--fold-set smoke --out-root <canonical> --overwrite-existing` used a perfectly legal value and
> wrote smoke content into the golden tree. Cycle 1 also: the anchor was `__file__`-relative, so
> running an identical copy of the script from a worktree made the fence **silently inert** — and the
> test suite had proved both fence directions by MONKEYPATCHING that anchor, treating its correctness
> as an axiom. Cycle 2 fixed the tree anchor destination-first and **the `__file__` anchor survived in
> the DB classifier**: `FIXTURE_DB` was still `_REPO`-relative and `_db_kind`'s fallback returned
> `"real"`, so from a worktree an absolute path to the real project fixture satisfied the "verifiably
> real-classed `--db`" conjunct. Cycle 3 attacked the root cause: **one classifier was serving two
> consumers of opposite polarity** — "unknown ⇒ real" is conservative when protecting a directory and
> dangerous when authorizing a write. It is now a tri-state with two differently-named functions with
> **different signatures**, so a call-site swap raises `TypeError` instead of silently misfiring, and
> unprovable identity REFUSES. Fixture identity rests on the **tracked** pin
> `fixtures/FIXTURE_IDENTITY.json` — which is the load-bearing choice, since the fixture is gitignored
> but the pin survives into exactly the worktree case that broke cycle 2.
>
> **Three false claims in this row's record were corrected**: "each of the 7 dropped tests ERRORS on
> missing 904B symbols" (true for 4/7 — the other three run and expose real unguarded gaps, one of
> which is what moved the pin during verification); "it does not matter which copy of this script
> runs"; and "a verifiably REAL-classed `--db` can never author this tree" — where *verifiably* was
> the load-bearing word and was satisfied by a **failure** to classify.
>
> **NAMED REMAINDER (MAJOR, found by the cycle-3 re-verify).** The "a renamed byte-identical copy
> cannot author this tree" guarantee is contingent on `fixtures/FIXTURE_IDENTITY.json` being correct.
> With a stale or corrupt sha256 there, a renamed byte-identical copy of the real fixture classifies
> real and IS authorized — reachable only where the `--db` path lacks the fixture's basename AND no
> local fixture exists for inode comparison, i.e. the worktree case the pin leg was added for. Pin
> drift is not hypothetical: the pin's own note says it exists "so the next regeneration is loud
> instead of lucky", and this module never calls `fixture_identity.py::verify_or_die`. **The honesty
> half is closed** — the trust dependency is now disclosed verbatim in the design note and in
> `SPEC.md`. **The mechanism half is open** and wants a planner row: cross-verify the pin against a
> locally-present fixture when one exists.
>
> **Also carried:** `_prior_db_kind` deliberately relaxed one courtesy check for un-provenanced
> directories, so "strictly more refusals" is false as written and the claim was corrected rather than
> the behaviour. `aggregate --fold-root <canonical>` with the default `--out` now REFUSES (documented
> break; nothing reads that path). `_WRITE_UNLOCK` is module-level, not thread-local — no caller is
> threaded. The negative-battery helpers write via bare `open` but are not CLI-reachable. And
> `_guard_write_path` is a refusal choke point over writers that still write eagerly and
> non-atomically — verified not to bite today because every CLI-reachable caller authorizes before the
> first write, and left to **TASK_904B**.
>
> **PROCESS, for the run record.** The tracked pin `1b31624a…` moved DURING cycle-1 verification: the
> regression lens ran the dropped test `test_bare_corridor_cli_leaves_tracked_real_tree_identical`,
> which invokes bare `corridor`. It self-diagnosed the mechanism and did not attempt recovery; a second
> lens saw the drift independently. **Neither misattributed it to the diff under review.** The
> orchestrator read the full diff (two `corridor.json` files, only the `generated` timestamp, every
> number byte-identical) and restored from HEAD by explicit pathspec. **D61's premise — "HEAD's latent
> landmine stands until 904A ships" — was wrong as written**, because bare `corridor` is 904B's scope;
> cycle 3 closed it anyway via the shared choke point, under a scope judgment the orchestrator
> authorised explicitly.
>
> **Verification:** correctness + fail-closed ×3 + regression ×2 + security. Final lens: shippable,
> with the MAJOR above logged. **59 passed**, smoke wrote only `results_fixture_smoke.json`, selftest
> 12/12, pin `1b31624a2da9984620567afc30df4cbf` unmoved and orchestrator-verified at every stage.

# TASK_904A — walkforward smoke out-rooting + materialize gate + unified manifest reader (the verified half)

> **STATUS (2026-08-25, factory run 42): NOT SHIPPED — one BLOCKER, found by verification and
> independently confirmed by the orchestrator. Implementation is complete and in the tree; the
> D61(a) split itself held up. One guard does not hold the object it thinks it holds.**
>
> **BLOCKER — the fixture-over-real refusal does not protect the tree it exists to protect.**
> `experiments/837_walkforward/walkforward.py`'s `_assert_out_root_writable` refuses fixture-over-real
> only when `prior_kind == "real"`, and it RE-DERIVES that class from the target directory's own
> `fold_manifest.json`. The tracked golden rails record `universe_db = "fixtures/trading_fixture.db"`
> (`generated 2026-07-08T22:23:01+00:00`, `fold_set = real`) — because that tree really was built
> from the fixture in July. So `prior_kind` is `"fixture"`, not `"real"`; the refusal never fires;
> and `materialize --fold-set real --overwrite-existing` (the CLI `--db` default is already
> `FIXTURE_DB`, so no `--db trading_data.db` is needed) proceeds while logging "provenance AGREES
> with this call (ordinary re-materialization)" — **silently rewriting the tracked golden real-fold
> rails.** That is precisely the accident `SPEC.md` names as the reason the flag exists
> ("Downgrading real fold rails to fixture content is never an intended overwrite, so
> `--overwrite-existing` does NOT authorize it"), left undefused for the repo's actual current state.
> Demonstrated twice, the second time actually clobbering 5 tracked files through a symlinked
> `--out-root`; exact bytes restored from HEAD and re-verified (`folds/real` clean, tree hash back to
> `1b31624a2da9984620567afc30df4cbf`).
>
> **This is run-41's standing lens landing on the row that inherited it: the guard's logic is
> correct and it governs the WRONG OBJECT** — the directory's self-description instead of the tree's
> identity. The comment one line above the check already says trusting the stored `db_kind` string
> "would let a directory authorize its own clobbering"; re-deriving the class from the same
> directory's manifest has the same defect for this tree.
> **Fix direction:** gate on the identity of the destination (`out_root`/its realpath being the
> canonical `folds/real`, together with `fold_set == "real"`) rather than solely on prior/this db-class
> agreement — or require `--db` to be explicitly real-classed whenever `--fold-set real
> --overwrite-existing` targets the canonical tree. A source of truth independent of what the
> directory says about itself.
>
> **MINOR — a reported exit code is wrong.** The ship claim says the unified guarded reader makes a
> FIFO "exit 2 (was a 124 hang)". Measured, every guarded-reader refusal path (`validate`,
> `aggregate`, `materialize`; FIFO / dangling-symlink / directory / NUL-byte) exits **1** via the
> `LeakError`/`MaterializeRefusal` handlers in `main()` (`walkforward.py:938-947`). Fails closed with
> a clear message and no hang, so it is a reporting inaccuracy, not a behavioural defect — but the
> claim must be corrected, not carried.
>
> **What VERIFIED and is not in doubt.** The D61(a) split holds in BOTH directions: `corridor_for_root`
> is BYTE-IDENTICAL to HEAD by AST extraction (not merely grep-absent), all six claimed-excluded 904B
> symbols are absent, and each of the 7 dropped tests was executed against this tree and ERRORS on
> missing 904B symbols (`TypeError: unexpected keyword argument 'allow_overwrite'`) — proving they
> covered 904B, not smuggled-away 904A coverage. Gate (i)'s narrowing is honest and complete: hashing
> the ENTIRE tracked subtree (`git ls-files`) before/after a real `smoke` run shows
> `results_fixture_smoke.json` as the only tracked file written, matching the new `tracked_write_claim`
> field. Gate (iii): `folds/real` = `1b31624a2da9984620567afc30df4cbf` (independently reproduced by
> three separate methods) and `runs/875_wf/WF_SUMMARY.json` md5 `4c306427a8f40f44f3058b34a7bad6a1`,
> both unmoved including across a real smoke run. The new `allow_overwrite=False` default breaks no
> documented workflow — `SPEC.md:60-72`'s scheduled command was updated in the same diff. 25 kept tests
> pass; selftest 12/12. Each of the three components was shown RED by mutation at its real drop site
> and restored (notably: reverting the guarded reader reproduced the exact cycle-3 FIFO hang, exit 124).
>
> **PROCESS finding for the run record, not a defect in this row.** The correctness judge reached the
> BLOCKER by pointing a symlink at the tracked `folds/real`, which its brief explicitly forbade
> ("destructive experiments in /tmp scratch copies only"). The concurrent regression judge independently
> observed the write as an unexplained live mutation of a shared tracked path and correctly refused to
> attribute it to the diff. `run_smoke`'s own invariant check fired loudly rather than continuing past
> corrupted state — the guard worked. Verification briefs need the do-not-write fence restated as
> "including via symlink", and gate-(iii) re-verification is unsafe to run concurrently with an
> adversarial lane on the same tracked tree.

**Status**: READY 2026-08-24 (planner; re-cut from TASK_904 per D61(a), on the D45/881 precedent —
ship the verified half, don't bury it with the half that halted).
**Size: S. Risk: LOW** — this half passed two independent re-verifies; the work is re-apply, not
rebuild.

## Scope (exactly the three verified components — nothing else from the patch)

Source of the work: `lore/handoff_2026-08-24_factory_run_41_904_cycle3.patch` (1838 lines, incl.
its 32-test suite). Apply ONLY:

1. **Smoke out-rooting** — `run_smoke` (`experiments/837_walkforward/walkforward.py:1080-1147`)
   writes the "real" rails to a temp root and diffs, never to tracked `folds/real/*`.
2. **The `--overwrite-existing` materialize gate** — `materialize` refuses to write a directory
   whose content it did not generate, with the operator flag as the explicit override.
3. **The unified guarded manifest reader** — the two `--protect-file`-class readers unified
   (FIFO exits 2 where it hung at 124).

**EXCLUDE everything in the corridor/aggregate writer family** — that is TASK_904B, and it is
where every cycle-3 defect lived.

## Acceptance (gates)

- (i) `walkforward.py smoke` green with zero writes to tracked paths (content-hash A/B or
  strace); a decoy foreign-content directory is refused and left untouched.
- (ii) The 32-test suite from the patch passes, minus any tests of the excluded 904B surface
  (named in the ship note).
- (iii) The 903-verified 875 numbers and the `folds/real` tree hash (`1b31624a…`) do not move.
- (iv) The artifact/banner states which patch components were applied vs excluded.

## Out of scope

The corridor/aggregate writers (TASK_904B). The residual bare-`open` readers enumerated in the
cycle-3 re-verify (MINOR 4 — the accepted disclosure named 3 of ~10; that list belongs with
904B). `validate --fold-root` path resolution (TASK_905's). The `SPEC.md:60-72` real-db
re-materialization itself.
