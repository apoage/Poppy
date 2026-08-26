> **STATUS (2026-08-27): SHIPPED as `task881d-d64-code-gate`** — factory run 43. The D62(a)/D63(a)
> work that sat green-but-unshipped since run 42 is committed, and D64's `--apply` freeze is now a
> CODE GATE rather than a written convention.
>
> **What shipped.** (1) `--apply` refuses at parse/entry time — before any filesystem work, exit 2,
> empty stdout — unless `--i-have-read-d64-and-accept-the-unbound-coverage-risk=I-HAVE-READ-D64` is
> given in full, exactly once, with that literal token. The refusal names D64, TASK_881E and the
> lift condition (881E ships AND 880's D35 opens). (2) The `--keep-best N` MAJOR: `plan()`'s
> per-metric arithmetic is byte-UNCHANGED and the `keeping N` line now DISCLOSES the per-metric
> multiplication when more than one metric is present — option (b), not a real global cap, because
> `plan()` is a live per-epoch `apply=True` path (`train_jax_fast.py:757`) and a real cap would have
> converted the bug into NEW DELETIONS, the one forbidden direction. (3) The tie-break MINOR: exact
> score ties now break on recency, reusing `_order_newest_first`'s ordering rather than a second
> rule. (4) The three `experiments/*/PROTECTED.txt` corpus files are now TRACKED — untracked
> manifests never reach gpu-server, which is where `ckpt_gc.py` actually runs.
>
> **Three cycles, two of them because the gate kept being bypassable in the same shape.**
> Cycle 1: `argparse` defaults `allow_abbrev=True` and no other flag starts with `i`, so
> **`--apply --i` armed the override** — proven with real deletion. Cycle 2's fix required the exact
> token AND the full spelling present in `sys.argv`, and was bypassed again:
> `FLAG=NOT_THE_TOKEN --i=I-HAVE-READ-D64` — the "full spelling present" test was an unconditional
> `any()` over argv while the value came from argparse's last-occurrence-wins namespace, so **two
> different tokens each satisfied one condition.** Cycle 3 closed the CLASS: `spelled_in_full` is now
> an equality against the whole occurrence list, AND multiplicity refuses first, so the occurrence
> that supplies the arming value must itself be the full spelling. Measured to be mutually redundant
> (severing either alone still refuses; only both restores the bypass) and pinned as behaviour.
> **The gate was also extracted from `main()` and PINNED** to hardcoded sha256 digests of its AST
> entity text — it previously lived inside the ~40% of the file nothing checks.
>
> **Two false numbers in this row's own record were corrected, one of them the orchestrator's.**
> "17 pre-existing `--apply` call sites mass-edited" → the true count is **1**. The removal-pin
> coverage figure was stale: **1549 lines / 40.5% unpinned, 2280 / 59.5% protected** (not
> 28.8%/2657), with `main` at 382 lines content-unchecked by design. Both now stated accurately.
>
> **Named remainders, all disclosed in code and pinned by tests where applicable.** The D64 gap
> itself (manifest coverage has no authority binding) is unfixed BY DESIGN — that is what the freeze
> exists for; it lifts only via **TASK_881E** + 880's D35. **The freeze covers the offline CLI
> ONLY** — the in-run pruner is not behind it, disclosed in four places, deliberately not gated
> because prune failure propagates to a process-level `sys.exit(1)` and a naive freeze would convert
> "do not prune" into "kill the training run"; recommended for 881E. `main()` remains unpinned (the
> new pin covers the gate, not the delete loop). `train_jax_fast.py:757` carries no D64 comment.
> The implementer's 1204-shape argv sweep was ephemeral, not a checked-in test.
>
> **Verification: fail-closed x3 cycles + regression x1 — NOT the full HIGH tier**; correctness,
> security and flow-verifier did not run against this resume's diff (run 42 ran all five against the
> pre-resume work, which this did not modify; the security question IS D64, already ruled). Final
> lens: NO FINDINGS, with a shape table where positive controls genuinely delete and every bypass
> shape leaves all checkpoints intact, plus a 13-shape sweep confirming the class closed. Suites
> **454 passed / 1 skipped** (HEAD 288/1 → 393/1 pre-resume → 454/1, every delta itemised, 0 removed),
> selftest 12/12 — both run by the orchestrator, not taken on report.

> **STATUS (2026-08-25, planner): RESUME per D64 — the in-tree work SHIPS with ONE addition: the
> `--apply` freeze becomes a CODE GATE** (parse/entry-time refusal of `--apply` unless the
> explicitly-named override is present; refusal text names D64/TASK_881E). The freeze lifts only
> when TASK_881E (authority binding, seeded) ships AND 880's D35 opens. Everything else is
> already green in the tree; the resume is: commit the in-tree work + the code gate + the
> PROTECTED.txt corpus (D63(a)'s gate-vi note), re-run the suites, ship.**
>
> **STATUS (2026-08-25, factory run 42): HALTED ON A NEW WALL — see D64. The D62(a)/D63(a) work
> is BUILT, GREEN and IN THE TREE; it does not ship, and the reason is not a defect in it.**
>
> **What landed (both rulings satisfied).** D63(a)'s narrow scope amendment was implemented as
> specified: `--known-dangling PATH…` (with `_spellings_of` + `_AcknowledgedIndex`, matching by
> realpath/abspath set-intersection, never by prefix) so `--verify-manifests` can reach exit 0 on
> the real tree despite this row's 3 acknowledged dangling 828/scratch disk entries; and the
> walk-pruned-dirname report (`MANIFEST NOT SCANNED:` + `manifest_files_unscanned_pruned`) so a
> manifest under `.git`/`__pycache__` is no longer a silent withdrawal. `INGESTION_NAMES` stays the
> full 43-name inventory with the two sanctioned functions re-pinned to their NEW bytes via
> HARDCODED sha256 digests in `AMENDED_INGESTION_NAMES` (verified not self-derived — a digest
> computed from its own input would be a tautology). D62(a)'s gate (iv) amendment landed too:
> `test_canon_functions_are_byte_identical_to_head` RESTORED over `UNTOUCHED_GUARD_NAMES`
> (`_safe_target`, `_du`, `free_gb` + the five named constants) and
> `test_diff_against_head_removes_no_content_line` UN-SKIPPED behind a named
> `_881D_KNOWN_REWRITES` entity allowlist — measured at 11 entities covering **28.8%** of
> `ckpt_gc.py`'s 3734 lines, leaving **2657 lines still protected** against unexplained removal, so
> the restored pin is load-bearing and not a rubber stamp. Real-tree measurement reproduced:
> `--verify-manifests --known-dangling jax_ckpt_fast jax_ckpt_fast/e5exit2_pq_s{0,1,2}` → **exit 0**,
> all 4 conditions `ACKNOWLEDGED`; without the flag it still exits 2. Suites **393 passed / 1
> skipped**; selftest 12/12. Gate numbering normalized against the parent task file.
>
> **Why it does NOT ship: D64.** The manifest COVERAGE predicate has no AUTHORITY BINDING, and this
> row is what first makes it gate an irreversible deleter. Measured by the security lens and
> reproduced independently by the orchestrator on an isolated `/tmp` tree: a run dir with no
> manifest correctly REFUSES; planting one well-formed `covers: runs/target_run` line in a
> **totally unrelated** `experiments/hostile_unrelated/PROTECTED.txt` flips it to `keeping 2`; and
> `--apply --keep-best 1` then really deletes `runs/target_run/checkpoint_1`. No malformation is
> needed — a grammatically perfect manifest in the wrong place suffices.
> **Stated precisely, because the lens overstated it:** this row's docstring premise "missing
> coverage is never read as 'nothing to protect'" is **NOT falsified** — the no-manifest refusal
> holds exactly as claimed. What is false is the *converse* a reader assumes: that "covered" means
> someone WITH STANDING over that run dir reviewed it. And the flat namespace is **not a defect this
> row introduced** — it is the SETTLED 881I/881G grammar (D53), documented verbatim in
> `tools/ckpt_gc.py` ("Either name, anywhere under a scanned directory, is read as a manifest") and
> quoted verbatim in THIS FILE at line 92; the tool's own refusal text instructs the operator to add
> the line to "an `experiments/<id>/PROTECTED.txt`" — any id. **The consequence is new; the cause is
> settled.** Binding manifest identity to run-dir authority is an ingestion-SEMANTICS change, and
> D63 ruled those "are not reopened — the amendment is additive-only". So the factory halts here
> rather than guessing past a ruling. D64 carries three options; (a) a narrow written amendment
> binding coverage to the manifest's own experiment id is recommended and is fail-closed (it can
> only ever REMOVE coverage).
>
> **FLOW-VERIFIER (re-run after the incident): NO BLOCKER, NO MAJOR in the behavioural half.** The
> TTL BLOCKER — this row's whole reason for existing — is now verified in ALL THREE directions
> through the real `prune_run_dir_by_score`/`get_protection` (no stubs), on `/tmp` scratch trees:
> (A) warm the cache with a dry-run that would delete `ckpt_a`, add `protect:` for it, then call
> `apply=True` 0.001 s later inside the 300 s TTL with no `sleep` → `keeping 2`, ckpt_a **survives**
> (the exact D63-fixed shape); (B) break the manifest so the fresh scan FAILS → `protection scan is
> STALE … REFUSING to prune`, `protection_cache_is_stale()==True`, the prunable ckpt **survives**,
> and `had_failure=False` (a refusal is correctly not reported as a delete failure); (C) offline tool
> and in-run pruner run against the same tree in the same second **agree** (`keeping 2`). Also
> verified: `--apply` implies `--write-index` and appends a well-formed `prunes.jsonl` (one object
> per line); a read-only run dir degrades rather than raising (`[warn] could not write index`,
> `FAILED:` line, exit 1, no traceback, checkpoint left intact); and `write_run_index` atomicity
> holds under SIGKILL mid-`json.dump` — a worker writing a ~60k-row index was killed ~80 ms in with
> the temp file caught partial at 4.26 MB, and the real `checkpoints_index.json` was COMPLETELY
> unchanged and still valid JSON (`os.replace` atomicity confirmed; a torn index was never
> produced). Gate (vi) holds in ALL FOUR report modes (`--verify-manifests`, dry-run,
> `--write-index`, `--apply` on a scratch mirror): 7 protect / 3 dangling named in every one.
> **`--protect-file` unification is confirmed on the DELETE path for all four shapes** — missing
> (`CRITICAL … could not be read`; `--apply` exits 2 rather than proceeding with an empty set),
> directory (refused, **no `IsADirectoryError` traceback` — old bug closed), live symlink (refused,
> not read — old bug was silently read-and-honoured), FIFO (refused, never opened, well under an 8 s
> timeout on all three invocations — the old exit-124 hang is closed). No shape silently un-protects
> anything before a deletion.
> **One item disclosed as inspection-only, correctly:** `train_jax_fast.py`'s literal process-level
> `sys.exit(1)` on a failed prune was verified by executing the exact production function (returning
> `had_failure=True`, the boolean `:768-771` assigns to `ckpt_prune_failed`) plus reading the two
> unbranched propagation lines at `:2185-2186` — NOT by a full CLI process run, because the default
> path only recency-deletes under a 5 GB free-space floor with no CLI knob to lower it, making a safe
> repro disproportionate. Flagged rather than glossed.
>
> **REGRESSION LENS: NO BLOCKER, NO MAJOR — and it corrected the ORCHESTRATOR's own bookkeeping.**
> Method worth reusing: it reconstructed true git-HEAD via `git archive` into a scratch copy rather
> than any destructive git command on the live tree, then diffed real CLI output byte-for-byte.
> **Corrected suite accounting:** true git-HEAD (4 files; `test_ckpt_881_protection_model.py` does
> not exist at HEAD at all) = **288 passed / 1 skipped**; working tree (5 files) = **393 passed / 1
> skipped**; delta **+105 passed, 0 removed anywhere**, fully additive — +100 wholly-new tests in the
> untracked protection-model suite and +5 new named tests in 881i (`comm` over collected test IDs:
> 5 added, 0 removed), those 5 being exactly D63(a)'s pin machinery. **The "baseline 340 passed / 2
> skipped" the orchestrator carried into the briefs was MISLABELLED**: 340/2 was measured over FOUR
> files on the tree with the run-41 patch already re-applied, i.e. a post-restore/pre-amendment
> state, NOT a pre-881D baseline. The judge was right to refuse it — an unreconciled baseline is how
> a phantom delta gets waved through. Use 288/1 → 393/1 as the accounting of record.
> Gates A-G all verified with no regression: D63(a) really is additive (`_ack.covers()` returns
> False on an empty `--known-dangling`, so all consult sites are inert unless the flag is passed;
> exactly 2 consult sites, the `except ManifestScanIncomplete` malformed path untouched; full 881i
> 166 + 881g 72 corpora pass, no accept/refuse verdict flipped); the peek changed no COUNT (D53's
> `--experiments-dir experiments experiments/__pycache__` collapse reports `1 protect` on HEAD and
> on the working tree alike, the new stat and `MANIFEST NOT SCANNED` line being the only additions);
> `--known-dangling` is byte-identical-inert when omitted; both D62 restored tests are live and
> non-vacuous with real mutation proofs; the 876 suite lost no coverage (the
> `..._refuses` → `..._now_has_an_implementation` rename is an honest behaviour change, and
> `final_ckpt_save_failed` alone is still asserted sufficient); a non-score training run touches
> zero new code paths (`_maybe_prune_by_score` early-returns when `ckpt_retention != 'score'`); and
> the suspected `ckpt_prune_failed = False` ordering bug **does not survive into the working tree**
> (the binding precedes every call site — closure definition order is not the constraint).
> selftest 12/12.
>
> **MAJOR (correctness lens) — `--keep-best N` does not keep N, and this repo's OWN trainer triggers
> it.** `plan()` (`tools/ckpt_gc.py` ~L1233-1247) partitions scored checkpoints by `metric` and
> applies `keep_best` INDEPENDENTLY WITHIN EACH GROUP (`ranked[:max(1, keep_best)]` per group). That
> is a deliberate design ("ranked WITHIN a metric, never across metrics") but it was never tested
> with >1 checkpoint per metric across >1 metric in one run dir — and it is not an adversarial
> shape: `train_jax_fast.py:1501` sets `ckpt_metric = 'train_reward_ma20'` every epoch and `:1975`
> overrides it to `f'eval_reward_L{config.eval_level}'` on eval epochs, both written to the SAME run
> dir's `scores.jsonl` through the same `append_checkpoint_score` call. So an ordinary run with
> `--eval-every` alternates metric strings against itself. Measured against the real `plan()`: 10
> ckpts (8 `train_reward_ma20`, 2 `eval_reward_L4`), `--keep-best 3 --keep-latest 1` → **5 survive**,
> reported as `keeping 5` at exit 0 with no warning. Extreme reachable form: 5 ckpts with 5 distinct
> metrics and `--keep-best 1` → **all 5 survive**. Direction is fail-SAFE (over-retention, never
> deletes what it shouldn't) so this is not BLOCKER-class — but it silently defeats the tool's own
> stated purpose (the mechanism that exists to stop the disk filling) while printing a `keeping N`
> line that reads as the honoured cap. Unguarded by the suite. **In scope to fix on resume** —
> `plan()` is a function this row is sanctioned to rewrite.
>
> **MINOR (correctness lens) — the tie-break under equal scores is filesystem-order-dependent.**
> `sorted(members, key=..., reverse=True)` is a stable sort over `members`, whose order follows
> `collect()`'s `os.walk`/`os.listdir` order, which is never sorted. Measured: 4 exactly-tied
> checkpoints, `--keep-best 2 --keep-latest 1` deleted `checkpoint_3` (second-newest) while keeping
> the two OLDEST. Exact float ties are rare but reachable (rounded eval metrics); an irreversible
> unsupervised deleter should have a documented deterministic tiebreak (recency).
>
> **Attacks the correctness lens drove to conclusion and did NOT break** (recorded so a resume does
> not re-spend them): `_is_protected`/`resolve_coverage` prefix boundaries are `os.sep`-anchored
> (`ckpt_s1` vs `ckpt_s10`, `exp1` vs `exp10` both correct); dangling entries withdraw coverage
> without poisoning siblings; `_order_newest_first`'s fix reproduced against the exact docstring
> repro (own-mtime primary, number only as sub-second tiebreak) — the disputed scope-4 item now has
> its THIRD independent measurement and the old borrowed-mtime defect is refuted; the
> `AMENDED_INGESTION_NAMES` digests are hardcoded 64-hex literals with a non-tautological mutation
> proof and no name was dropped from the 43-name inventory; `UNTOUCHED_GUARD_NAMES` resolves all 8
> names via AST (a renamed name would raise, not pass vacuously) with real red-on-mutation proofs
> for 4 of 8; `_entity_ranges` tiles the file with NO gap (inter-entity comments fold into the
> following entity; no top-level decorators exist to create an AST-lineno gap); and every arithmetic
> edge (N == keep_best, N+1, zero, one, all-unscored, NaN/Inf via `_finite`, negative scores) matches
> the documented policy.
>
> **Verification status: ALL 5 HIGH-tier lenses COMPLETED — 4 clean, 1 produced the D64 wall, plus one in-scope MAJOR.** The **fail-closed**
> lens found NO defects and reported all six gates MET, having failed to break three specific
> attacks: a parent-only `--known-dangling` acknowledgement does NOT blanket-silence descendants
> (`_spellings_of` is exact set-intersection, not prefix); `protect:`/`covers:` manifests planted in
> `.git`/`__pycache__` grant NOTHING (the peek records path strings only and never opens them); and
> no caller can reach the deleter with `force_rescan=False` (`get_protection` has exactly one call
> site, `prune_run_dir_by_score:303`, passing `force_rescan=apply`; that function has exactly one
> production caller, `train_jax_fast.py:757`, with literal `apply=True`). The **security** lens
> produced D64. **Regression** and the re-run **flow-verifier** are clean (detailed above);
> **correctness** found one in-scope MAJOR and one MINOR (detailed above). The first flow-verifier
> attempt did not complete — it caused the incident below and was re-run from scratch under a
> hardened brief.
>
> **INCIDENT, 2026-08-25 ~09:05 CEST — recorded here because it happened while verifying THIS row.**
> The flow-verifier destroyed this repository's `experiments/` and `runs/` directories: it ran
> `rm -rf` on a RELATIVE path intending its scratchpad, while the working directory was the repo
> root because a `cd` in an earlier tool call had not persisted. 255 tracked files and every
> gitignored checkpoint were wiped. It was caught within minutes by a DIFFERENT agent (the
> fail-closed judge), which correctly refused to attribute the damage to the diff under review and
> escalated instead of attempting recovery. Recovery: all 255 tracked files restored from git (the
> deletions were tracked — the initial "unrecoverable" read was wrong in the direction that
> mattered); all 7 load-bearing checkpoints re-pulled from gpu-server and verified BYTE-IDENTICAL
> (`2d6d167a…`, `9a8a9085…`, `cca94aff…`, `243f87e8…`, `46f34475…`, `753f6902…`, `5272638e…`);
> `runs/806_predictability_probe` and `runs/875_wf` re-pulled; this row's 3 `PROTECTED.txt` corpus
> files restored from the committed lore patch. `trading_data.db`, the fixture and all three backups
> were never in the blast radius. **Genuinely lost:** `experiments/906_oos_nonstationarity/` (7
> untracked files) and `runs/875_wf/WF_SUMMARY.json` (srv holds only a 2-byte `[]` stub, md5
> `d751713988987e9331980363e24189ce`, against the real `4c306427a8f40f44f3058b34a7bad6a1`) — both
> operator-snapshot items. **Standing correction to every verification brief from here: destructive
> commands take ABSOLUTE paths only, `cd` does NOT persist between tool calls, and nothing under the
> repo is ever an `rm -rf` target — including via a symlink** (a separate lens clobbered tracked
> `folds/real` through a symlinked `--out-root` the same morning). Both lenses that caused damage
> had explicit do-not-write fencing; the fencing was not specific enough about MECHANISM.
>
> ---
>
> **(Superseded — factory run 41 banner, kept for the record:)**
>
> **STATUS (2026-08-24, factory run 41): PARKED — GREEN, BLOCKER DEAD, NOT SHIPPED. See D63 (and
> D62, which pairs with it).** Work preserved in
> `lore/handoff_2026-08-24_factory_run_41_881D_cycle2.patch` (4321 lines, incl. the 1341-line suite
> and the 3-file corpus); tree reverted to HEAD.
>
> **The BLOCKER is dead, and it was severe: a manifest-protected checkpoint WAS being deleted.**
> The in-run pruner consumed a protection set cached up to 300 s; the `stale` flag fires only when
> a scan FAILS, so a scan that SUCCEEDED 299 s ago against a *different* tree was authoritative for
> an irreversible deleter. Measured in the same second on the same tree: offline `ckpt_gc.py` said
> `keeping 4` while the in-run pruner deleted the protected snapshot with `stale=False` and no
> warning. Fixed by `get_protection(..., force_rescan)` with `prune_run_dir_by_score` passing
> `force_rescan=apply` — keyed off `apply` itself, so the delete branch cannot be reached with a
> cached protect set; a failed fresh scan still refuses. Proven by mutating the real drop site
> (reverted → `1 failed`; restored → `1 passed`) at the DEFAULT 300 s TTL with `apply=True` and no
> `sleep`. A full fresh scan costs 1.694 s against saves that are minutes apart.
>
> Also closed: the two `--protect-file` readers are UNIFIED (FIFO now exit 2 where it hung at 124);
> the falsified "can never disagree about the same tree" claim strings are corrected to what holds;
> ingestion drift is back to **EMPTY** (AST entity-diff over all 43 `INGESTION_NAMES` = `[]`); and
> D62's measured consequence is closed by an ADDITIVE `UNTOUCHED_GUARD_NAMES` pin — `_safe_target`
> ×2, `_du`, `free_gb`, `_WALK_PRUNED_DIRNAMES` and the four `_WHY` constants all mutate RED again.
> **376 passed / 2 skipped** (baseline 351), selftest 12/12 both hosts, all 7 load-bearing
> checkpoints byte-identical throughout, no `--apply` outside a scratch copy.
>
> **WHY IT DID NOT SHIP.** Two items have their ONLY drop site inside ingestion functions pinned by
> `INGESTION_NAMES` — which THIS ROW'S scope item 2 forbids reopening. (i) MAJOR:
> `--verify-manifests` exits 2 permanently, because this row's own corpus carries 3 acknowledged
> dangling 828/scratch disk entries — so the one pre-flight signal the delete path does NOT repeat
> carries zero information. `--known-dangling PATH…` was built and measured working, then reverted
> at the wall. (ii) MINOR: the `experiments/__pycache__` silent withdrawal; same wall, same
> treatment. **A verdict cannot be green with an open MAJOR** — and at HEAD `--ckpt-retention
> score` is refused at parse time, so this row is what makes an unsupervised `apply=True` per-save
> deleter reachable at all. Shipping a newly-reachable deleter whose pre-flight gate can never go
> green is the "an always-red gate gets ignored" hazard, on exactly the path whose refusals the
> deleter does not repeat.
>
> **Gate (vi) is answered but not executed:** the three `PROTECTED.txt` files now carry a `TRACKING`
> block recording WHY they must be tracked (untracked files never reach `gpu-server` under the
> deploy contract, and that is where `ckpt_gc.py` runs). Committing them is a one-line action once
> the row ships; they are deliberately un-tracked meanwhile (`5e6f2ff`).
>
> **Resume per D63** — the factory recommends a narrow, WRITTEN scope amendment permitting
> `_run_manifest_check` and `load_manifests` to change for exactly those two items, with
> `INGESTION_NAMES` re-pinned to their new bytes. Both changes are already built and measured; the
> wall is procedural, not technical.

# TASK_881D — the checkpoint protection model + deleter, on 881I's settled ingestion

**Status**: READY 2026-08-17 (planner; D45 re-cut). **Amended 2026-08-19 (D50):** the ingestion
881D consumes is settled by 881I's D50 re-cut — but the SCORE-SOURCE side is NOT settled
(TASK_881S owns it) and this row must not be told otherwise. **NEW PRECONDITION (D50): there is
no live manifest corpus at all** — 0 of 56 SPEC.md/PROTECTED.txt files carries a
`protected:`/`covers:` block, so the protection model currently protects NOTHING and its first
real exercise would be whatever an operator writes after it ships. **Scope item 0 below authors
the corpus.** Also carried: the tool is `tools/ckpt_gc.py`, NOT `tools/ckpt_archive.py`.
**Sequenced AFTER TASK_881I (shipped run 37) — AND BLOCKED UNTIL TASK_881G SHIPS (D53,
2026-08-20):** the corpus this row's scope item 0 authors must be written against a grammar that
exists in writing; today the MD032 form refuses and the accepted grammar is documented nowhere.
**Size: M. Risk: HIGH in consequence** — this row owns the `--apply` path of a tool that deletes
checkpoints. The D32/D34 `--apply` freeze STANDS until this ships AND 880's D35 gate opens.
**Resume from the tree state after 881I ships** (the `stash-881-halted-d45` tag is superseded;
run 36's durable copies are `lore/handoff_2026-08-19_factory_run_36_881I_cycle3.patch` for the
ingestion and the D45 stash for the protection/deleter halves, which six opus judges across three
cycles verified — take them as settled; re-verify, don't rebuild).

## What is already verified in the stash (from D45 — re-run each kill from scratch, do not take on report)

- All six prior kills DEAD: symlinked manifest FILE/DIR refusals; torn `scores.jsonl` per-run-dir
  refusal with peak survival; the cycle-2 killer's full four-step sequence (healthy
  `--write-index` 42.0 → tear → `--apply` refuses AND index still reads 42.0 → follow the
  refusal's own remediation → re-run under the floor → **peak survives**); missing/symlinked/
  directory `--protect-file` → exit 2.
- `write_run_index` atomicity under SIGKILL mid-`json.dump`; degrade-not-raise on read-only dirs.
- `root_resolution()` closes the D38-c3 asymmetry — survivor sets BYTE-IDENTICAL under `--apply`
  on both deleters with a simulated mounted scratch disk.
- `plan()`'s 4th return value honoured by both production callers.
- MAJOR E's dangling-entry refusal correctly scoped (3 unresolvable `828_stage3_train` entries
  refuse 0 run dirs; all 7 checkpoints across `runs/832_e5` + `runs/861_{ref,imprint}` survive).

## Scope

0. **PRECONDITION — author the real `PROTECTED.txt` corpus** (D50): the three
   `experiments/828_stage3_train` entries D45 names (scratch disk-unmounted, they stay loudly named
   in every report), at minimum, over the run dirs the deleter can actually see. The protection
   model's acceptance below is then exercised against a LIVE corpus, not rebuilt inputs alone —
   and the `--apply`-freeze conversation happens with eyes open about what is actually protected.

   **The grammar to author it against (TASK_881G, D53 — SHIPPED; mirrored verbatim from
   `tools/ckpt_gc.py`'s module docstring, "MANIFEST GRAMMAR" section):**

   MANIFEST BASENAMES: exactly `PROTECTED.txt` and `SPEC.md` (`MANIFEST_BASENAMES`). Either name,
   anywhere under a scanned directory, is read as a manifest. Both share the SAME per-line ITEM
   grammar (below, "one path, one line, `$`-anchored"); PROTECTED.txt additionally supports a
   plain `#` comment line and SPEC.md does not — SPEC.md is Markdown, where a leading `#` is
   heading TEXT, and that divergence is deliberate, not a bug.

   WHOLE-FILE PRECONDITIONS, identical for both basenames, checked before either parser ever runs:
   - the basename must be a real, regular, non-symlink file at a real, tracked path. A symlink
     (to a file OR to a directory), a real directory, a fifo/socket/device, or a manifest reached
     only by crossing a symlinked ANCESTOR directory (including a symlinked scan root) is
     REFUSED, never read.
   - the file must decode as UTF-8 with no byte-order mark and no NUL byte.
   - ANY malformed line, anywhere in the file, WITHDRAWS THE WHOLE FILE's directives — not just
     the offending line.
   - a well-formed entry naming a path that does not exist on disk is NOT malformed — it is
     DANGLING: it protects nothing and is reported on its own, but does not poison the rest of
     the file the way a malformed LINE does. "NOT malformed" is about INTRA-FILE poisoning ONLY
     (the file's other entries still parse and still grant) and is NOT an acceptance: a dangling
     entry makes `--verify-manifests` print `DANGLING ENTRY` and EXIT 2, never 0, exactly like a
     refusal. Both are true at once; reading only the first is how an author concludes a typo'd
     path will pass the check.
   - ENTRY PATHS RESOLVE AGAINST THE PROCESS'S CWD, NOT against the manifest's own directory.
     Every entry — either basename, `--protect-file`, relative or absolute — goes through
     `os.path.realpath(<the text as written>)`, relative to wherever the tool was INVOKED. So
     `- runs/a` inside `experiments/999_z/SPEC.md` names `$CWD/runs/a`, NOT
     `experiments/999_z/runs/a`; from another CWD the same manifest names something else, and
     usually something DANGLING. Author the corpus for, and run the tool from, the repo root.

   PROTECTED.txt GRAMMAR — one directive per line:
   - `protect: <path>` or `covers: <path>` — exactly one directive keyword, a colon, and a
     single-token path, optional surrounding whitespace, nothing else on the line. ACCEPTED:
     ```
     protect: runs/some_run/snapshot_ep10
     covers: runs/other_run
     ```
   - WHITESPACE AROUND THE DIRECTIVE IS FREE — the same micro-grammar the SPEC.md item form uses
     (below). The line is STRIPPED before it is matched, so leading indentation
     (`  protect: runs/a`) and trailing whitespace are ACCEPTED, and the space after the colon is
     optional (`protect:runs/a` is ACCEPTED). What is NOT free is the number of tokens after the
     colon — exactly one.
   - a blank line — ACCEPTED, produces no entry. A line of nothing but SPACES or TABS counts as
     blank here exactly as an empty one does (the test is on the STRIPPED line).
   - a line whose STRIPPED form starts with `#` (indentation before the `#` is fine) — ACCEPTED
     as a comment, produces no entry.
   - anything else is MALFORMED (whole file withdrawn) — including two tokens after the
     directive (`protect: a b`), a bare `protect:` with no path, an unrecognized directive word,
     or plain prose.

   SPEC.md BLOCK GRAMMAR — a `protected:` or `covers:` block:
   - the header is column-0, lower-case, exact: `protected:` or `covers:`, optionally followed on
     the SAME line by nothing but whitespace to end-of-line/file (a torn final line with no
     trailing newline still counts as a header).
   - immediately after the header — with EITHER zero blank lines OR one-or-more blank lines
     before the first item — a list of items, one path per line, each written `- <path>` (a
     single `-`, optional whitespace, exactly one non-whitespace token, nothing else on the
     line). BOTH of the following are ACCEPTED and equivalent (the second is the canonical MD032
     form — a list surrounded by blank lines — and is the FIRST shape an author who runs a
     markdown linter will type; accepting it is what D53 required):
     ```
     protected:
     - runs/some_run/snapshot_ep10
     - runs/other_run
     ```
     ```
     protected:

     - runs/some_run/snapshot_ep10
     - runs/other_run
     ```
   - THE ITEM MICRO-GRAMMAR IS DELIBERATELY FORGIVING, in deliberate ASYMMETRY with the strictly
     column-0, lower-case, exact HEADER above it. An item line is matched STRIPPED-EQUIVALENTLY,
     so all three of these are ACCEPTED and name exactly the path they appear to: LEADING
     INDENTATION before the dash (`  - runs/a`, the Markdown nested-list shape); NO whitespace
     between the dash and the path (`-runs/a`); TRAILING whitespace after the path
     (`- runs/a   `). The asymmetry is on purpose: an odd HEADER can silently withdraw a whole
     declared block, so it refuses; an odd but unambiguous ITEM inside an already-matched block
     can only go into that block, so it is read. What an item may NEVER do is name a second token
     (see INTERRUPTED below) — the one thing whitespace cannot buy.
   - A LINE OF NOTHING BUT SPACES OR TABS IS A BLANK LINE, uniformly, in ALL THREE places this
     grammar says "blank": the MD032 gap between a header and its first item, the SPLIT gap after
     at least one item, and the blank line that legitimately ENDS a list. There is no shape where
     a whitespace-only line behaves differently from an empty one.
   - once at least one item has been consumed, a blank line followed by more PROSE, a heading, or
     end-of-file legitimately ENDS the block. ACCEPTED.
   - once at least one item has been consumed, a blank line followed by ANOTHER `- <path>` item —
     the list resuming after a gap — is REFUSED (SPLIT). This is the only place a leading-vs-mid-
     list blank line matters: BEFORE the first item, a gap is the MD032 fence above; AFTER at
     least one item, a gap is a SPLIT.
   - a header carrying inline content (`protected: runs/a`) — REFUSED, whole file withdrawn.
   - a header with ZERO items — REFUSED as EMPTY, whole file withdrawn.
   - WHICH REFUSAL WINS WHEN BOTH COULD APPLY (EMPTY vs INTERRUPTED): the INTERRUPTED diagnosis
     requires at least ONE already-consumed item to have been interrupted, so a non-item,
     non-blank line that is the FIRST content after the header (`protected:` / `hello` /
     `- runs/a`) refuses as EMPTY ("it names nothing"), not as INTERRUPTED, even though a
     well-formed item follows it. Both are exit-2 whole-file withdrawals; only the message
     differs. The ONE exception is an attempted item naming more than one token (`- a b`), which
     refuses wherever it sits under its OWN distinct message ("list item at line N carries MORE
     THAN ONE token"), because that line unambiguously IS an attempted item and "names nothing"
     would be false about it.
   - TWO OR MORE headers for the SAME key in one file — REFUSED (duplicate), whole file
     withdrawn. `protected:` and `covers:` are independent keys — one of each is fine.
   - an item line naming MORE THAN ONE token (`- runs/a runs/b`, or `- runs/a # note` — a
     TRAILING COMMENT on an item is this same shape) — REFUSED (the INTERRUPTED-list class, under
     its own distinct message: "list item at line N carries MORE THAN ONE token"), whole file
     withdrawn.
   - any OTHER non-blank, non-item line inside a started list (a stray word, a `*` bullet, a bare
     `-`, or a `#` line — NOT a comment here, unlike PROTECTED.txt) — REFUSED as INTERRUPTED,
     whole file withdrawn.
   - a line that is ENTIRELY two-or-more `-` characters (a markdown THEMATIC BREAK, e.g. `---`)
     is NEVER a list item, even though its first character reads as a bullet marker — it falls
     through to whichever of the two bullets above applies.
   - a header that misses column-0/lower-case/exact spelling by ANYTHING (indentation, a
     capitalized `PROTECTED:`, a leading `>`/`*`/`-`, a byte-order mark) is a NEAR-MISS: REFUSED
     (never silent), whole file withdrawn, EVEN beside a well-formed block elsewhere and EVEN
     inside a fenced code block — both accepted-deliberately for the same reason (a well-formed
     header inside a fence is ALSO read as a real block; this parser does not try to detect
     fences at all).
   - FENCES ARE NOT INVISIBLE, THEY ARE UNSEEN — and the difference bites in the most natural
     shape an author writes. Because no fence is detected, the CLOSING BACKTICK-FENCE LINE of a
     fenced EXAMPLE is just another line, and if it comes IMMEDIATELY after the last item it is a
     non-item line inside a started list: the whole file is REFUSED as INTERRUPTED. So "a header
     inside a fence can only ADD protection" is FALSE as written — it can also WITHDRAW the
     entire file. REFUSED:

     ````
     ```
     protected:
     - runs/a
     ```
     ````

     ACCEPTED (one blank line TERMINATES the list before the closing fence; the block inside the
     example parses as a real block and its entries are real):

     ````
     ```
     protected:
     - runs/a

     ```
     ````

     The practical rule for a document that must both illustrate and be ingested: leave a blank
     line after the last item of any example block, and expect every path written in an example
     to be ingested as a real entry (a nonexistent one lands in DANGLING and exits 2).

   `--protect-file` GRAMMAR — AS READ BY `--verify-manifests` (`load_protect_file`), the ONLY
   reader this grammar describes (a flat list — no `protect:`/`covers:` prefix): one path per
   line, realpath'd. A blank (stripped) line, or a line whose STRIPPED form starts with `#`
   (indentation before the `#` does not exempt it), is skipped; anything else is one path,
   trusted verbatim. A missing, symlinked, or non-regular-file `--protect-file`, WHEN THE FLAG IS
   GIVEN, is REFUSED.

   NOT TRUE OF THE DELETE PATH (dry-run / `--apply`), stated because a doc that over-promised
   here would become this row's false precondition: `main()`'s deletion loop does NOT call
   `load_protect_file`. It reads the file inline, guarded by `os.path.exists` — every non-blank
   line is realpath'd, `#` is NOT a comment there, and the four shapes `--verify-manifests`
   refuses do NOT all fail the same way here. A MISSING `--protect-file`, or one that is a
   DANGLING SYMLINK (`os.path.exists` reports False once a symlink's target is gone), is silently
   treated as EMPTY at exit 0 rather than refused. A DIRECTORY is not caught at all: `open()`
   raises `IsADirectoryError`, uncaught, so the process CRASHES with a traceback at exit 1 — a
   crash, not "empty at exit 0". A LIVE SYMLINK to a real file is not refused either: it is READ
   AND HONOURED, an unrefused ACCEPT, the opposite failure mode from the missing/dangling case. A
   FIFO `--protect-file` is worse than either: `open()` on a reader-less FIFO in read mode BLOCKS
   FOREVER, the same hang `load_protect_file`'s own docstring warns about. That inline read is
   canon and 881G did not change it; UNIFYING the two readers (routing the deleter through
   `load_protect_file`, so a missing protect file cannot silently un-protect anything before a
   deletion) IS THIS ROW'S WORK. Until then: run `--verify-manifests` FIRST and act on its exit
   code, because the delete path will not repeat its refusals.

   `--experiments-dir` SEMANTICS: the flag OMITTED and absent-by-default is a silent no-op; the
   flag EXPLICITLY GIVEN a nonexistent value now REFUSES (D53 — matches `--protect-file`); two
   values that resolve to the same or a NESTED directory are collapsed to one before scanning
   (never a refusal — a redundant request is not an operator error). The collapse is REPORTED on
   stdout and the scan line still counts the dirs the OPERATOR named ("1 of 2 scan dir(s)
   visited"). A COLLAPSE IS A COUNTING FIX AND MAY NEVER CHANGE A VERDICT, so it happens ONLY
   when (a) NEITHER dir is reached by crossing a SYMLINK (`realpath != abspath` — a symlinked
   scan dir is the input that produces the "manifest reached only by crossing a SYMLINKED
   DIRECTORY" refusal, and collapsing it on realpath ERASED that refusal), and (b) the SURVIVING
   dir's walk PROVABLY REACHES the dropped one (being underneath it is not enough: the walk does
   not descend into `.git` or `__pycache__`, so collapsing `experiments experiments/__pycache__`
   silently dropped a manifest, "1 protect" -> "0 protect", at exit 0). Otherwise BOTH dirs are
   kept and the duplicate stat is tolerated. Even together, those two conditions only prove the
   surviving walk REACHES the dropped directory against its own name-based prune list — not
   against every I/O outcome: a non-listable intermediate directory (measured — an ancestor at
   mode 111) still stops the walk and still drops the nested dir it can no longer reach, "1
   protect" -> "0 protect" on the same host. That is NOT a silent withdrawal, though: it raises a
   `ManifestScanIncomplete` and the whole run REFUSES at exit 2 ("Permission denied" printed on
   both the collapsed and uncollapsed invocation), so no verdict ever changes — the fail-closed
   direction holds — but "nothing under a collapsed directory goes un-scanned" is not literally
   true; only "nothing goes un-scanned without the scan itself loudly refusing" is.

1. Resume the protection model + deleter from the stash; re-run every kill above against the
   resumed tree.
2. Consume 881I's shipped ingestion as a settled input (do not re-open it; do not assume the
   score-source side is settled — that is 881S).
3. The score reader resolving by `path`/`step` (D35's original item), the `_is_protected`
   descendant rules, the TTL re-scan, the failed-`rmtree` audit row, and the `followlinks=False`
   BLOCKER — all as previously specced in 881, unchanged.
4. The disputed pre-existing `_order_newest_first` — needs a third independent measurement
   (carried from 881).
5. The documented side effect from D45 planner note 4: the dangling-advisory-doc refusal also
   disables the IN-RUN pruner process-wide via the stale cache — independently judged CORRECT,
   but its blast radius is global by assertion rather than necessity, and the docstring omits the
   in-run pruner effect. Fix the docstring; scope the refusal's blast radius if cheap, document
   it if not.

## Acceptance (gates)

- (i) Every kill from the verified list reproduced on the final tree, independently rebuilt.
- (ii) A `--apply` dry-run on the REAL tree keeps all 7 load-bearing checkpoints
  (`runs/832_e5` + `runs/861_{ref,imprint}`), printed and diffed against the pre-881 survivor set.
- (iii) 881I's malformed-input corpus still refuses identically when driven through 881D's
  callers (integration, not just unit).
- (iv) **AMENDED (D62(a), 2026-08-25).** Byte-identity of every non-`--apply` code path vs HEAD
  **EXCEPT** the sanctioned no-manifest refusal (881D's own scope item 2: "a run dir this tool
  finds on disk that no manifest covers is REFUSED"), whose output changes vs HEAD are enumerated
  here in full — "except the sanctioned refusal" with the exception left un-enumerated is the hole
  this amendment exists to close:
    1. a new per-run-dir line, printed for every run dir no manifest covers, dry-run or `--apply`,
       that did not exist at HEAD at all (HEAD's canon never gated on manifests): `` [ckpt_gc]
       REFUSE run dir <rd>: no manifest declares coverage for it (or a manifest entry pointing
       into it names a path that does not exist) — add a `protect: <path>` (real checkpoints,
       cited from a LEDGER/SPEC reference — never invented) or `covers: <rd>` (reviewed, nothing
       to protect) to an experiments/<id>/PROTECTED.txt, or a `protected:`/`covers:` block in its
       SPEC.md. Nothing under <rd> will be pruned or written until then, dry-run or --apply. ``;
    2. the closing summary line HEAD already prints (`` [ckpt_gc] {reclaimed|would reclaim} X GB
       across N checkpoints ``) gains one appended clause, `` ; <n> run dir(s) REFUSED (no
       manifest); <m> run dir(s) REFUSED (score read incomplete) `` — an ADDITION to an existing
       HEAD line, not a byte-identical carry-over.
  Restored by D62(a) to close the gap the "except the sanctioned refusal" wording briefly opened:
  `test_canon_functions_are_byte_identical_to_head` is back (narrowed to the guards 881D does not
  touch — `_safe_target`, `_du`, `free_gb`, and the five named WHY/constant strings,
  `UNTOUCHED_GUARD_NAMES`), and `test_diff_against_head_removes_no_content_line` is un-skipped,
  allowlisted by NAMED ENCLOSING ENTITY (`_881D_KNOWN_REWRITES`: the module docstring, `plan`,
  `collect`, `_epoch_of`, `_load_scores`, `_is_ckpt_dir`, `CKPT_PATTERNS`, `main` (the delete loop
  + argparse block both live inside it), and — D63(a)'s own additive amendment —
  `_run_manifest_check`/`load_manifests`) rather than by individual line text, since 881D
  legitimately rewrote hundreds of HEAD lines across those entities. A removal outside every named
  entity still fails loudly.

  **Gate-numbering note (D62(a), item 4):** `tools/tests/test_ckpt_881_protection_model.py`'s own
  module docstring numbers its FIRST set of gates against `tasks/TASK_881_CHECKPOINT_PROTECTION_
  MODEL.md` (the parent task) — a DIFFERENT (i)-(vi) than this file's own, e.g. that scheme's
  (iii) is "both deleters agree" where THIS file's (iii) is "881I's malformed corpus, INTEGRATION".
  Every section header and gate reference in that suite is now prefixed to say which scheme it is
  ("881(parent) gate (n)" vs "TASK_881D gate (n)") so the two roman-numeral sequences can no
  longer be read as one.
- (v) Standing clauses: guards go red when removed at the real drop site; real CLI invocations,
  no stubbed internals; mutations independently rebuilt.
- (vi) **The corpus of scope item 0 exists, is tracked-or-deliberately-untracked with the reason
  recorded (D48's class: cited-but-untracked evidence is a named structural pattern now), and
  appears in every report the tool emits.**

## Out of scope

Manifest ingestion semantics (881I's, settled). Score-source enumeration (881S). TASK_880 (gated
by D35 on THIS row shipping + real score files the new reader resolves). Lifting the `--apply`
freeze (operator's, post-ship).
