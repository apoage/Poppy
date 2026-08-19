# Factory run 36 — 2026-08-19 (IN PROGRESS at time of writing)

Implementer lane. Queue taken in rank order: 892 (rank 1) → 881I (rank 3) → 891, 896 (filler).
894 (rank 2) deliberately NOT started — it edits `eval_vector.py`, which 892 holds; serial by 894's
own sequencing note, and the TASK_851 incident (a concurrent edit to shared `eval_vector.py`
mid-review made headline numbers unreproducible) is the precedent for not overlapping them.
881D not started — sequenced after 881I by D45. 880 stays gated on D35.

## The finding that outran every row: cited-but-untracked evidence

Four independent instances surfaced in ONE run. This is a structural pattern, not four coincidences,
and it is the thing most worth the planner's attention:

1. **`runs/819_single_asset/portfolio.py`** — the file D44 ordered a comment into is gitignored
   (`.gitignore:54` → `runs/*/*`). `git ls-files runs/819_single_asset/` returns NOTHING: not one
   file of the 819/820 prototype lineage is tracked. So TASK_896 has no diff to commit, and D44's
   "the claim must exist and be true" is satisfied on this workstation only. → **D48 seeded.**
2. **`runs/893_prefix_cells/prefix_cells_real.json`** — untracked, yet `experiments/LEDGER.md`'s
   `893/prefix-cells` row pins its sha256 (`6bede1e9…`). The LEDGER pins a digest of a file no
   clone contains. It DOES still reproduce bit-identically (the 896 fix regenerated it and matched
   the pin), so this is a durability gap, not a rotted artifact.
3. **`runs/860_codistill/manifest_860.txt`** — the file D46 clause 5 NAMES as the shape to fix is
   untracked, so the clause's own "tracked manifests" wording excludes it. The gate came back
   EMPTY partly by construction.
4. **`runs/819_single_asset/__pycache__`** — a fresh `.pyc` from a `py_compile` check is likewise
   invisible to every gate. Harmless here, same class.

Prior instances the memory already records: run 25's fixture re-base, run 32's 14 stamped scratch
provenance documents that a `git stash` could not un-write and that MASKED a real 861 red.

## Verified first-hand by the orchestrator (not taken on report)

- **819 fill extent, at both granularities.** `build_close_aligned` builds a ONE-MINUTE grid and the
  leading-gap branch fills `out[:vr[0], col]` — measured leading minute rows: GRT 24, KAVA 22,
  MANA 19, ZEC 16, EOS 15, XDG 7, ALGO 7, LINK 6, FIL 6, XMR 4, TRX 3, BCH 2, XTZ 2, SC 2, KSM 1,
  CRV 1. **Only KSM and CRV are genuine 1-row fills; 14 of 16 span 2–24 rows.** "1 row, row 0 only"
  is a property of the DOWNSTREAM daily subsample (`close_1m[::1440]`), which keeps minute row 0 and
  discards the rest. D44's own phrasing conflated the two, and TASK_896's first comment inherited it.
- **The 16/10/13 split**, from `prefix_cells_real.json`: 16 assets prefix==1 with 0 FULL cells; 10
  with earlier history admitted exactly at `grid_start` (AAVE ADA ATOM DOT ETH LTC UNI XBT XLM XRP);
  13 late listings holding **all 4904** FULL decision cells (PEPE 865 … FLOW 26), summing exactly.
- **`tools/ckpt_gc.py` is the real 881 surface**, NOT `tools/ckpt_archive.py` as `TASK_881I:6-7` and
  D45 both state. ckpt_gc holds `PROTECTED.txt`/`SPEC.md` ingestion, `plan()`, `root_resolution()`,
  the TTL re-scan and the `--apply` deleter; ckpt_archive is TASK_880's remote archive tool, under
  its own freeze. An implementer obeying the task file literally would have edited 880's tool — in
  the lane where four of the last five planner errors were boundary calls. **Spec fix needed.**
- **881I BLOCKER (dangling-symlink invisibility).** A broken symlinked `experiments/<id>` is
  invisible: `os.walk` classifies a dangling symlink-to-dir as a FILE (its `is_dir()` follows the
  link and fails), so it never enters `dirnames` where the symlink guard lives. Reproduced both as a
  child of the scan root and as the `--experiments-dir` root: silent `check-manifests: OK`, exit 0.
  **Control:** a LIVE symlinked dir refuses correctly at exit 2 — so the guard exists and the
  dangling case slips past it. Live today: `jax_ckpt_fast`/`jax_ckpt_easy` point at
  `/mnt/scratch`, which is NOT mounted (real mount `/run/mnt/scratch`).
- **881I MAJOR (torn SPEC header).** `printf 'protected:'` with no trailing newline → silent OK,
  exit 0, because the header regex requires a literal `\n`. Control with the newline refuses.
- **881I MAJOR (argparse abbreviation).** Adding `--check-manifests` made `--c` ambiguous with the
  pre-existing `--critical-free-gb`: works at HEAD, dies on the candidate. Fails CLOSED and no repo
  caller uses abbreviations, but it breaks a previously-valid invocation of an unmodified flag.
- **891 BLOCKER premise, all three pillars.** `trader_env/bin/python` is a `lukas:lukas` 0777 symlink
  in `drwxrwxr-x` `lukas`-owned dirs; `tools/safe_run.sh:50` is `exec systemd-run --user --scope`, so
  every queued job runs as the SAME UID with only a cgroup RAM cap; and the `protected` list in
  `.dark-factory/config.json` has NO enforcement code anywhere. The code comment claiming
  `trader_env/` is "never job-writable … no deferred-window forgery risk left" is therefore FALSE.
  A fail-closed judge demonstrated it live in a sandbox: a plain `passthrough` row overwrote the venv
  python after the one-time parse-time check, and a later row labelled `interpreter=venv` executed
  the forged payload at rc=0.
  **Scope it honestly:** this is NOT a security boundary — a malicious same-user job could already
  rewrite `run_queue.sh` or the training code. The defect is the false CLAIM; the real value of the
  fix is honest text plus catching accidental mid-queue venv breakage.
- **891 MAJOR (D46 clause 4 half-met).** `run_queue.sh:339` (dry-run) prints `-> $cmd`;
  `:342` (live launch) prints only `interpreter=$interp log=$log`. In a live run the absolute path
  survives solely in the transient `/proc` cmdline. The clause says the operator must see the
  absolute path, "not a description of one".
- **892's 22 MB `smoke_ckpt/` hazard: CLOSED by the implementer**, correctly and per repo
  convention — the dir is gone and `experiments/892_action_ev/.gitignore` now carries
  `smoke_ckpt*/`, matching the per-experiment pattern used by 857 (`ckpt_smoke/`) and 874
  (`smoke_runs/`). Had it been committed, 22 MB of checkpoint blobs would have entered git.

## Process events worth carrying forward

- **A judge is not a rubber stamp, twice over.** 896 passed its implementer's own checks and a
  correctness judge still found its most concrete sentence false. 881I and 891 both self-reported
  fully green and each had a BLOCKER found by an independent lens.
- **An implementer destroyed its own diff on the live tree.** 892 ran
  `git checkout -- experiments/835_stage4_eval/eval_vector.py`, lost its whole working diff, and
  reconstructed it. It flagged this itself. The standing rule (never stash/checkout/reset the live
  tree; use a worktree) exists precisely for this, and it now has a second incident behind it. The
  reconstructed diff was NOT accepted on report — its regression judge was briefed to hunt
  reconstruction damage specifically.
- **A 529 killed an escalated opus fix mid-flight** (896, first attempt). It had made no edit, so the
  retry was a retry of the same cycle, not a new one. Worth distinguishing infrastructure death from
  work failure when counting cycles against `halt_after_failures`.
- **Leftover worktrees:** `/var/tmp/judge865/{BASE,CAND}` are stale from an earlier run, pinned at
  `086b1c4`/`aef1d98`, keeping those objects alive. Also `/home/lukas/trdr03` and `/home/lukas/trdr2`
  are NOT worktrees but independent full copies of the repo; a judge used trdr03 as a pristine
  baseline (sound here — it differed only by the comment block — but only valid at a matching
  revision; trdr2 differs by 105 lines and was correctly rejected).

- **ORCHESTRATOR ERROR, mine: I spawned TASK_892's fix pass while its own regression judge was still
  running**, so that judge audited a moving target — the exact TASK_851 concurrent-edit class I had
  cited, one turn earlier, as the reason NOT to start 894 against the same file. It caught
  `eval_vector.py` mid-write twice (`NameError: name 'finalize_universe_pin_record' is not defined`,
  a torn read consistent with a non-atomic edit landing a call site before its definition) and watched
  the diff grow 145+/10- to 206+/36- underneath it. Its torn-read observations are therefore artifacts;
  its structural findings (the `--a` abbreviation collision, the `random-matched` silent no-op) survive
  and were re-verified independently. **Rule for next time: a fix pass and a verification pass must
  never hold the same file at the same time. Freeze the snapshot (worktree or copy) before handing it
  to a lens, or serialise fix-then-verify.** The judge itself recommended exactly this.
- **A judge's own suite run can be a false red.** That regression judge reported the 892 acceptance
  suite exit 1 at 21:16 — real at that instant, but it was the fix's NEW mode-aware fail-closed budget
  guard firing against a test harness the same in-flight fix had not yet rewired. A red observed
  mid-fix is not a verdict on the fix; it was sent back for confirmation-at-rest rather than recorded
  as a defect.

## The sharpest finding of the run: the VERIFICATION MACHINERY produced a fake green

While assessing TASK_891's tree after its cycle-3 agent died on a connection error, the orchestrator
ran the interpreter suite directly instead of accepting a reported count. It printed
`RESULT: 18 passed, 0 failed`, exit 0 — and buried in the output:

```
  PASS: MUTATION KILLED: M7 launch-time re-probe removed (poison-after-check exploitable) (4 check(s))
MUTATION-BUILD-ERROR: expected exactly 1 occurrence of the target text, found 0
cp: cannot stat '/tmp/rqitest.XXXX/mutant_m8_raw.sh': No such file or directory
chmod: cannot access '/tmp/rqitest.XXXX/sbx_mut_.../tools/deploy/run_queue.sh': No such file or directory
  PASS: MUTATION KILLED: M8 chained-python detector blinded (accepted limit becomes invisible) (9 check(s))
```

Cycle 3's detector rewrite moved the text M8's mutation anchor targets, so **the mutant was never
built**. The harness then observed "the scenario failed" — of course it did, the mutant script does
not exist — and scored it **MUTATION KILLED / PASS**. **A mutation that cannot be constructed was
being counted as proof that a guard is live.**

This is the run's theme in its purest form. Every row self-reported green and every row had real
defects found by an independent lens; here the thing producing the green was the mutation harness
itself. Note cycle 2's own report had claimed "`_mutant`'s uniqueness assertion fails loudly if an
anchor ever moves" — it *does* print `MUTATION-BUILD-ERROR`, but the harness does not FAIL on it, so
the loud print was decoration on a silent pass.

**The fix commissioned is the general one, not the instance:** a mutation-BUILD failure must FAIL the
suite, distinguished from "mutant built AND scenario failed" (a real kill). Re-anchoring M8 alone
would leave the identical trap armed for M1–M7 on the next edit. **Transferable rule: a mutation
harness must never treat "I could not build the mutant" as "the guard is live", and a suite count is
not evidence until someone has run it and read the output, not the total.**

## TASK_881I: HALTED on the 3-cycle rule (D50 seeded)

Cycle 1 → BLOCKER + 2 MAJOR. Cycle 2 → BLOCKER. Cycle 3 → MAJOR. Three failed cycles, so the row
stopped rather than taking a fourth. It was tempting to argue for one more small fix — the cycle-3
finding is narrower than its predecessors and touches a fallback source nothing uses — and that
argument is exactly what the halt rule exists to override. A fourth cycle would have been the fourth
consecutive one whose fixing code contained the next defect.

**The layer that killed cycles 1 and 2 IS closed** and should not be re-opened: the correctness lens
confirms the manifest / protect-file / co-located-score surfaces refuse every constructed shape
(socket, FIFO in three contexts, symlink→file, live symlink→dir, symlink→dangling, real directory)
with no hang and NO false refusals, and both prior killers genuinely refuse.

**What halted it is a different layer the row never scoped: SCORE-SOURCE ENUMERATION.**
`_find_score_dirs` enumerates a directory only if it contains `SCORE_BASENAMES`
(`scores.jsonl` / `checkpoints_index.json`), so a run scored PURELY via `--score-regex`/`--log-glob`
— the module's own documented THIRD fallback source — is never handed to `_load_scores_strict`, and
that reader's log-file shape guard is dead code for exactly that input. Measured:
`--verify-manifests … --log-glob <FIFO>` with no `scores.jsonl` → `verify-manifests: OK`, **exit 0**;
the same flags on the real path → **`EXIT=124`, hangs forever**. So the mode is not the pre-flight
guarantee its own `--help` advertises.
**The tests could not have caught it:** all three new `--score-regex` tests construct a redundant
`scores.jsonl` alongside the log, and the suite's own comment documents that dependency instead of
closing it — so they cannot tell "the branch is reachable" from "reachable only when redundant".
*A test for a fallback source must not construct the primary source beside it.* That is the
transferable rule.

**Code preservation.** The work could NOT be tagged (`git tag` is blocked by the commit gate) nor
stashed (a `git stash` would have swept three other rows' concurrent work out of the tree). The
cycle-3 state is preserved as a two-part patch:
`lore/handoff_2026-08-19_factory_run_36_881I_cycle3.patch` (2470 lines; part 1 = `git diff` of the
tracked `tools/ckpt_gc.py`, 954 lines; part 2 = the untracked
`tools/tests/test_ckpt_881i_manifest_ingestion.py` inlined verbatim; sha256 `f9c6b4f2…`).
**NOTE: the working tree still CARRIES this work** — the patch is a durable copy, not a removal.

- **ORCHESTRATOR ERROR, mine (second one): every agent was pointed at ONE shared scratchpad.**
  TASK_892's fix reported that its A/B outputs (`ab_head*/ab_live/wt`) were **deleted mid-run by a
  concurrent agent** and that its git worktree was **pruned out from under it** (`git worktree list`
  lost the entry); it recovered by redoing the byte-identity A/B in `/var/tmp/t892ab`. With 6-8 agents
  live, `/tmp/claude-1000/.../scratchpad` is a shared mutable directory and any agent doing a cleanup
  `rm -rf` can destroy another's evidence. **Rule: give each agent a private `/var/tmp/<task>` dir**
  (which is what the surviving `judge865` worktrees happen to use) and reserve the session scratchpad
  for single-agent runs. Later briefs this run were corrected to say so.
- **A verification gate can assert something that BREAKS other agents.** 892's suite now asserts
  `experiments/892_action_ev/` gains no new entries ("tracked directory clean"). That is a good gate,
  but it means any judge or agent writing scratch into that directory silently fails someone else's
  test. Judges were briefed accordingly. Worth generalising: a no-new-files assertion is only safe if
  every concurrent actor knows about it.

## Open decisions this run touches

- **D43** (pre-existing, OPEN) — the 857 smoke-assert. Gated nothing here; no 857 number quotable.
- **D48** (NEW, seeded this run) — two calls, both one-word:
  (1) should the 819 prototype's `.py` sources be tracked? Until answered, TASK_896 is
  fix-complete-but-unshippable and its verified comment sits in the tree unversioned.
  (2) correct the instrument's two "verbatim" strings + the docstring granularity and RE-PIN the
  LEDGER digest deliberately, or leave both and accept that the measurement instrument overclaims
  inside a pinned artifact. (`tools/measure_leading_prefix.py:115-118` and the emitted
  `db.admission_predicate` both say "verbatim"; it is functionally equivalent, textually different —
  the modulo-lattice term is in SQL in one and in numpy in the other. Fixing the wording MOVES the
  sha256 the LEDGER pins, and D39's rule says a repair tool refreshing its own pin pins nothing.)
- **892's verdict number:** the SPEC pre-registers **D47** for the action-EV pivot ruling. The queue's
  row-1 note says "seeds D44" and `TASK_892`'s header says "seeds D42" — both already taken by ruled
  decisions. D47 is the live one; the two stale references want correcting.

## Panel findings by row (verification did the work this run)

Every row self-reported fully green and every row had real defects found by an independent lens.
Nothing below was self-reported by the implementer that wrote it.

### TASK_892 — 2 BLOCKER + 2 HIGH, all four biting the REAL LEG specifically
The instrument is well-built; its defects were all in the configuration the SPEC actually mandates,
which the shipped test suite never exercised (no `ext_windows`, no `--vector-weight-rebalance`).
1. **BLOCKER — degenerate zero-width CI whenever n < block(=21).** `n_blocks == 1`, and a single
   circular block of length ≥ L is just a ROTATION of the full sample; a mean is rotation-invariant,
   so all 10,000 resamples return the same value and `ci_lo == ci_hi == delta`. Any nonzero sample
   delta then reads EDGE or ANTI-EDGE with fake certainty and **can never read NULL**. The only size
   gate (`MIN_NON_HOLD_DECISIONS=30`) is on the POOLED count while per-window buckets are unguarded —
   and SPEC.md itself pre-registers that the within-regime match "thins further" from a low-hundreds
   base. Sub-21 buckets are the SPEC's predicted regime, not a corner case.
2. **BLOCKER — the budget/PV instrument-fail check is wired to DEAD CODE and would fail the real leg
   every time.** It uses `position_size_pct`, which is not read for sizing under
   `--vector-weight-rebalance` (the SPEC's mandated mode); sizing is `w_unit * portfolio_value`
   (`jax_trading_env.py:1806`). The repo already knew — `882_instrument_reconciliation/
   test_instrument_agreement.py:19` says "`position_size_pct` is DEAD CODE" and proves an 8× sweep is
   inert. At ~57 open positions × default 0.1 = 5.7 ≫ 1.0, so it flags a violation on essentially
   every decision, and `report.py` sums those into `instrument_fail_reasons` ⇒ **guaranteed
   INSTRUMENT FAIL, swallowing the verdict the experiment exists to produce.**
3. **HIGH — scope 2 not delivered, and the artifact lies about why.** `trace.py`'s `main()` never
   calls `ev.enforce_universe_pin(...)`; `run_walk(args, ev)` is always 2-arg, so `universe_pin` is
   `null` on EVERY walk. Proven on a genuine non-smoke `--checkpoint` run, where the emitted note
   still read "no checkpoint provenance to certify (fresh-init --smoke walk)" — false on that run.
   The "first consumer of TASK_886" claim is false as shipped.
4. **HIGH-MODERATE — per-window bands leak past the instrument-fail gate.** `regime_bucket_report()`
   calls `band_for()` with no knowledge of `instrument_fail`, and per-window entries carry no
   disclaimer. Demonstrated: an n=2 case tripping all three fail conditions still emitted
   `per_window["ALL"]["band"] == "EDGE"`; the real fixture chain emitted top-level INSTRUMENT FAIL
   beside `per_window["ALL"]["band"] == "NULL"`.
**Adjudicated NOT self-certification:** the implementer loosened a unit-test bound 0.5pp → 1.0pp. The
judge simulated 300 independent seed pairs (mean max-dev ≈0.65pp, P(>0.5pp)≈80%) and found the
committed 0.5593pp sits BELOW the median of pure sampling noise, while `report.py` still enforces the
SPEC's real 0.5pp bar on a different, far-larger-n quantity. Legitimate — different quantity,
different appropriate bound. Recorded because the reflex to call it self-certification was wrong.
**→ D49 seeded** for the one thing the factory must not fix itself: the forward-return estimator
deviates from the frozen pre-registration, and amending a pre-registration to match the build is the
self-certification this program forbids.

### TASK_881I — 1 BLOCKER + 2 MAJOR, then a clean fix
See the verified-first-hand section above for the dangling-symlink BLOCKER, the torn-SPEC-header
MAJOR and the argparse-abbreviation MAJOR. The fix (+651, pure insertion) closed all four with
per-function sha256 canon parity, a line-level SequenceMatcher showing ONLY insert opcodes,
behavioural parity across `<no flags>` / `--apply` / `--protect-file --apply` on a scratch tree with
real deletions, 39+20 tests, selftest 12/12, and every guard mutation-proven at its real drop site.
Flag renamed `--check-manifests` → `--verify-manifests` (letters `e`/`v`, unused at HEAD) with a
two-layer class-closing abbreviation test; `--c` again resolves to `--critical-free-gb`.
**Two residual risks the planner owns:** (a) `--verify-manifests` on DEFAULT `--roots` now refuses on
this workstation, because `jax_ckpt_fast` dangles while scratch disk is unmounted (both it and
`jax_ckpt_easy` dangle, but the default `--roots` is `["jax_ckpt_fast", "runs"]`, so only the former
is in scope for that invocation — flow-verifier's precision note) — the
intended loud signal, but a cron caller on defaults sees exit 2 continuously until an unrelated mount
is fixed; (b) a `protected:`/`covers:` line appearing in SPEC.md PROSE is now a refusal (none exists
today, but it is a foot-gun for a future SPEC author).
**Unrelated, found not fixed:** `experiments/832_linear_imprint/ckpt_smoke/ckpt_smoke` is a
self-referential symlink pointing at its own parent — harmless (the realpath `visited` set terminates
it) but it looks like a smoke-run accident.

**Then a FRESH fail-closed lens found a SECOND BLOCKER in the same layer — the run-34 precedent
repeating exactly (a fresh lens catching what three prior lenses missed).** A manifest BASENAME
(`PROTECTED.txt`/`SPEC.md`) that is itself a **directory-type symlink** is invisible to every guard:
silent exit 0. Crossvalidated by the orchestrator with the control set that pins the diagnosis:
| shape | result |
|---|---|
| `PROTECTED.txt` → symlink to a DIRECTORY | silent `OK`, exit 0 |
| no manifest at all | **byte-identical output** |
| `PROTECTED.txt` → symlink to a FILE | correctly REFUSED, exit 2 |
An operator therefore cannot distinguish "nothing to protect" from "a protection intent I silently
dropped", going into a deleter.

**The root cause is one mistake, not two bugs, and it is the transferable lesson of this run:
security-relevant decisions were being made DOWNSTREAM of a link-FOLLOWING classification.**
`os.walk` buckets entries via `DirEntry.is_dir()`, which follows symlinks. Cycle 1 died because a
DANGLING link-to-dir gets `is_dir() == False` and is misfiled into `filenames`. Cycle 2 died on the
mirror image: a LIVE link-to-dir named `PROTECTED.txt` gets `is_dir() == True` and is misfiled into
`dirnames`, where nothing evaluates manifest-basename identity (`load_manifests` only tests
`fn in MANIFEST_BASENAMES` over `filenames`). Two cycles, two shapes, one root. That is precisely
D38's halt signature — "the same layer, each cycle's fix containing the next defect".
**Cycle 3 was therefore directed at the ROOT, not a third shape:** classify every entry with
`lstat`/`islink` independently of how `os.walk` bucketed it, and evaluate manifest-basename identity
BEFORE the dir-vs-file split, over both lists. Cycle 3 is the last cycle under
`halt_after_failures: 3` — if it does not close, the row halts to a DECISIONS row rather than shipping.
**Two findings deliberately NOT fixed, both ruled by the judge:** the SPEC-prose header strictness is
fail-SAFE (a bogus path hits the loud dangling-entry refusal; it can only make the deleter more
conservative), and the default-`--roots` refusal is the unmounted-scratch disk ops item. Quieting a
correct refusal on a checkpoint deleter to make output tidier is the wrong direction.
**MINOR also closed in cycle 3:** `_load_scores_strict`'s docstring claimed a torn record RAISES,
but its `--score-regex` branch did `except Exception: continue` — as permissive as canon, and
untested. Same overclaiming class the program keeps walking back.

**Cycle 3 fixed the ROOT and found a THIRD defect of that same root that no cycle had caught: a FIFO
named `PROTECTED.txt` HUNG the tool forever** (`open()` on a reader-less FIFO blocks; measured
`EXIT=124` under `timeout 15`, and likewise for a FIFO `scores.jsonl` and a FIFO `--protect-file`).
Three cycles, three shapes, one root — which is exactly why the third cycle was directed at the
classification layer rather than at another shape. The fix: one `lstat`-based `classify_entry()` that
never follows a link to decide a shape (one following `stat` types the END of a chain; `ELOOP` ⇒
dangling), `_walk_scan_dirs` yielding `dirnames + filenames` **un-split** so nothing downstream can
inherit a bad bucket, manifest-basename identity tested FIRST over every entry, and shape checks
before every `open()` on the manifest, score-file and protect-file paths. Ten shapes × both basenames
are now tabulated with actual output; the three previously-silent shapes (live symlink→dir, real
directory, symlink→symlink→dir) each refuse with a DELIBERATELY DISTINCT message so the three
diagnoses can never conflate. 7 mutations rebuilt at real drop sites, three of which HANG when their
guard is removed. Tests 39 → 86 (+1 skip), 106 with the 876 suite; selftest 12/12. Fence re-proven:
0 removed content lines (924 insertions), per-function byte-identity vs `git show HEAD:`, identical
`--apply` survivors, and `--verify-manifests --apply` deletes nothing. Real tree read-only: the walk
reaches **56/56** manifest basenames, 0 missed. Honest self-limit: the descent-decision refactor is
"a principle refactor with no behavioural delta — I am not claiming it as a mutation-proven guard."

**PLANNER ITEM, and it matters BEFORE TASK_881D runs: there is no live manifest corpus at all.**
`experiments/828_stage3_train/PROTECTED.txt` — the artifact D45 note 2 / scope item 5 requires to stay
"loudly named in every report" — **does not exist in this tree**, and of the **56** `SPEC.md` /
`PROTECTED.txt` files under `experiments/`, **NONE contains a `protected:` or `covers:` block**. So
scope item 5 is unsatisfiable as written, the dangling-entry machinery is exercised only on rebuilt
inputs, and — the part that bears on 881D — the protection model currently protects NOTHING, because
nobody has authored a manifest. That changes the risk calculus for lifting the D32/D34 `--apply`
freeze: 881D's protection layer would ship with an empty live input, so its first real exercise would
be whatever manifest an operator writes afterwards. Worth deciding whether authoring the real
`PROTECTED.txt` corpus (the three 828 entries D45 names, at minimum) is a precondition for 881D.
This is the **third** instance this run of a spec naming something the tree cannot supply
(the others: `TASK_881I`/D45 naming `tools/ckpt_archive.py`, and D46 clause 5 naming an untracked
manifest).

### TASK_891 — 1 BLOCKER (fail-closed) + 1 MAJOR + 2 routed; correctness found NO blocker
The BLOCKER and the clause-4 logging gap are in the verified-first-hand section. The correctness
lens verified every gate live and confirmed the rewrite preserves `--tag python`, a quoted
`"a python b"` argument and the `&&` tail byte-exact; 6/6 mutations real (M4 and M6 hand-rebuilt).
**The manifest finding got sharper on inspection and changed the fix:** in
`runs/860_codistill/manifest_860.txt` the SECOND `python` is the eval that reads the corridor result,
and on gpu-server bare `python` is python2 (`884_rung1/manifest_884_battery.txt:6` says so), so
re-running that file there silently runs the eval half under python2 — the exact failure D42/D46
exist to eliminate — while the operator sees `interpreter=passthrough:python`, identical to a
harmless row. Because that manifest is UNTRACKED, an inline comment there is not durable, so the
primary fix was redirected to a **runtime warning in tracked code for chained-bare-python passthrough
rows** (kept a WARNING, not a refusal: D46 priced chained pythons as an accepted NOT-COVERED limit,
and the factory does not start refusing rows the ratified contract permits).
**MINOR:** the sweep's discovery step used a FILENAME heuristic (15 files) and missed a 16th tracked,
run_queue-format file, `experiments/828_pathway_r0r1/jobs.txt` — verified tracked and CLEAN (all 9
python occurrences spell `trader_env/bin/python`), so the conclusion held but the method would have
missed a dirty file the same way.

**The fix pass (cycle 2) is the best-argued artifact of this run, and its method is worth copying.**
It did not merely add the launch-time re-probe (shared `_venv_usable_ok` at BOTH call sites so they
cannot drift, refusal by name, GPU released via `_free_gpu`, `venv-refused` recorded durably). It
**measured the strengthened probe's limits and documented the forgery that still defeats it**:
nonce + `sys.executable` + `sys.prefix` refuse `/bin/true`, a fixed-output stub, a plain
`exec`-delegator and a foreign python — **but `exec -a <our path> <real python> "$@"` plus a planted
3-line `pyvenv.cfg` passes all four**, and the comment says exactly that instead of claiming the hole
is shut. That is the opposite of the overclaiming habit this program keeps having to walk back.
It also proved the guard live rather than asserting it: mutation **M7** removes the re-probe at its
real drop site, gate **P** goes red, and it re-ran the original exploit on that mutant to show
`FORGED_PAYLOAD_RAN` come back.
The replacement claim text states plainly that **this is not a security boundary and never was**,
that `safe_run.sh:50`'s `systemd-run --user --scope` gives the queue no privilege boundary over its
jobs, that `protected` in `.dark-factory/config.json` means "agents are told not to write it" and NOT
"jobs cannot write it", and that the residual TOCTOU is shrunk "from hours to milliseconds" while
catching "nothing that races the exec — which, per the THREAT MODEL above, it is not trying to."
D46 clause 4 was closed in a durable way (the live launch line prints the rewritten absolute command
AND the summary carries a rewritten-commands block, since a launch line only reaches tmux scrollback).
The chained-python signal became a **warning, not a refusal** — deliberately, because D46 priced
chained pythons as an accepted NOT-COVERED limit and the factory does not start refusing rows a
ratified decision permits. Detector documents `n=0` as "none visible, never none present".
**The exhaustive re-sweep settled the question:** content-shape predicate over all **1438 tracked
files** → 18 contain run_queue-format rows, **0** contain a NOT-COVERED bare-python spelling; the
untracked `runs/**` scan found exactly one dirty file, `runs/860_codistill/manifest_860.txt` — i.e.
**the file D46 named is the only NOT-COVERED manifest anywhere in the repo.**
Suites: interpreter 18/18 (10 gates + 8 mutations M1–M8 all killed), scheduler 16/16, watchdog 18/18
(at load average 13.04 — the known flake did not bite), selftest 12/12. Gate (v) byte-identity
re-verified in paired sandboxes vs `git show HEAD:` — raw diff is ONE line (the new `interpreter=`
label) and byte-identical after normalising it.
**Two behaviour changes flagged by the implementer and sent to re-verification, not accepted on
report:** (a) `launch()` gained a return contract (0/1) where it "previously returned 1 for every CPU
row by accident", and four call sites now gate `_watch_register`/`confirm_vram_or_warn` on it;
(b) the probe now asserts `sys.prefix == dirname(dirname(interp))`, which assumes a normal venv
layout — a false refusal there would block a real GPU sweep, a worse regression than the hole closed.
**New noise, intentional:** the chained-python warning fires 3 stderr lines on every re-run of the
860-shaped manifests.

### TASK_896 — fix-complete, ship-blocked
A correctness judge found its most concrete sentence false (the granularity BLOCKER) even though the
implementer's own checks were green. The opus fix re-derived every number independently, matched the
orchestrator's measurement element-for-element, and added correctly-scoped clauses (the 10
zero-prefix majors; the 13 late listings owning all 4904 FULL cells) — all verified. Blocked only by
D48.

## Shipped

| Version / tag | Row | Verification outcome |
|---|---|---|
| `task891-run-queue-venv-python` (`706ce97` + banner `440f238`) | **TASK_891** — run_queue venv python, D46 second re-cut | 3 cycles (2-3 on opus). Gates (i)-(vii). 1 BLOCKER + 3 MAJOR + 2 MINOR closed. Suites interpreter **25/25**, scheduler 16/16, watchdog 18/18, selftest 12/12, gate (v) byte-identity vs HEAD. Orchestrator verified the three cycle-3 claims first-hand: no hollow mutation passes, detector 12/12, every header number true. |

## TASK_892 — the panel earned its cost twice

Cycle 1: 2 BLOCKER + 2 HIGH (see above). Cycle 2's fix was thorough, honest and self-consistent —
and two FRESH lenses still found 1 BLOCKER + 5 MAJOR in it. The two worth carrying forward:

- **A code comment falsified by a six-line probe.** `report.py:234` said "Nothing here can emit a
  zero-width interval." Constant inputs at/above the new floor (`s=np.full(42,0.02)`, `n=np.full(42,0)`)
  give `ci_lo == ci_hi == 0.02`, `insufficient_n=False`, band **EDGE**. The floor guarantees "enough
  blocks", not "enough variance", and the comment conflated them.
- **A guard whose STATED claim was true and whose IMPLIED claim was off by 4-5x.** `MIN_BOOTSTRAP_BLOCKS=2`
  does kill the literal rotation-invariant case, and `BOOTSTRAP_POWER_DISCLOSURE` claims only that. But
  simulated against the real code (i.i.d. N(0,1), true delta=0): 95% CI coverage is **80.7% at 2 blocks
  (n=42)**, 81.7% at 3, 90.0% at 4, ~91% at 10, and only **94-95% at 15-20 blocks**. Under a true null at
  the shipped floor, **22.3% of comparisons band EDGE or ANTI-EDGE** against a nominal 5%. The suite's
  gate asserted only `ci_hi > ci_lo`, never calibration. So the floor "prevents nonsense, not
  overconfidence" — and per-regime buckets, which SPEC itself says "thin further", are exactly where n
  lands in that range. The primary pooled read is not at risk (~80 blocks per the operator's audit).

Also found: `report.py`'s ONLY soft `.get(..., 0)` in the whole comparison meant a trace missing
`budget_conservation_violations` — with no `schema` validation either — produced a complete unflagged
**EDGE** (BLOCKER); `NaN > x` is False so a NaN allocation was silently not a violation, contradicting
the module's own "cannot miss an overdraft/leverage defect"; and `build_trace_doc`'s two positive-match
raises let a THIRD `walk_kind` through, emitting a note falsely claiming a smoke walk on a document
carrying a real checkpoint label.

**ORCHESTRATOR ERROR, mine (third instance, different substrate).** The `--smoke` checkpoint race is
downstream of my own instruction. I told the implementer to move default `--out` writes out of the
tracked directory; it did so by consolidating every default onto ONE shared literal path
(`_scratch/smoke_ckpt`), and gate (v) runs two `trace.py --smoke` subprocesses back-to-back, each doing
an orbax save+reload against it. Live: run 1 FAILED (`ocdbt read failure`), an immediate re-run PASSED.
**A suite whose green depends on disk timing is not evidence.** All three of my errors this run were the
same shape — shared mutable state between concurrent actors (one file, one scratchpad, one default path).
Cycle 3 was told to run the acceptance suite TWICE back-to-back and report both, because a single green
no longer counts for this row.

Verified clean by those lenses and left alone: the budget identity (`1 - cash/PV > 1 <=> cash < 0`
exactly, same post-step state, `max_allocated=0.999963` being the intended `1e-4` fee haircut rather
than saturation); the opt-in collector across every `rollout` call site; `finalize_universe_pin_record`
as a verbatim extraction with two callers; 886's record inherited not synthesized; operator coherence
across 11 adversarial CLI shapes; the undroppable disclosure; and the 57-option abbreviation table as an
exact match.

### TASK_892: HALTED on the 3-cycle rule (D51 seeded)

Cycle 1 -> 2 BLOCKER + 2 HIGH. Cycle 2 -> 1 BLOCKER + 5 MAJOR. Cycle 3 -> 1 BLOCKER. **All three
cycles died in the SAME layer: `report.py`'s banding/exclusion logic.** Every TRACE-layer finding
closed and held under independent re-verification.

**The cycle-3 defect, crossvalidated by the orchestrator with a control.** `report.py:423` is
`zero_width = not (float(hi) > float(lo))` — bit-exact:

| case | CI width | zero_width | band |
|---|---|---|---|
| bit-identical `np.full(k, 0.02)` (what the shipped gate constructs) | 0.000e+00 | True | ZERO-WIDTH-CI |
| same data + per-block `N(0,1e-9)` jitter, 35 blocks | 6.687e-10 | False | **EDGE** |
| same, 33 blocks | 7.890e-10 | False | **EDGE** |

So the guard catches only bit-identical floats, while `disclosures.py` claims it catches inputs that
"collapse the interval at ANY block count" and names near-duplicate autocorrelated per-regime buckets
as **its own plausible real route**. Reproduced through the full `compare()` pipeline on schema-valid
traces, not a hand-built dict.
**Severity stated honestly:** the MECHANISM is broken and the CLAIM is false, but real-data
REACHABILITY is unestablished in both directions — real overlapping forward returns would not agree to
nine decimals. It is still a false claim inside a fail-closed guard, and the third consecutive cycle
whose fix contained the next defect in one layer.

**The transferable rule** (the three failures share it): *an exclusion guard must be a tolerance
relative to the data's own scale, never an exact-equality or bit-identity test — and its test must
construct the NEAR case, not the exact one.* Every one of these guards was tested against the exact
case it already handled.

**What is verified and should NOT be re-opened by the re-cut:** the realized-allocation budget check
(the dead-field form reported 30/30 violations at alloc 5.70 where the realized form reports 0 at
0.8906 — the real leg WOULD have read INSTRUMENT FAIL every time); `universe_pin` actually consumed
with 886's split inherited not synthesized; the NaN guard; the `walk_kind` enum; the per-invocation
smoke checkpoint path (RED 2/3 concurrent runs died, GREEN 3/3); atomic writes; the
`--marginals-sidecar` rename with a 470-prefix class-closing gate; and the operator-coherence refusal
that closed a fail-open where a NONEXISTENT sidecar exited 0 with a complete plausible report. Gate (i)
byte-identity re-proven against a real HEAD worktree: 0 scored-block diffs, canonical JSON identical.
Suite 93 PASS / 0 FAIL twice back-to-back; 8/8 mutation reverts caught; selftest 12/12.

**Code preservation:** `lore/handoff_2026-08-19_factory_run_36_892_cycle3.patch` (3886 lines; part 1 =
379-line `git diff` of the tracked `eval_vector.py`, part 2 = all six untracked `892_action_ev` files
inlined; sha256 `e8bc83b0…`). The working tree still CARRIES the work.

## Remainders -> planner

- **D48** (untracked-but-cited evidence; 819/893/860): one-word call on tracking the 819 prototype's
  `.py` sources, and a second on correcting + re-pinning the instrument's two "verbatim" strings.
- **D49** (892's estimator vs its frozen pre-registration) — now carries the **interval-method
  escalation**: coverage plateaus at ~88% under realistic dependence and never reaches 95% at any
  attainable n, because pre-registered `block = 21` EQUALS the dependence length. Remedy is the block
  length or the interval method, not more data. The instrument's two-scenario calibration harness is
  reusable and is the cheapest way to settle it before the real leg.
- **D50** (881I halt): re-cut keeping manifest ingestion, new row for score-source enumeration; plus
  the finding that **there is no live manifest corpus at all** (none of the 56 `SPEC.md`/`PROTECTED.txt`
  files carries a `protected:`/`covers:` block), which bears on TASK_881D and on lifting the `--apply`
  freeze.
- **D51** (892 halt): re-cut keeping the trace/instrument layer; the `report.py` banding layer should
  land TOGETHER with SPEC Amendment 1, since Amendment 1 changes what the report computes and the
  interval method is under review.
- **TASK_891 shipped**: redeploy `tools/deploy/` to gpu-server (as TASK_833).
- Housekeeping: stale worktrees `/var/tmp/judge865/{BASE,CAND}` pin old commits; the watchdog suite's
  load-sensitive flake remains an unspecced candidate row (it did not fire once this run).
- The operator's `docs/pathfinder_audit_2026-08-19.md` ranks the next real value elsewhere than this
  queue: **G1** (the 806->811 minute signal, whose 809 PASS was an estimator artifact but whose
  CORRECTED number was never computed — "died of venue economics and a broken estimator, not of
  evidence"), **G3** (the fallback rule measured dead in recent data, so `pathway.md`'s "fixed rule
  stays the fallback shippable at every rung" is currently FALSE; a long-flat/no-short spot-legal
  variant never measured), **G6** (data rot as the binding constraint on any future OOS claim). All
  CPU-hours.

## Run status

4 rows taken. **1 SHIPPED** (891). **2 HALTED on the 3-cycle rule** (881I, 892) — both correctly, both
with the failing layer isolated and the verified layer preserved. **1 fix-complete but ship-blocked**
(896, on D48). **4 decision rows seeded** (D48, D49, D50, D51). Tree carries the two halted rows'
work plus the planner's audit; both halted rows are preserved as patches beside this handoff.
