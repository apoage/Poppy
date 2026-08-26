# Factory run 42 — 2026-08-25

**Nothing shipped. Nothing committed. One new wall (D64), one destructive incident (contained and
recovered), two rows left mid-flight with their findings preserved.**

The tree is deliberately left DIRTY. The commit gate requires a fresh green verdict and there is no
green verdict, because nothing reached green. Writing one to satisfy the hook was the one move not
available — that is the failure mode this entire run was about.

---

## 1. The incident (read this first)

**~09:05 CEST: the TASK_881D flow-verifier destroyed `experiments/` and `runs/`.** It ran `rm -rf`
on a RELATIVE path intending its own scratchpad, while the working directory was the repo root
because **a `cd` in an earlier tool call had not persisted**. 255 tracked files and every gitignored
checkpoint were wiped, leaving only its own pytest fixtures (`experiments/exp1`, `runs/r1`).

**How it was caught: by a different agent, not by the one that did it.** The fail-closed judge
noticed mid-review, correctly refused to attribute the damage to the diff it was reviewing, declined
to attempt recovery under its read-only mandate, and escalated. That is why this cost minutes rather
than the session. Both behaviours are worth keeping: the escalation, and the refusal to misattribute.

**One assessment in that escalation was wrong, in the direction that mattered.** It reported the loss
as unrecoverable, assuming the deletions were gitignored. They were not — **all 255 deleted files
were tracked**, including the 29 under `runs/`. Checking that assumption before acting is what turned
"catastrophic" into "recoverable".

### Recovery, verified

| Asset | Status | How |
|---|---|---|
| 255 tracked files | **RESTORED** | `git restore --source=HEAD` on the explicit deleted-path list; 0 deletions remain |
| 7 load-bearing checkpoints | **RESTORED, byte-identical** | rsync from gpu-server; both sides hashed with the same method, all 7 match |
| `runs/806_predictability_probe` | **RESTORED** (6 files) | rsync from srv |
| `runs/875_wf` | **RESTORED** (1.4 G, 269 files) | rsync from srv — but see the loss below |
| 881D's 3 `PROTECTED.txt` corpus | **RESTORED** | extracted from the committed run-41 lore patch |
| `experiments/837_walkforward/folds/real` | **INTACT**, hash `1b31624a2da9984620567afc30df4cbf` | restored from git; 904A's gate-(iii) baseline holds |
| `trading_data.db` (25 G), fixture, all 3 backups | **NEVER IN THE BLAST RADIUS** | verified present |

Restored checkpoint hashes (sha256 of sorted per-file sha256s, 16-hex prefix), all matching srv:
`2d6d167ada2148a0` `9a8a90851700d611` `cca94aff05f7adb3` `243f87e8f05d6f63` `46f344759897c57c`
`753f6902a6a11b4e` `5272638e51aebee2`

Post-recovery the 881D suites run **393 passed / 1 skipped**, identical to the pre-incident count.

### Genuinely lost — OPERATOR SNAPSHOT ITEMS

A snapshot of `/home/lukas/Trader` from **before 09:00 on 2026-08-25** covers both.

1. **`experiments/906_oos_nonstationarity/`** — 7 untracked files created 08:39–08:45:
   `PREREGISTRATION.md`, `independent_construction.py`, `measure_causes.py`, `variance_profile.py`,
   `GATE1_INDEPENDENT_CONSTRUCTION.json`, `GATE4_CANDIDATE_CAUSES.json`, `VARIANCE_PROFILE.json`.
   Worth recovering for a reason beyond convenience: two judges verified the **pre-registration
   ordering by file mtime** as evidence of discipline. Rebuilding creates new mtimes and destroys
   that property; only a snapshot preserves it.
2. **`runs/875_wf/WF_SUMMARY.json`** — must be the version with md5
   `4c306427a8f40f44f3058b34a7bad6a1`. **srv holds only a 2-byte `[]` stub** (md5
   `d751713988987e9331980363e24189ce`, confirmed by hashing `[]` directly). Never tracked, so git has
   nothing either. **Do not accept the stub as a restore** — it satisfies a file-exists check and
   would silently re-base 904A's gate-(iii) pin, the same class as the gitignored-fixture-regen
   incident in this project's own records. The rest of 875_wf came back real (193 files over 1 k; that
   stub is the only file under 10 bytes).

`experiments/837_walkforward/folds/smoke` is also gone but regenerates from a `walkforward.py smoke`
run — needs neither snapshot nor srv.

### A second, smaller containment failure the same morning

The 904A **correctness judge** clobbered 5 tracked files under `folds/real` by pointing a **symlink**
at the tracked tree and running `materialize` through it — its brief said "destructive experiments in
/tmp scratch copies only". It disclosed this, restored the exact bytes from HEAD, and the concurrent
regression judge independently observed the write without misattributing it. `run_smoke`'s own
invariant check fired loudly rather than continuing past corrupted state. Three layers held.

### STANDING CORRECTION TO EVERY VERIFICATION BRIEF FROM HERE

Both agents that caused damage **had explicit do-not-write fencing.** The fencing named the target
but not the MECHANISM. Every brief must now carry, verbatim:

- **Destructive commands take ABSOLUTE paths only.** Every `rm`/`rm -rf`/`mv`/`chmod`/`>`/`rmtree`
  target is one absolute path beginning `/tmp/`.
- **`cd` does NOT persist between tool calls.** Start each command with an absolute path, or
  `cd <repo> &&` in the SAME command.
- **Nothing under the repo is ever an `rm -rf` target — including via a symlink.**
- **Do not create symlinks that resolve into the repo.**
- Gate re-verification must not run concurrently with an adversarial lane on the same tracked tree.

This is already applied: the re-run flow-verifier, correctness, security and regression briefs all
opened with that preamble, and none of them touched a tracked path.

---

## 2. D64 — the new wall (blocks TASK_881D)

**The manifest COVERAGE predicate has no AUTHORITY BINDING, and 881D is what first makes it gate an
irreversible deleter.** Found by the security lens, reproduced independently by the orchestrator on
an isolated `/tmp` tree:

```
no manifest anywhere            → REFUSE run dir … nothing pruned or written, dry-run or --apply
+ 'covers: runs/target_run' in experiments/hostile_unrelated/PROTECTED.txt
                                → covers …, keeping 2      (refusal gone)
--apply --keep-best 1           → DELETE runs/target_run/checkpoint_1
```

No malformation needed. A grammatically perfect manifest in an unrelated directory is sufficient.

**Two points of precision, because the lens overstated one.** (1) 881D's premise "missing coverage is
never read as 'nothing to protect'" is **NOT falsified** — the no-manifest refusal holds exactly as
claimed. What is false is the *converse* a reader assumes: that "covered" means someone **with
standing over that run dir** reviewed it. (2) The flat namespace is **not a defect 881D introduced**
— it is the settled 881I/881G grammar (D53), documented verbatim in `ckpt_gc.py` ("Either name,
anywhere under a scanned directory, is read as a manifest"), quoted verbatim in 881D's own scope at
`tasks/TASK_881D_PROTECTION_DELETER.md:92`, and the tool's own refusal text instructs the operator to
add the line to "an `experiments/<id>/PROTECTED.txt`" — any id.

**The consequence is new; the cause is settled.** Binding manifest identity to run-dir authority is an
ingestion-SEMANTICS change, and D63 ruled those "are not reopened — the amendment is additive-only".
So the factory halted rather than guessing past a ruling. D64 carries three options; **(a)** a narrow
written amendment binding coverage to the manifest's own experiment id is recommended and is
fail-closed (it can only ever REMOVE coverage, never add it). **(b)** ship 881D but explicitly
decouple it from lifting the `--apply` freeze — the right interim, and it composes with (a).
**(c)** rule the gap acceptable by design, but then say so in the docstring, which currently reads as
a stronger guarantee than it gives.

Relevant to the stakes: today proved `runs/*` is gitignored and unrecoverable without a second host.
The `--apply` freeze standing is the only reason none of this was live.

---

## 3. Row-by-row

### TASK_881D — HALTED on D64. Work BUILT, GREEN, IN THE TREE, NOT SHIPPED.

Both rulings were satisfied. D63(a): `--known-dangling PATH…` (with `_spellings_of` +
`_AcknowledgedIndex`, matching by realpath/abspath set-intersection, never by prefix) so
`--verify-manifests` reaches exit 0 on the real tree despite the 3 acknowledged dangling
828/scratch disk entries; plus the walk-pruned-dirname report so a manifest under `.git`/`__pycache__`
is no longer a silent withdrawal. `INGESTION_NAMES` keeps its full 43-name inventory with the two
sanctioned functions re-pinned to their NEW bytes via **hardcoded** sha256 digests (verified not
self-derived). D62(a): `test_canon_functions_are_byte_identical_to_head` restored over
`UNTOUCHED_GUARD_NAMES`, and `test_diff_against_head_removes_no_content_line` un-skipped behind a
named entity allowlist — measured at 11 entities covering **28.8%** of `ckpt_gc.py`'s 3734 lines,
leaving **2657 lines still protected**, so the restored pin is load-bearing.

**All 5 HIGH-tier lenses completed. 4 clean.**

| Lens | Result |
|---|---|
| fail-closed | clean; 6 gates MET; 3 named attacks failed to break it |
| regression | clean; gates A–G; corrected the orchestrator's baseline (below) |
| flow-verifier (re-run) | clean; TTL BLOCKER verified in all 3 directions; one item disclosed as inspection-only |
| correctness | 1 MAJOR (in scope, fixable) + 1 MINOR |
| security | **BLOCKER → D64** |

**MAJOR to fix on resume — `--keep-best N` does not keep N.** `plan()` (~L1233-1247) caps
independently *within each metric group*. That is deliberate ("ranked WITHIN a metric, never across")
but untested with >1 checkpoint per metric across >1 metric — and **this repo's own trainer reaches
it**: `train_jax_fast.py:1501` sets `ckpt_metric='train_reward_ma20'` every epoch, `:1975` overrides
to `f'eval_reward_L{...}'` on eval epochs, both into the same run dir's `scores.jsonl`. Measured: 10
ckpts (8+2 metrics), `--keep-best 3 --keep-latest 1` → **5 survive**, printed as `keeping 5` at exit
0. Five distinct metrics with `--keep-best 1` → **all 5 survive**. Fail-SAFE (over-retention) so not
BLOCKER-class, but it silently defeats the tool's stated purpose. **MINOR:** the equal-score
tie-break follows `os.walk` order, so which of several tied checkpoints survives is
filesystem-dependent; wants a documented recency tiebreak.

**Scope item 4 is now CLOSED:** the disputed `_order_newest_first` got its third independent
measurement (own-mtime primary, number only as sub-second tiebreak), refuting the borrowed-mtime
defect.

**Baseline correction (the regression lens caught the orchestrator).** The "340 passed / 2 skipped"
carried into several briefs was **mislabelled** — it was measured over FOUR files on the tree with
the run-41 patch already re-applied, i.e. post-restore/pre-amendment, not a pre-881D baseline. True
git-HEAD is **288 passed / 1 skipped** (`test_ckpt_881_protection_model.py` does not exist at HEAD);
working tree is **393 / 1**; delta **+105 passed, 0 removed anywhere**, fully additive. **Use 288/1 →
393/1 as the accounting of record.** An unreconciled baseline is how a phantom delta gets waved
through, and refusing it was correct.

Work preserved at `lore/handoff_2026-08-25_factory_run_42_881D_d64_halt.patch` (5037 lines: tracked
diff + all 4 untracked files), second copy in the session scratchpad.

### TASK_904A — implemented, 1 BLOCKER, fix pass killed by the incident. Tree reverted to HEAD.

The D61(a) split **verified clean in both directions**: `corridor_for_root` byte-identical to HEAD by
AST extraction (not grep), all 6 claimed-excluded 904B symbols absent, and each of the 7 dropped
tests ERRORS against this tree on missing 904B symbols — proving they covered 904B, not smuggled-away
904A coverage. Gate (i)'s narrowing is honest and complete (hashing the entire tracked subtree
before/after a real smoke run shows `results_fixture_smoke.json` as the only tracked write, matching
the new `tracked_write_claim` field). Gate (iii) held. All three components shown RED by mutation at
their real drop sites — notably, reverting the guarded reader reproduced the exact cycle-3 FIFO hang
(exit 124).

**BLOCKER — the fixture-over-real refusal does not protect the tree it exists to protect.**
`_assert_out_root_writable` refuses fixture-over-real only when `prior_kind == "real"`, and
re-derives that class from the **target directory's own** `fold_manifest.json`. The tracked golden
rails record `universe_db = "fixtures/trading_fixture.db"` (generated 2026-07-08, `fold_set = real`)
because that tree really was built from the fixture in July. So `prior_kind` is `"fixture"`, the
refusal never fires, and `materialize --fold-set real --overwrite-existing` — with the CLI's DEFAULT
`--db` — rewrites the tracked golden rails while logging "ordinary re-materialization".

**Run-41's lens, landing on the row that inherited it: the guard's logic is correct and it governs
the WRONG OBJECT** — the directory's self-description instead of the destination's identity. The
comment one line above already says trusting the stored `db_kind` "would let a directory authorize its
own clobbering"; re-deriving from the same directory's `universe_db` has the identical defect here.
**Fix direction:** an additional identity-based fence (realpath of `out_root` being the canonical
`folds/real` together with `fold_set == "real"`), keeping the existing db-class check, which correctly
catches other cases.

Note how the two lenses reconciled rather than contradicted: the flow-verifier found fixture-over-real
**is** refused on decoys whose manifest records a real DB, and separately reported it could not reach
the `--fold-set real` default path. That unreached input is exactly the one that defeats the guard.

Also open: **MAJOR** — `aggregate` on a missing manifest raises a raw `FileNotFoundError` traceback
instead of a parseable `LEAK GATE:` refusal (`validate_folds` pre-checks `os.path.lexists`,
`aggregate` does not). **MINORs** — the "FIFO exits 2" claim is wrong (every refusal path exits **1**;
correct the claim, not the code), and the excluded-test disclosure names 6 of the 7 actually dropped.

Full fix brief is preserved in the task file's banner. Re-apply from
`lore/handoff_2026-08-24_factory_run_41_904_cycle3.patch`.

### TASK_906 — implemented, 3 BLOCKERs, work LOST to the incident.

The findings survive in the task banner even though the artifacts do not; that banner is now the
authoritative record. Three BLOCKERs: the banner claimed scope-4 exclusions its own
`GATE4_CANDIDATE_CAUSES.json` contradicted (`alias_4a` = `NOT EXCLUDED`); the forward-fill test was a
**temporal-misalignment bug** correlating 26 decision-chunks against 217 raw-bar-chunks by truncation,
so it compared the full-window Sharpe series against only the first ~12% of the window (reproduced to
14 significant figures, proving that is the code that ran); and the `experiments/LEDGER.md` row was
left contradicting the corrected banner — which the orchestrator's own banner fix created, and which
matters because the doctrine paragraph declares the ledger to be *where verdicts are read*.

Three MAJORs: even realigned, 4(b) tested the **wrong target statistic** (mean-based Sharpe, when the
row's own headline is a variance effect); the Levene p-values (to 4.3e-274) are meaningless on an
anti-persistent series — though a judge then **closed that gap favourably** by block-permutation
(observed 29.62 vs a permutation max of 14.70 over 1500 replicates, p ≤ 0.002, so the conclusion
survives a fair test and only the quoted strength was wrong); and "directionless" understates a real
localized tail event (final 3 chunks lowest of all 24 consecutive 3-windows, ~1/2600) which an OLS
slope test cannot detect — compounded by the three horizons **not** being independent confirmations
(`chunk_std` profiles correlate 0.79–0.93).

**Two attacks were attempted and REFUTED, strengthening the row:** the flat mean is real and not
underpowered (MDE Cohen's f ≈ 0.042–0.056, below conventional "small"), and 901's VR = 7.19 reconciles
mechanically via `corr(chunk_sharpe, chunk_std) = 0.895`. Causes (c) and (d) survived, (d) cleared as
non-tautological.

**Thread worth following on resume:** the tail contraction may be construction-linked, and the alias
twin id 1144's rows begin exactly **2025-01-01**. If the unexcluded alias gap and the variance
structure are the same phenomenon, that speaks directly to scope 4's instrument-versus-market
question. A null is still a result.

---

## 4. Off-lane: 897 leg #2 landed

The srv leg the planner launched 2026-08-24 ~19:01 **completed 22:12**: `864/864 cells recorded
(ok=864)`, runtime 11438.5 s, artifact `runs/897_corrected_809/result_897_2026-08-24.md` on srv
(423 KB). Leg #1 was the `ok=0` honest NULL of D60, so the D60 guard re-base worked. **The verdict
read is planner-lane and was not done here.** gpu-server has been idle since.

---

## 5. State at disarm

- **Tree DIRTY, nothing committed, nothing pushed, no verdict written.** `.dark-factory/verdicts/`
  has no green verdict because nothing reached green.
- Working tree holds: 881D's complete D62(a)/D63(a) work (9 modified + 1 untracked + 3 corpus files),
  the three task banners, D64 in `tasks/DECISIONS.md`, and the two preservation patches.
- 904A's edits were reverted to HEAD by the recovery restore; re-appliable from the run-41 patch.
- **To commit this, the operator must either resolve D64 or explicitly authorise a documentation-only
  commit** (banners + D64 + patches), which the gate will still refuse without a green verdict.

### Queue implication

`tasks/QUEUE.md` was NOT re-groomed (planner-owned). On the current evidence the order wants:
**D64 first** (it blocks row 1), then **904A's one-BLOCKER fix** (cheapest real ship — the split is
verified and only the identity fence is missing), then **906 rebuilt from the snapshot** if the
operator recovers it, else rebuilt from the banner.
