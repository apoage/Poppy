> **STATUS (2026-08-20): SHIPPED as `task881i-manifest-ingestion`** — factory run 37, 3 cycles
> (implement=sonnet, fix=opus, fix2=opus), medium panel + a fresh fail-closed lens on opus.
> Diff a pure insertion **1430 / 0**; 8 canon functions and `main()`'s post-guard body byte-identical
> to HEAD; suite **162 passed / 1 skipped twice back-to-back**; selftest 12/12; 10 mutation-reverts at
> real drop sites; delete-nothing fence re-proven by contrast; real corpus still **56 manifest files
> read, exit 0**. The `--apply` freeze STANDS — this mode deletes nothing.
> **REMAINDERS → planner, all grammar/spec calls deliberately NOT taken by the factory:** MD032's
> `protected:`/blank/list form is still REFUSED (message corrected only) and is the first shape 881D's
> corpus author will type; a trailing comment on an item is now refused (consequence of anchoring);
> the near-header detector fires inside fenced code; a near-miss header withdraws a whole file even
> beside a valid block. NITs left: `---` ingests as item `--`, nested scan dirs double-count,
> `load_protect_file` turns an indented `#` into an entry, `str.splitlines()` line-number drift.
> **Escalated asymmetry:** an explicitly passed NONEXISTENT `--experiments-dir` exits 0 (deliberate
> tested control, `test_absent_scan_root_is_not_a_refusal`) while an explicitly passed nonexistent
> `--protect-file` REFUSES — same operator error, opposite outcomes.
> **D50 finding, load-bearing for 881D:** "settled after five lens-runs" was scoped to the REFUSE
> path. The ACCEPT path had never been attacked, because 0 of 56 real manifest files carries a block —
> and four pre-existing silent-OK holes were sitting in it. 881D's precondition is therefore worse than
> "unmet": the corpus is empty AND a manifest `PROTECTED.txt` refuses both plausible syntaxes, so the
> grammar it DOES accept is undocumented anywhere an author would look. Write the grammar down first.

# TASK_881I — manifest ingestion, and ONLY manifest ingestion (RE-CUT per D50)

**Status**: READY 2026-08-19 (planner; **D50 re-cut — the D33/D36/D38/D39/D45 template, sixth
application**; every prior application shipped on the attempt after its cut).
**Size: S. Risk: MEDIUM.**
**TOOL CORRECTION (carried from run 36, verified first-hand): the 881 surface is
`tools/ckpt_gc.py`, NOT `tools/ckpt_archive.py`** as this file and D45 previously wrote —
ckpt_archive is TASK_880's remote archive tool under its own freeze.
**Resume from the tree state** — the working tree CARRIES run 36 cycle-3's work; durable copy
`lore/handoff_2026-08-19_factory_run_36_881I_cycle3.patch` (sha256 `f9c6b4f2…`). The
`stash-881-halted-d45` tag is superseded as the resume point.

## What is settled (D50 — five lens-runs could not break it; re-prove, don't rebuild)

The manifest-ingestion surface as cycle 3 left it: one `lstat`-based `classify_entry()` that
never follows a link to decide a shape (one following `stat` types the END of a chain; ELOOP ⇒
dangling); `_walk_scan_dirs` yielding `dirnames + filenames` UN-SPLIT so nothing downstream
inherits a bad bucket; manifest-basename identity tested FIRST over every entry; shape checks
before every `open()` on manifest, score-file and protect-file paths (a FIFO named
`PROTECTED.txt` hung the tool forever before this); ten shapes × both basenames each refusing
with a DELIBERATELY DISTINCT message; the delete-nothing fence (proven by contrast); canon
functions byte-identical to HEAD (924 insertions, 0 deletions); the real tree reaching 56/56
manifest basenames at exit 0; the argparse abbreviation class-closing test (`--c` restored to
`--critical-free-gb`); 86 (+1 skip) / 106 tests; selftest 12/12.

**How to decode "86 (+1 skip) / 106 tests"** (added on the escalated cycle — the number was real
but undecodable without `lore/handoff_2026-08-19_factory_run_36.md:316`, so the reference is
inlined here for the next grader): `tools/tests/test_ckpt_881i_manifest_ingestion.py` ALONE
collects 87 = **86 passed + 1 skipped**. The **106** is that file run TOGETHER with
`tools/tests/test_ckpt_876_score_emission.py` (20 tests, an unrelated TASK_876 suite) →
**106 passed, 1 skipped**. Reproduce either half directly:

```bash
python -m pytest tools/tests/test_ckpt_881i_manifest_ingestion.py -q          # -> 86 passed, 1 skipped
python -m pytest tools/tests/test_ckpt_881i_manifest_ingestion.py \
                tools/tests/test_ckpt_876_score_emission.py -q                # -> 106 passed, 1 skipped
```

Both numbers are the CYCLE-3 baseline; the escalated cycle's four fixes add tests to the 881I
file, so its own count moves up while the 876 suite's 20 stays fixed — as shipped it is
**114 passed, 1 skipped** alone and **134 passed, 1 skipped** for the pair. Separately: the 3 failures
in the wider `tools/tests` directory are PRE-EXISTING and out of scope (identical failure set at
HEAD and on the candidate).

## The rule this row enforces on itself (unchanged)

**It may PARSE, CLASSIFY and REFUSE — it deletes nothing.** The `--apply` freeze (D32/D34)
STANDS.

## Scope of this re-cut

Ship the settled layer. That means: re-run the verified list above against the carried tree
(every refusal independently rebuilt, the fence re-proven, suites green twice back-to-back),
banner it, commit. **Score-source enumeration is explicitly OUT — it is TASK_881S.**

## Accepted residuals (documented in run 36's handoff; not defects of this row)

- `--verify-manifests` on DEFAULT `--roots` refuses on this workstation while scratch disk is
  unmounted (`jax_ckpt_fast` dangles) — the intended loud signal; a cron caller on defaults sees
  exit 2 until the mount returns.
- A `protected:`/`covers:` line in SPEC.md PROSE is now a refusal — a documented foot-gun for
  future SPEC authors, fail-safe in direction.

## Acceptance (gates)

- (i) Every shape in the D50 verified list refuses identically, each malformed input
  independently rebuilt (never taken on report).
- (ii) The delete-nothing fence re-proven by contrast (with `--verify-manifests` all checkpoints
  survive; without it the unprotected one is genuinely deleted).
- (iii) Byte-identity of every non-ingestion code path vs HEAD.
- (iv) Suites green TWICE back-to-back (run-36 rule: a single green no longer counts for rows
  with a concurrency-shaped history).

## Out of scope

Score-source enumeration (TASK_881S). The protection model, the deleter, `plan()`,
`root_resolution()`, the TTL re-scan — all 881D. Authoring the real `PROTECTED.txt` corpus
(881D's precondition — **there is no live manifest corpus at all**: 0 of 56 SPEC.md/PROTECTED.txt
files carries a `protected:`/`covers:` block). Any write or delete anywhere.
