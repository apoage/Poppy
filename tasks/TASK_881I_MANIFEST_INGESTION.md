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
