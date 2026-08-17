# TASK_881I — manifest ingestion, and ONLY manifest ingestion

**Status**: READY 2026-08-17 (planner; D45 re-cut, fifth application of the capacity-cut template
— D33/D36/D38/D39 all shipped on the attempt after their cut).
**Size: S. Risk: MEDIUM** — this is the layer every 881 cycle died in, isolated.
**Sequenced before TASK_881D; disjoint from every other Ready row** (`tools/ckpt_archive.py`
ingestion surface only).
**Resume from tag `stash-881-halted-d45`** — but take ONLY the ingestion-relevant parts; the
protection model and deleter in that stash are 881D's.

## The rule this row enforces on itself

**It may PARSE, CLASSIFY and REFUSE — it deletes nothing.** The moment this row touches a
deletion path it has re-welded the two layers that killed 881 three times. If a finding lands in
the deleter, it is 881D's, and this row waits.

## Scope

1. **One function owns both sides of the manifest.** Parsing and SPEC-block stripping are one
   pass returning `(entries, spans)`: the strip can only ever remove exactly what the parse
   consumed. A line that is neither consumed nor refusable is a refusal, never a silent skip.
2. **A malformed line WITHDRAWS coverage** — the run dir it names is treated as unprotected-and-
   unscoreable (fail-safe direction), never reported-and-ignored.
3. **A second SPEC block is a refusal**, not a silent drop.
4. **A dangling manifest entry refuses at NON-ZERO exit** (D45 planner note 3: today it refuses
   at exit 0, so a cron caller cannot see that an operator's protection line is non-functional).
5. **The three `experiments/828_stage3_train/PROTECTED.txt` entries that currently resolve
   MISSING** (scratch disk unmounted) stay loudly named in every report; when the mount returns
   they must be corrected or deleted — the tool's output must make that an operator-visible item,
   not a footnote (D45 planner note 2).

## Acceptance (gates)

- (i) For every malformed-input class in the halted work's corpus (symlinked manifest FILE,
  symlinked DIRECTORY incl. `--experiments-dir` pointed AT the symlink, torn `scores.jsonl`,
  missing/symlinked/directory `--protect-file`): the same refusal behaviour as the verified
  halted work, reproduced by rebuilding each input independently.
- (ii) Parse-then-strip round-trip: stripping `spans` from the input yields a document that
  re-parses to zero entries; stripping anything MORE is impossible by construction (the strip
  consumes `spans`, not text).
- (iii) Dangling entry → refusal, exit ≠ 0, entry named.
- (iv) Every guard goes red when removed at its real drop site; mutations independently rebuilt;
  no test stubs the mechanism under test; the documented CLI invocation is run, not imported.

## Out of scope

The protection model, the deleter, `plan()`, `root_resolution()`, the TTL re-scan — all 881D.
Any write or delete anywhere.
