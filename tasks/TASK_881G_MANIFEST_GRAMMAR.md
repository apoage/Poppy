# TASK_881G — the manifest grammar: written down, then accepted (881D's unblocker)

**Status**: READY 2026-08-20 (planner; seeded by D53 after run 37's finding that 881I's "settled"
was scoped to the REFUSE path and the accepted grammar exists in writing NOWHERE).
**Size: S. Risk: MEDIUM** — same tool (`tools/ckpt_gc.py`), same refusal discipline; this row
WIDENS the accept path, which is the dangerous direction, so every acceptance is mutation-tested.
**Sequenced AFTER TASK_881S** (same file, serial). **BLOCKS TASK_881D** — its corpus is authored
per the grammar this row writes.

## The situation (D53)

0 of 56 real manifest files carries a `protected:`/`covers:` block, so nothing had ever exercised
a manifest that parses — and the accept path shipped with four silent-OK holes (closed in
`bf1f8d5`; the sharpest: an unanchored item regex where `- runs/a\nruns/b` granted ONE protection
at exit 0 while `strip_spans` erased the token the parse never consumed, the identical shape in
PROTECTED.txt refusing because that regex was anchored — two parsers, one grammar). The shipped
state refuses the canonical markdown form (MD032: `protected:` / blank line / list items) — the
first shape a corpus author will type — and corrects only the misleading message. So the grammar
the tool DOES accept is undocumented anywhere an author would look.

## Scope

1. **Write the grammar down.** One doc block (in `tools/ckpt_gc.py`'s module docstring AND
   `tasks/TASK_881D_PROTECTION_DELETER.md`'s corpus precondition): exactly what a
   `PROTECTED.txt` and a SPEC.md `protected:` block accept — header form, item form, comments,
   blank lines, examples of each accepted and each refused shape. If it is not written here, it
   is not accepted.
2. **Accept the canonical MD032 form.** `protected:` followed by a blank line and a markdown
   list. The first shape an author types must not refuse.
3. **One parser, one grammar.** The PROTECTED.txt and SPEC.md ingestion paths share the
   item grammar (the run-37 unanchored-regex class); anchoring is uniform, and a test proves the
   two paths accept and refuse identically on a shared corpus of shapes.
4. **The accept-path remainders from 881I's banner:** a trailing comment on an item; the
   near-header detector firing inside fenced code; a near-miss header withdrawing a whole file
   even beside a valid block; the ingestion NITs (`---` ingesting as item `--`, nested scan dirs
   double-counted, `load_protect_file` turning an indented `#` into an entry,
   `str.splitlines()` line-number drift). Each is either accepted-deliberately or
   refused-loudly, and the grammar doc says which.
5. **The asymmetry (ruled in D53):** an explicitly-passed nonexistent path REFUSES on both
   `--experiments-dir` and `--protect-file`; the exit-0 control
   (`test_absent_scan_root_is_not_a_refusal`) is deleted and replaced.

## Acceptance (gates)

- (i) The grammar doc exists and a fixture corpus authored FROM THE DOC ALONE (by an agent that
  did not read the parser) ingests with zero surprises — the doc is the spec, tested.
- (ii) Every newly-accepted shape has a mutation that turns it into a refusal and is caught;
  every still-refused shape refuses with its distinct message.
- (iii) The two ingestion paths accept/refuse identically on the shared shape corpus.
- (iv) The delete-nothing fence and every 881I-shipped refusal re-proven unchanged; suites green
  twice back-to-back; byte-identity outside the ingestion surface.
- (v) The MD032 corpus shape (the one 881D will author) is in the acceptance fixtures.

## Out of scope

Score-source enumeration (TASK_881S, sequenced before this). The protection model and deleter
(881D). Authoring the real corpus itself (881D scope item 0 — it consumes this row's grammar).
Lifting the `--apply` freeze.
