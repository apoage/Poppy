> **STATUS (2026-08-22): SHIPPED as `task881g-manifest-grammar` (4341586). TASK_881D IS UNBLOCKED** —
> its corpus precondition now carries the written grammar. Gates (i)–(iv) PASS. `tools/tests`:
> **529 passed, 1 skipped, 0 failed** (orchestrator-run first-hand).
>
> **Gate (i) FAILED the doc on first execution, and that is the headline.** An isolated agent authored
> 46 cases from the doc alone (never reading the parser); a second ran them through the real CLI:
> **40 AGREE, 3 SURPRISE, 3 UNDERSPECIFIED — every one a DOC defect, the tool self-consistent
> throughout.** The doc was amended six times and each amendment re-verified against actual behaviour.
> What a reader hit at the *first realistic shapes they would type*:
> a **dangling entry exits 2** (the doc's "not malformed / does not poison the file" is about
> intra-file poisoning and never said so); **"a fenced header can only ever ADD protection" is FALSE**
> when the closing fence immediately follows the last item — it withdraws the WHOLE FILE, and one
> blank line before the fence makes it accept; EMPTY-vs-INTERRUPTED was undecided; the item
> micro-grammar (leading indent, no space after the dash, trailing whitespace — **all accepted**, in
> deliberate asymmetry with the strictly column-0 header) was unwritten; whitespace-only-as-blank was
> unwritten; and **entry paths resolve via `realpath` against the process CWD, not the manifest's own
> directory** — unwritten, and it made a verifier's whole first pass falsely refuse.
> **Lesson for 881D's corpus author:** the grammar is now authorable, but it was not before it was
> tested this way. Write the corpus from the doc, and if you find yourself guessing, that is a doc bug.
>
> **Two defects the row itself introduced, both caught and closed:**
> **(BLOCKER)** the nested/duplicate `--experiments-dir` collapse **suppressed an 881I-shipped
> refusal** — it ran on `realpath` before `load_manifests`, erasing the input the symlinked-scan-root
> refusal needs, taking `--experiments-dir experiments experiments_link` from **exit 2 to exit 0**.
> It survived review only because no test had ever passed TWO scan dirs. **(MAJOR)** the collapse's
> claim that "nothing under a collapsed directory goes un-scanned" was false — the walk prunes
> `.git`/`__pycache__`, so a manifest under `experiments/__pycache__/` went **1 protect → 0 protect,
> silently, at exit 0**: the wrongly-WITHDRAWN direction, which in 881D's world makes a
> manifest-protected checkpoint deletable. **(MAJOR)** the new MD032 acceptance emitted a misaligned
> span, so `strip_spans` cut INSIDE the last item line — on the exact shape gate (v) says 881D's
> corpus will use.
>
> **Remainders — both doc-accuracy, both FIXED before shipping rather than carried** (this block IS
> 881D's corpus precondition, and its own header says "if it is not written here, it is not
> accepted"): the delete-path `--protect-file` paragraph was false for two of the three shapes it
> named (a **directory CRASHES** at exit 1, a **live symlink is READ AND HONOURED**, a **fifo HANGS**
> — only missing/dangling is "silently empty at exit 0"), and the collapse's reachability claim is
> proven only against the walk's **name-based** prune list, not past an I/O error (a non-listable
> intermediate does drop a protection — loudly, exit 2, so no verdict changes).
>
> **Still open, named in the doc, not closed here:** the `---`-after-a-gap SPLIT wording; the two
> parsers' line-number drift on exotic Unicode separators (never changes an accept/refuse decision).
> **Inherited from 881S and still live:** MAJOR-A (one shared absolute `--log-glob` reported once per
> co-located score dir) and its three fail-closed count-inflation MINORs.

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
