> **STATUS (2026-08-22): SHIPPED as `task894-frame-symbols-digest` (8388c24).**
> All four scope items landed; gates (i), (iii), (iv), (v) PASS. The run-34 forgery — reversed
> `frame_symbols` with `param_fingerprint` intact — now **REFUSES** for a stamped checkpoint and still
> certifies-with-`NOT PROVEN` for an unstamped one, rebuilt through the real CLI by three independent
> agents. `NOT PROVEN: slot order` is deleted from the emitted row only when the digest genuinely
> matched (`is True`), so the claim narrows only where the mechanism binds.
>
> **Gate (ii) shipped on an ACCEPTED PROXY, not its literal text — remainder for the planner.** The
> spec asks for byte-identity vs HEAD on every scored field for all 7 real checkpoints pre-stamp.
> What was run: a direct read proving none of the 7 carries the new key, ONE real HEAD-control-
> worktree A/B on a pre-894 checkpoint (result artifacts diffed to **zero** differences), and an
> exhaustive trace showing the vacuous `None` is inert in every consumer. Two judges independently
> called this a substitution. **If the literal 7-checkpoint A/B is wanted it needs authorizing as a
> scheduled run** — it was not run inside the factory loop.
>
> **Remainders (none blocking, none opened as rows):** (a) the retro-stamp tool writes the digest only
> into the shared `trainer_metadata.json`, never per-leaf — exact today because per-leaf stays absent,
> but a future multi-leaf retro-stamp path must write per-leaf too; (b) stripping the digest from BOTH
> sidecars downgrades a `PROVEN` checkpoint to `NOT PROVEN`-but-`verified: true` with no signal — by
> design, identical to the pre-894 baseline, named because it is the one place absence is deliberately
> silent; (c) `\x1f` could in principle collide a symbol containing the separator against a
> differently-sliced list — no upstream loader produces such a symbol; (d) 886 remainder (2), moving
> the `.sha256` pins somewhere git-tracked, remains out of scope and unstarted.
>
> **Consequence disclosed, not hidden:** `imprint.py` now stamps on every pinned run, so the certified
> sentence changes for every freshly-trained checkpoint. That forced 886's `(order-tamper-a)` and
> `(pin-sync)` gates to flip from certifying to asserting REFUSAL — the hole they demonstrated is
> closed for stamped checkpoints. A regression lens interrogated each edit and confirmed it was forced
> by real behaviour rather than bent to fit the code.
>
> **Pre-existing, NOT caused by this row:** `tools/test_stamp_checkpoint_fingerprints_889.py` carries 2
> standing failures, confirmed identical at HEAD `730c629` in a detached worktree (a local
> `runs/832_e5/ckpt_imprint_full/trainer_metadata.json`, mtime 2026-08-14, already records
> `vector_action` against that test's own precondition). Not a queue row; flagged for the planner.

# TASK_894 — record the slot order INSIDE the checkpoint: the save-time `frame_symbols` digest

**Status**: READY 2026-08-16 (planner; TASK_886's remainder (1), named in its shipped banner and
anticipated by its own Out-of-scope).
**Size: S. Risk: MEDIUM** — touches the checkpoint SAVE path (889's surface) and 886's
certification predicate. Default behaviour for old checkpoints must stay byte-identical.
**Sequenced after TASK_892** — this row adds a fifth conjunct to `_certification_verdict` in
`eval_vector.py`, and 892 adds a flag to the same file; serial, 892 first (the research row).
(878, the original sequencing constraint, shipped in run 35.)

## The hole this closes (measured, not argued)

`param_fingerprint` hashes parameter shape/dtype/bytes — **zero dependence on symbol names**.
Run 34 measured the consequence: reversing `frame_symbols` in a checkpoint's own provenance with
the fingerprint intact still certified `verified: true`, and moved **net@21 −1.7963 → −1.6575**.
886 ships an honest `PROVEN:` / `NOT PROVEN:` split that names this; this row is the structural
close that deletes the `NOT PROVEN` half.

## Scope

1. **Save time (889's surface):** when a checkpoint is saved under a pinned frame, record a
   `frame_symbols_sha256` (digest of the ordered symbol list, canonical separator, documented
   encoding) in the same save-time record 889 owns (`trainer_metadata_<step>.json`), alongside
   the architecture record. No pinned frame at save time ⇒ the key is absent — never null,
   never guessed.
2. **Retro-stamp:** extend `tools/stamp_checkpoint_fingerprints.py` to stamp the digest into the
   7 surviving load-bearing checkpoints' records — derived from each checkpoint's own tracked
   frame-provenance document, with the derivation named in the tool's output. `--apply` remains
   the operator's (the freeze discipline from D39 stands for this tool's new write too: dry-run
   default, planner authorization to apply).
3. **Certification (886's predicate):** `_certification_verdict` gains a fifth conjunct — the
   provenance document's `frame_symbols` order must digest-match the checkpoint's own recorded
   digest. A mismatch REFUSES; an absent recorded digest keeps today's behaviour exactly
   (`verified: true` earnable with the `NOT PROVEN` sentence intact) so every pre-894 checkpoint
   grades byte-identical to HEAD.
4. **The emission sentence:** when the fifth conjunct holds, `NOT PROVEN: slot order` is deleted
   from the emitted row — the claim is only narrowed when the mechanism actually binds.

## Acceptance (gates)

- (i) The reversed-`frame_symbols` forgery from run 34 (fingerprint intact) now REFUSES for a
  stamped checkpoint, and certifies-with-`NOT PROVEN` for an unstamped one.
- (ii) Byte-identity vs HEAD on every scored field for all 7 real checkpoints pre-stamp.
- (iii) The retro-stamp runs its documented bare invocation (run-33/34 rule: real CLI, no
  monkeypatched internals) in dry-run, and refuses a checkpoint whose provenance digest does not
  match what it would stamp.
- (iv) A test that removes the fifth conjunct goes red at the REAL drop site (run-34's standing
  lesson: a guard whose removal keeps the suite green is not a guard).
- (v) Regression tests for (i)–(iv), mutations independently rebuilt.

## Out of scope

Moving the `.sha256` pins somewhere git-tracked (886 remainder (2) — a separate planner call).
Any change to what `param_fingerprint` covers. Re-grading historical LEDGER rows.
