# TASK_886 — bind the universe claim to what was LOADED and what was RETURNED

> **STATUS (2026-08-11): HALTED on the 3-cycle rule (factory run 31) — nothing shipped, code preserved in
> `git stash` ("TASK_886 … HALTED on the 3-cycle rule (D38)"). Decision seeded: D38 OPEN.**
>
> **The row welded two concerns and only one ever failed.** (A) the CERTIFICATION CONTRACT — built, and
> verified harder than anything this program has shipped: `verified: true` binds to the LOADED parameter
> fingerprint AND the provenance's own recorded checkpoint path AND the realized panel AND `n_padded == 0`;
> **two independent opus fail-closed panels could not forge it**, both cycle-1 reproductions now REFUSE with
> no artifact, and the predicate is proven non-vacuous. Also built: D37's load detector via OBJECT IDENTITY
> (0/204 leaves aliased across 4 real checkpoints; 15 real checkpoints, zero false refusals), hole-1
> resolution (all three spellings → identical `universe_pin`), the mechanism-typed record, gate 0c, a named
> `n_assets` refusal (17/204 leaves) replacing a bare `ScopeParamShapeError`, and one coherent `UNIVERSE …`
> sentence in `ledger_row`. Byte-identity proven on 12 A/B pairs; all six suites exit 0.
>
> **(B) hardening the checkpoint LOAD PATH (D37's folded-in item 0) is where all three cycles died**, each
> fix containing the next defect: (c1) numeric param equality as the load detector false-positived on an
> honestly-saved fresh-init checkpoint and broke `861/smoke_e2e` + `835/scorecard_smoke`; (c2) the hang guard
> stat'd 5 named files while the loader opened the payload — three strace-proven infinite hangs; (c3)
> **`realpath` guard vs `abspath` loader** — a `--checkpoint` crossing a symlink then `..` has the guard bless
> directory A while `trainer.load_checkpoint` opens directory B (proven hang exit 125; arch check reading the
> wrong checkpoint's metadata). This repo ships exactly those symlinks. That is this row's own thesis, one
> layer down: a guard verdict about checkpoint A stamped on a run that graded checkpoint B.
>
> **Load-bearing consequence of the verified contract: `verified: true` is UNREACHABLE for every checkpoint
> that exists today** — no emitter writes a `param_fingerprint` into provenance. Fail-closed and consistent
> with D36/D37, but it means the positive claim stays unearnable until an emitter row lands.
>
> **Found and NOT fixed (out of surface, seeded in D38):** the D37 fail-open also lives in
> `train_jax_fast.py:544`, `experiments/857_vhat_distill/distill.py:632` and **`real_trade_jax.py:145`
> (live)** — a multi-hour resume can silently discard its checkpoint; and `_is_fixture_db` keys the LEDGER
> caveat on a PATH, so a byte-identical fixture COPY ships "Scheduled-run numbers" — strictly worse than HEAD.

**Status**: READY 2026-08-09 (planner; created by the D36 ruling out of TASK_883's halt).
**Size: M. Risk: HIGH in consequence** — this row decides when the program is allowed to say a
number was measured on a known universe. Surface: `experiments/835_stage4_eval/eval_vector.py`
(checkpoint resolution + the panel return path + artifact emission).
**Sequenced AFTER TASK_883R.** 883R makes the tool unable to lie; this row makes it able to
certify.

## Thesis (D36 option (a))

**Argv is not evidence.** Both fail-open holes that halted TASK_883 have one root: the guard
reasons about the command line — the `--checkpoint` *string*, the `--symbols` *list* — and never
about the checkpoint that was actually loaded or the panel `build_panel` actually returned.

- **Hole 1**: provenance is located as a path-SIBLING of the `--checkpoint` string, so pointing at
  the orbax step subdir (`…/snapshot_ep2000`) makes the guard report the checkpoint
  provenance-less — and a **fully reversed** pin then grades to exit 0. Both spellings appear in
  committed manifests, and the score is identical either way, so nothing else flags it.
- **Hole 2**: `explicit-symbols` stamps its label from argv. A DB-absent symbol is silently dropped
  (`WARNING:root: Symbol … not found`, buried in absl noise), every later slot shifts, and the
  artifact read `padded: true`, `universe_pinned: true`, **6 of 8 slots wrong**, net@21 −3.953,
  paste-ready.

883R already refuses on hole 2's *mechanism* (realized panel ≠ requested). This row closes the
remaining gap: making a POSITIVE claim that is bound to reality rather than to intent.

## Scope

**0. (D37, RULED 2026-08-09 — do this FIRST.) Refuse a checkpoint that cannot be LOADED, and record
the evidence that it was.** At HEAD a nonexistent `--checkpoint` dir grades to **exit 0 on
random-init weights with a paste-ready LEDGER row** (`check_checkpoint_arch` only WARNs;
`load_checkpoint_into` "starts fresh"). This is the degenerate case of this row's own thesis and it
is ordered first because it is the cheapest and highest-severity item here.

Refusing is only half. **The artifact must record load evidence**: the RESOLVED checkpoint path,
the step, and a fingerprint of the loaded parameters. The planner audit that produced this ruling
found 367 results JSONs carrying a `checkpoint` field and **not one** carrying any load/step/
param/hash key — so the 48 whose path is absent today cannot be classified (checkpoints are
legitimately deleted after grading), and **no historical number in this program can be exonerated
of D37**. Unfalsifiable is worse than common. Without a fingerprint we fix the bug and remain
unable to answer this question about any *future* number either.

*One number is already clean and it is the load-bearing one:* `+0.9914` is exonerated by
`882/decisive-legA` — two independently-written harnesses agreeing to **2.1e-4** on the same
weights, which random-init cannot do.

Same family, folded in: `make_ledger_row` keys the fixture caveat on `tag.startswith("fixture")`,
so a custom `--tag` on fixture data ships "Scheduled-run numbers" wording. A label derived from
argv rather than from what happened — which is this row's whole subject.


1. **Resolve the checkpoint, then look for provenance.** Locate provenance relative to the
   RESOLVED checkpoint directory — walking up from an orbax step subdir to its run dir — not
   relative to the argv string. Every spelling of the same checkpoint must reach the same verdict.
   Prove it with a test that grades the same checkpoint under both committed spellings and asserts
   identical `universe_pin` output.
2. **Certify against the realized panel.** The claim is a statement about what `build_panel`
   returned: the ordered symbol list, the active/inactive mask, and the pad state. Anything the
   panel did that the request did not ask for invalidates the claim.
3. **Replace the boolean with a mechanism-typed record** (D36 rider). `universe_pinned: true|false`
   conflates universe IDENTITY with PANEL CONSTRUCTION — measured: the same checkpoint on the same
   8-symbol order reads **−1.796** via pin-file and **−1.866** via `--symbols`, both stamped
   `true`, the difference being the frozen-cell mask. Emit instead:
   `universe_pin: {mechanism: "checkpoint-provenance"|"explicit-symbols"|"periods-file"|"none",
   verified: bool, realized_matches_requested: bool, n_slots, n_padded, provenance_path, sha256}`.
4. **Only now may `verified: true` exist**, and only when (1)+(2) both hold. 883R ships with the
   positive claim structurally absent; this row is what re-introduces it, earned.
5. **`ledger_row` carries the mechanism inline** — not just the JSON. It is the field humans paste,
   and the 882 complaint was precisely a paste-ready row with no warning.

## Comparability — the program call D36 answered

**`--periods-file`-pinned runs and checkpoint-provenance-pinned runs are NOT cross-comparable.**
Comparability holds WITHIN a mechanism. The artifact must make that visible so a reader does not
have to know it; the Δ0.070 above is the measured price of pretending otherwise. This row does not
make the mechanisms agree — it makes the difference legible.

## Acceptance (gates)

- (0a) A nonexistent / unloadable `--checkpoint` REFUSES before any panel is built, naming the path.
- (0b) Every emitted artifact carries resolved checkpoint path + step + parameter fingerprint, and two
  runs of the same checkpoint produce the same fingerprint while a different checkpoint produces a different
  one (proven, not asserted).
- (0c) The fixture caveat in `ledger_row` is keyed on the DB actually opened, not on the `--tag` string.
- (i) Both committed `--checkpoint` spellings of the same checkpoint produce identical
  `universe_pin` output, and a reversed pin REFUSES under both.
- (ii) A `--symbols` list containing a DB-absent symbol REFUSES, naming the dropped symbol and the
  first shifted slot — never `verified: true`.
- (iii) `universe_pin.mechanism` is present on every emitted artifact, including `none`.
- (iv) `ledger_row` names the mechanism inline in every case.
- (v) Byte-identity of every scored block vs HEAD when the pin is supplied and valid.
- (vi) Regression tests for (i)–(v) against the fixture, with the mutations independently rebuilt.

## Out of scope

Making the two pinning mechanisms produce the same number — they don't, and this row's job is to
say so, not to fix it. Re-grading historical LEDGER rows.
