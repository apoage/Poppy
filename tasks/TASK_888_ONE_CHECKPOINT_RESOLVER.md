# TASK_888R — one checkpoint resolver, and a loader that refuses

> **STATUS (2026-08-14): SHIPPED as `task888r-one-checkpoint-resolver`** — factory run 33, 2 cycles
> (implement `sonnet` → escalated fix `opus` → fresh `sonnet` fail-closed re-verify). One resolution
> function (`mcts_trader/jax_agent/checkpoint_io.py`) feeds BOTH guard and loader at all five sites; the
> loader REFUSES via object identity instead of silently starting fresh; a bounded hang guard covers the
> parameter payload AND the run-dir sidecars; load evidence lands in the run's own artifact only.
>
> **THE CAPACITY LINE HELD, proven mechanically rather than asserted:** `checkpoint_io.py` contains no
> `open(...,'w')`, no `json.dump`, no `os.replace`, no `.write()`. Gate (v) verified against an
> orchestrator-held baseline captured before any agent ran and re-checked after two full batteries and
> seven agents — the 7 `runs/*/trainer_metadata.json` hash byte-identical in mtime and size, the 4 tracked
> `runs/*frame_provenance*.json` are zero-diff vs HEAD, no provenance document carries `param_fingerprint`.
> **Everything that writes into a persisted artifact was SUBTRACTED to TASK_889** — save-time fingerprint,
> save-time architecture record, the retro-stamp tool, `distill.py`'s hand-built target, the
> `config/real_trade_config.py` fossil — and a re-verifying judge diffed the tree against the stash
> specifically to confirm none of it leaked back. **`stash@{0}` is LEFT INTACT: TASK_889 resumes from it.**
>
> **1 BLOCKER, found by the fail-closed lens at the site this row ranks FIRST — and it was this row's own
> defect class one line upstream of the module that closes it.** `train_jax_fast.py` gated the only call
> into `checkpoint_io` on TRUTHINESS, so `--load-checkpoint ""` (an unset shell variable in the
> orchestration scripts that build this flag) collapsed to "flag not given" and training proceeded on fresh
> weights in silence — verbatim the scenario this row exists to prevent. Fixed by gating on whether the flag
> was GIVEN, via a named predicate so the test exercises the expression `main()` runs; regression test shown
> RED then GREEN. Four other lenses, one explicitly hunting fail-opens, cleared every escape hatch inside
> `checkpoint_io` and were right to — the hole was in the conditional guarding entry to it. Also closed:
> uniform clip discipline, the dead `trainer.restore_checkpoint` alias DELETED (a second unguarded entry
> into the loader), a read-permission gap that surfaced as a raw orbax `ValueError` instead of a
> `CheckpointRefusalError`, and a docstring claiming a flax symmetry the code did not have.
>
> **ACCEPTANCE, stated at its true strength.** (i)/(iii) proven — 9 honest spellings resolve to one leaf and
> one fingerprint through a REAL site, 3 escapes refused by BOTH guard and loader, the object-identity
> detector mutation-tested by reproducing the historical D37 defect on a scratch copy. (iv) MEASURED, not
> inferred — `83b8c96126fac…` agrees across nine spellings, the checkpoint's own recorded value, and a real
> training epoch's printed evidence; a second checkpoint differs and matches its own record; all 7 real
> checkpoints load with zero over-refusal after the readability guard. (v)/(vi) as above; scored fields
> bit-identical. **(ii) is proven END-TO-END AT 4 SITES AND BY DIRECT IMPORT AT `distill.py`, NOT at five —
> and that limit is stated rather than rounded up.**
>
> **REMAINDER → planner, a MAJOR this row deliberately did NOT fix.**
> `experiments/857_vhat_distill/distill.py`'s CLI has been DEAD since TASK_874: `run()` unpacks 3 values
> from `imprint.load_training_panel`, which returns 5, so `python distill.py …` dies with `ValueError: too
> many values to unpack (expected 3)` before any refusal is reachable. Pre-existing at HEAD, reproduced by
> three independent lenses. Repairing it decides whether distill consumes 868's `open_gate` and the alias
> union — which changes what a distill run measures, out of this row's scope. **The 857/860 LEDGER numbers
> predate the break and are unaffected.** Other remainders: the 7 orphaned `param_fingerprint` values are now
> CORROBORATED by two independent re-derivations (889 must still recompute);
> `tools/parallel_epoch_training.py:45` remains a 5th unguarded load site; `distill.py::resolve_warm_start`
> still uses truthiness for `--warm-start` (verified NOT a fail-open).
>
> **`861/smoke_e2e.py` is 11 legs PASS / `vii_baseline_pins` FAIL on the single leaf
> `/provenance/alias_union_rule`** — TASK_877's deliberate provenance-description rewrite, pre-existing since
> run 31, `results` pin IDENTICAL, bit-identical before and after this row's fix pass. Clearing it needs ONE
> operator authorization: `python experiments/861_search_teacher_imprint/smoke_e2e.py --promote-baselines`.

> **RE-CUT BY D39 (2026-08-12) after the run-32 halt. This row is now 888R: the VERIFIED half.**
> The original row welded PATH IDENTITY + REFUSAL to ARCHITECTURE BINDING, and every cycle-3
> survivor was in the latter. **The architecture layer has moved IN FULL to
> [`tasks/TASK_889_CHECKPOINT_ARCHITECTURE_BINDING.md`](TASK_889_CHECKPOINT_ARCHITECTURE_BINDING.md)
> — together with every persisted write: the save-time fingerprint, the save-time architecture
> record, the retro-stamp tool, and `distill.py`.** Do NOT build any of it here.
>
> **THE CAPACITY LINE, which is the whole point of the re-cut: this row may only RESOLVE, REFUSE and
> DESCRIBE. It writes into no checkpoint, no provenance document and no `runs/` artifact.** A row
> that cannot write cannot corrupt what TASK_886 later certifies against. This is the same cut that
> shipped 876 on attempt 3 and 883R on the first try.
>
> **RESUME FROM `git stash`, do not rebuild** — find it by MESSAGE ("TASK_888 … HALTED on the
> 3-cycle rule (D39)"), never by index. What is in there held under 8 lens-runs across 3 rounds,
> including two final `opus` panels: ONE resolution function (`checkpoint_io.resolve_checkpoint_dirs`,
> realpath) feeding BOTH guard and loader — **9 honest spellings** (run dir, orbax leaf, trailing
> slash, `..`, relative, symlinked run dir, symlink+`..`, and the shipped `jax_ckpt_fast/../..`
> divergence) resolve to the SAME leaf and the SAME fingerprint, **3 escapes** (absolute symlink out,
> prefix-confusion sibling, relative `../` symlink out) refused by BOTH; the loader REFUSES via
> OBJECT IDENTITY (never numeric equality — flax returns its `target` unchanged on a miss); a bounded
> recursive hang guard covering the parameter payload **and** the run-dir sidecars; load evidence
> emitted at load time; refusal echoes clipped 1.6M → <4k with the path still named; byte-identity vs
> a HEAD worktree on every SCORED field across 2 real checkpoints; the 861 legs `i_canon_835` and
> `results_pin` restored to green. 105 checks across 6 new suites.
>
> **Subtract from the stash before shipping:** the architecture-binding work, the save-time
> fingerprint write, the provenance emitter, and `tools/stamp_checkpoint_fingerprints.py`. All of
> it belongs to 889.
>
> **Ranking corrected by D39.** The original file ranked this row on `real_trade_jax.py:145`. That
> was wrong: the file self-declares BROKEN/LEGACY (2026-07-05) and depends on `RealPortfolioManager`,
> absent from the tree — the live-trading consequence class **cannot be instantiated**. This row is
> still RANK 1, on `train_jax_fast.py:544`.
>
> **`861/smoke_e2e.py`'s red IS 877's, as run 32 reported — but a second failure was masking it.**
> Run 32's own stashed emitter stamped `param_fingerprint` into 14 gitignored smoke-scratch
> `frame_provenance*.json` documents, and HEAD's `imprint.py` does not emit that field, so 874's
> refusal fired and the suite died at the canon-832 leg, never reaching
> `collection_pin_fresh_vs_committed`. Planner quarantined the 14 and re-ran: 11 legs PASS,
> `vii_baseline_pins` fails on exactly that leaf, awaiting a one-command operator re-base. See gate
> (v) below — that is the gate this history bought.

**Status**: READY (re-cut) 2026-08-12 — planner, D39. **RANKED FIRST.**
**Size: S/M. Risk: HIGH in consequence.**
Surface: the shared resolver module · `train_jax_fast.py:544` ·
`experiments/857_vhat_distill/distill.py:632` · `experiments/835_stage4_eval/eval_vector.py` ·
`real_trade_jax.py:145`.

## Why this is one row and not two

Two defects were reported separately. They are the same defect.

- **Path identity** (D38, what killed 886 cycle 3): a `realpath` guard against an `abspath` loader.
  A `--checkpoint` that crosses a symlink and then `..` has the guard bless directory **A** while
  the loader opens directory **B** — proven hang, exit 125, and `check_checkpoint_arch` reading the
  wrong checkpoint's metadata. **This repo ships exactly those symlinks** (`jax_ckpt_fast`,
  `jax_ckpt_easy` → `/mnt/scratch`).
- **The production fail-open** (D37): `train_jax_fast.py:544`, `distill.py:632` and
  `real_trade_jax.py:145` all print **"loaded successfully"** on a bad path and continue on
  **random-init weights**.

One is "two functions disagree about which directory this path names". The other is "the loader
continues instead of refusing". Both are fixed by one resolution function and one refusal.

## Consequence order (corrected by D39)

1. **`train_jax_fast.py:544`.** A multi-hour resume that silently discards its checkpoint and starts
   fresh — days of GPU, and the failure is invisible in the training curve because a fresh run looks
   like a fresh run. This is the lane the 884/885/887 batteries run in.
2. `distill.py:632`, then the eval harness (which 886 already covers).
3. `real_trade_jax.py:145` — still fixed here, because the fix is free once the resolver exists and
   the file is worse than D38 recorded (below), but it ranks last: the module cannot be constructed.

## Scope

1. **ONE resolution function**, used by the guard *and* by the loader, at every site. Not two
   functions that agree today — one function. Realpath semantics, given the shipped symlinks.
2. **The loader REFUSES.** A checkpoint that cannot be found or cannot be loaded is a hard failure
   at every site, naming the resolved path. Never "starting fresh", never "loaded successfully".
   Deliberate fresh-init stays available but must be **explicitly requested**, never a fallback.
   Detect the miss by **object identity**, not numeric parameter equality — cycle 1's numeric
   version is the known-wrong one that false-positived on an honestly-saved fresh-init checkpoint
   and broke `861/smoke_e2e` and `835/scorecard_smoke`.
3. **A bounded hang guard** covering the parameter payload **and** the run-dir sidecars — a FIFO at
   `trainer_metadata.json` / `popart_state.npz` / `d/<hash>` / `ocdbt.process_0` / `array_metadatas`
   must be named, not hung on. (Cycle 2 stat'd 5 named files while the loader opened the payload:
   three strace-proven infinite hangs.)
4. **Load evidence into the RUN's own artifact** — resolved path, step, and a fingerprint of the
   loaded parameters, recorded in whatever result the site emits. **Into the run's output only.**
   Writing a fingerprint into a checkpoint or a provenance document is TASK_889.
5. **`real_trade_jax.py` is worse than D38 recorded** and the fix belongs here: it DISCARDS
   `trainer.load_checkpoint`'s return value, and `load_checkpoint` never assigns `self.state` — so
   the live path inferred with fresh-init `PRNGKey(0)` weights on **every** path, not merely a bad
   one. Assign the returned state.
6. **Clip refusal echoes.** A refusal that prints 1.6M characters is not a refusal anyone reads; the
   resolved path must still be named.

## Acceptance (gates)

- (i) A `--checkpoint` crossing a symlink then `..` resolves identically in guard and loader —
  proven against the shipped symlink layout, with the 9 honest spellings and 3 escapes from the
  stash kept as the test corpus.
- (ii) All four sites REFUSE on a nonexistent and on an unloadable checkpoint, naming the resolved
  path. Proven per-site, not by inspection.
- (iii) No site can reach fresh-init weights without an explicit flag; the mutation that removes the
  refusal is rebuilt independently and goes red.
- (iv) Every emitted RUN artifact carries resolved path + step + parameter fingerprint; same
  checkpoint twice ⇒ same fingerprint, different checkpoint ⇒ different.
- (v) **Nothing under `runs/`, no `trainer_metadata.json` and no `frame_provenance*.json` is written
  or modified by this row.** Prove it: record mtimes across those trees before and after the suites
  and assert none moved. This gate exists because run 32's cycle-1 `--apply` silently rewrote 7
  checkpoint metadata documents and 14 smoke-scratch provenance documents, all gitignored, and the
  clean-tree gate saw nothing — then the resulting suite failure was attributed to a shipped row.
- (vi) Byte-identity vs HEAD on every path where the checkpoint loads correctly.
- (vii) Suites green: selftest, 883R, 882, 861 smoke, `scorecard_smoke`, 863, 865, 874.

## Out of scope

**Everything that writes into a persisted artifact — TASK_889.** Save-time architecture and
fingerprint records, the refusing recorded-vs-loaded comparison, the retro-stamp tool, `distill.py`'s
hand-built target, and the `config/real_trade_config.py` fossil. If a BLOCKER here lands in that
surface, it belongs to 889 and this row waits.

TASK_886's certification contract. Changing what a correctly-loaded run measures.
