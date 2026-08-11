# Factory run 32 handoff — TASK_888 HALTED on the 3-cycle rule (D39)

**Date:** 2026-08-11/12 · **Harness:** Claude Code (Opus 5, 1M context) · Model policy per
`.dark-factory/config.json`: implement on `sonnet`, fix passes escalated to `opus`, final-cycle
panel on `opus`.

## Outcome

| Row | Result |
|---|---|
| TASK_888 — one checkpoint resolver | **HALTED** on the 3-cycle rule. Code in `git stash@{0}`. **D39 OPEN.** |
| TASK_886 / 878 / 881 | Not started. 886 is BLOCKED by 888's halt; 878 is next takeable. |

Twelve subagent runs: 3 implement/fix passes (1 `sonnet`, 2 `opus`) and 8 lens-runs across three
verification rounds (5-lens `sonnet` panel → 3-lens `sonnet` re-verify → 2-lens `opus` final), plus
five long suite runs driven by the orchestrator directly.

Nothing shipped. Nothing committed to code. Nothing pushed.

---

## The finding: the row welded two layers, and only one ever failed — for the third time on this lane

TASK_888 carried (A) **path identity + a refusing loader** and (B) **architecture binding** between
the target model and the checkpoint. (B) was never named in the task file; it arrived implicitly with
the parameter fingerprint.

**Every survivor of the final cycle was in (B). (A) was attacked by 8 lens-runs across 3 rounds and
held every time.**

This is D33 → D36 → D38 → D39: four consecutive rows on the checkpoint/eval lane whose halt was a
DECOMPOSITION finding, not an implementation one. Each previous re-cut shipped on the next attempt.

### (A) — verified, and worth resuming rather than rebuilding

- **ONE resolution function** (`checkpoint_io.resolve_checkpoint_dirs`, realpath) feeds BOTH the guard
  and the loader. D38's c3 divergence is closed: **9 honest spellings** — run dir, orbax leaf, trailing
  slash, `..`, relative, symlinked run dir, symlink+`..`, and the shipped `jax_ckpt_fast/../..`
  divergence — resolve to the SAME leaf and the SAME fingerprint; **3 escapes** (absolute symlink out,
  prefix-confusion sibling, relative `../` symlink out) refused by BOTH guard and loader.
- **The loader REFUSES**, by OBJECT IDENTITY, never numeric equality (flax returns its `target`
  unchanged on a miss). A checkpoint saved from a fresh init still loads — the cycle-1 defect of the
  predecessor row, not repeated.
- **Hang guard extended to the run-dir sidecars.** A FIFO at `trainer_metadata.json`,
  `popart_state.npz`, `d/<hash>`, `ocdbt.process_0/d/<hash>` or `array_metadatas/process_0` is named,
  not hung. The security lens proved the sidecar hole by strace in round 1; round 2 closed it and
  round 3 could not reopen it.
- Load evidence emitted; save-time fingerprint; provenance emitter; backfill with a dry-run default
  and an `--apply` gate; refusal echoes clipped from 1.6M chars to <4k with the path still named.
- **Byte-identity vs a HEAD worktree on every SCORED field**, measured on 2 real checkpoints.
- 105 checks across 6 new suites; selftest 12/12, 883R, 882, scorecard_smoke, 863 13/13, 865 30/30,
  874 8/8 — all green and all re-run by the orchestrator, not taken on report.

**Also fixed here, and worse than D38 recorded:** `real_trade_jax.py` **discarded**
`trainer.load_checkpoint`'s return value, and `load_checkpoint` never assigns `self.state`. The live
path therefore inferred with fresh-init `PRNGKey(0)` weights on **every** path, not merely a bad one.

### (B) — what halted it

Reproduced first-hand by the orchestrator, `config/real_trade_config.py` **exactly as shipped**:

```
SHIPPED CONFIG: USE_VECTOR_ACTION=False  USE_PREDICTION_HEAD=True
ckpt meta:      use_prediction_head=False   recorded fp= 83b8c96126fac493…
⚠️  WARNING: Checkpoint has NO prediction head … Prediction weights will be RANDOM
NO REFUSAL.  evidence fp = 594538e970c91933…   MATCH? = False
total leaves = 253   leaves STILL AT RANDOM INIT = 52
by module: {'price_predictor': 49, 'latent_to_market': 2, 'action_embedding': 1}
```

The contradicting value is inside the `trainer_metadata.json` the loader opened three lines earlier —
and that **this task put there**. `load_checkpoint_into` never compares it. The mismatch *is* detected
at `trainer.py:1908` and then downgraded to a stdout `print` while the success goes through `logging`
to the run's log file. **That is D37's exact shape, re-created by the row built to close it.**

Two more with the same root: the backfill hardcodes `vector_action=True`, so a scalar checkpoint is
fingerprinted over a target carrying two extra leaves at the tool's own `PRNGKey(0)` init — and that
wrong value is written into the document TASK_886 certifies against; and a recorded-vs-recomputed
DISAGREEMENT (precisely the "a checkpoint changed underneath you" signal scope 3 exists to produce) is
silently overwritten instead of refused. `distill.py` still hand-builds its target and drops 51 leaves
on a prediction-head warm start.

**One fix closes all three:** record the architecture (`vector_action`, `use_prediction_head`) at SAVE
time, and make `load_checkpoint_into` REFUSE when the checkpoint's own recorded fingerprint disagrees
with the loaded one.

## Findings that outrun the row

- **D38's ranking basis is factually wrong.** `real_trade_jax.py` self-declares "BROKEN / LEGACY … not
  being resurrected … Do not use this file for live/paper trading" (2026-07-05) and depends on
  `RealPortfolioManager`, which is absent from the tree — the live-trading consequence class **cannot
  currently be instantiated**. 888 still ranks first, but on item 2 (`train_jax_fast.py:544`, the lane
  the 884/885/887 batteries run in), not on live trading.
- **`861/smoke_e2e.py` has been RED since TASK_877 shipped in run 31.** `collection_pin_fresh_vs_committed`
  differs on the single leaf `/provenance/alias_union_rule`: the baseline was promoted at `894ab36`,
  877 shipped at `72c0c87`, and 877 changed the rule text without re-promoting. Measured across two
  independent runs and attributed by three converging routes (git dates, the string's own content, and
  zero occurrences of `alias_union_rule` in this diff). Needs a planner-gated `--promote-baselines`;
  that is not a factory action. A HEAD-control run to settle it is impossible from a worktree — 861's
  own provenance gate correctly refuses because the fixture path string differs.
- **Applying the backfill to the 4 tracked provenance documents moves EMITTED ARTIFACTS**, not just the
  two hand-kept `.sha256` pins the tool warns about: `eval_vector.py:819`'s `provenance_meta_block`
  copies every non-`frame_symbols` key into every artifact that reads that frame.
- **The 7 `runs/*/trainer_metadata.json` carry a `param_fingerprint` written by a bare `--apply` during
  cycle 1** (mtime Aug 11 21:24). They are gitignored, so the clean-tree gate cannot see it — the
  `feedback_gitignored_fixture_invisible` class again. Payloads verified untouched and the values
  independently re-derived correct by two lenses; but with the code stashed the field is now orphaned.
- `tools/parallel_epoch_training.py:45` calls `checkpoints.restore_checkpoint` directly, bypassing every
  guard — a fifth load site.
- `experiments/857_vhat_distill/distill.py:669` unpacks 3 values from `load_training_panel`, which
  returns 5 (`imprint.py` unpacks 5 correctly). Pre-existing since TASK_868/874; it blocks that CLI
  entirely, so gate (ii) at that site is verifiable only by direct import.

## Process notes

- **The panel earned its cost three times over.** Round 1's five lenses produced four BLOCKERs by four
  different routes — and two of them (the multi-leaf run dir) converged independently, which is the
  strongest signal this program has. Round 3's `opus` fail-closed lens found the halt-causing defect
  that three prior rounds and a green 105-check suite had missed.
- **A measured suite failure outranked every gate pass, again.** The first 861 run exited 1 while all
  gates read green; the notification's "exit code 0" was the bash pipeline's status, not the suite's.
  Reading the suite's own exit line is the only safe habit.
- **Attribution was worth the 20 minutes.** Two of the four 861 legs were 888's and two were not; the
  fix pass closed exactly the two that were, verified across two independent runs.
- **The fix pass escalated a scope question instead of deciding it** (whether ~20 lines in `imprint.py`
  beyond the emitter were in scope). That is the behaviour the two-thread contract wants.
- **A read-only reference worktree with `stash@{0}` applied, plus a clean HEAD control worktree, both
  with gitignored `fixtures/` and `runs/` symlinked in** — this made "reuse, do not re-derive" and the
  byte-identity A/B possible without ever touching the live tree. Worth keeping as the standard rig.

## Invariants

`trading_data.db` never opened — size 26245337088, mtime still 2026-07-26 20:47:32. Fixture never
regenerated — sha256 `b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9`, 79310848 B,
re-verified by every agent at the end of its own run and by the orchestrator at the end of this one.
No GPU jobs. No live-trading execution; `real_trade_jax.py` verified as a unit only.

## State at end of run

- Tree clean. Halted 888 work is `stash@{0}`; the 886 halt is now `stash@{1}`, 883 `stash@{2}`.
- `tasks/DECISIONS.md`: **D39 OPEN** with the re-cut recommendation and the blocking operator question.
- `tasks/QUEUE.md`: run-32 note; row 1 (888) HALTED with "do not retake as written"; row 2 (886) marked
  BLOCKED behind it.
- **Ready and takeable without a ruling: TASK_878** (failed-fill semantics), then TASK_881.

## The one question that unblocks the most

`USE_PREDICTION_HEAD=True` in the shipped live config, against `use_prediction_head: false` in every
checkpoint that exists — is the config wrong, or are the checkpoints? The answer decides whether the
architecture-binding row REFUSES that combination or normalizes it, and it is the difference between a
guard that protects the live path and one that refuses every artifact the program owns.
