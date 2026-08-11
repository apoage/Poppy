# TASK_888 — one checkpoint resolver, and a loader that refuses

> **STATUS (2026-08-12): HALTED on the 3-cycle rule (factory run 32) — nothing shipped, code preserved in
> `git stash` ("TASK_888 … HALTED on the 3-cycle rule (D39)"). Decision seeded: D39 OPEN.**
>
> **The row welded two layers and only one ever failed — the third time this lane has produced that finding
> (D33, D36, D38).** The **PATH-IDENTITY + REFUSAL** layer was attacked by 8 lens-runs across 3 rounds,
> including two final `opus` panels, and held every time: ONE resolution function feeds both guard and loader
> (**9 honest spellings** → same leaf and same fingerprint; **3 escapes** refused by both), the loader REFUSES
> via object identity, the hang guard covers the parameter payload *and* the run-dir sidecars, load evidence
> is emitted, refusal echoes are clipped 1.6M→<4k with the path still named, byte-identity vs a HEAD worktree
> holds on every SCORED field across 2 real checkpoints, and the 861 legs `i_canon_835` + `results_pin` were
> restored to green. 105 checks across 6 new suites; selftest 12/12, 883R, 882, scorecard_smoke, 863 13/13,
> 865 30/30, 874 8/8.
>
> **Worse than this file recorded, and fixed:** `real_trade_jax.py` discarded `trainer.load_checkpoint`'s
> return value, and `load_checkpoint` never assigns `self.state` — so the live path inferred with fresh-init
> `PRNGKey(0)` weights on **every** path, not merely a bad one.
>
> **WHAT HALTED IT — one unscoped layer: ARCHITECTURE BINDING between target model and checkpoint.**
> Reproduced first-hand with `config/real_trade_config.py` **exactly as shipped**: `USE_PREDICTION_HEAD=True`
> against every checkpoint's recorded `use_prediction_head: false` gives **NO REFUSAL**, an evidence
> fingerprint of `594538e970c9…` against the checkpoint's own recorded `83b8c96126fa…`, and **52 of 253
> leaves left at random init**. The contradicting value sits in the `trainer_metadata.json` the loader opened
> three lines earlier and that this task populated; `load_checkpoint_into` never compares it, and the
> mismatch is detected at `trainer.py:1908` then downgraded to a stdout `print` while the success goes to the
> log file — **D37's shape, re-created by the row built to close it.** Also: the backfill hardcodes
> `vector_action=True` (wrong fingerprint for scalar checkpoints, written into the document 886 certifies
> against) and silently overwrites a disagreeing recorded fingerprint instead of refusing; `distill.py` still
> hand-builds its target. **All three share one root and one fix** — record the architecture at SAVE time and
> REFUSE on recorded-vs-loaded fingerprint disagreement.
>
> **This file's consequence ranking is factually wrong and D39 corrects it:** `real_trade_jax.py`
> self-declares BROKEN/LEGACY (2026-07-05) and depends on `RealPortfolioManager`, absent from the tree — the
> live-trading class **cannot be instantiated**. 888 still ranks first, on `train_jax_fast.py:544`.
>
> **Found, not caused by this row:** `861/smoke_e2e.py` has been RED since TASK_877 shipped (run 31) on the
> single leaf `/provenance/alias_union_rule`; needs a planner-gated `--promote-baselines`.

**Status**: READY 2026-08-11 (planner; created by the D38 ruling out of TASK_886's halt).
**Size: S/M. Risk: HIGHEST CONSEQUENCE IN THE QUEUE** — one of the four sites is live trading.
**RANKED FIRST, ahead of TASK_878.**
Surface: `experiments/835_stage4_eval/eval_vector.py` · `train_jax_fast.py:544` ·
`experiments/857_vhat_distill/distill.py:632` · `real_trade_jax.py:145` · the shared resolver.

## Why this is one row and not two

Two defects were reported separately. They are the same defect.

- **Path identity** (D38, what killed 886 cycle 3): a `realpath` guard against an `abspath` loader.
  A `--checkpoint` that crosses a symlink and then `..` has the guard bless directory **A** while
  the loader opens directory **B** — proven hang, exit 125, and `check_checkpoint_arch` reading the
  wrong checkpoint's metadata. **This repo ships exactly those symlinks** (`jax_ckpt_fast`,
  `jax_ckpt_easy` → `/mnt/scratch`).
- **The production fail-open** (D37, wider than the eval harness): `train_jax_fast.py:544`,
  `distill.py:632` and `real_trade_jax.py:145` all print **"loaded successfully"** on a bad path
  and continue on **random-init weights**.

One is "two functions disagree about which directory this path names". The other is "the loader
continues instead of refusing". Both are fixed by the same thing, and fixing them in separate rows
would be the seventh instance of the boundary error this planner has now made six times.

## Consequence order — why this outranks everything queued

1. **`real_trade_jax.py:145`.** A live-trading process that prints "loaded successfully" and then
   trades **random-init weights**. The dry-run-only invariant is a policy, not a mechanism, and a
   policy does not stop a typo. Nothing else in the queue has this consequence class.
2. **`train_jax_fast.py:544`.** A multi-hour resume that silently discards its checkpoint and
   starts fresh — days of GPU, and the failure is invisible in the training curve because a fresh
   run looks like a fresh run. This is the lane the 884/885/887 batteries run in.
3. `distill.py:632`, then the eval harness (which D37/886 already covers).

## Scope

1. **ONE resolution function**, used by the guard *and* by the loader, at every site. Not two
   functions that agree today — one function. Whatever semantics it picks (realpath is the correct
   choice given the shipped symlinks), the guard and the loader are the same code path, so they
   cannot diverge again.
2. **The loader REFUSES.** A checkpoint that cannot be found or cannot be loaded is a hard failure
   at every one of the four sites, naming the resolved path. Never "starting fresh", never
   "loaded successfully". Deliberate fresh-init stays available but must be **explicitly
   requested**, never a fallback.
3. **Load evidence, computed at LOAD time, always** (D38): resolved path, step, and a fingerprint
   of the loaded parameters, recorded in whatever artifact the site emits. This is evidence
   independent of provenance — it makes two runs of the "same" checkpoint comparable and detects a
   checkpoint changing underneath you.
4. **Emit the fingerprint at SAVE time** into provenance, so future checkpoints are certifiable by
   TASK_886's contract.
5. **Backfill what survives.** A one-shot script that computes and records fingerprints for
   existing checkpoints. **Measured cost: 7 locally** — `runs/861_ref/ckpt_ref_s{0,1,2}`,
   `runs/861_imprint/ckpt_s{0,1,2}`, `runs/832_e5/ckpt_imprint_full` — every load-bearing one, plus
   whatever gpu-server still holds. Checkpoints that no longer exist stay uncertifiable, which
   costs nothing: a checkpoint you do not have cannot be graded.

**Reuse, do not re-derive:** 886's load detector is already correct and verified — object identity
(flax returns its `target` unchanged on a miss), 0/204 leaves aliased across 4 real checkpoints,
15 real checkpoints loading with zero false refusals. Take it from the stash; do not rebuild it.
Cycle 1's numeric param-equality detector is the known-wrong version — it false-positives on an
honestly-saved fresh-init checkpoint and broke `861/smoke_e2e` and `835/scorecard_smoke`.

## Acceptance (gates)

- (i) A `--checkpoint` crossing a symlink then `..` resolves identically in guard and loader —
  proven by a test that constructs exactly that path against the shipped symlink layout.
- (ii) All four sites REFUSE on a nonexistent and on an unloadable checkpoint, naming the resolved
  path. Proven per-site, not by inspection.
- (iii) No site can reach fresh-init weights without an explicit flag; the mutation that removes
  the refusal is rebuilt independently and goes red.
- (iv) Every emitted artifact carries resolved path + step + parameter fingerprint; same checkpoint
  twice ⇒ same fingerprint, different checkpoint ⇒ different.
- (v) Backfill runs over the surviving checkpoints and is idempotent.
- (vi) Byte-identity vs HEAD on every path where the checkpoint loads correctly.
- (vii) The suites 886 already exercises stay green: selftest, 883R, 882, 861 smoke,
  `scorecard_smoke`.

## Out of scope

TASK_886's certification contract — it ships separately and is already verified. Changing what a
correctly-loaded run measures.
