# TASK_889 — bind a checkpoint to the ARCHITECTURE it was saved under

**Status**: READY 2026-08-12 (planner; created by the D39 ruling out of TASK_888's halt).
**Size: S/M. Risk: HIGH in consequence** — this row owns every write into a persisted artifact.
**Sequenced AFTER TASK_888R, BEFORE TASK_886.**

> **NAMING (2026-08-12, operator correction).** This operation is called a **retro-stamp**, never a
> "backfill". In this repo *backfill* already means TASK_858's append-only INGEST of historical minute bars,
> and *back-fill* in the data sense is a FORBIDDEN look-ahead — the standing rule is forward-fill only, a stale
> price is valid while nothing trades. This operation touches **no price data at all**: it computes a hash over
> parameters already on disk and writes it into `trainer_metadata.json`. Naming ambiguity has now killed four
> consecutive cycles on this lane; it does not get to kill a fifth.

Surface: `mcts_trader/jax_agent/trainer.py` (save-time metadata + `load_checkpoint`) ·
the shared checkpoint-IO module 888R ships · `tools/stamp_checkpoint_fingerprints.py` ·
`experiments/857_vhat_distill/distill.py` · `config/real_trade_config.py`.

## Why this row exists, and why it is separate from 888R

TASK_888 welded two layers. The PATH-IDENTITY + REFUSAL layer survived 8 lens-runs across 3 rounds
and is being shipped as 888R. **Every cycle-3 survivor was in a layer the row never named:
architecture binding between the target model and the checkpoint.** This row is that layer, and
nothing else.

The capacity line, which is what D39 cut on: **888R may only resolve, refuse and describe — it
writes into no checkpoint, no provenance document and no `runs/` artifact. This row owns every
persisted write.** The retro-stamp is untrustworthy *because* it builds a target from an architecture
it never records; that is the same defect as B1, so the writer and the thing that makes writing
safe belong in one row.

## The defect

`load_checkpoint` reads `trainer_metadata.json`, and at `trainer.py:1854` it already *detects* the
prediction-head mismatch — then downgrades it to a stdout `print` and continues:

```
if not ckpt_pred and self.model.use_prediction_head:
    print("   ⚠️  WARNING: Checkpoint has NO prediction head, but model has prediction enabled!")
    print("      Prediction weights will be RANDOM (not from checkpoint)")
```

Reproduced by the run-32 orchestrator with `config/real_trade_config.py` **exactly as shipped**:
`USE_PREDICTION_HEAD=True` against every checkpoint's recorded `use_prediction_head: false` gives
**no refusal**, an evidence fingerprint `594538e970c9…` against the checkpoint's own recorded
`83b8c96126fa…`, and **52 of 253 leaves left at random init** (`price_predictor` 49,
`latent_to_market` 2, `action_embedding` 1). The contradicting value is inside the file the loader
opened three lines earlier. **This is D37's exact shape, re-created by the row built to close it** —
a mismatch that is detected, printed, and then not acted on.

Two more instances of the same root:

- **The retro-stamp tool hardcodes `vector_action=True`.** A scalar checkpoint is then fingerprinted over a
  target carrying 2 extra leaves at the tool's own `PRNGKey(0)` init, and that wrong value is
  written into the document TASK_886 certifies against. Worse, a recorded-vs-recomputed
  DISAGREEMENT — precisely the "a checkpoint changed underneath you" signal the fingerprint exists
  to produce — is silently **overwritten** instead of refused.
- **`distill.py` hand-builds its target** via `imprint.build_model_and_trainer` and drops 51 leaves
  on a prediction-head warm start.

## The ruled answer to the design question (D39, 2026-08-12)

**The config is wrong; the checkpoints are right. REFUSE the mismatch — do not normalize it.**
Verified: `config/real_trade_config.py:55` was committed as *legacy working-tree state* by `6ea7fe7`
(2026-07-06) and last authored dojo-era; its only consumer, `real_trade_jax.py`, self-declares
BROKEN/LEGACY and cannot be constructed; the model default is `False` (`muzero_model.py:358`); the
active line ASSERTS the prediction loss OFF as "the 97%-of-loss trap" (`train_jax_fast.py:371`); and
all 7 surviving checkpoints record `false` as genuine save-time values. **Zero artifacts and zero
live paths say `True`.** Refusing costs the program nothing; normalizing means accepting 52/253
random leaves in silence.

## Scope

1. **Record the architecture at SAVE time.** `vector_action` and `use_prediction_head` (plus
   whatever else determines leaf structure) go into `trainer_metadata.json` alongside the existing
   `h_cycles`/`l_cycles`/`state_dim`/`physics_only`. Today `vector_action` is recorded **nowhere**,
   which is why no existing fingerprint can be validated.
2. **Record the parameter fingerprint at SAVE time**, in the same document, computed over the target
   the checkpoint was actually saved from.
3. **`load_checkpoint_into` REFUSES on disagreement.** Recorded architecture ≠ target architecture,
   or recorded fingerprint ≠ loaded fingerprint ⇒ hard failure naming both values and the resolved
   path. No `print`-and-continue path may remain at `trainer.py:1854-1859`. Deliberate mismatch
   stays available only behind an explicit flag.
4. **Fix the retro-stamp tool.** Build the target from the checkpoint's OWN recorded architecture, never a
   hardcode. A recorded-vs-recomputed disagreement **REFUSES and writes nothing**. `--apply` stays
   planner-gated. Note that applying it to the 4 TRACKED provenance documents moves EMITTED
   ARTIFACTS, not just the two hand-kept `.sha256` pins, via `eval_vector.py:819`'s
   `provenance_meta_block` — so the blast radius is every artifact reading that frame.
5. **Fix `distill.py`** to build its target through the shared path instead of hand-rolling it.
6. **Flip the fossil**: `config/real_trade_config.py` `USE_PREDICTION_HEAD = False`, with the reason
   in a comment. The `True`/`false` mismatch then becomes a CONSTRUCTED case in the regression test,
   not a landmine in a shipped config.

## Acceptance (gates)

- (i) A save writes `vector_action`, `use_prediction_head` and `param_fingerprint`; loading that
  checkpoint into a matching target succeeds and into a mismatched target REFUSES, naming both.
- (ii) The specific halt case is a test: prediction-head-enabled target vs `use_prediction_head:
  false` checkpoint REFUSES. Constructed in the test — do not rely on the shipped config, which
  item 6 flips.
- (iii) A checkpoint whose payload is altered after save produces a fingerprint disagreement and
  REFUSES; the disagreement is never overwritten.
- (iv) The retro-stamp run over the 7 surviving checkpoints (`runs/861_ref/ckpt_ref_s{0,1,2}`,
  `runs/861_imprint/ckpt_s{0,1,2}`, `runs/832_e5/ckpt_imprint_full`) is idempotent, refuses on
  disagreement, and is a **no-op without `--apply`**. **The 7 already carry a `param_fingerprint`
  written by run 32's cycle-1 bare `--apply` and it is UNTRUSTED** — no code in the tree produced
  it, and none of the 7 records the architecture it was computed against. Recompute; do not trust.
  Values on record in D39 if a comparison is wanted.
- (v) `distill.py` warm-starts with zero dropped leaves, proven by leaf count.
- (vi) Byte-identity vs HEAD on every scored field wherever the checkpoint loads correctly.
- (vii) Suites stay green: selftest, 883R, 882, 861 smoke, `scorecard_smoke`, 863, 865, 874, and
  888R's own suites.

## Out of scope

Path identity, the resolver, the hang guard, refusal echoes — **all TASK_888R**, which ships first.
TASK_886's certification contract. Changing what a correctly-loaded run measures.

## Note for whoever builds this

Every side effect this row writes lands in artifacts the clean-tree gate cannot see — `runs/` and
the smoke scratch dirs are gitignored. Run 32 left 14 stamped `frame_provenance*.json` documents in
smoke scratch and 7 stamped `trainer_metadata.json` files under `runs/`; the code was stashed, the
files were not, and the result was a suite failure attributed to the wrong row. **`git stash`
restores code; it does not un-write files.** If this row writes, it records what it wrote.
