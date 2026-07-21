# TASK_851 — fixture-leg acceptance artifacts (reproducibility pin)

Committed per coordinator review (correctness-judge/flow-verifier panel, 2026-07-21):
acceptance gate (i)'s headline numbers previously reported only in prose. This directory
pins them to committed result JSONs + the EXACT reproduction recipe, mirroring the
830/831/838/847/849 convention (LEDGER row + committed/pointed-to results JSON).

## Checkpoint (gitignored — rebuild with the command below; verified deterministic 3x
## across independent re-runs, byte-identical downstream Sharpe to 1e-6)

```
source trader_env/bin/activate
python experiments/832_linear_imprint/imprint.py
# -> experiments/832_linear_imprint/ckpt_smoke (fixture-scale BC imprint of the 819
#    order-code book teacher; all-default args: seed=0, iters=80, n_assets=8,
#    bar_minutes=240, window 2025-09-14..2025-12-12, position_size_pct=0.5/8=0.0625)
```

All RNGs in `imprint.py` derive from `--seed` (default 0); no time-based or
hardware-nondeterministic sources were found. Re-run three times independently (twice at a
scratch `--ckpt-out`, once at the canonical path) and got the IDENTICAL sanity-replay number
every time: policy net@21 **-6.778**, teacher floor net@21 **-3.437**, corridor position
**-3.341**.

## Gate (i) — the fixture inversion leg

```
source trader_env/bin/activate

# leg A: policy-argmax replay (the floor reference)
python experiments/835_stage4_eval/eval_vector.py \
  --checkpoint experiments/832_linear_imprint/ckpt_smoke \
  --start 2025-09-14 --end 2025-12-12 --bar-minutes 240 --n-assets 8 \
  --position-size-pct 0.0625 --eval-seed 0 \
  --operator policy-argmax --tag 851_fixture_inversion_policyargmax \
  --out-dir experiments/851_true_env_leaf

# leg B: search with --leaf-eval true-env (default R=3, tail ON)
python experiments/835_stage4_eval/eval_vector.py \
  --checkpoint experiments/832_linear_imprint/ckpt_smoke \
  --start 2025-09-14 --end 2025-12-12 --bar-minutes 240 --n-assets 8 \
  --position-size-pct 0.0625 --eval-seed 0 \
  --operator search --leaf-eval true-env --tag 851_fixture_inversion_trueenv \
  --out-dir experiments/851_true_env_leaf
```

**Result** (committed JSONs in this directory; each self-documents checkpoint/window/panel/
operator — see the `panel`/`operator` blocks):

| leg | net@21 | gross |
|---|---|---|
| policy-argmax replay (floor reference) | **-6.778** | -6.409 |
| search, `--leaf-eval true-env` | **+6.801** | +7.761 |

Preserves-or-beats: BEATS, by a wide margin (directionally mirrors the real-panel E1 result,
-6.92 model-leaf-search → +5.29 true-env-leaf-search, LEDGER `832/E5-autopsy` / `831/E1`).
Reproduced identically (net@21 to 3 decimal places) across 3 independent checkpoint
rebuilds + eval runs on 2026-07-20/21.

**IMPORTANT — a note on a review-time discrepancy.** The correctness-judge/flow-verifier
panel that reviewed this task reported reproducing DIFFERENT numbers on the same nominal
gate (net@21 -18.829 policy-argmax vs -12.621 true-env; direction still held). Root cause
was NOT determinism in this task's own code: this pin (rebuilt from a clean canonical
checkout with the exact commands above) reproduces -6.778 / +6.801 on every attempt, and
`imprint.py`'s checkpoint construction was independently verified deterministic (see
above). The panel's run coincided with a concurrent, unrelated in-flight edit to
`experiments/835_stage4_eval/eval_vector.py` from a different task (TASK_852) sharing the
same working tree — the coordinator independently confirmed this file was mid-edit and
briefly `git checkout --`'d by a reviewer before being restored. The most likely explanation
is that the panel's reproduction ran against a transient, partially-edited state of
`eval_vector.py` rather than either this task's clean diff or true canon. Flagged here so a
future reader isn't puzzled by conflicting numbers in the review record; not something this
task's own code needed to change.

## Gate (ii) — E1-consistency anchor (zero-knowledge net)

Same window/panel; checkpoint = a FRESH-INIT (never-trained) model saved via
`imp.build_model_and_trainer(args, n_assets=8)` + `trainer.save_checkpoint(...)` (the exact
architecture-matching construction `imprint.py` itself uses, just with zero training
iterations — `imprint.py --iters 0` crashes on a metrics-formatting bug unrelated to this
task, so the checkpoint was built directly rather than through the CLI).

| operator | net@21 |
|---|---|
| search, `--leaf-eval model` (canon) | -10.287 |
| search, `--leaf-eval true-env` | **+2.198** |
| in-window teacher floor (819 book, `imprint.py`'s `order_floor_reference`) | -3.437 |

Even a knowledge-free net clears the floor under the true-env leaf (matches 831/E1's own
"even uniform `independent` reaches the floor at K>=16 under the perfect R=3 value")
while the canon model leaf does not — reproducing 831's behavior class now that the shipped
path shares its operator.
