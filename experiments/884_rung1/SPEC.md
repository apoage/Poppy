# 884/rung1-selfplay-base — the long self-play base-training block

**PRE-REGISTRATION. Staged in the scratchpad because the factory is ARMED** (an untracked
`experiments/884_rung1/` would dirty the tree the ship ritual gates on). Lands verbatim as
`experiments/884_rung1/SPEC.md` at disarm, with the sha256 recorded below so the pre-registration
is auditable rather than merely claimed.

**sha256 of this SPEC at launch time: `2b2657014d96a61cc7f598816d569e2b1ae6b1d251af6432753d51ced26ee9ca`** (self-excluding: this line was
appended after hashing the pre-launch body; the body hash above covers everything the run was
registered against).

Supersedes `scratchpad/RUNG1_DRAFT_SPEC.md` (2026-08-06). Two things changed since that draft:
TASK_876 shipped and is DEPLOYED (its gate), and `882/decisive-legA` settled the instrument
question — so the score read no longer has to be deferred, and the grading command can be written
now instead of improvised later.

---

## 1. Question

LEDGER `828s3/env-wall` (2026-07-09) asked "why did every phase-1 instrument fail or degenerate?"
and answered: `step_vector` charged every buying/shorting asset against ONE cash snapshot with no
aggregate budget, so a near-uniform action vector bankrupts 2/4 envs **at step 0** and exploration
is lethal from the very first uniform-policy step. **That defect is FIXED** (framework epoch E-a,
TASK_840/843/845/846 → the weight-rebalance env). The phase-1 battery was never repeated, and its
checkpoints were deleted from gpu-server on 2026-08-05 precisely because they were trained inside
the defect. There is nothing to re-grade — only to re-run.

This is the **self-play lane**. The `875/wf-generalization` verdict (no OOS edge) closed
hard-target BC on the 832 recipe out-of-sample; it says nothing about this lane, which has never
run on the fixed framework.

## 2. What the frame investigation found — and why it changes the design

The training universe is **fully pinned by the periods file**, not coverage-ranked at runtime:
`mcts_trader/training/setup.py:355-365` passes `specific_symbols = periods[0]['assets']` into the
env, and the period-swap loader at `:417-423` passes the same list. So there is no live ranking to
drift under this run.

**Measured 2026-08-08:** `experiments/821_planner_daily/period_821_train.json`'s 64-symbol list is
**ORDER-EXACT identical** to `runs/832_e5/frame_provenance_imprint.json`'s `frame_symbols`. Two
consequences, both load-bearing:

1. **Rung-1 checkpoints are pinnable with an existing provenance file** — no 882-class hazard, no
   hand-derivation, and the grading command below can be written in advance.
2. **Rung 1 and the imprint lane share a frame**, so their numbers are directly comparable on the
   same 64 slots. That is what makes the design change in §3 worth making.

### STOP condition found while checking this

`mcts_trader/training/setup.py:429` pads a short price matrix with
`pad_indices = [i % n_loaded for i in range(64)]` — which **duplicates columns and silently
re-maps slot → asset**. Positionally universe-locked encoder, no error message: an 882-class
identity scramble that announces nothing. **Preflight (`scratchpad/rung1_preflight_frame.py`)
asserts `prices.shape[1] == 64` and `env.symbols == periods[0]['assets']` order-exact.
If the pad would fire, this run does NOT launch.**
Preflight result: **CLEARED 2026-08-08, on the 884 periods file itself** (not inferred from the
821 window — a subset load is not the same load). `prices.shape = (2495650, 64)`; 64/64 symbols
loaded; `env.symbols` order-exact vs `periods[0]['assets']`; **PAD WOULD FIRE: False**; missing
symbols: none; zero-price cells: **0**; non-finite: **0**. The 821 window was checked first and
also cleared (`(2578638, 64)`), which is what established the frame identity in the first place.

## 3. Design

Command form = `experiments/828_stage3_train/jobs_phase1c.manifest` **with two deliberate
deviations, both stated here rather than discovered later**:

| # | Deviation | Why |
|---|-----------|-----|
| 1 | `trader_env/bin/python` instead of bare `python` | bare `python` on gpu-server is python2; this cost two launches on the 879 legs. An XS runner fix is queued (`run_queue.sh` does not activate the venv). |
| 2 | training window truncated `2025-12-01` → **`2025-10-04`** | So the held-out window is **2025-10-05..2026-03-31** — byte-identical to the OOS window the imprint lane was graded on (`861/ref-oos-rebase`, `875/wf-generalization`) and for which floors already exist. The verbatim window would leave only ~4 unseen months AND make the comparison to the BC lane inexact. The pre-fix phase-1 checkpoints are deleted, so nothing is gained by preserving comparability to them. Costs ~4% of the training span. |

Deviation 2 is what turns this from "does it survive?" into "does self-play generalize where BC
did not?" — which is the program's actual open question. Requires a new periods file,
`experiments/884_rung1/period_884_train.json`, identical to the 821 file except `end_date`
(staged as `scratchpad/period_884_train.json`; asset list copied byte-for-byte, order asserted).

Otherwise verbatim: 3 seeds, `--epochs 2000`, `--vector-action --vector-proposal-scheme
dithered-book --vector-k 16 --vector-hold-bias 0.0 --vector-dither-scale 1.0`, `--reward-hybrid
--reward-attributable`, `--model-size small --sims 32 --batch-size 256 --parallel-envs 4
--steps 256`, `--fee 0 --slippage 0` (cost0 first — fee16-from-scratch is REJECTED, LEDGER
`828s3/preflight`), `--snapshot-every 100`, `--bar-minutes 1440`, **`--vector-weight-rebalance`**
(the E-a fix; it is `store_true` DEFAULT OFF and omitting it is exactly what voided the 879
rung-3 leg — see §6).

### Step 0 — a 40-epoch smoke on ONE GPU, before committing three for days

Nothing on file gives the epoch wall-time for this recipe. The smoke measures it and re-checks the
collapse signature early. **Only if the smoke passes S1–S3 at epoch 40 does the 3-seed battery
launch.** Launching a multi-day 3-GPU job on an unmeasured epoch cost is how you find out on day
three.

## 4. Bars — SURVIVAL first, and they are written before launch

The pre-fix failure was degeneracy, not a weak score, so the primary read is whether the policy
stays alive now that breadth is expressible.

**Measurability was checked against the emitted columns BEFORE launch, and it moved two bars.**
`train_jax_fast.py:888-914` writes `logs/training_<name>.csv` with `pct_hold`, `policy_entropy`,
`agent_return`, `alpha`, `n_sell/n_hold/n_buy`, `pct_buy/pct_sell` — so S2 and S3 read straight off
the training log, but **S1 has no bankruptcy column and S4 has no simultaneous-position column**.
A bar that cannot be measured from anything the run emits is not a bar, so both are re-stated
against artifacts that do exist. Caught here rather than at grading time.

- **(S1) No bankruptcy.** Training-log proxy: no epoch with `agent_return <= -0.99`. Confirmed on
  the final snapshot by the pinned eval's `bankrupt_at == -1` (the same field that exposed the
  857/860 corridor runs as dead at step 27). Direct test of E-a; the pre-fix cascade hit at step 0.
- **(S2) Hold share does not converge**: `pct_hold` < 95% at the final logged epoch on >=2/3 seeds
  (pre-fix: 94.7% -> 99.97% by epoch 40). Straight from the CSV.
- **(S3) Policy entropy does not collapse**: `policy_entropy` > 0.05 at the final logged epoch
  (pre-fix arms landed at 0.005-0.05). Straight from the CSV.
- **(S4) Breadth is actually used**: `avg_long + avg_short > 10` on the final-snapshot eval. The
  CSV's `n_buy`/`n_sell` count ACTIONS, not simultaneous positions, so this bar is read from the
  eval artifact instead — the same `avg_long`/`avg_short` fields that read 25.8/30.9 on the
  imprint reference. The pre-fix cap was ~10 positions by construction, so exceeding it shows the
  wall is *gone*, not merely *survivable*.

**Only if S1–S3 hold does any score read matter.** Then, and pre-registered now:

- **In-window**: net@21 median over 3 seeds vs the on-window teacher floor, ≥2/3 seeds above floor
  to count. Reference points, reported never as gates (D17): imprint in-window **+0.9914**
  (instrument-independent as of `882/decisive-legA`) and the search ceiling **+8.118**
  (clean-measured, non-deployable).
- **OOS**: net@21 median over 3 seeds on **2025-10-05..2026-03-31** vs the same-window teacher
  floor. This is the read that matters, and it is registered NOW so it cannot be quietly skipped
  if the in-window number is pretty. The comparison it earns: the BC lane cleared its floor
  **1/5** folds out-of-sample.

Grading command (both windows), pinned — written now, not improvised:

```
trader_env/bin/python experiments/835_stage4_eval/eval_vector.py \
  --db trading_data.db --start <W0> --end <W1> \
  --bar-minutes 1440 --n-assets 64 \
  --panel-frame-from-checkpoint runs/832_e5/frame_provenance_imprint.json \
  --panel-frame-source minute \
  --operator policy-argmax \
  --checkpoint runs/884_rung1/<ckpt> \
  --vector-weight-rebalance --teacher-floor \
  --tag r884_<leg> --out-dir runs/884_rung1_eval
```

The pin is mandatory and is the whole lesson of `882/decisive-legA`: unpinned, `eval_vector`
silently grades on today's coverage-ranked universe and the number is meaningless (63/64 slots
differ, first divergence at decision 0). `eval_vector` still **fails open** on this — TASK_883.

## 5. Checkpoint economics (the operator's rule: unscored weights are garbage)

2000 epochs × `--snapshot-every 100` = 20 snapshots/seed ≈ 1.8 GB/seed, **5.4 GB for 3 seeds**;
gpu-server has 162 GB free. TASK_876 is SHIPPED and DEPLOYED (`60f77f7`), so every save appends a
`scores.jsonl` row unconditionally.

Worth stating plainly because it is the one piece of good news D35 leaves standing: this run writes
**`snapshot_ep<N>`** directories, and `snapshot_ep<N>` is the **one** naming scheme that
round-trips through the shipped GC reader. The flax `checkpoint_<step>` dirs the training path also
writes do NOT (D35) — they read UNSCORED forever until TASK_881's path-first reader lands. 876's
`flax_keep = 3` bounds those to three per run, so the unreadable set stays small. **No deletion
tooling is run against this run's output** — the `--apply` freeze on `tools/ckpt_archive.py`
STANDS, and D35 rewrote 880's gate.

## 6. Framework epoch — the doctrine check, run BEFORE launch

Per the transferable finding from `879/rung3-distill-remeasure`: **a verbatim re-run does not
deliver "the fixed framework" when a fix is FLAG-GATED.** E-b/E-c/E-d are unconditional semantics
and a verbatim command picks them up; **E-a is a flag** (`--vector-weight-rebalance`, `store_true`,
default OFF), and omitting it is what made all seven 879 rung-3 jobs bankrupt while faithfully
reproducing the recorded command.

- `--vector-weight-rebalance` — **PRESENT** on every line below. Asserted, not assumed.
- Deployed tree at launch: recorded in the manifest header at launch time.
- Epochs in force: **E-a..E-d**. **E-e (TASK_877, XBT fold) and E-f (TASK_878, failed-fill /
  coinflip execution) are still QUEUED**, and E-f changes execution semantics.

**Stated up front, and it is a real limitation:** a rung-1 result is itself PROVISIONAL under the
very epoch rule this program established. That is acceptable for the SURVIVAL read (S1–S4 are
structural, not semantics-sensitive) and **not** acceptable for the score read, which must be
re-graded after E-f rather than quoted as final.

## 7. What this can and cannot show

It tests the self-play lane, untouched by the 875 verdict. But note the adjacency: if rung 1
survives and scores, the natural next step is phase 2 warm-started from the imprint — and **the
imprint has real in-window learning and no demonstrated OOS edge**. So a rung-1 pass must be
followed by an OOS read of its OWN policy before anyone warm-starts anything from anything. That
is why §4 registers the OOS read now rather than leaving it to be proposed after a good in-window
number exists.
