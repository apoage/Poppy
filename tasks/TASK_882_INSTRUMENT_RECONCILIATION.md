# TASK_882 — Two "argmax replay" instruments disagree by 1.6 net@21 on the same weights

> **STATUS (2026-08-08): SHIPPED as `task882-instrument-reconciliation`.** Mechanism confirmed
> and reproduced; ruling deliberately NARROW. 2 cycles (implement=sonnet, fix=opus,
> panel=sonnet), 3-lens panel + a fresh correctness re-verify: cycle 1 returned 1 BLOCKER
> (the ruling overclaimed) + 1 in-scope MAJOR (the regression test tested the wrong path),
> both closed; re-verify returned **0 BLOCKER / 0 MAJOR** and judged the re-scoped ruling
> defensible. selftest 12/12; suite 4/4, mutations independently rebuilt and red.
>
> **What is settled:** a missing `--panel-frame-from-checkpoint` silently routes eval_vector
> onto a coverage-ranked default universe; the encoder is positionally universe-locked, so the
> model gets a scrambled slot→asset mapping (63/64 positions differ, 24 symbols absent).
> First divergence at **decision 0**. The unpinned **−0.6145 is withdrawn**, as is every
> unpinned eval_vector number. Sizing (`position_size_pct`) ELIMINATED with a working control.
>
> **What is deliberately NOT claimed:** imprint.py is **not** vindicated in general, and
> **+0.8394 is NOT restored**. `+0.9914` is **CORROBORATED, not restored** — by the same pinned
> instrument reading 12/12 positive on the clean-trained seeds. **The OLD checkpoint's
> disagreement is UNRESOLVED**, and this row does **not** close `861/ref-oos-rebase`; its
> four-candidate list stands, "frame pin/source" included.
>
> **PLANNER ACTION — the decisive measurement is pre-registered and NOT run** (§6): eval-only,
> `ckpt_ref_s{0,1,2}` pinned over the continuous 2021-01-01..2025-10-04 vs imprint.py's own
> sanity number, bands fixed in advance. **Blocker to scheduling it:**
> `runs/861_ref/frame_provenance_ref_s{0,1,2}.json` are cited by shipped artifacts but are
> **absent locally** — recover them first.
>
> **Determinism hazard for the whole program:** the same probe reads net@21 **−4.9678** on GPU
> vs **−4.9876** on CPU — 0.4%, enough to break this suite's own 1e-4 tolerance. Production
> paths self-guard `JAX_PLATFORMS`, but an in-process import of `eval_vector` bypasses that.
>
> Recommended follow-up rows are listed at the end of the RESOLUTION — chiefly that
> **`eval_vector.py` fails OPEN on universe identity** while `imprint.py` hard-refuses.
> Full detail in the CHANGELOG entry.

**Status**: READY 2026-08-07 (planner; from the 879 rung-3 control leg). **Blocks every
cross-harness comparison in the program until it lands.**
**Size: S/M. Risk: low to run, HIGH in consequence** (it decides which of two load-bearing
reference numbers is real). Surface: `experiments/832_linear_imprint/imprint.py` sanity replay ·
`experiments/835_stage4_eval/eval_vector.py --operator policy-argmax` · whatever the 862
scorecard actually runs.

## The measurement (LEDGER `882/instrument-disagreement`)

Same checkpoint `runs/832_e5/ckpt_imprint_full`, same bar size, same cost tier, both claiming to
replay per-asset argmax:

| instrument | window | net@21 | final PV | book |
|---|---|---|---|---|
| `imprint.py` sanity replay | 2021-01-01..2025-09-01 | **+0.8394** | **299,547** (3× gain) | 28.7L / 32.0S ≈ 61 of 64 held |
| `imprint.py` sanity replay (861 ref s0) | 2021-01-01..2025-10-04 | **+0.9914** | **465,545** (4.7× gain) | 25.8L / 30.9S |
| `eval_vector --operator policy-argmax` | 2021-01-01..2025-10-04 | **−0.6145** | **41,919** (58% LOSS) | 5.9% hold |

1673/1673 decisions, no bankruptcy, weight env ON, in-run teacher floor on the eval_vector side.
**The two harnesses produce opposite economic outcomes on identical weights.** The window was
already eliminated as the cause — that is what the third row measures.

## Why this is urgent rather than interesting

`+0.9914` is the program's current in-window reference (LEDGER `861/ref-rebase`), the number that
superseded `+0.839` and the one the imprint lane's "real in-window learning" claim rests on. If
eval_vector is right, that claim is an instrument artifact. If imprint.py is right, then every
eval_vector-based read — including the 875 walk-forward battery and the 879 rung-3 corridor —
is measuring something other than what it claims. **One of those two families is wrong and both
are load-bearing.**

## Scope

1. **Classify every instrument.** Enumerate which LEDGER numbers came from which harness:
   `imprint.py` sanity replay · `eval_vector --operator policy-argmax` · `eval_vector --operator
   search` · the 862 scorecard (which wraps one of these — determine which, and whether
   eval-MCTS is a fourth). Output: a table in this task file, one row per LEDGER row, naming its
   instrument. This is mostly artifact reading, not compute.
2. **Per-decision diff on the fixture** (small, deterministic, no GPU needed at scale): run both
   instruments on the same fixture checkpoint and window, dump per-decision `(action_vector,
   position, cash, pv, fee)`, and find the FIRST decision where they diverge. Report that
   decision's full state from both sides. A single divergent step localises the defect far
   faster than arguing about aggregates.
3. **Test the leading hypothesis explicitly**: the imprint artifact records
   `position_size_pct 0.0078125` (1/128) while eval_vector runs its own `w_unit`/
   `vector_weight_max`. If sizing differs, the same argmax produces different leverage and the
   PV paths separate exactly as observed. Confirm or eliminate — and if it IS sizing, state which
   one matches the trained policy's own semantics, because that is the one that is correct.
4. **Rule which instrument is canonical for which claim**, and write it where it cannot be
   missed: in-sample learning claims, OOS claims, and floor comparisons must each name their
   harness. A number without an instrument is not a result.

## Acceptance (gates)

- (i) The instrument table exists and covers every LEDGER row that carries a net@21 number.
- (ii) The first divergent decision is identified, with both instruments' state printed side by
  side, and the mechanism named.
- (iii) A regression test pins the agreed semantics so the two cannot silently drift apart again.
- (iv) Any LEDGER row whose number is invalidated gets a correction stamp naming the instrument
  (the `857/860` MEASUREMENT VOID stamps are the format).
- (v) A one-line rule added to the LEDGER header: every future row names its instrument.

## Out of scope

Deciding whether the imprint lane has an edge — that question cannot even be asked until this
lands. Changing either harness's behaviour beyond what (iii) needs to pin it.

## RESOLUTION (2026-08-08, implementer; **RE-SCOPED after review** — the first draft of this
section over-claimed, see "What this does NOT establish")

### 1. What IS established: a confirmed, reproduced defect

**Mechanism: a missing/mismatched `--panel-frame-from-checkpoint` pin, not sizing.** Slot k IS
asset k positionally in the encoder (27-feat × n_assets flattened kernel), so grading a
checkpoint on a different universe/order than it trained on feeds every slot's learned weights
another asset's price history.

Evidence, all reproduced on `fixtures/trading_fixture.db` (CPU, hermetic, read-only DB):

- **Frame-matched ⇒ the two instruments agree to float32 ulp.** Same fixture checkpoint, single
  variable changed (the frame path): `imprint.py` sanity net@21 `-4.9876119298103765` /
  final_pv `77559.4140625` vs `eval_vector.py --operator policy-argmax`
  `-4.987614330179929` / `77559.4140625` (PV bit-identical).
- **Frame permuted ⇒ `-0.6440382361256386`.** Same 8 symbols, reversed slot order.
- **Frame pin OMITTED ⇒ `-5.325664210970868`**, a *third* distinct number, and the artifact has
  **no `panel_frame` key at all** (see §6 — this is the literal production failure path, and it
  is now CHECK 4 of the regression test).
- **First divergence is decision 0**, retained and re-verifiable:
  `experiments/882_instrument_reconciliation/dump_first_divergence.py` →
  `first_divergence.{json,txt}`. Counts, with their definitions (the earlier "466/472" was off
  by one on the window boundary and did not name which stream it counted):
  **465/471 on the RAW stream** (the pre-mask per-asset argmax out of
  `policy_decision_actions`) and **471/471 on the EXECUTED stream** (raw plus the TASK_853
  price-validity HOLD and the TASK_868 causal open-gate HOLD). The two counts differ because
  the masks are themselves frame-dependent, so the executed vectors can differ at a decision
  where the raw argmax agrees. Quote the count *with* its definition; they are not
  interchangeable.
- **Real-panel confirmation** (`trading_data.db`, read-only): the checkpoint's recorded frame
  (`runs/832_e5/frame_provenance_imprint.json`, 64 symbols) and today's unpinned
  coverage-ranked universe over the identical window/n_assets disagree on **63/64 positions**
  (24 frame symbols absent from the unpinned set entirely) — the 858 backfill moved the ranking
  after the frame was recorded.

**The defensible claim, in full:** *a missing `--panel-frame-from-checkpoint` is a confirmed,
reproduced defect that invalidates the specific unpinned `−0.6145` read in
`882/instrument-disagreement`, and makes every unpinned `eval_vector` number non-comparable to
any frame-pinned read.* That is the whole of what the mechanism buys.

**Leading hypothesis (position_size_pct 0.0078125 vs eval_vector's own w_unit/cap) ELIMINATED.**
Both real invocations ran `--vector-weight-rebalance`, under which `step_vector` overwrites
cash/positions wholesale and sizes limit legs at `w_unit*PV`
(`jax_trading_env.py:1791-1798,1785,1561,1418`) — `position_size_pct` is dead code there. On the
fixture, an 8× `--position-size-pct` spread (0.0625 → 0.5) left net@21 **bit-identical**
(`-4.987614330179929` both runs); the same spread with `--vector-weight-rebalance` DROPPED moved
net@21 from `-3.824` to immediate bankruptcy, so the probe is sensitive and the ON-case equality
is a real inertness finding.

### 2. What this does NOT establish (the review blocker, engaged)

The first draft of this section claimed that `imprint.py`'s reads are "CORRECT for this
checkpoint's own economics" and that `+0.9914` "IS RESTORED". **Neither is established by the
mechanism above**, because a correctly-frame-pinned `eval_vector` read of the same checkpoint
still disagrees:

LEDGER `861/ref-oos-rebase`, leg **`old_ckpt_clean`**
(`runs/861_ref_scorecard/old_ckpt_clean/windows/*/results_sc862_imprint_full_*.json`) runs the
SAME checkpoint `runs/832_e5/ckpt_imprint_full` through `eval_vector --operator policy-argmax`
with `panel_frame.provenance_path = runs/832_e5/frame_provenance_imprint.json` — the exact frame
this RESOLUTION treats as ground truth — **and** `vector_weight_rebalance: true`, ruling out the
sizing confound. It reads net@21 **−0.8805** (bear22w), **−1.3883** (bull23w), **−0.4102**
(y2024), **−0.1697** (recent): negative in all four regime windows, spanning 2021-12 → 2025-09,
i.e. nearly the whole imprint training window. **The pin was present, so the pin mechanism
cannot explain it.** `861/ref-oos-rebase`'s own 2026-08-05 correction already lists "frame
pin/source" as one of four UNISOLATED candidate causes; this task must engage that, not pass it.

### 3. Reconciling hypothesis A — is the gain 2021-concentrated? (tested; numbers below)

Every regime window starts at/after 2021-12-15 while `imprint.py`'s sanity replay starts
2021-01-01. If essentially all the positive PnL sat in the excluded first ~11.5 months of 2021
(a violent crypto bull), a continuous 4.7-year window could be strongly positive while four
~1-year sub-windows were each negative, with no contradiction.

Tested on retained artifacts only — no new run:
`experiments/882_instrument_reconciliation/period_decomposition.py` →
`period_decomposition.{txt,json}`. Sub-windows restart the book at initial cash, so PV multiples
cannot be chained; the script converts each window's `final_pv` to a CAGR, which is comparable
across window lengths.

| checkpoint | imprint.py continuous | pinned eval_vector in-window sub-windows (CAGR band) | full-window CAGR | inside band? |
|---|---|---|---|---|
| `ckpt_ref_s0` (clean-trained) | +0.9914, PV 465,545, **+38.2%/yr** | +30.7 / +22.3 / +52.8 / +36.0 → **[+22.3, +52.8]** | +38.2%/yr | **YES** |
| `ckpt_ref_s1` | +0.9941, PV 466,883, **+38.3%/yr** | +23.8 / +28.0 / +65.1 / +46.9 → **[+23.8, +65.1]** | +38.3%/yr | **YES** |
| `ckpt_ref_s2` | +0.9533, PV 432,664, **+36.1%/yr** | +14.4 / +22.6 / +52.2 / +41.0 → **[+14.4, +52.2]** | +36.1%/yr | **YES** |
| `ckpt_imprint_full` (OLD) | +0.8394, PV 299,547, **+26.5%/yr** | −26.7 / −29.0 / −15.3 / −9.7 → **[−29.0, −9.7]** | +26.5%/yr | **NO** |

**Result, two halves.**

**(a) For the `+0.9914` anchor itself, 2021-concentration is REFUTED — and the anchor gains an
independent corroboration.** On all three clean-trained reference seeds the frame-pinned
`eval_vector` scorecard reads **POSITIVE on 12/12 in-window regime legs** (+0.59 … +1.59), and
the continuous `imprint.py` growth rate lands *inside* the sub-window band every time. So the
imprint lane's in-window number is **not** carried by one bull year — growth is spread across
2021-2025 — and the two instruments agree in sign and magnitude band whenever checkpoint, pin
and semantics epoch are matched. (This is corroboration, not numeric identity: a window-sliced
Sharpe is not the same statistic as a continuous one. §5 pre-registers the identity check.)

**(b) For the OLD checkpoint the arithmetic does NOT reconcile, and it is untestable from
artifacts.** Its full-window +26.5%/yr sits far outside its sub-window band [−29.0, −9.7]. With
the covered stretch (3.71y) running at the mean sub-window rate (−20.2%/yr ⇒ ×0.433), the
excluded 2021-01-01..2021-12-15 stub would have to deliver **×6.92 in 11.4 months** to reach the
observed ×2.995. Large, not absurd for 2021 crypto on a 64-slot 4×-max-weight long/short book —
but **not testable**: `imprint.py`'s sanity block records only aggregate net@21 and `final_pv`,
with no PV path, no per-period breakdown and no equity curve. So hypothesis A is **neither
confirmed nor refuted for the old checkpoint**.

### 4. Reconciling hypothesis B — the EPOCH confound (better supported; explains the pair)

The `old_ckpt_clean` comparison is not instrument-vs-instrument. It is **2026-07-17 vs
2026-08-03**, across the entire D30/D31 semantics wave and the 858 data backfill:

| | produced | pins/semantics in force |
|---|---|---|
| `runs/832_e5/imprint_full.json` (+0.8394) | **2026-07-17** | no `panel_frame` / no `panel_frame_from_checkpoint` key at all — the run was **UNPINNED**, on the then-current coverage-ranked minute panel. Pre-853 (2026-07-22), pre-858 backfill (2026-07-24), pre-865/D30 fill (2026-07-31), pre-868 alias union + causal open-gate (2026-08-03 01:48), pre-869 UTC windows (2026-08-03 04:55). |
| `old_ckpt_clean` battery (−0.88 … −0.17) | **2026-08-03 23:51** | all of the above in force. |

Two consequences.

- **There is no matched-epoch pair for the old checkpoint, and one cannot be produced without
  retraining.** `imprint.py` has no `--checkpoint` / eval-only path: it trains and sanity-replays
  in-process, so its every number is epoch-stamped at training time and a foreign checkpoint
  cannot be re-graded on that instrument at all. The `+0.8394` therefore cannot be re-measured
  under today's semantics, by construction.
- **`frame_provenance_imprint.json` is a POST-HOC reconstruction, not something the training run
  emitted.** Its own `source` field says so: "835 minute-panel slot order on the training window
  (`results_e5autopsy_imprint`, same ckpt lineage)". Verified here: it is order-exact with that
  autopsy artifact's slot order (64/64). But that autopsy ran on window
  `2021-01-01..2025-12-01`, while the checkpoint trained on `2021-01-01..2025-09-01` — a
  different window, hence potentially a different coverage ranking — and `imprint_full.json`
  retains **no symbol list** to check against (the provenance-emission gap the 861 derivation
  note already flagged). So "the old checkpoint was graded on its true training universe" is
  **assumed, not verified**, and "frame pin/source" stays live on `861/ref-oos-rebase`'s
  four-candidate list.

**This task therefore does NOT close `861/ref-oos-rebase`.** It removes nothing from that row's
candidate list; it only establishes that the pin was *present* on the `old_ckpt_clean` legs, so
whatever sank the old checkpoint is not a *missing* pin.

### 5. RE-SCOPED RULING (final wording)

1. **CONFIRMED.** A missing `--panel-frame-from-checkpoint` is a real, reproduced defect. It
   invalidates the specific unpinned `−0.6145` read in `882/instrument-disagreement`, and every
   unpinned `eval_vector` number is non-comparable to any frame-pinned read. The row's stated
   "instruments disagree" finding is **explained for that pair** and its `−0.6145` leg is
   WITHDRAWN as a measure of the checkpoint's in-window performance.
2. **NOT ESTABLISHED, and explicitly not claimed.** `imprint.py` is *not* vindicated in general.
   `+0.8394` is *not* restored. The pin mechanism says nothing about a leg that carried the pin.
3. **CORROBORATED (new, on evidence).** The `+0.9914` in-window anchor is corroborated by an
   independent, frame-pinned `eval_vector` instrument on the SAME checkpoints in the SAME
   semantics epoch: 12/12 in-window regime legs positive across three seeds, with the continuous
   growth rate inside the sub-window band in all three cases (§3). The `882` row's consequence
   (2) — "'real in-window learning' may itself be an instrument artifact" — rested on a read
   that differed from the anchor in *three* ways at once (different checkpoint, unpinned, and
   across the D31 wave), and is **withdrawn**. The **"+0.9914 in-window, no OOS edge" framing
   STANDS**: the in-window half is now corroborated across two instruments, and the OOS half
   (875) was frame-pinned throughout.
4. **UNRESOLVED, stated plainly.** The old checkpoint's `+0.8394` vs the pinned `−0.88 … −0.17`
   battery is **NOT reconciled**. 2021-concentration is untestable for it; the epoch confound is
   the better-supported account but is not proven; and the frame reconstruction itself is
   unverified. This is exactly `861/ref-oos-rebase`'s standing verdict — "the anchor is NOT
   REPRODUCIBLE under current semantics, cause UNISOLATED" — and this task leaves it there.
5. **UNAFFECTED.** `875/wf-generalization` STANDS UNCHANGED (frame-pinned throughout, per its own
   row text). `879/rung3-distill-remeasure`'s internal argmax-vs-search comparison remains valid
   as a RELATIVE, INTERNAL comparison (both arms share the identical unpinned universe by
   disclosed design, `experiments/879_rerun/SPEC.md` AMENDMENT 2), but its ABSOLUTE numbers
   (search −2.166, argmax −0.564, floor +0.5164) are NOT cross-comparable to any frame-pinned
   read — previously suspected, now confirmed.

### 6. PRE-REGISTERED decisive measurement — SCHEDULED, **DO NOT RUN IN THE FACTORY LOOP**

Real-data eval; per this repo's invariant ("expensive real-data validation is SCHEDULED —
server/overnight one-shot, never polled inside the factory loop"). Pre-registered here so the
planner can schedule it. **Not executed by this task.**

**Leg A — the identity check (decisive, and cheap: eval-only, no training).** Grade the
clean-trained reference checkpoint on the SAME continuous window its `imprint.py` sanity replay
used, correctly pinned, and compare the two numbers directly.

```
python experiments/835_stage4_eval/eval_vector.py \
  --db trading_data.db \
  --start 2021-01-01 --end 2025-10-04 \
  --bar-minutes 1440 --n-assets 64 \
  --panel-frame-from-checkpoint runs/861_ref/frame_provenance_ref_s0.json \
  --panel-frame-source minute \
  --operator policy-argmax \
  --checkpoint runs/861_ref/ckpt_ref_s0 \
  --vector-weight-rebalance --teacher-floor \
  --tag 882decisive_s0 --out-dir runs/882_decisive
```
Repeat for `s1` / `s2` (`frame_provenance_ref_s{1,2}.json`, `ckpt_ref_s{1,2}`). Note: those three
frame files are referenced by the 862 scorecard artifacts but are **not present in the local
`runs/861_ref/`** — they were derived planner-side on the run host; the scheduler must recover
or re-derive them (order-exact against `runs/832_e5/frame_provenance_imprint.json`, which is
what the 861 derivation asserted) before launching.

**Comparison:** `tiers["21"]["sharpe_full"]` and `tiers["21"]["final_pv"]` vs the same
checkpoint's `imprint.py` sanity numbers — s0 **+0.9914 / 465,545**, s1 **+0.9941 / 466,883**,
s2 **+0.9533 / 432,664** (`runs/861_ref/results_s{0,1,2}.json`,
`sanity.per_tier["21"]`).

**Pre-registered bands and what each means:**
- **|Δ net@21| ≤ 0.1 on ≥2/3 seeds** ⇒ the instruments AGREE on the continuous window under a
  matched pin. The `882` row closes fully; `+0.9914` is confirmed cross-instrument, not merely
  corroborated; the "in-window learning is an instrument artifact" line is dead.
- **0.1 < |Δ| ≤ 0.5** ⇒ residual harness difference (float32 reduction order, warm/`start_row`
  convention, teacher-floor masking). Not the 1.6-scale disagreement; localise it, do not
  re-open the ruling.
- **|Δ| > 0.5, or sign flip** ⇒ **a SECOND, pin-independent instrument defect exists.** §5.3 is
  withdrawn, the `+0.9914` anchor goes back to WITHDRAWN, and this task's ruling is reduced to
  §5.1 alone.

**Leg B — the old checkpoint (optional, expensive, low value).** There is no way to re-measure
`ckpt_imprint_full` on the `imprint.py` instrument without retraining, and retraining under
today's semantics just reproduces the 861 reference seeds. Recommend **not** scheduling Leg B;
leave `861/ref-oos-rebase`'s isolation ladder (legs B/C, tags `task867…` / `task868…`) as the
route to that question.

**Guard rails for whoever schedules this:** single GPU on a verified-idle device (or CPU),
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`, under `safe_run`/tmux, never bare; `trading_data.db`
read-only.

### 7. Instrument table (gate i)

Every LEDGER row carrying a net@21 number, by harness. **Read the "Frame-pinned?" column
together with §4: a pin makes a row's universe *identity* explicit, but it does not make the row
comparable across the D30/D31 semantics wave (2026-07-22 → 2026-08-03) or the 858 data backfill
(2026-07-24). Cross-epoch is a second axis, not covered by this column.**

"862 scorecard" = `experiments/835_stage4_eval/scorecard.py`, confirmed a pure subprocess DRIVER
over `eval_vector.py` (`OPERATOR_ARGS = ["--operator", "policy-argmax", ...]`, always with its
own `--panel-frame-from-checkpoint` manifest field) — NOT a distinct instrument. "eval-MCTS"
(`tools/benchmark_easy.py`, `experiments/759_mcts_eval`) is a DIFFERENT, older evaluator for the
retired scalar-action architecture (Foxglove, experiments 000-780) — it never appears in any row
below and is NOT a fourth instrument in this ledger's scope.

| LEDGER row(s) | Instrument | Frame-pinned? |
|---|---|---|
| `830/E0`, `830/E0-REAL` | `corridor.py` (819 linear-book direct replay through `step_vector`) — no trained checkpoint | n/a |
| `831/E1`, `831/E2`, `831/E1-REAL`, `831/E2-REAL`, `831x/A-D` | `search_probe.py` (perfect-model K-sample search, no learned net) | n/a |
| `828s3/preflight`, `828s3/env-wall`, `844/postmortem`, `834/E3-p1{,b,c}`, `836/E4-p1{,b,c}`, `849/leaf` | bespoke diagnostic probes — carry no net@21 (own metrics: R_model/R_true, Spearman ρ, per-candidate calibration) | n/a |
| `835/S4-p1`, `835/S4-p1b`, `835/S4-p1c` | `eval_vector.py --operator search` (pre-832, phase-1 checkpoints) | n/a (pre-832; no checkpoint frame to pin against) |
| `832/E5-gate` | `imprint.py` (`--iters 1` gate-only mode) | n/a (gate-only, no checkpoint grading) |
| `832/E5-imprint`, `832/E5-inversion3` (`+0.839`/`+8.024` policy anchors) | `imprint.py` embedded sanity replay | **NO** — `runs/832_e5/imprint_full.json` carries no `panel_frame` / `panel_frame_from_checkpoint` key; the run was UNPINNED on the 2026-07-17 coverage-ranked minute panel, and `frame_provenance_imprint.json` is a post-hoc reconstruction of it (§4) |
| `861/ref-rebase`, `861/search-teacher-imprint` | `imprint.py` embedded sanity replay | YES — `runs/861_ref/results_s*.json` carry `panel_frame_from_checkpoint: runs/832_e5/frame_provenance_imprint.json`, minute source |
| `832/E5-exit`, `832/E5-erosion`, `832/E5-autopsy`, `832/E5-inversion`, `832/E5-inversion2`, `832/E5-inversion3` (search legs), `832/E5-exit2` | `eval_vector.py --operator search` (various leaf/proposal configs, graded against the imprint policy-argmax anchor cited, not re-measured) | pre-858 backfill — unpinned was contemporaneously correct |
| `832/E5-oos` (thin n=31 preliminary) | `eval_vector.py --operator policy-argmax` (848 operator) | pre-858 backfill |
| `853/numeric-leg` | `eval_vector.py --operator policy-argmax`, SAME continuous window as the imprint sanity (`2021-01-01..2025-09-01`), read −0.344 / PV 46,394 | YES (853's own mechanism) — but **daily-source**, and the row itself attributes the miss to `ohlcv_daily` covering only 27/64 universe assets (33 active slots), so it is NOT a clean pinned counter-example to `+0.8394` |
| `832/E5-oos-wf`, `832/E5-oos-real`, `856/topm-oos`, `856/rule-grid` | `eval_vector.py --operator policy-argmax` (848 operator, 853 fixed-frame panel) | YES — verified via `runs/858_oos/manifest_wf.txt` / `manifest_oos.txt` |
| `850/eqgate`, `852/reconcile` | `eval_vector.py --equivalence-gate` (daily-vs-minute teacher-floor comparison; no trained-checkpoint grading) | n/a |
| `851/fixture-inversion`, `851/e1-anchor` | `eval_vector.py --operator search --leaf-eval true-env` vs the imprint-lifted policy-argmax baseline, fixture only | n/a (fixture, single fixed universe) |
| `857/vhat-corridor`, `860/codistill-corridor` (original) | `eval_vector.py --operator search --leaf-eval vhat-distilled` | **NO** — confirmed via `runs/860_codistill/manifest_860.txt` (0 occurrences of the flag); already superseded/VOIDed for a *different*, additionally-stacked defect (missing `--vector-weight-rebalance` ⇒ bankruptcy cascade). This task does not re-open them. |
| `879/rung3-distill-remeasure` | `eval_vector.py --operator search` and `--operator policy-argmax` (control) | **NO, by disclosed design** — `experiments/879_rerun/SPEC.md` AMENDMENT 2. Internal A/B comparisons stay valid (both arms share the identical uncorrected universe); ABSOLUTE numbers are NOT cross-comparable to any pinned read. |
| `882/instrument-disagreement` | Both, side by side | eval_vector.py leg: **NO** (the defect this task diagnoses); imprint.py leg: **NO** (the 2026-07-17 unpinned run) |
| `875/wf-generalization` | `eval_vector.py --operator policy-argmax` (848 operator, per-fold) | YES — per the row's own text (frame `runs/832_e5/frame_provenance_imprint.json`) |
| `861/ref-oos-rebase` (both legs), `861/ref-oos-maskoff` | "862 scorecard" (`scorecard.py` driving `eval_vector.py --operator policy-argmax`) | YES — `panel_frame.provenance_path` present in every window artifact. **The `old_ckpt_clean` leg is §2's counter-example: pinned, weight-rebalance ON, negative in all four in-window regimes.** |

### 8. Regression test (gate iii)

`experiments/882_instrument_reconciliation/test_instrument_agreement.py` — **four** checks,
fixture-only, hermetic (trains its own tiny checkpoint per run, ~2.5 min CPU):

1. **Frame-matched agreement** between `imprint.py` sanity replay and `eval_vector.py --operator
   policy-argmax` to 1e-4 relative (the pre-existing float32 walk-score tolerance). The
   invariant that was silently violated in production.
2. **Frame-mismatch divergence** (pinned but permuted) — pins the mechanism as current behaviour.
3. **`position_size_pct` inertness under `--vector-weight-rebalance`**, with an OFF-control
   proving the check can detect sizing changes at all.
4. **Pin OMITTED entirely** — the literal incident, which routes through `build_panel`'s
   unpinned coverage-ranked branch (`eval_vector.py:3282-3286`), a *different* code path from
   check 2. Asserts BOTH halves of the silent-drift signature: net@21 does not match the imprint
   reference (`-5.325664210970868` vs `-4.9876`), **and** the artifact carries no `panel_frame`
   key at all — absent, not present-but-wrong.

Mutation results (each mutation run to completion; all red, exit 1):

| # | mutation | observed |
|---|---|---|
| CHECK 1 | point CHECK 1's eval at the mismatched frame | 2/3 checks red (pre-CHECK-4 run) |
| CHECK 4 / M1 | pass the MATCHED pin where the check omits it | red — `diverges=False, panel_frame_absent=False` (both halves bite) |
| CHECK 4 / M2 | pass the MISMATCHED pin where the check omits it | red — `diverges=True, panel_frame_absent=False` (isolates the `panel_frame`-absence half) |
| CHECK 4 / M3 | invert only the divergence oracle | red — `diverges=False, panel_frame_absent=True` (isolates the divergence half) |

### 9. Retained evidence (gate ii)

- `dump_first_divergence.py` + `first_divergence.{json,txt}` — per-decision raw/executed action
  vectors, realized book, `pos_size`, cash and PV for both legs; first divergent decision; both
  legs' full state at decision 0 side by side.
- `period_decomposition.py` + `period_decomposition.{txt,json}` — §3's CAGR decomposition.
- Both are fixture-only / artifact-only, assert the fixture is unchanged, and re-run in minutes.

**Trap for the next reader of `dump_first_divergence.py`:** `eval_vector.py` sets
`JAX_PLATFORMS=cpu` immediately before it imports jax. Any script that imports jax *first* gets
a different backend and different numbers — measured here: net@21 `-4.9678` on the accidental
GPU leg vs `-4.9876` on CPU, a 0.4% gap that would silently break CHECK 1's 1e-4 tolerance. The
script now sets the two env vars at its own top, before any import.

### 10. LEDGER changes (gates iv/v)

An `INSTRUMENT DISCIPLINE` rule in the LEDGER header, and a `[RESOLVED — SCOPED]` stamp on
`882/instrument-disagreement` carrying §5's ruling verbatim in summary, plus the cross-ref note
on `879/rung3-distill-remeasure`.

## Recommended follow-up rows (NOT implemented here — out of this task's scope)

**(A) `eval_vector.py` fails OPEN on universe identity.** It can be run today, unpinned, against
a frame-locked checkpoint and will emit a clean `STAGE-4 EVAL` banner, a paste-ready LEDGER row
and a `results_*.json` — with no warning, no refusal, and no marker. Confirmed:
`print_summary` prints its `FIXED-FRAME (TASK_853)` line only when `panel_frame` is present
(`eval_vector.py:3620-3627`), so an unpinned run prints *nothing* about the frame; and CHECK 4
shows the artifact has no `panel_frame` key at all. Meanwhile `imprint.py` has a hard
`RuntimeError` frame gate, fail-closed in BOTH directions (TASK_863 scope 2,
`imprint.py:1096`). That asymmetry between the two harnesses is the fail-open. The checkpoint
cannot vouch for itself either: `trainer_metadata.json` carries no frame identity, which is also
why the old checkpoint's true training universe is unverifiable (§4).
*Suggested shape:* (i) stamp `panel_frame_pinned: bool` into **every** `eval_vector` artifact
unconditionally, and print it in the banner; (ii) have training write frame identity into
`trainer_metadata.json` so the evaluator can compare and at least WARN; (iii) once (ii) exists,
consider promoting the warning to `eval_vector`'s own fail-closed gate, matching `imprint.py`.
Note: CHECK 4's `panel_frame`-absence assertion is deliberately coupled to today's behaviour and
must be updated as part of (i).

**(B) The `INSTRUMENT DISCIPLINE` LEDGER rule is prose with no mechanical enforcement** — the
same "describes but does not correct" class the repo's standing rule warns about. Gate (v) only
asked for the rule, so this is a follow-up candidate, not a defect in this task. It is largely
subsumed by (A)(i): once every artifact carries `panel_frame_pinned`, the row's instrument line
can be generated rather than remembered.

**(C) `imprint.py` emits no frame provenance for the checkpoint it writes.** Flagged already in
the 861 derivation note; §4 shows the cost — `frame_provenance_imprint.json` had to be
reconstructed from a *different* artifact on a *different* window, and the reconstruction cannot
be validated. Fold into (A)(ii).
