# 2026-07-29 — dark factory run 21 handoff (TASK_863 + TASK_864)

Neutral technical continuity record. Factory/implementer lane. Both queue rows shipped; the run's
most consequential output is a measurement that **blocks** the 861 real-leg chain (D29).

## Shipped

| Tag | Commit | What |
|-----|--------|------|
| `task863-imprint-frame-pin` | `609d6d3` | `--panel-frame-from-checkpoint` on `imprint.py` (D28 opt (a)) |
| `task864-parity-baseline-hygiene` | `6b71e0e` + refresh `086b1c4` | tracked baseline unwritable by default; promotion recipe- and validity-gated; drift reconciled |
| — | `64ecb62` | D29 seeded OPEN + queue groomed |

## D29 — the wall (READ FIRST if picking up the 861 chain)

**D28 opt (a) cannot be satisfied as written.** It ruled "pin the frame on both 861 legs" to remove a
59/64 positional-slot universe drift. But the frame pin only exists on a **daily** loader
(`eval_vector.load_daily_prices_framed` / `create_daily_panel_framed`); no minute-framed counterpart
exists, and `eval_vector`'s frame machinery was out of 863's scope.

Measured read-only (`file:trading_data.db?mode=ro`, using the loader's own coverage definition) over
the leg-1 window `2021-01-01..2025-10-04` against `runs/832_e5/frame_provenance_imprint.json`:

- **`ohlcv_daily` covers 33 / 64 slots. 31 slots have ZERO daily rows** → inactive, forced HOLD for the
  entire run. 0 unknown symbols, so the loader's fail-loud never fires. Median 1517 rows among covered.
- The **858 backfill brought MINUTE to 62/64** for the same frame.
- The `+0.839` bar leg 3 grades against is **minute-panel-native**: the frame provenance's own `source`
  reads "835 minute-panel slot order on the training window", and LEDGER 832/E5-imprint records it with
  a 64-asset book (29L/32S) and 97.5% agreement.

So pinning as ruled swaps one confound for two: **source** (D20 measured |Δ|=0.0555 net@21
coverage-symmetric and ruled cross-source comparisons forbidden — daily admitted for within-panel
grading only) and **breadth 64→33**, first-order here (selftest ST3 puts a 24-asset portfolio at
**2.76×** the mean single-asset Sharpe), direction vs the bar unestablished. That reproduces exactly
the uninterpretable-number outcome D28 existed to prevent.

**Recommendation: teach the pin to the MINUTE path** (queue row A). Alternatives: re-measure +0.839 on
the daily 33-slot panel first (costs a full imprint, and the number still won't line up with +0.898
OOS / the WF quarters / 862's cards), or proceed as-is (not recommended).

**Spec error found and corrected**: D28 and TASK_861 line 179 both state `imprint.py:620` "only WARNs"
on a frame-pinned collection. Shipped 627–631 is a hard `RuntimeError` (verified against the tag).
Scope 2's "keeps the existing WARN" was therefore unimplementable, and adding a WARN would have broken
byte-identical stdout. 863's banner carries the correction; **the 861 pre-registration still leans on
the wrong claim and should be fixed where authored.**

## TASK_863 — what shipped

Ported the 853 mechanism by **calling** `eval_vector._load_frame_provenance` / `load_daily_prices_framed`
rather than re-deriving them (`eval_vector.py` byte-unmodified). The 861 provenance gate is now
fail-closed in **both** directions, and a same-membership **reorder** is refused — the failure no
set-based check can see, since slot k IS asset k.

Legacy parity proven **five independent ways**: two orchestrator re-proofs at matched `_REPO` with
identical `--out`/`--ckpt-out` (stdout byte-identical, 170 leaves / 0 differing); the fixer's mirrored
tree (0-of-759, 0-of-545); the regression judge's **30-leg** legacy battery (zero rc/artifact
divergence); the fresh judge's mirrored-tree control. Smoke **8 → 11 → 13 legs**. selftest 12/12.

**1 MAJOR + 6 MINOR + 3 NIT closed over two fix rounds**, every new guard demonstrated to fail when
disabled:

- **MAJOR (pre-existing code, fixed pin-gated)**: a degenerate pinned run exited **0** with a
  green-looking artifact — null Sharpes beside `all_losses_finite: true` / `policy_ce_decreased: true`,
  no failure marker, and the 861 fit gate passing on it too — whenever `--smoke-assert` was off, which
  is the default for **any db that is not the repo fixture, i.e. the real leg**. Now any null
  Sharpe-valued `per_tier` field marks `sanity.degenerate_replay` with distinct STUDENT/TEACHER
  diagnoses, exits non-zero regardless of `smoke_assert`, and stamps `DEGENERATE_REPLAY.json`. The
  checkpoint is written before the replay exists, so it is **marked, not prevented**. The marker is
  **advisory and unwired** — 862's `scorecard.py` never opens an imprint artifact.
- **Masked-cell dilution**: `teacher_action_agreement` 0.8929 vs **0.8571** active-cells-only (56/224 =
  25% masked on the fixture; ~48% on the real frame). Same dilution hit `policy_action_hist`,
  `avg_long`/`avg_short`, and the 861 fit block whose `modal_action_rate` floor is itself inflated.
  Matters because the reference (+0.839 run's 97.5%) was measured on a fully-covered minute panel.
  Active-only companions + `masked_cell_fraction` + an explicit non-comparability note now emitted.
- **The `panel_source` pin was false AND vacuous** under the pin (found independently by two judges):
  it refused frame-matched pinned pairs while clearing the smoke's own pass leg only because
  `eval_vector` records the flag *default* string rather than the actual source.
- **"flat-zero features" was factually false**: both dead slots carry a constant **0.2** on encoder
  channels 8–11 (896 of 6048 cells). No leak (constant ⇒ no signal), but shipped 853 never claimed it.
  Reworded **and asserted** in the smoke, so a future encoder change leaking signal into a dead slot
  fails loudly.
- **The force-HOLD mask had no regression guard at all**: it is behaviorally **inert** on the
  `--teacher-vectors` path (the collector already masks — both judges confirmed deleting the mask left
  artifacts byte-identical), and the smoke had **zero partial slots**. An off-by-one would have passed
  every shipped leg silently. Now boundary-asserted (`[0 11 0 0 28 0 0 0]`, k=10 HELD / k=11 TRADEABLE).
- 853's historical **partial-slot leak verified NOT reintroduced** — both judges built the case by hand
  (`MONUSD`, ZEC/USD); boundary exact, no fabricated pre-listing price tradeable.

## TASK_864 — the premise was wrong, and the fix flag was the bigger hazard

**The run-21 finding that motivated this task was a misdiagnosis (mine).** "~74 commits stale / 479
differing leaves / not a usable parity oracle" compared the baseline against a **flagless default** run.
The committed artifact is the **weight-ON** deliverable (`--vector-weight-rebalance`), already
documented at `SPEC.md:205-211`. **~461 of ~468 leaves were the flag, not drift** — the finding
overstated the drift by ~2 orders of magnitude in leaf count, and it propagated into the queue note and
the task brief.

Real drift: ~7 fields + one conditional block, **all from `c2f1ba5`/TASK_846**, matching the ratified
**D16** ruling verbatim (`teacher_weight_net21 −4.687 → −3.111`; gate `passed` False→True, which removed
the 10-key `weight_native_regate` block as a *consequence*, not a schema change). Bisection: old bytes
reproduce at `e17dc2d` (846^) weight-ON with 0 differing leaves, new bytes at `c2f1ba5` and HEAD with 0,
and the whole 460-moved + 10-removed delta appears at that one commit. `842`/`843` ruled out (both
predate the file). Zero unattributable fields ⇒ refresh justified under acceptance (iii), committed
separately (`086b1c4`) with the attribution table in its message.

Then the fail-closed judge found the new promote flag was worse than the staleness:

- **BLOCKER**: `--write-parity-baseline` validated *where* output went and *whether* the file was dirty,
  never *which experiment produced it*. The flagless **weight-OFF** default promoted onto the
  **weight-ON** baseline — 462/545 leaves, `vector_weight_rebalance True→False`, exit 0, "PROMOTED" —
  contradicting the module's own `imprint.py:104` invariant, with **no downstream detector** (sole
  consumer is a within-run byte-identity check; `CANON_832_PINS` uses hardcoded pins + explicit
  `--out`). Permanent and silent, and it *looks* freshly reconciled. Now recipe-pinned with expected
  values **re-derived from the module's own argparse defaults** so the check cannot drift; 9 attack
  invocations refuse at argparse time naming the field.
- **MAJOR**: promotion did not require the run to have passed its own validity gate —
  `--iters 1 --no-smoke-assert` installed an untrained artifact (`policy_ce_decreased: false`,
  trajectories truncated 80→1) as the oracle. Now gated pre-run *and* post-run independent of
  `--smoke-assert` (863's precedent). The guard caught its own author's first-draft bug by refusing the
  canon promote before any bytes moved.
- **MINOR**: non-atomic `copyfile` truncating in place + re-reading a shared path → now `.tmp` +
  `os.replace` from `run()`'s returned value.

## Why the 864 drift hid for ~80 commits (the real lesson)

**`CANON_832_PINS` is a weight-OFF oracle guarding a weight-ON baseline.** The 861 smoke's
`i_canon_832` leg pins the *flagless default*; the committed artifact is the *weight-ON* one. So nothing
in the repo pins the weight-ON numbers. The file was not unguarded by accident — it was guarded by an
oracle for a **different configuration**, which is exactly how 846 moved it while every gate stayed
green. → queue row B.

## Ops notes / footguns confirmed this run

- **`is_fixture` compares `abspath(db)` to the repo fixture path**, so running against a fixture COPY
  *or* from a mirrored tree silently switches `batch_size` to 256, `smoke_assert` **OFF**, and the
  recorded `tag`. Cost one confusing parity measurement (a `/tag real-panel` vs `fixture-smoke` diff
  that was purely a mirror's `_REPO`-relative `FIXTURE_DB`). Pass `--batch-size 32 --smoke-assert`
  explicitly when you mean fixture semantics.
- **`smoke_e2e.py --quick` silently DOWNGRADES `861/results_fixture_smoke.json`** — writes falsification
  seed 0 where the committed baseline holds seeds 0–3, leaving a file that still reads complete and
  green. Clobbered once, restored via `git checkout --`. → queue row C.
- **Concurrency**: the 861 smoke was run while the verification panel was live; both touch the same
  tracked baselines. No damage (restored), but it is the TASK_851 contamination pattern — serialize
  orchestrator-side smoke runs against the panel, or point them at a scratch tree.
- **Untracked scratch under `experiments/`** (`_parity863`, `_parity863c`, `_ovfy`, `_fin`) is a live
  hazard: ungitignored, so a `git add -A` would commit a stale duplicate of `imprint.py`. All removed;
  prefer a scratchpad mirror with symlinks.
- Two **session-limit interruptions** hit mid-run. Both times agent work survived because it was
  read-only analysis and the tree was verified clean before resuming. The git gate also correctly blocked
  an agent's `git checkout` while the verdict file was stale (373 min) — verification freshness is a
  precondition.
- `--p` no longer abbreviates unambiguously (`--panel-frame-from-checkpoint` vs `--position-size-pct`).
  Fails closed with a clear argparse message; no repo script uses it. Acceptance (i) violated *by the
  letter* — recorded, not fixed.
- Pre-existing, unchanged: the **UNPINNED** silent-collapse path (null Sharpe, exit 0, no marker) — only
  the pinned path was made fail-loud, to preserve invariant #1. A matched-stride mixed-pin leg dies in
  `trader/strategy.py:150` with a raw broadcast `ValueError` at `--bar-minutes 1440`. Partial slots have
  return channels pinned at the `return_clip ±10` saturation for post-listing decisions whose 60-row
  window still contains pre-listing rows, and those decisions are tradeable and BC-labeled — faithful to
  853, but 863 moves it from inference into gradients.

## State at handoff

Clean tree, **nothing pushed** (`allow_push: false`; review is the operator's). HEAD `086b1c4`. Five tags
`task860`…`task864`. Ready queue **EMPTY**, factory IDLE, one wall: **D29 OPEN**. Four recommended next
rows are listed in `tasks/QUEUE.md` for the planner to spec (A minute-path frame pin — the D29 unblock;
B weight-ON pin; C port the promote gate to 861's baselines; D frame pin for the ExIt continuation).
Verdict for 864 in `.dark-factory/verdicts/current.json`; 863's is in CHANGELOG + the run log.
