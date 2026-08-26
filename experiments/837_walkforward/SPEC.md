# SPEC — TASK_837: walk-forward split harness (the R4 graduation instrument)

**Pathway**: R4 (`docs/pathway.md`); definition-of-working = UNSEEN YEARS. Before this
module nothing in the repo could produce an unseen-years number (training and 835-eval both
run the pinned 2021–2025 window). **Pre-registered 2026-07-08, before any fold result
exists.** This harness BUILDS RAILS — it trains nothing, runs no agent eval, and applies no
verdict; the leak-free gates AS CODE are the deliverable.

## Pre-registered split policy (planner, 2026-07-08 — implemented, NEVER invented)

| Fold | Train | Eval |
|------|-------|------|
| `fold1` | 2021-01-01 .. 2023-12-31 | 2024-01-01 .. 2024-12-31 |
| `fold2` | 2021-01-01 .. 2024-12-31 | 2025-01-01 .. 2025-09-01 |

Eval grading: the 835 scorecard on the eval window ONLY, vs that window's OWN corridor
floors (830 pattern, computed by this harness) and the engine-rule ref on the same window.
The harness fail-louds (`LeakError`) if a fold definition is edited into overlap — it
implements splits, it does not repair them.

## PIT universe (load-bearing) — the `build_pit_universe` verdict

**Verdict: `trader/data.py:401 build_pit_universe(db_path, min_bars)` CANNOT derive a
universe as-of train-end.** Its signature carries no as-of date and its coverage floor
counts ALL 1-min bars (`SELECT COUNT(*) FROM ohlcv WHERE asset_id=?` — unbounded in time),
so any universe it returns embeds post-train-end knowledge. The PIT derivation is therefore
implemented inside `walkforward.py::derive_pit_universe` with read-only queries; **ZERO
edits to `trader/data.py`**.

**Pre-registered qualification rule** (mirrors build_pit_universe's *filter*,
time-bounded):

1. symbol contains `/`, quote == `USD`, base not in `trader.config.STABLE_FIAT`
   (build_pit_universe's filter, imported — not copied);
2. coverage floor: ≥ `--min-days` (default **180**) DAILY bars with
   `timestamp <= as_of` (= train-end, UTC midnight);
3. rank by that same time-bounded daily-bar count DESC (tie: asset_id ASC), cap at
   `--n-assets` (default **64** real / **8** smoke). **The coverage RANKING + cap are NEW
   policy, not part of build_pit_universe's rule** (it has no ranking or cap — it returns
   ALL qualifying assets, asset_id-sorted); the ordering instead mirrors the training
   loader's own coverage-ranked universe selection. Pre-registered here, pinned by
   `POLICY` (below).

**LOUD DEVIATION**: the coverage counts run on **`ohlcv_daily`**, not build_pit_universe's
1-min `ohlcv`. Reasons: (a) `ohlcv_daily` is the canonical audited daily store of the
819/830 lineage (one bar per asset-day regardless of source, includes agg1m-only
dead/delisted assets — the survivorship control the PIT universe exists for); (b) fixture
daily == db daily is LEDGER-official (830/E0-REAL), so the REAL fold universes are
derivable *and leak-assertable* inside the factory without touching `trading_data.db`
(whose minute slice the fixture lacks pre-2025); (c) minute-bar counts are
source/backfill-dependent. Consequence carried, not hidden: the training loader reads the
1-min table, so a daily-qualified member could in principle lack minute bars — the emitted
`universe_pit.json.minute_coverage_check` reports this (VACUOUS on the fixture for
pre-2025 as-of dates, says so loudly; re-checked at the scheduled re-materialization).

The **pinned 64-asset 821 universe** (selected with knowledge through 2025 —
survivorship-leaky for fold evals) is emitted alongside as `period_*_leaky.json`, every
`_comment` carrying `LEAKY-UNIVERSE`, order byte-identical (TASK_740 landmine), for
continuity with 821-lineage numbers only.

**Scheduled cross-check (pre-registered; M4 identity contract)**:

```bash
# BOTH writing steps need the full three-part unlock (--fold-set real + a positively
# proven real-classed --db + --overwrite-existing) to land in the tracked tree — the
# identity fence binds at the shared write choke point, so `corridor` is gated too.
python experiments/837_walkforward/walkforward.py materialize \
    --fold-set real --db trading_data.db --overwrite-existing
python experiments/837_walkforward/walkforward.py corridor \
    --fold-set real --db trading_data.db --overwrite-existing
```

on the server must reproduce `folds/real/` **content-identically**, where the contract
is: all four `period_*.json` per fold **byte-identical** (they embed no volatile fields),
and `universe_pit.json` identical on the DERIVATION fields (`as_of`, `as_of_ts_utc`,
`rule`, `n_qualifying`, `n_selected`, `symbols`, `asset_ids`, `counts`). The volatile
fields — `generated` (timestamp), `db` (source path), `minute_coverage_check` (VACUOUS on
the fixture, CHECKED on the real db) — are **excluded from the contract** by definition (a
literal byte-identity contract would be unsatisfiable). Any diff on a contract field is a
finding (daily-parity break). Run it against a scratch `--out-root` first and diff, if you
want the cross-check without touching the tracked tree at all.

**`--overwrite-existing` (TASK_904A)**: `materialize` refuses to write into a non-empty
`--out-root` unless you pass this flag — the gate on overwriting an existing fold tree is
the operator's explicit intent, because the default invocation is
`materialize --db fixtures/trading_fixture.db --fold-set real --out-root folds/real` and
an accidental bare `materialize` would otherwise silently replace real fold rails with
fixture content. The flag authorizes the write above; it does **not** authorize
fixture-over-real — a re-materialization whose recorded provenance says the target was
built from a real database and whose `--db` is the fixture is refused outright (exit 1),
flag or no flag. The manifest's `_provenance` block (db, db class, fold set, content hash)
is **disclosure**: a re-materialization prints what it is replacing and flags any
disagreement, but nothing inside the directory grants write permission.

**The canonical `folds/real/` identity fence (TASK_904A, destination-first)** — scope, said
precisely, because an earlier draft of this section over-claimed. The tracked
`folds/real/` tree is fenced by the **destination's own identity**, never by this script's
location and never by what the directory's manifest says about itself:

- **Identity** = the canonical real-fold path of the git checkout the *destination* lives
  in (found by walking up from `realpath(dest)` to the nearest `.git`), matched by path and
  by `(st_dev, st_ino)`. So a copy of `walkforward.py` run from `/tmp` — or from a
  HEAD-control worktree — that points `--out-root` at the real tree by absolute path is
  fenced exactly the same. A genuine `/tmp` scratch copy outside any checkout is *not*
  fenced, which is what scratch copies are for. If identity cannot be resolved the guard
  **refuses**; it never goes inert.
- **Unlock** = a conjunction that must AGREE with the destination: `--fold-set real` AND a
  **positively proven** real-classed `--db` AND `--overwrite-existing`. "Positively proven"
  is load-bearing and is the cycle-3 fix: the db class is a **tri-state**
  (`fixture` / `real` / `unproven`) and the two consumers of it are deliberately *separate
  functions of opposite polarity* — `_db_kind_protective` (collapses `unproven` → "real":
  protect harder) for the recorded-provenance comparison, `_require_positively_real_db`
  (collapses `unproven` → refuse) for authorization. Authorization is never granted by a
  fallback, a default, or a failure to classify. Fixture identity is resolved
  **destination-first** (the destination's own repo root, `_REPO` only as an extra
  candidate) by path, by `(st_dev, st_ino)`, and by content against the tracked pin
  `fixtures/FIXTURE_IDENTITY.json`; if it cannot be established at all, that is *not
  positively real* and the write is refused. `FIXTURE_DB` is the CLI default value only,
  never an identity oracle — as a `__file__`-anchored oracle it let an absolute path to the
  real project fixture classify `real` from a worktree and overwrite the tree at exit 0.
  Caller *disagreement* with the
  destination is disqualifying, not exculpatory: `materialize --fold-set smoke --out-root
  folds/real --overwrite-existing` is refused (it would write 8-asset smoke content into
  the 64-asset real rails). A falsy `--db` is refused outright rather than silently
  reclassifying.
- **Where it binds**: at the shared write choke point, so it covers **`materialize` *and*
  `corridor`** — `corridor` therefore also takes `--overwrite-existing` and needs the same
  three-part unlock to write `folds/real/fold*/corridor.json` (bare `corridor` previously
  rewrote those files with no gate at all). **`aggregate` has no unlock**: point its `--out`
  at a run directory (e.g. `runs/<id>/WF_SUMMARY.json`) — the tracked fold tree is
  pre-registered rails, not a results directory. The corridor/aggregate writers themselves
  are **not** redesigned here (no idempotent recompute, no content-change detection, no
  two-phase write) — that is TASK_904B.
- **Net effect on refusals**, said precisely (an earlier draft over-claimed "only ever
  refuses more"): **strictly more refusals for the canonical tree; one courtesy check for
  un-provenanced directories was deliberately narrowed.** A hand-made non-empty scratch
  directory whose `fold_manifest.json` records no `universe_db`/`fold_set`/`generated` used
  to hit the fixture-over-real refusal, because the old prior-class derivation *invented*
  `"real"` for it; `_prior_db_kind` now returns `(None, None)` for a directory that makes no
  provenance claim, and the write proceeds. That is intended (see its docstring): inventing
  a class for a directory that never recorded one locks out every hand-made scratch
  out-root, the canonical tree is fenced separately and unconditionally by destination
  identity (`_authorize_canonical_write` runs first, independent of `_prior_db_kind`), and
  the relaxation only touches directories the operator already opted to overwrite.
  A second, narrower consequence of the same shape, disclosed for completeness: because the
  PRIOR-side classifier now recognises the fixture by basename and by pinned content as well
  as by path, a recorded `universe_db` that names a fixture *copy* (e.g.
  `backups/trading_fixture.db`, or a renamed byte-identical copy) reclassifies from "real" to
  "fixture", so the fixture-over-real refusal no longer fires for it. That is the correct
  class — such a tree really is fixture-provenance, and the refusal exists to protect
  REAL-provenance trees — but it is a change in the *fewer refusals* direction and is not
  **Trust dependency (cycle-3 re-verify, named remainder).** The "a renamed byte-identical
  copy cannot author this tree" guarantee is contingent on the tracked pin
  `fixtures/FIXTURE_IDENTITY.json` being correct. If that pin's sha256 is stale, hand-edited
  or corrupt, a renamed byte-identical copy of the real fixture classifies as real and IS
  authorized. It is reachable only where the `--db` path does not carry the fixture's
  basename AND no local fixture file exists for inode comparison at any candidate root --
  the worktree/reconstruction case the pin leg was added for. The bare-default
  `--db fixtures/trading_fixture.db` path is unaffected (spelling evidence wins). This module
  does not call `fixtures/fixture_identity.py::verify_or_die`, so nothing cross-checks the pin
  against a live fixture file before trusting it. Open remainder, not fixed here.

  covered by "strictly more". No manifest in the tree today is affected: the only recorded
  values are `fixtures/trading_fixture.db` and `trading_data.db`, and neither changes class.

**Pre-flight (M5 — REQUIRED before launching any fold training)**: on the fixture the
`ASSERT-2 ingest-max ohlcv` gate is VACUOUS for the real folds (no pre-2025 minute rows;
`validate` prints a WARN line saying exactly this). The manifests are launch-ready
against `trading_data.db`, so on the server run
`python experiments/837_walkforward/walkforward.py validate --fold-root
experiments/837_walkforward/folds/real --db trading_data.db` (read-only) BEFORE
`run_queue.sh` consumes `train_jobs.manifest` — the minute-table ingest assert and the
minute-coverage cross-check gain teeth only there.

## Fold artifacts (821 periods-file mechanism, mirrored exactly)

Per fold under `folds/<set>/<fold_id>/`: `universe_pit.json`, `period_train.json`,
`period_eval.json`, `period_train_leaky.json`, `period_eval_leaky.json` — all in the exact
`experiments/821_planner_daily/period_821_train.json` schema
(`periods[0]{name,start_date,end_date,assets}`; `train_jax_fast.py:223-231` and 835's
`--periods-file` read `periods[0]` only). Plus per set: `train_jobs.manifest` +
`eval_jobs.manifest` (the `tools/deploy/run_queue.sh` whitespace format
`<job_id> <gpu|cpu> <mem> <command…>`) and `fold_manifest.json` (machine-readable master).

- Training arm mirrored VERBATIM from `experiments/828_stage3_train/jobs_phase1.manifest`
  (cost0 phase-1; the preflight finding killed fee16-from-scratch — fee16 continuation is
  a planner phase-2+ decision, out of scope). Seeds {0,1,2}; `--name 837_<fold>_cost0_s<k>`
  ⇒ checkpoints at `jax_ckpt_fast/837_<fold>_cost0_s<k>` (`train_jax_fast.py:554`).
- Eval jobs call `experiments/835_stage4_eval/eval_vector.py` (imported by manifest, never
  modified) with the fold's `period_eval.json`, `--n-assets` = the fold universe size, and
  the operator carried from the TRAINING arm — `--vector-hold-bias 0.0` explicit (835's
  CLI default 3.0 is the jobs.manifest arm; explicit beats default), `--checkpoint` at the
  run root (= final; snapshots selectable by editing), `--out-dir runs/837_walkforward/…`.

## Fold corridor (830 machinery, per-eval-window, eval-slice numbers)

`experiments/830_corridor/corridor.py` is loaded by importlib (the 832 pattern) and
**never modified**; reused pieces: `load_panel` (audited daily panel + 819 liquidity
filter, UNMODIFIED defaults incl. `min_active=365`), `book_target_signs`, `replay_floor`,
`oracle_positions`.

Pre-registered conventions:
- **Backward pad `--pad-days` = 366**: the raw eval windows are shorter than the
  `min_active=365` filter tolerates (the 831-EXT amendment precedent: windows <366d zero
  the universe) and the 819 book needs warmup (`warm`≈31). The pad is PAST data only; the
  panel **never extends beyond eval-end** (asserted at compute time AND re-checked from
  the emitted `corridor.json` by `validate` — `panel_last_day_ts`).
- Every reported number is computed on the **eval-window decision slice only**
  (backtest half-open `[eval_start, eval_end)` convention):
  - `floor_inenv` — 830 `replay_floor` over the padded panel (the env book enters the eval
    window already positioned, as a live floor strategy would), per-decision rets index
    `k` ↦ row `warm+k` (the 830 `_regime_slices` convention), sliced by decision-row
    `day_ts`;
  - `reference` — `trader.backtest.window_metrics` (the gate-2 scorer: frozen 819
    positions on the whole panel incl. pre-history, metrics only on the slice);
  - `ceiling` — 830 `oracle_positions` (path_solver = measurement instrument ONLY, H5)
    scored by `trader.backtest.regime_series` on the identical `[r_lo, r_hi)` slice.
- Tiers = the 830 `GATE_COSTS` (gross / net@21 = fee16+slip5); annualization everywhere =
  `backtest._sharpe(., H_rebal·K = 1440)`; replay sizing = the 830 default `0.5/A`.
- Corridor universe = the 830-pattern liquidity-filtered full daily panel (NOT the fold's
  64) — the same yardstick caveat as 835 SPEC #10: per-fold floors are a calibrated
  yardstick for the 64-asset agent runs, indicative, never a promotion trigger alone.
- Real-fold corridors computed here run on the fixture's full daily panel
  (fixture daily == db daily, 830/E0-REAL); the scheduled confirmation against
  `trading_data.db` is a planner option, expected identical.
- **Edge convention vs the 835 fold scorecard (M6 — documented, with sign, not tuned)**:
  the corridor slice is backtest half-open `[eval_start, eval_end)` on UTC-midnight
  `day_ts`, and the daily panel load is end-EXCLUSIVE, so the last graded corridor
  decision is `eval_end − 2d` (the final panel row only labels the previous decision).
  The 835 fold scorecard's minute loader is end-INCLUSIVE at LOCAL midnight
  (`vectorbt_env.py:584`) and consumes `window_size` (60) rows of the eval panel for obs
  warmup, so its graded decisions run `~eval_start+60d .. ~eval_end−1d` (padding the eval
  periods-file backward to recover those 60 rows would put train data inside the
  scorecard's obs windows — rejected). **Sign of the mismatch**: the corridor covers ~60
  MORE daily decisions at the LEFT edge and ~1 FEWER at the RIGHT edge (timezone skew
  ≤1d). Neither instrument ever touches data past `eval_end`. The exact convention is
  embedded in every `corridor.json` (`slice_convention`).

## Aggregation

`aggregate` grades each fold's 835 scorecard against that fold's OWN `corridor.json` →
`walkforward_summary.json` + one LEDGER-convention row. Per-fold corridor positions are
ALWAYS shown (printed table renders per-fold FIRST); the pooled block (mean/min net@21,
all-beat-floor) exists only beneath the per-fold rows — **never a pooled Sharpe without
the per-fold view**. 835's completeness-guard convention carries over: a partial rollout
withholds the corridor booleans. HOLD-collapse flag at pct_hold > 99 (the 821 signature).
`worst_graded_fold` is computed over the SAME graded set (non-partial folds) that feeds
mean/min — a partial fold can never name the worst while being excluded from the stats.

## Leak-free gates AS CODE (`validate` — parses artifacts, never trusts construction)

- **POLICY-PIN (MAJOR-1 fix, 2026-07-09)**: every check runs against the PRE-REGISTERED
  constants (`POLICY` in `walkforward.py`), never against the artifacts' own tamperable
  rule blocks: (i) only pre-registered fold sets (`real`/`smoke`) validate at all;
  (ii) fold-set completeness — the manifest must carry exactly the pre-registered
  fold_ids; (iii) each fold's PARSED train/eval dates must equal the pre-registered
  windows; (iv) `universe_pit.json.rule.{n_assets_cap,min_days}` AND the manifest's
  `n_assets`/`min_days` must equal the pre-registered values. An internally-consistent
  tamper (rule block + periods-files edited coherently) and an honest
  `materialize --min-days 30` typo are both REJECTED.
- **ASSERT-1 (no train/eval overlap)**: train + eval periods-files (pit AND leaky) are
  re-parsed; train end must strictly precede eval start under the CONSERVATIVE DUAL PARSE
  (training loader's local-midnight `strptime` with END-INCLUSIVE bounds —
  `vectorbt_env.py:584` `timestamp <= end_ts` — AND the backtest UTC parse).
- **ASSERT-2 (no eval data can enter a train periods-file)**:
  (a) the loader's ingest query is run against the db with the PARSED train-file bounds;
  max ingestable timestamp (minute AND daily tables) must precede eval start (VACUOUS +
  loud when the db has no rows in-bounds; on real folds the vacuous minute check
  additionally prints the M5 pre-flight WARN);
  (b) PIT re-derivation equality: the universe re-derived from data ≤ train-end **with
  the POLICY-pinned rule parameters** (never the artifact's) must equal
  `universe_pit.json` exactly, symbols and order — a tampered or future-informed
  universe is REJECTED;
  (c) manifest cross-checks: every train job's `--periods-file` must be one of the fold's
  train files AND parse to a window ending before eval start; every eval job's
  `--periods-file` must be the fold's `eval_pit` file **IDENTICALLY** (MAJOR-2 fix:
  window equality alone would admit any same-window substitute, incl.
  `period_eval_leaky.json` — the survivorship-leaky universe silently becoming the
  graded unseen-years number) AND carry assets == the PIT universe symbols;
  (d) corridor panel bounded at eval-end.
- Violations are collected and raised together (`LeakError`, exit 1, loud banner) — the
  harness FAILS LOUD on any fold whose generated artifacts violate the gates.

## Pre-registered fixture smoke (`python walkforward.py smoke`)

The fixture's minute table covers 8 symbols × 2025-09-13..2025-12-12 (daily table is
full-history). Smoke folds are sized to that slice — **pre-registered**:

| Fold | Train | Eval |
|------|-------|------|
| `smokefold1` | 2025-09-14 .. 2025-10-31 | 2025-11-01 .. 2025-11-30 |
| `smokefold2` | 2025-09-14 .. 2025-11-30 | 2025-12-01 .. 2025-12-11 |

Smoke PIT cap = 8, min_days = 180 (same rule). Sequence: materialize → validate (gates
PASS on good folds) → per-fold corridors → tiny STUB scorecards (835-schema keys, marked
`stub`) → aggregate → the **NEGATIVE battery with teeth** (clones under
`folds/smoke/_negative/`; every clone gets ALL its repo-relative paths re-pointed —
fold_manifest AND both job manifests — the MAJOR-3 de-confound):
0. `control_noop` — an UNMUTATED clone must validate **CLEAN**; if it is rejected, the
   negative results are path confounds and the smoke FAILS;
1. `train_into_eval` (train end_date pushed into the eval window) → must be REJECTED
   naming **ASSERT-1**;
2. `tampered_pit_universe` (symbol swapped in `universe_pit.json`) → REJECTED naming
   **ASSERT-2 PIT universe re-derivation**;
3. `train_job_on_eval_file` (train manifest re-pointed at `period_eval.json`) → REJECTED
   naming the **train periods-file gate**;
4. `consistent_universe_retamper` (universe truncated to 3 with rule block, counts AND
   both pit periods-files edited to match — internally consistent) → REJECTED naming
   **POLICY-PIN universe rule**;
5. `eval_job_on_leaky_universe` (eval manifest re-pointed at the same-window
   `period_eval_leaky.json`) → REJECTED naming the **eval_pit identity gate**.
A rejection by the WRONG gate fails the smoke (each case asserts its targeted check is
named in the rejection) — a refactor that de-tooths any single gate breaks the smoke.
Then the REAL fold rails (materialize + corridor + validate → `folds/real/`, tracked).
Since TASK_904A both writing steps of that sequence need the three-part unlock spelled out
above (`--fold-set real` + a real-classed `--db` + `--overwrite-existing`) to land in the
tracked tree; `smoke` itself never writes there at all — it materializes the real rails
into a per-run gitignored scratch root and *diffs* against the tracked tree.
Artifact: `results_fixture_smoke.json`. PASS = correct window boundaries + the leak
gates demonstrated in BOTH directions. Stub numbers are plumbing, not verdicts; smoke
corridor Sharpes (9–28 decisions) are short-window noise by construction and graded by
nothing.

## Compute / IO envelope

CPU-only (`JAX_PLATFORMS` setdefault `cpu`); fixture db only in this task (read-only
`file:…?mode=ro` URIs); never `trading_data.db` here (scheduled steps may point `--db` at
it read-only); no GPU, no network, no npz; writes only inside this experiment dir (smoke)
or the chosen `--out-root`. Full smoke ≈ 15 s wall.

## Deliverables

- `walkforward.py` — materialize / validate / corridor / aggregate / smoke. Zero canon diffs.
- `folds/real/` — the default manifest: pre-registered fold rails (periods-files, PIT +
  LEAKY universes, run_queue + 835 manifests, fold corridors), validated.
- `results_fixture_smoke.json` — smoke artifact (incl. the no-op control record and the
  five targeted negative-test records).
- `.gitignore` — `folds/smoke/` (regenerated), bytecode.

## Out of scope

Running fold trainings (scheduled, phase-2+ planner decision); changing the
definition-of-working bar; touching 835/830 internals (imported); applying the R4
graduation verdict (planner, gate ladder).
