# 898/trend-decay — pre-registration

**PRE-REGISTRATION. Written before any score in this experiment is read.** This document is
hashed (sha256 of these exact bytes) by `decay_harness.py preregister`, and the resulting hash
is written to `PREREG.json` in this directory. `decay_harness.py`'s variant-scoring path
(`require_preregistered`) refuses to run unless `PREREG.json` already exists on disk and its
recorded hash matches the current bytes of this file — the ordering ("registered before its
first score is read") is therefore enforced by the code, not by discipline alone. If this file
changes after registration, the hash check fails loudly and `preregister` must be re-run
deliberately (a visible act, not a silent drift).

> **AMENDMENT 1 — 2026-08-22. This file was amended after its first registration and
> DELIBERATELY RE-REGISTERED with `python decay_harness.py preregister --force`.** The
> `spec_sha256` in `PREREG.json` is therefore the AMENDED spec's hash. Every artifact this
> harness writes carries `gate_iii_preregistration.amendment_note` saying so, so the amendment
> travels with the numbers rather than living only in this file's history.
>
> **What changed:** Scope 2 previously claimed the long-flat variant was "identical to Scope 1's
> frozen strategy in every respect ... with exactly one change". That is true at the line-diff
> level and FALSE at the portfolio level (see Scope 2 below). The amendment corrects the claim,
> requires realized gross exposure to be reported beside every Sharpe, and adds a third
> leverage-matched comparator series (Scope 2b). No verdict band, window family, cost tier,
> seed, bootstrap count, or rule parameter was touched.
>
> **Why this is a legitimate amendment and not p-hacking:** no real-data leg has run. At
> amendment time the only numbers in existence are fixture-scale instrument-verification
> numbers — the fixture is a small sampled panel used to prove the instrument works,
> and its Sharpes are not this row's scores. There is therefore nothing for an amendment to be
> tuned *against*: the pre-registered bands in `tasks/TASK_898_TREND_DECAY_SPOT_LEGAL.md`
> (FALLBACK STANDS / FALLBACK DEAD, VARIANT LIVES / BOTH DEAD) are unchanged and remain
> unscored. The real-data measurement (D54, gpu-server) runs strictly AFTER this hash. Had a
> real-data score already been read, the honest move would have been a new row, not an
> amendment.

This row is the FACTORY-EXECUTABLE HALF of `tasks/TASK_898_TREND_DECAY_SPOT_LEGAL.md`. It
builds and verifies the instrument at fixture/synthetic scale. **It emits no verdict.** The
real-data measurement over the full production `trading_data.db` daily table is planner-scheduled
on gpu-server (D54) and is explicitly out of scope here.

## Scope 1 — decay aggregation (the frozen 819/820 rule, unmodified)

The `trader.config.StrategyConfig` defaults (lookbacks=(30,), sizing="equal", signal="sign",
vol_window=30, H_rebal=1, gross_cap=1.0, costs_bp=(0,3,8,21), gate_costs=(8,21), n_boot=2000,
seed=819) — **no parameter is changed.** Positions are built by the unmodified
`trader.strategy.positions_from_config`. Three window families, all scored on whatever the
loaded daily table's actual coverage supports (never on a hardcoded end date):

1. **Calendar-year windows** `[Y-01-01, (Y+1)-01-01)` UTC, for every `Y` from 2021 (819's
   original research-window start) through the calendar year containing the table's present
   end.
2. **Rolling half-year folds** `[Y-01-01, Y-07-01)` ("YH1") and `[Y-07-01, (Y+1)-01-01)` ("YH2"),
   same naming convention as `875/wf-generalization`'s folds, from 2021H1 through the half
   containing the table's present end.
3. **One master forward window** `[2021-01-01, table_end)` — the single number that answers
   "is the 819 edge still alive, aggregated across everything since its original window."

Windows whose nominal end exceeds the table's present end are still scored (the underlying
`regime_series` clips to available rows) but are marked `"partial": true` with the row count
`n` reported plainly — never silently dropped, never silently treated as if they had full
coverage.

Scorer: `trader.backtest.window_metrics`, called unmodified, once per window.

## Scope 2 — pre-registered LONG-FLAT / no-short spot-legal variant

**Rule:** the frozen rule's inputs are all unchanged — same lookbacks, same vol_window, same
present-mask, same `weights()` call with the same `sizing="equal"`, same `H_rebal=1`, same
`gross_cap=1.0`, same cost tiers, same seed, same bootstrap count — and the signal is
transformed by exactly one line:

    s'_t = max(s_t, 0)          where s_t = sign(logc_t - logc_{t-30d})  (819 signal, unchanged)

i.e. every SHORT signal is replaced by FLAT (zero position) before weighting; LONG and no-signal
rows are untouched. This is the only spot-legal variant of the rule (no perps/margin needed) and
concentrates exactly on the thesis under test: the frozen rule's edge is suspected to live in
crash-avoidance shorting, which a spot-only book can never hold anyway.

**What that one line actually does to the PORTFOLIO (Amendment 1 — this paragraph replaces an
earlier claim that the variant was "identical in every respect ... with exactly one change",
which was true of the diff and false of the book).** `trader.strategy.weights(pres, rv, sizing)`
is a pure function of the present-mask and realized vol; it never sees the signal's sign. The
budget freed by clipping a short is therefore **NOT reallocated onto the surviving longs — it
becomes implicit cash.** The long-flat book consequently runs at a **lower, STATE-DEPENDENT
gross exposure** than the frozen rule, and it de-leverages hardest exactly where the frozen rule
was most short — i.e. in bear regimes, the very windows the "is the edge just crash-avoidance
shorting?" question turns on. Measured on `fixtures/trading_fixture.db` (mean realized
`Sum_i|pos[t,i]|` on the scorer's own decision grid): FULL 0.998 → 0.430 (43%), BULL21
1.000 → 0.611 (61%), BEAR22 0.999 → 0.276 (28%).

Sharpe is invariant to a CONSTANT leverage multiplier; it is **not** invariant to a
state-dependent one, which damps mean and variance unevenly across the window. So the naive
"variant Sharpe vs baseline Sharpe" comparison is confounded. This spec's response is
deliberately NOT to change the variant:

1. **The de-leveraged book stays the PRIMARY variant.** It is the honest reading of "shorts
   replaced by flat" and it is what could actually be traded on a spot venue. Silently
   re-normalizing it would report a book nobody can hold.
2. **Exposure is reported everywhere (mandatory).** Every scored series in every window and
   every regime carries `gross_exposure_mean` / `_min` / `_max` / `_n`, plus a per-row
   `leverage` block giving baseline vs variant vs leverage-matched mean exposure and the
   `variant_over_baseline` ratio. No Sharpe in this artifact is readable without its leverage
   beside it.
3. **A third, leverage-matched comparator is scored alongside** — Scope 2b.

### Scope 2b — leverage-matched comparator (DIAGNOSTIC, not deployable)

`variant_long_flat_lev_matched`: the long-flat positions rescaled **row-wise** to the frozen
rule's own realized gross exposure on that same decision,

    pos_lm[t,:] = pos_var[t,:] * ( Sum_i|pos_base[t,i]| / Sum_i|pos_var[t,i]| )

so `Sum_i|pos_lm[t,:]| == Sum_i|pos_base[t,:]|` wherever the long-only book holds anything. A
per-decision rescale is required: a single constant multiplier would leave Sharpe numerically
unchanged and answer nothing. This series isolates **signal quality** from **sitting in cash**,
which is the question the row exists to ask; it is explicitly labelled as a diagnostic, not as a
shippable book (its concentration is unbounded even though its gross stays `<= gross_cap`).

**Residual, disclosed confound:** on decisions where every present asset is short, the long-only
book has nothing to hold and cannot be matched; those rows stay in cash. Their frequency is
reported as `lm_unmatched_frac` (with `lm_scale_mean` / `lm_scale_max` for how hard the rescale
had to push). Leverage-matching narrows the confound; it does not abolish it, and the artifact
says so rather than pretending otherwise.

Both extra series are built from the SAME unmodified `trader.strategy` / `trader.backtest`
primitives; `trader/` is not touched (gate i).

**Construction:** built in `decay_harness.py::build_long_flat_positions` from the SAME
unmodified `trader.strategy` primitives the frozen rule uses (`rolling_vol`, `first_active`,
`trend_signal`, `present_mask`, `weights`) — the diff against `trader.strategy.build_positions`
is exactly the one `sig = np.maximum(sig, 0.0)` line, nothing else (with the portfolio-level
consequence spelled out above: that line de-leverages the book, it does not merely re-sign it).
`trader/` itself is not touched; the composition happens entirely in this experiment's own file.
The leverage-matched comparator is built in `decay_harness.py::build_leverage_matched_positions`
from `build_long_flat_positions`' own output — no new signal, no new parameter.

**Scoring:** `trader.backtest.cell_metrics` accepts caller-supplied `(pos, pres, warm)` — it was
already built that way for TASK_820/893 — so the variant's REGIME table (Scope 3, the fixed
819 `REGIMES` windows) is scored by calling `trader.backtest.cell_metrics` completely
unmodified with the variant's own `(pos, pres, warm)` in place of the frozen rule's. For the
Scope 1 decay windows (which extend past `REGIMES`' fixed FULL end and therefore cannot go
through `cell_metrics`'s hardcoded regime table), `trader.backtest.window_metrics` cannot be
reused directly because it always rebuilds `pos` internally via
`strategy.positions_from_config` and has no parameter for an externally supplied `pos`.
`decay_harness.py::score_window` supplies that missing composition — it is the *same* four
primitives `window_metrics` itself calls (`trader.backtest.regime_series`, `.net_series`,
`._sharpe`, `._block_bootstrap_sharpe_ci`), called directly, with zero re-derived arithmetic.
`test_decay_harness_898.py` proves this equivalence directly: scoring the FROZEN rule's own
`(pos, pres, warm)` through `score_window` and through `window_metrics` on identical bounds
must — and does — produce byte-identical dicts.

Same windows as Scope 1 (calendar years, rolling halves, master forward window), same regime
table as Scope 3.

## Scope 3 — regime + B&H + cash-floor report, with CIs

For every regime in `trader.config.REGIMES` (BULL21, BEAR22, REC23, BULL24, CHOP25, FULL) and
for every Scope-1/2 decay window: report net Sharpe at every cost tier, gross Sharpe, B&H Sharpe
(`bh` field — already computed by the unmodified scorer), the cash floor (0 return / 0 Sharpe,
reported as the trivial comparator net{c} > 0 and cum{c} > 0), and — at the two gate costs
(8bp, 21bp) — the block-bootstrap 95% CI on net Sharpe, from the unmodified
`_block_bootstrap_sharpe_ci` with the frozen `n_boot=2000`, `seed=819`.

All three series (`baseline`, `variant_long_flat`, `variant_long_flat_lev_matched`) are reported
for every regime and every decay window, each carrying its own realized exposure fields
(`gross_exposure_mean/_min/_max/_n`), plus a per-row `leverage` block summarizing the three mean
exposures, `variant_over_baseline`, `lm_scale_mean/_max` and `lm_unmatched_frac`. Exposure is
measured on the SCORER'S OWN decision grid: `decay_harness.py::decision_grid` recovers the exact
rows `trader.backtest.regime_series` scores by calling that unmodified primitive on a synthetic
unit-ramp panel (for which every forward label equals `H`, so its `gross` output degenerates to
a readable function of `pos`); no window/warmup/stride arithmetic is re-derived here.

## Gate (i) — unmodified scorer

Enforced by construction: `decay_harness.py` imports `trader.backtest` and `trader.strategy`
and never copies, monkeypatches, or re-derives any of their arithmetic. `test_decay_harness_898.py`
asserts the `score_window`/`window_metrics` equivalence above, and separately runs
`python -m trader selftest` (must stay green — nothing in `trader/` is touched by this row).

## Gate (ii) — data-epoch assertion, stale/short table discipline

Every run asserts, at the top of its output:
- `ohlcv_daily`'s actual `MIN(timestamp)`/`MAX(timestamp)` over the WHOLE table (independent of
  which assets survive the USD-quote/liquidity/PIT filters) — the literal table epoch.
- the loaded, filtered panel's own `ts_sub` min/max (may differ from the table epoch if the
  scored universe's coverage is narrower). Emitted by `assert_panel_epoch` into
  `gate_ii_data_epoch.panel_epoch` as `panel_min_date` / `panel_max_date` / `panel_n_rows` /
  `panel_span_days` / `head_gap_days` / `tail_gap_days` / `narrower_than_table` / `note`, and
  printed by the CLI. When the filtered panel is narrower than the table at either end, the
  `note` says so explicitly and instructs the reader to read "through `<table_max>`" claims
  against `panel_max_date` instead — a "recent" claim resting on a universe whose coverage
  stopped earlier is the same failure mode as a stale table, and must not pass silently.
- `staleness_days = now_utc - table_max` (or `--now-utc` override, for reproducible testing).

Thresholds (this row's own encoded discipline — the task file does not fix numbers):
- `staleness_days > HARD_REFUSE_DAYS (120)` **or** `table_span_days < MIN_SPAN_DAYS (365)`
  (the table's OWN `MAX(timestamp) - MIN(timestamp)` coverage, not measured against
  2021-01-01) ⇒ **loud refusal**: the run exits non-zero, prints the reason, and writes NO
  results JSON, unless `--allow-stale` is passed.
- `WARN_DAYS (14) < staleness_days <= HARD_REFUSE_DAYS` ⇒ the run proceeds but every result
  carries a top-level `"epoch_caveat"` block naming the staleness and instructing the reader not
  to treat any number as "current."
- Below `WARN_DAYS` ⇒ the epoch assertion is still recorded (never omitted) but is not a caveat.

This is this row's mechanism for G6 (data rot as a named failure mode): a "recent" claim on a
stale table is a loud refusal or an explicit recorded caveat, never a silent pass.

## Gate (iii) — pre-registration ordering

See the top of this file. `PREREG.json`'s presence + hash match is a precondition
`decay_harness.run()` checks (`require_preregistered`) **before it constructs the variant at
all** — so no reported variant number, in any artifact this harness writes, can exist without a
matching registration.

**Scope of that claim (narrowed — Amendment 1).** It is a statement about `run()`, the only
producer of results JSON. It is NOT a claim that the module offers no unguarded entry point:
`build_long_flat_positions`, `build_leverage_matched_positions`, `score_window` and
`decision_exposure` are plain importable functions, and `test_decay_harness_898.py` calls them
directly. The earlier wording ("there is no code path that produces a variant number without
it") over-claimed. The protection that actually holds is: **the gate precedes variant
construction on the artifact-producing path**, and it fails closed (missing `PREREG.json`, or
hash drift, both raise). `test_decay_harness_898.py`'s CHECK 6 additionally asserts the live
gate passes before it builds a variant, so the checked-in test observes the ordering it
describes rather than quietly stepping around it.

## Out of scope (verbatim from the task file)

Changing the 819 rule's parameters (tuning). Running the full real-data measurement over
`trading_data.db` as this row's deliverable (planner-scheduled on gpu-server, D54). Writing to
`trading_data.db`. Regenerating `fixtures/trading_fixture.db`. Emitting FALLBACK STANDS /
FALLBACK DEAD / BOTH DEAD or any other verdict. Editing `docs/pathway.md`. GPU jobs.
