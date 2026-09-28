# TASK_923 SPEC — post-listing event book, phase A

Pre-registration, frozen BEFORE the first real-panel book run (G1 SPEC-PIN). `run_book.py`
refuses to execute unless this file's sha256 matches its hardcoded `EXPECTED_SPEC_SHA256`.
Every number below that depends on the real panel was computed by `spec_power.py` and
`age_decay.py` — **no book P&L, no cost, no verdict was computed to produce them** (D85);
they are read-only measurements of P3's own instrument and of the panel's price/volume data.

## 0. Provenance

- Direction and headline age-bucket profile: `experiments/917_second_moment_and_event_
  instruments/p3_result.json`, sha256
  `17dcf3cc424f81400ec9d210b788f8f5cf2b25b5fe3318efc1a247abb8e1e0c9`, commit `8f411d1`
  (P3: age-conditional IC +0.0304, NW-t +4.15, n=946 dates, median 24 assets/date, placebo-clean).
- Panel loader / market-residual construction: `experiments/917_second_moment_and_event_
  instruments/daily_panel.py` (sha256
  `e85fff4af4e1dc9b97fb68e09f4d12dd2e8addee9c4fae24b627cff8cca12f09`) and
  `p3_post_listing_event_study.py` (sha256
  `2b28b42a4aea767da74d2cafd652f49fb4a2229bbfd1cfdc97c1274718d4ca3d`), imported UNCHANGED by
  `experiments/923_post_listing_book/loader.py`.
- Cost stack: `experiments/916_cost_truth/` (`TASK_916_COST_TRUTH_PACK.md`,
  shipped `18d42d3`) — `trader/backtest.py`'s `BORROW_RATE_SCENARIOS` and the 916-verified
  80bp base-taker round-trip tier, imported UNCHANGED (read-only) by `book.py`.
- Fixture: `fixtures/trading_fixture.db`, sha256
  `b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9` (verified before AND
  after every run in this row; never regenerated, never opened read-write).
- `spec_power.json` (this SPEC's numeric source): sha256 computed and recorded in
  `RESULTS.md` alongside this file's own hash.

## 1. Direction (pre-committed, not re-derived here)

**SHORT.** P3's market-residual age-bucket profile (mean bp/day by age, dead-included panel,
398 listed assets, `p3_result.json` `profile`):

| age bucket (days) | n | mean residual bp/day |
|---|---|---|
| 0–9 | 3893 | −78.30 |
| 10–19 | 3926 | −43.84 |
| 20–29 | 3941 | −24.11 |
| 30–44 | 5930 | −12.22 |
| 45–59 | 5911 | −10.71 |
| 60–74 | 5909 | −16.69 |
| 75–89 | 5909 | −1.30 |

Every bucket is negative: new listings **under**-perform the equal-weight PIT market
residual. The book is therefore: **short the new listing from entry, hedged long the
equal-weight market at beta=1** (the same beta=1 demeaning P3's residual uses — a plain
cross-sectional mean, not a fitted beta; P3's own disclosed simplification, inherited here
unchanged). The hedge leg is costed identically to the short leg (own fee round-trip; no
hedge-side borrow since the hedge is LONG, not short).

## 2. Universe / listing-date source

Identical to P3: `daily_panel.load_raw_pit_panel(fixture, include_dead=True)`, listing date
= first real (non-NaN) print per asset in the RAW (un-ffilled) store. **Left-censoring
exclusion**: an asset whose first real print is the panel's own grid-start row (index 0) is
excluded from the event universe — it is not an observed listing, it is the panel's own
left edge. P3's own audit found only 2/398 such assets on the dead-included panel and
reported that excluding them left the IC unchanged to 6 decimals (`RESULTS.md:798-799`);
this book applies the same exclusion on the same criterion (`book.EXCLUDE_FIRST_ROW_IDX`).
Measured on this panel (`spec_power.json`): **397 candidate events, 1 excluded** as
left-censored — consistent with P3's count.

## 3. Entry

- **Binding**: `entry = first_real_print_row + 1` — the close of the bar AFTER the first
  PIT-tradable post-listing bar (918-lesson execution timing: same-close entry systematically
  overstates edges that reverse within a day).
- **Non-binding sensitivity**: `entry = first_real_print_row` (same-close), reported, not
  claimed.

Both go through the identical book code (`book.entry_indices`, `book.compute_event_returns`).

## 4. Exits (fixed set, pre-registered, no tuning; every exit reported, no "best" selection)

- Fixed trading-day-row-offset horizons from entry: **{5, 10, 21, 60}**. "Trading day" here
  means one row of the daily grid (P3's own H=1-day forward-return convention extended to
  multi-day cumulation by summing daily residuals — no skip-NaN adjustment; disclosed).
- **Age-decay exit**: derived from P3's own age-bucket profile by a rule stated BEFORE
  running the real book and never revisited: **age\* = the right edge of the FIRST bucket
  (in bucket order 0–9 … 75–89) whose mean residual is within 1 standard error of zero**,
  where the per-bucket SE = bucket residual-return SD / sqrt(bucket n), computed from the
  SAME panel P3 used (`age_decay.py`). Default `age* = 89` (P3's own window end) if no
  bucket qualifies. **Measured** (`spec_power.json` `age_bucket_profile`):

  | bucket | mean bp | SD bp | n | SE bp | within 1 SE of zero |
  |---|---|---|---|---|---|
  | 0–9 | −78.30 | 1169.30 | 3893 | 18.74 | no |
  | 10–19 | −43.84 | 713.39 | 3926 | 11.39 | no |
  | 20–29 | −24.11 | 667.11 | 3941 | 10.63 | no |
  | 30–44 | −12.22 | 651.51 | 5930 | 8.46 | no |
  | 45–59 | −10.71 | 669.51 | 5911 | 8.71 | no |
  | 60–74 | −16.69 | 667.83 | 5909 | 8.69 | no |
  | 75–89 | **−1.30** | 676.29 | 5909 | **8.80** | **yes** |

  → **age\* = 89 days** (the last bucket is the first, and only, one within 1 SE of zero).

  So the pre-registered exit family is **{5, 10, 21, 60, 89}** trading days from entry (89
  coincides with the panel-defined window end, so this exit and a literal "age-decay" exit
  are the same number here — both computed independently, both landing on 89).

**N_FAMILY_EXITS = 5** for the Bonferroni family-wise correction below.

## 5. Costs (D81(c) denomination: per-notional per-unit-time for borrow, per-round-trip for
fees — never conflated, always summed)

- **Binding**: 80 bp round-trip base taker fee (916-verified) charged on **both** the short
  leg and the hedge leg (2 x 80 = 160 bp total fee per event) + margin-rollover borrow on
  the short notional at the **ALT-HIGH** scenario
  (`trader.backtest.BORROW_RATE_SCENARIOS["high"]["ALT"]` ≈ 87.6%/yr — new listings are
  alts, not BTC), charged for the full holding period
  (`notional * rate_annual * hold_days/365`).
- **Non-binding, reported**: legacy 21 bp round-trip fee (record-keeping only, no claim may
  cite it — task file, G3); ALT-LOW borrow scenario (≈43.8%/yr); unhedged naked short (fee on
  one leg only, still borrow-charged); same-close entry (§3).

## 6. Estimand, clustering, and the D85 power computation

**Estimand**: per-EVENT (listing) net markout in bp, at each pre-registered exit, mean with
event-clustered CI. Clustering unit = **listing cohort WEEK** (`floor(entry_ts / 7 days)`).

**Clustering formula** (disclosed simplification, used identically for the SPEC-time MDE and
the observed CI so the two are directly comparable): mean is the equal-weighted mean over
EVENTS; dispersion SD is the event-level SD (not cluster-mean SD); **SE = event-level SD /
sqrt(n_clusters)**. This is deliberately simple — no cluster-size weighting — the by-cohort
table in the results reports cluster-size spread so a reader can judge the approximation.

**Band**: mean net markout > **+100 bp** round-trip (the pre-registered tradability bar).

**MDE** (two-sided 5%, 80% power ≈ 2.8016·SE, `book.Z_MDE_80PWR_5PCT_TWOSIDED`) computed at
SPEC time from the **gross** (pre-cost) market-hedged residual return per event — subtracting
the binding costs shifts every event's value by a near-constant (fee is exactly constant;
borrow varies only through `hold_days`, which is pinned by the horizon, not free-floating),
so it does not materially change the SD. **Measured** (`spec_power.json`, T+1 entry, 397
candidate events minus per-horizon invalid trailing/gap drops):

| horizon | n_events | n_clusters | gross residual SD (bp) | SE (bp) | MDE (bp) | vs +100bp band |
|---|---|---|---|---|---|---|
| h5 | 386 | 134 | 1942.8 | 167.8 | 470.2 | **UNDERPOWERED** |
| h10 | 386 | 135 | 2655.1 | 228.5 | 640.2 | **UNDERPOWERED** |
| h21 | 388 | 135 | 3323.1 | 286.0 | 801.3 | **UNDERPOWERED** |
| h60 | 387 | 134 | 5355.9 | 462.7 | 1296.2 | **UNDERPOWERED** |
| age\*(89) | 386 | 134 | 6499.4 | 561.5 | 1573.0 | **UNDERPOWERED** |

**Every pre-registered exit is UNDERPOWERED against the +100bp band, by pre-registration,
before the binding-table run is even executed.** Per D85 and the task's own instruction, this
means the row refuses a "tradable" verdict at every horizon regardless of what the real
binding-table point estimates show; the deliverable is the distribution plus the re-powering
requirement (§8). The cause is not sample size in the ordinary sense (386-388 events is not
tiny) — it is that new-listing idiosyncratic residual-return dispersion (SD 1943–6499 bp) is
one to two orders of magnitude larger than the +100bp band itself, and clustering by cohort
week collapses the effective sample from ~387 events to ~134-135 independent weeks.

## 7. Family verdict rule (Bonferroni over the 5 exits)

Per-exit read (only for exits that clear the MDE screen — none do here, see §6): Bonferroni-
adjusted two-sided 95%-family CI (`z = z_bonferroni(5, 0.05)` ≈ 2.576, i.e. each exit's CI is
built at the 99%-per-test level so the FAMILY-WISE coverage is 95%). CI-lower > +100bp →
TRADABLE; CI-upper < 0 → COST_NEGATIVE; else → NULL. **UNDERPOWERED overrides every other
label** (D85: "an underpowered PASS is reported as UNDERPOWERED, never as tradable"). The
family verdict is the MAX (most favorable) reading over the 5 exits, stated with every
individual exit's own reading alongside it — never a single number standing in for the set.

## 8. Honesty battery (mandatory, same book code as the binding table)

1. **Circular-shift placebo (G2)**: per horizon, 200 i.i.d. draws, each admissible asset's
   own listing date replaced by `first + shift_days` (uniform fraction of its own life in
   [0.10, 0.90], clipped to the per-asset admissible band `min_days=90 <= shift_days <=
   life-90` — P3 revision 3's own bound, reused verbatim so the longest exit (89d) always has
   90 days of post-shift life to score). Same universe exclusion as §2 (left-censored assets
   never enter the placebo pool either). PASS requires `|real mean| > max|placebo mean|` AND
   empirical two-sided p <= 0.05 (P3 revision 3's own criterion, reused). If not met at a
   powered horizon, that horizon's verdict is NULL per the task file's G2 text; none of this
   row's horizons are powered, so this reading is disclosure, not a gate, here.
2. **Leave-one-listing-out**: jackknife over events at the binding cell; reports the min/max
   leave-one-out mean and the largest single-event influence.
3. **Capacity disclosure**: n_events, an ASSUMED (not measured) notional of $10,000/event, and
   a first-5-day dollar-volume ceiling read from the panel's own `volume` column at 1% of
   summed dollar volume (a standard, disclosed participation-rate heuristic, not a measured
   limit).
4. **By-cohort-year table** (2021→2026, entry-date calendar year, h21 binding cell).
5. **Zero/one-event degenerate guards**: `book.clustered_mean_se` never divides by zero and
   never fabricates an SD/SE from <2 events or <2 clusters (returns NaN + `degenerate=True`
   instead); `book.listing_events` on an all-excluded input returns an empty array rather
   than raising; `run_book.py` refuses outright (raises) if the REAL panel's candidate event
   count is exactly zero. Unit-tested in `test_book.py` on synthetic zero/one-event inputs.

## 9. Out of scope (unchanged from the task file)

Maker/limit entry (924's question), multi-venue listings, any portfolio/Sharpe framing, any
live trading, any DB write. Shortability of a freshly-listed asset on Kraken margin is
UNVERIFIED (no margin-eligibility universe in this repo, exchange APIs forbidden this run) —
**every verdict line carries "conditional on shortability at listing (unverified)."**

---

## POST-HOC ERRATA (cycle-2 fix pass, 2026-09-28) — appended AFTER the real-panel run

**This section is NOT pre-registration.** It was written after `run_book.py`'s real-panel
results were seen. Everything above this rule is the pre-registered SPEC exactly as frozen:
bytes `[0, 12138)` of this file hash to
`8c16e13a60528f6a5f01557adbd5a2a20d9efc32657afc6f81b89b55c7af3063` (the value
`run_book.py:EXPECTED_SPEC_SHA256` pins; verify with `head -c 12138 SPEC.md | sha256sum`).
Because this errata is appended, the WHOLE file no longer matches that pin, so `run_book.py`
will now refuse to re-run; that refusal is intended and the pin is left unchanged.
`g2_corrected.py` verifies the 12138-byte prefix instead.

**E1: the G2 criterion in §8.1 deviated from the task's own G2 wording.** §8.1 pre-registered
"`|real mean| > max|placebo mean|` AND empirical two-sided p <= 0.05", taken from P3 revision 3.
That criterion only works for a statistic whose null is centred at zero, which holds for P3's
IC. It does not hold for this book. Each horizon carries a deterministic cost (160 bp fee plus
borrow fixed by hold length: h5 ≈ 280 bp, h60 ≈ 1600 bp, age\* ≈ 2296 bp), so the placebo
book's null is centred near −cost, not zero. Comparing absolute values about zero therefore does
not test the real book against the placebo. TASK_923's G2 says: "the circular-shift placebo, run
through the SAME book code, must produce a distribution the real book's CI excludes; if the
placebo book matches the real book, the row's verdict is NULL".

**Correction applied (post-hoc, after seeing the data):** the task's own G2 criterion, per
horizon, on NET (binding costs) and on GROSS (fee = 0, borrow = 0) both. The rule checks
whether the real book's Bonferroni family CI (§7) excludes the placebo's central 95% interval
(the 2.5–97.5% quantiles of the 200 per-draw means). The report also gives the placebo mean and
the one-sided empirical p = share of placebo means ≥ the real mean. The placebo draws are the
SAME design as §8.1: same code (`battery.run_placebo`), seed 923, 200 draws. The gross placebo is
the same call with zero costs, so it uses the same RNG sequence and the same placebo dates.
Implemented as `battery.placebo_separation_task_g2`, run by `g2_corrected.py`, output
`results/g2_corrected.json`. The §8.1 criterion's output is kept alongside, labelled
"as-implemented (deviates from task G2)". The criterion was chosen after the data were seen,
but it is the task file's own criterion, not a new one.

Unchanged by this errata: exits, costs, estimand, band, MDE, the §7 verdict rule, D85's
UNDERPOWERED override, and the shortability condition.
