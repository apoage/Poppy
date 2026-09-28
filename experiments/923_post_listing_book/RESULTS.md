# TASK_923 — post-listing event book, phase A — RESULTS

**Verdict: UNDERPOWERED (D85, pre-registered, unchanged). G2 (POST-HOC corrected to the
task's own wording, applied after the data were seen): the placebo book does NOT match the real
book. The real book's Bonferroni CI excludes the placebo's central 95% interval, on the
favourable side, at 3 of 5 exits (h21, h60, age\*), on net and on gross alike. At h5 and h10 the
CI overlaps the placebo interval. At all 5 exits, 0 of 200 placebo means reach the real mean.
Conditional on shortability at listing (unverified). This is no tradability claim either way:
every binding net CI straddles zero and the +100bp band.**

**Cycle-3 closing pass (orchestrator transcription of judge-collected evidence, D76-style, run
57 — no new computation by the author; judge scratch `/var/tmp/923-judge-c2/` on gpu-server).**
(1) *The G2 placebo interval above is too narrow.* The 200 placebo draws reuse the same ~397
assets and only re-date them, so they are correlated, not independent replicates. Measured:
between-draw SD of the placebo means 48.5 / 70.2 / 92.8 / 158.5 / 190.2 bp (h5/h10/h21/h60/age\*)
vs a single draw's own clustered SE 79.1 / 112.4 / 155.7 / 254.7 / 308.2 bp vs the real book's
clustered SE 167.8 / 228.5 / 286.0 / 462.7 / 561.5 bp. With a pooled SE
√(real_se² + median single-draw SE²), (real − placebo)/SE = 1.59 / 2.68 / 3.17 / 3.11 / 2.91
against Bonferroni 2.576: the favourable reading at h21/h60/age\* survives with thin margins
(~0.35–0.6σ over threshold), and h10 also separates under this denominator. "3 of 5" is
therefore soft (3–4 of 5 depending on the null-variance estimator); the as-implemented z's of
6–11 in `results/g2_corrected.json` overstate the margin and must not be quoted. (2) *Population
mismatch refuted.* Restricting the real book to the placebo's admissible population (life ≥ 180
d) through `run_book.score_cell` unchanged: only 1 of 397 events is excluded; every mean moves
< 2% (h21 net +357.3 → +363.8, gross +1021.3 → +1027.8) and the G2 exclusion reading is
unchanged at every horizon. Neither item changes the verdict: UNDERPOWERED, no tradability claim.

Every one of the 5 pre-registered exits {5, 10, 21, 60, age\*=89} trading days is
**pre-declared UNDERPOWERED** against the +100bp tradability band (D85, computed at SPEC
time, before the binding-table run — see `SPEC.md` §6): MDE ranges 470bp (h5) to 1573bp
(age\*), one to two orders of magnitude above the +100bp band, because new-listing
idiosyncratic residual-return dispersion (gross SD 1943–6499 bp/event) swamps the band and
cohort-week clustering collapses ~387 events to ~134-135 effective clusters. **Per D85 this
verdict is fixed BEFORE the real-panel run is even executed** and does not change no matter
what the binding-table point estimates below show.

**G2 correction (disclosed, post-hoc).** The G2 criterion as first implemented and
pre-registered in SPEC.md §8.1 was `|real mean| > max|placebo mean|`, taken from P3's IC test.
That deviates from the task's G2 wording and is mis-denominated for this book. The real and
placebo books pay the same deterministic per-horizon cost (h5 ≈ 280bp … age\* ≈ 2296bp), so
the placebo's null is centred near −cost, not zero, and an absolute-value comparison about zero
does not test real against placebo. The earlier headline, "G2 does not cleanly separate at any
exit, reads NULL", came from that defect and is withdrawn. The corrected reading is the task's
own criterion, applied after the data were seen (`SPEC.md` "POST-HOC ERRATA",
`g2_corrected.py`, `results/g2_corrected.json`). The same code, seed and draw count reproduce
the placebo draws exactly; the reproduction check against `book_result.json` passed at every
horizon.

SPEC.md sha256 `8c16e13a60528f6a5f01557adbd5a2a20d9efc32657afc6f81b89b55c7af3063`, frozen
BEFORE `run_book.py`'s first real-panel run (verified by the run's own sha256 pin check,
`run_book.py:EXPECTED_SPEC_SHA256`, which refuses to execute on a mismatch). Fixture
`fixtures/trading_fixture.db` sha256 `b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9`
— verified byte-identical before AND after the run, both on the workstation and on the
gpu-server staged copy; never regenerated, never opened read-write.

## G1 SPEC-PIN

`SPEC.md` (sha256 above) was written from `results/spec_power.json`'s numbers — the exit
family, cost constants, estimand, band, and MDE computation — and frozen before
`run_book.py`'s real-panel run. Provenance: `spec_power.json` was produced by a run at
`results/spec_power.json` sha256 `00d754f3594135a1f43f4c3693142bdb0708009938f728cc96f6a526ff69f74b`.

## D85 power table (SPEC-time, no book P&L computed to produce it)

| horizon | n_events | n_clusters | gross residual SD (bp) | SE (bp) | MDE (bp) | vs +100bp band |
|---|---|---|---|---|---|---|
| h5 | 386 | 134 | 1942.8 | 167.8 | 470.2 | UNDERPOWERED |
| h10 | 386 | 135 | 2655.1 | 228.5 | 640.2 | UNDERPOWERED |
| h21 | 388 | 135 | 3323.1 | 286.0 | 801.3 | UNDERPOWERED |
| h60 | 387 | 134 | 5355.9 | 462.7 | 1296.2 | UNDERPOWERED |
| age\*(89) | 386 | 134 | 6499.4 | 561.5 | 1573.0 | UNDERPOWERED |

Age-decay exit derivation (P3 profile, 1-SE-of-zero rule, pre-registered before running):
every bucket 0–74d has `|mean| > 1 SE`; bucket 75–89d is the first (and only) one within 1 SE
of zero (mean −1.30bp, SE 8.80bp) → **age\* = 89 days**, coinciding with P3's own window end.

**Re-powering requirement** (the deliverable D85 asks for when MDE > band): to bring h5's MDE
down to +100bp at the SAME per-cluster dispersion would need roughly `(167.8/100)^2 ≈ 2.8x`
today's 134 effective clusters (~375 cohort weeks / ~7.2 years of listing cohorts at this
panel's rate), i.e. either a materially longer sample window, a cross-venue pool (out of
scope here — Kraken-only data), or a fundamentally different (lower-dispersion) estimand.
Longer horizons need far more (h60: `(462.7/100)^2 ≈ 21x`; age\*: `(561.5/100)^2 ≈ 32x`) —
the family cannot be re-powered by simply waiting for more Kraken listings at the short
horizons alone, and the longer horizons are not realistically re-powerable on this venue at
all.

## Binding table (T+1 entry, hedged short+market, 80bp/leg taker + borrow-high)

| exit | n_events | n_clusters | mean net (bp) | Bonferroni-95% family CI (bp) | MDE (bp, observed) | verdict |
|---|---|---|---|---|---|---|
| h5 | 386 | 134 | +7.8 | [−424.5, +440.1] | ~473 | UNDERPOWERED |
| h10 | 386 | 135 | +283.9 | [−304.7, +872.5] | ~643 | UNDERPOWERED |
| h21 | 388 | 135 | +357.3 | [−379.4, +1094.0] | ~801 | UNDERPOWERED |
| h60 | 387 | 134 | +0.6 | [−1191.1, +1192.4] | ~1300 | UNDERPOWERED |
| age\*(89d) | 386 | 134 | −474.9 | [−1921.1, +971.3] | ~1573 | UNDERPOWERED |

Bonferroni z = 2.576 (5-exit family, alpha=0.05). Point estimates are directionally positive
(short+hedge profits) at h5/h10/h21, roughly flat at h60, and negative at age\* (89-day
holding period's borrow charge — ~87.6%/yr x 89/365 ≈ 2135bp — dominates by then). Every CI
straddles both zero and the +100bp band; none excludes zero, let alone clears the band.

## Non-binding sensitivities (same book code, different inputs — reported, never claimed)

**Same-close entry** (vs T+1, hedged, 80bp+borrow-high): h5 +251.1bp, h10 +385.3bp, h21
+392.6bp, h60 +4.3bp, age\* −442.6bp. Same-close is uniformly MORE favorable than T+1 at every
horizon except age\* (execution-timing sensitivity comparable in direction to 918's finding
that same-close overstates a reversing edge — here the gap is smaller and less consistent than
918's `reversal_21d_xs` collapse).

**Legacy 21bp fee** (NON-BINDING, record-keeping only, no claim may cite it): h5 +125.8bp,
h10 +401.9bp, h21 +475.3bp, h60 +118.6bp, age\* −356.9bp.

**Borrow-low scenario** (NON-BINDING, ~43.8%/yr alt rate instead of ~87.6%): h5 +67.8bp, h10
+403.9bp, h21 +609.3bp, h60 +720.6bp, age\* +593.1bp. Borrow scenario matters a great deal at
the longer horizons (age\* swings from −474.9bp at high to +593.1bp at low) — the 89-day hold's
borrow charge is the single largest cost line at that exit.

**Unhedged naked short** (reported per the task file's explicit instruction, NOT the binding
claim — still 80bp+borrow-high, one fee leg only): h5 +465.1bp CI[−69.0,+999.2], h10
+903.4bp CI[+187.9,+1618.8], h21 +1267.1bp CI[+319.4,+2214.8], h60 +1917.4bp
CI[+278.9,+3555.8], age\* +2050.5bp CI[+106.8,+3994.1]. The unhedged reading is uniformly
larger and its Bonferroni CI-lower bound actually clears zero at h10/h21/h60/age\* (and clears
the +100bp band at age\*) — but this is a NON-BINDING sensitivity computed on the SAME
pre-declared-underpowered universe/dispersion, not a separately powered claim, and it carries
NO market hedge (raw directional crypto exposure, not the market-neutral claim this row
pre-registered). It is disclosed because the task asked for it, not promoted.

## Honesty battery

**G2 circular-shift placebo** (200 i.i.d. draws/horizon, seed 923, per-asset uniform shift
fraction in [0.10,0.90] of own life, clipped to the admissible day band `90<=shift_days<=life-90`,
same left-censoring exclusion as the real universe).

*Corrected G2 (task wording, POST-HOC).* Columns: the real mean and its Bonferroni family CI
(z=2.576); the placebo mean and its 2.5/97.5% quantiles over the 200 per-draw means; "excludes"
= the real CI lies wholly outside the placebo central 95% ("above" means the real book is more
favourable); p1 = one-sided share of placebo means ≥ the real mean.
Source: `results/g2_corrected.json` sha256
`9e29d4da505571055b64a31a2155b1a115b4a0721c25a044360e1d759b9d8878`.

NET (binding: 80bp/leg taker x2 legs + borrow-high):

| exit | real mean | real CI | placebo mean | placebo 2.5% / 97.5% | excludes? | p1 |
|---|---|---|---|---|---|---|
| h5 | +7.8 | [−424.5, +440.1] | −286.7 | [−382.5, −189.0] | No (overlap) | 0/200 |
| h10 | +283.9 | [−304.7, +872.5] | −399.0 | [−540.0, −259.6] | No (overlap) | 0/200 |
| h21 | +357.3 | [−379.4, +1094.0] | −674.2 | [−871.9, −511.7] | **Yes, above** | 0/200 |
| h60 | +0.6 | [−1191.1, +1192.4] | −1642.0 | [−1934.7, −1327.9] | **Yes, above** | 0/200 |
| age\*(89d) | −474.9 | [−1921.1, +971.3] | −2340.9 | [−2686.9, −1955.2] | **Yes, above** | 0/200 |

GROSS (fee=0, borrow=0; the same calls with zero costs, so the placebo dates are identical):

| exit | real mean | real CI | placebo mean | placebo 2.5% / 97.5% | excludes? | p1 |
|---|---|---|---|---|---|---|
| h5 | +287.8 | [−144.5, +720.1] | −6.7 | [−102.5, +91.0] | No (overlap) | 0/200 |
| h10 | +683.9 | [+95.3, +1272.5] | +1.0 | [−140.0, +140.4] | No (overlap) | 0/200 |
| h21 | +1021.3 | [+284.6, +1758.0] | −10.2 | [−207.9, +152.3] | **Yes, above** | 0/200 |
| h60 | +1600.6 | [+408.9, +2792.4] | −42.0 | [−334.7, +272.1] | **Yes, above** | 0/200 |
| age\*(89d) | +1821.1 | [+374.9, +3267.3] | −44.9 | [−390.9, +340.8] | **Yes, above** | 0/200 |

Net and gross give the same exclusions, as they must: the real and placebo books pay the same
per-horizon cost, so the cost cancels in the comparison. The placebo's gross mean is near zero
at every exit; the real book's gross mean is +288 to +1821bp. Read against the placebo, the real
listing dates are favourable at every exit. The CI clears the placebo central 95% at 3 of 5
exits and overlaps it at h5 and h10. The overlap comes from the real CI's width (the same
under-power D85 flags), not from a match in location. G2's NULL condition ("the placebo book
matches the real book") is therefore NOT met. This is post-hoc. It says the listing-date
timing is distinguishable from arbitrary in-life dates. It says nothing about beating the cost
stack: every net CI straddles zero.

*As-implemented criterion (deviates from task G2; kept for the record):* `|real mean| >
max|placebo mean|` AND two-sided empirical p <= 0.05.

| exit | real net mean | placebo net mean | placebo max\|mean\| | p (two-sided, |.|) | clean? (net) | clean? (gross, same criterion) |
|---|---|---|---|---|---|---|
| h5 | +7.8 | −286.7 | 412.4 | 1.0000 | No | Yes |
| h10 | +283.9 | −399.0 | 588.7 | 0.9453 | No | Yes |
| h21 | +357.3 | −674.2 | 935.8 | 1.0000 | No | Yes |
| h60 | +0.6 | −1642.0 | 1991.4 | 1.0000 | No | Yes |
| age\*(89d) | −474.9 | −2340.9 | 2767.2 | 1.0000 | No | Yes |

The as-implemented criterion flips from "not clean anywhere" on net to "clean everywhere" on
gross. Only the constant cost differs between the two, so this flip is the defect itself: the
statistic depends on where zero sits relative to the cost, not on real vs placebo.

**Leave-one-listing-out** (jackknife over events, binding cell): max single-event influence on
the headline mean is small everywhere — h5 26.7bp, h10 35.6bp, h21 47.1bp, h60 68.5bp,
age\* 94.7bp (all far smaller than the CI half-widths above) — no single listing drives the
binding-table reading; the underpowered finding is not an artifact of one outlier event.

**Capacity disclosure** (binding entry, first-5-day window, ASSUMED $10,000/event notional,
1% of summed 5-day dollar volume as a disclosed participation-rate ceiling, not a measured
limit): n_events=386. First-5-day dollar-volume ceiling distribution: min $13.56, median
$3,639, mean $62,770, max $12.7M. **The median event's ceiling ($3,639) is well below the
assumed $10,000 notional** — this book's assumed size would not be absorbable at the median
new listing's own early liquidity without materially larger participation-rate assumptions;
only the right tail (large, liquid listings) could plausibly carry $10k/event at 1%
participation.

**By-cohort-year** (h21, binding cell, calendar year of T+1 entry): 2015 n=1 (−1452.6bp),
2016 n=3 (+7195.6bp), 2017 n=6 (+1394.9bp), 2018 n=3 (−780.8bp), 2019 n=12 (−639.6bp), 2020
n=22 (+625.6bp), 2021 n=43 (−1277.8bp), 2022 n=114 (+166.6bp), 2023 n=17 (+305.6bp), 2024
n=79 (+333.2bp), 2025 n=88 (+1259.2bp). No monotone decay pattern with listing-quality era;
early years (n<15) are too thin to read at all, and the two biggest cohorts (2022 n=114,
2025 n=88) both read modestly positive but neither is powered on its own (n well below the
~135-cluster requirement above, and cohorts within one year cluster more tightly in cohort
week than the full-sample count suggests).

**Zero/one-event degenerate guards**: `book.clustered_mean_se` returns
`{n_events:0, degenerate:True, mean:nan, se:nan}` on empty input and
`{n_events:1, degenerate:True, mean:<value>, sd:nan, se:nan}` on one event — never a
divide-by-zero, never a fabricated CI. `book.listing_events` returns an empty array (not an
exception) when every candidate is excluded. `run_book.py` raises outright if the REAL panel's
candidate-event count is zero. All four exercised in `test_book.py`
(`test_clustered_mean_se_zero_events_guard`, `test_clustered_mean_se_one_event_guard`,
`test_listing_events_all_excluded_is_empty_not_crash`, plus the single-cluster degenerate
case) — 30/30 fixture-smoke tests pass (26 original + 4 cycle-2 G2 tests) (see "Tests" below).

## Files, provenance, compute

| file | sha256 |
|---|---|
| `SPEC.md` (pre-registered, as frozen; = its first 12138 bytes now) | `8c16e13a60528f6a5f01557adbd5a2a20d9efc32657afc6f81b89b55c7af3063` |
| `SPEC.md` (with appended POST-HOC ERRATA, cycle 2) | `535a019b1c5271d7f96c8436335474594ebd546662fdbe18db2c1aec2f4d03a6` |
| `loader.py` | `59cc466d79de7fa08ce78db3a40c48aa281aeaed60198c9db6634edd2f5cfd7b` |
| `book.py` | `e3444a9ca9c8d98e38ddfc1ffacd3d065dc0d429c0cc0270903464a87ec230ce` |
| `age_decay.py` | `6bb876ec8ec3979384802543622b92719e1365ba3f651c82ef68150c0b44d1f3` |
| `battery.py` (cycle 2: + `placebo_separation_task_g2`; prior `2826245d…`) | `c54de847e9b111ded5bccd1018edd147ef1109eba72a75d6ef3c1e81d90bc973` |
| `spec_power.py` | `9678019f2ed960d7ad5e4947b66f6554a649ff54e25015ba703f75c7bd98520d` |
| `run_book.py` | `b4b95372cb1230b3d126daad7ca42a0ed6b62f66605710281030fd50630e4262` |
| `test_book.py` (cycle 2: + 4 G2 tests; prior `11cf1cf8…`) | `eee5b19e9f6549f66224609992e4f2b05a1c22d133da2aa447e0935b06c19fef` |
| `g2_corrected.py` (cycle 2, new) | `296814de40f54a343739df902fb6db3f3ac7bcf01e3a507d480153997332f102` |
| `results/spec_power.json` | `00d754f3594135a1f43f4c3693142bdb0708009938f728cc96f6a526ff69f74b` |
| `results/book_result.json` (unchanged) | `b92673b0fe31e0384bc7c0f1ac1536bfff57ca35e9a8464238bd8ef8bd52c4de` |
| `results/g2_corrected.json` (cycle 2, new) | `9e29d4da505571055b64a31a2155b1a115b4a0721c25a044360e1d759b9d8878` |

Upstream provenance (read-only imports, unchanged): `experiments/917_second_moment_and_event_
instruments/daily_panel.py` (sha256
`e85fff4af4e1dc9b97fb68e09f4d12dd2e8addee9c4fae24b627cff8cca12f09`),
`p3_post_listing_event_study.py` (sha256
`2b28b42a4aea767da74d2cafd652f49fb4a2229bbfd1cfdc97c1274718d4ca3d`), `p3_result.json`
(sha256 `17dcf3cc424f81400ec9d210b788f8f5cf2b25b5fe3318efc1a247abb8e1e0c9`, commit `8f411d1`);
`trader/backtest.py`'s `BORROW_RATE_SCENARIOS` and 80bp taker tier (TASK_916,
`18d42d3`).

Compute: `gpu-server`, private scratch `/var/tmp/923-implementer/` (staged workstation-HEAD
code + a 50MB fixture copy, verified byte-identical before and after every run). Every run
wrapped `tools/safe_run.sh --cpu --mem 8G` with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4`. `pytest` fixture-smoke suite (26 tests, synthetic data, no DB touched):
26 passed. `spec_power.py` and `run_book.py` both CPU/seconds-scale (well under a minute
each). Never touched `trading_data.db`; never wrote server-side `runs/`.

Cycle-2 fix pass: private scratch `/var/tmp/923-fix/` on gpu-server, a copy of
`/var/tmp/923-implementer/` (which was left untouched) with the four changed files overlaid.
Upstream 917 and `trader/` files were sha-matched to the workstation. The same wrapper and
thread caps were used; the fixture was `b0f8de8b…` before and after. `run_book.py` was NOT
re-run: its whole-file SPEC pin now refuses by design, and `book_result.json` is unchanged.

## Tests

`test_book.py`, run wrapped on gpu-server: **30 passed, 0 failed** (cycle 2; originally 26) (fixture smoke: entry
timing, listing-event exclusion incl. all-excluded degenerate, event-return mechanics vs
hand computation, out-of-range-exit dropping, hedge-doubles-fee-leg, borrow scaling with
hold-days, zero-notional zero-borrow, zero/one/single-cluster degenerate guards, MDE/CI
degenerate handling, Bonferroni z sanity, placebo admissibility + smoke, leave-one-out hand-
computed + degenerate-below-3, capacity disclosure incl. zero-events, by-cohort-year
bucketing; cycle 2 adds four tests. (1) A synthetic book with a 2000bp constant cost and a
+1500bp real edge, against a zero-edge placebo carrying the same cost: the corrected G2
separates it and the as-implemented criterion does not. This test asserts the defect. (2) A
zero-edge real book: no separation. (3) Degenerate inputs: no separation. (4) The unfavourable
side is reported).

## Deviations / scope notes

- The task asked for "n_events and the event-level SD ... computed from the panel at SPEC
  time" using the panel's price data; this row used the GROSS market-hedged residual return
  (pre-cost) as the dispersion proxy rather than a net-of-cost figure, on the disclosed
  argument (SPEC.md §6) that binding costs shift each event by a near-constant, not a
  variance-changing quantity. Not independently re-verified against the actual observed net
  SD; the binding table's own reported SD (implicit in its CI) is available in
  `results/book_result.json` for a reader who wants to check this directly (`stat['sd']`
  under each `binding_table` row) — I did not add a side-by-side gross-vs-net SD comparison
  table for lack of remaining scope budget; this is a real, disclosed gap, not a silent one.
- Capacity disclosure used a $10,000 assumed notional and 1% participation rate — both
  arbitrary, disclosed assumptions, not measured or task-specified numbers.
- The by-cohort-year table is reported only for h21 (the task's own "by-cohort-year table"
  singular; the full JSON carries every horizon's per-event data needed to rebuild the table
  at any other horizon, but I did not print all five).
- G2's placebo used 200 draws/horizon (matching P3's own N_IID_DRAWS) rather than a
  task-specified count; not swept for sensitivity to draw count (unlike P3's own revision-3
  saturation check on its diagnostic grid, which does not apply here since this placebo is
  the i.i.d. design from the start, not the deterministic linspace grid P3 later demoted).
- No re-verification that Kraken actually permits shorting/margin on assets in their first
  days post-listing (task explicitly forbids exchange-API verification this run) — carried
  as an unverified condition on every reading, per the task file.
