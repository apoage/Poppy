# TASK_917 — second-moment & event-conditional instrument pack: results

**REVISION 3, 2026-09-06 — CLOSING PASS (cycle 3).** Disclosed here, in the verdict itself and
not in a footnote, per the project's D76 convention: **this is a closing pass at cycle 3, the
last `halt_after_failures` allows.** Its entire remit was to make the artifacts' *stated
reasoning* match evidence that two adversarial judges had already collected and confirmed. **It
changed no verdict.** P1 is still INCONCLUSIVE (kill does not fire); P2 is still KILL FIRED with
κ₃ byte-identical (`0.036056622631943466`); P3 still survives placebo-clean; P4 is still
prepared-not-run. What it did change is that several arguments were **wrong even where the
conclusions were right** — most importantly P1's pro-kill counterfactual (refuted), P1's missing
power ceiling (the load-bearing correction: **P1 could not have fired a defensible kill at any
block length**), P2's placebo (a sample-size mixture, not a null) and P3's placebo design (a
deterministic grid whose p-value was a function of grid density). Every withdrawn claim is named
as withdrawn rather than silently replaced. Section-by-section detail is inline below and in each
module's `REVISION 3` docstring block; machine-readable in each `p*_result.json` under
`closing_pass`.

**REVISION 2, 2026-09-06.** Revision 1 (same day) shipped a **KILL FIRED** verdict for P1 that
an adversarial panel then showed does not survive a block-length sweep. This revision replaces
that verdict, ships the sweep as a first-class artifact, fixes a placebo that contained copies
of the statistic under test, adds the placebo P2 was missing, and widens P3's. **P1's verdict is
restated to INCONCLUSIVE; P2's and P3's verdicts are unchanged.** What changed and why is stated
in each section, not in a footnote.

## Provenance, epoch and side effects (read first)

**Data.** Read-only against `fixtures/trading_fixture.db` for P1–P3. Its `ohlcv_daily` table is
verified BYTE-IDENTICAL to `trading_data.db`'s — sha256 of the ordered row dump matches on both
sides, `90d67f40eb2aee31a4745c37669479c7f5701022d64facbb0bb4ac35a6f26abe`, 499430 rows —
independently confirmed, **so these are the real production numbers, not a small-sample proxy.**
`trading_data.db` was opened `mode=ro` only, never written, never DDL'd. Fixture sha256 unchanged
before and after every run in this revision:
`b0f8de8b108b50521f301df1d0856aed72bc760c57b67a842fa62fa918206cf9`.

**Framework epoch.** `experiments/LEDGER.md`'s epoch ladder **E-a…E-f does not apply to this
row**, and that is an explicit statement rather than a silence. E-a…E-f are all defects of the
`mcts_trader/` vector-env / training / eval lane (snapshot-cash wall, zero-price tails in the
828 panel builder, ingest identity split, local-time windows, XBT↔BTC fold, no-print fill
semantics). P1–P4 are `trader/`-lineage **data instruments**: they read `ohlcv_daily` / `ohlcv`
directly through `daily_panel.py` (raw, un-ffilled) and `trader.data.load_db_backtest_panel` for
universe membership only, train nothing, take no positions and grade no checkpoint. Two epoch
concerns *are* live and are handled inside the instruments rather than by epoch tag: the E-b
zero-price/ffill class (defused — `daily_panel.py` deliberately does NOT ffill, so no fabricated
zero-return day can bias a variance ratio toward VR→1 or invent a zero-magnitude "shock"), and
the E-e XBT↔BTC identity fold (the fixture carries `XBT/USD` id 1091; universe membership comes
from the canon loader, so this row inherits whatever fold state that loader has and does not
re-derive it).

**Instrument.** `experiments/917_second_moment_and_event_instruments/{p1,p2,p3,p4}_*.py` at
revision 3, numpy/pandas/CPU, run under `tools/safe_run.sh --cpu --mem 12G`. P1 runs in ~30 s
(the added tiling-origin sweep and 200-draw sign-flip null are cheap); P2 in ~4 min; P3 in ~1 min.

**Side effect, stated because two review agents tripped on it.** P1, P2 and P3 **WRITE** their
`p*_result.json` next to the script via `__file__`-relative paths — running them is not a
read-only act. Each now prints the artifact path twice: once at startup
(`SIDE EFFECT: this run will OVERWRITE …`) and once immediately before the write
(`OVERWRITING artifact: …`). P4 writes nothing and now says so.

---

## P1 — variance-ratio persistence — **INCONCLUSIVE, KILL DOES NOT FIRE**

### Revision 3, up front: the verdict is unchanged and the reason is stronger

Two adversarial judges accepted revision 2's verdict and then dismantled most of the arguments
revision 2 used to reach it. The corrected position, in one paragraph:

**P1 could not have fired a defensible kill at any block length.** A model-free sign-flip null on
the real panel cannot distinguish the observed block-VR dispersion from pure estimator noise at
any of the 11 block lengths, which caps the largest ρ̄ this instrument could report against a
*perfectly persistent* true VR at roughly the size of its own standard error, and puts the
anchor's power to separate ρ̄ = 0.10 from ρ̄ = 0 at **8.9% two-sided / 15.6% one-sided** (the
panel's independent figure: ~10%). Revision 2's framing — "the answer depends on a free
parameter with no principled setting" — was true but too kind to the instrument: the answer
depends on **two** free parameters (block length *and* the tiling origin, which revision 2 never
disclosed at all), and at no setting of either did the instrument have the power to resolve the
kill line. Six specific claims are **withdrawn** below and named as withdrawn: the pro-kill
counterfactual (H), the level-vs-breakeven shortfall (J), R1's "→ FAIL" as a data finding (K),
the ≤0.15 independence claim and the 44→11 multiplicity deflation resting on it (L), the
k-parity "open instrument concern" (M), and the claim that R2 is a power guard (H).

### What revision 1 got wrong

Revision 1 read the kill off a **single cell**: block=180d / pit_dead_retained / pooled →
ρ̄ = −0.0315 on 18 block pairs, and reported "ρ̄ < 0.10 on every one of 8 cells: every
regime-gating idea on this venue is closed". Three defects, all confirmed:

1. **The 180-day block length was TASK_917's own addition and was disclosed nowhere.** The
   source panel this row salvages, `lore/harvest_6b32_appendix/mech_B.md:44,48`, pre-registers
   **90-day non-overlapping windows** with the identical ρ̄ ≥ 0.10 threshold (and a stronger
   null: NW t ≥ 3, 1000 asset-label permutations). Nothing inherited 180d.
2. **The grid composition was the implementer's, not the task's.** The task file mandates
   "Circular-shift placebo + Bonferroni over 8 cells" as a *count*. The specific 8 cells —
   in particular including 180d and stopping the block axis at {90, 180} — were the
   implementer's own pre-registration. That is now disclosed here and in
   `p1_result.json:grid_provenance`.
3. **The estimator was fine.** Three independent reproductions confirmed ρ̄ = −0.0315;
   `pooled_block_vr_series` / `xsmean_block_vr_series` are unmodified in this revision. The
   failure was selection and non-disclosure of sensitivity, not arithmetic.

### The full block-length sensitivity sweep (44 cells)

Every block length that tiles cleanly into the 30-day VR sub-window, 60d…360d, × universe ×
estimator. `k` = 30-day sub-windows per block. `circ_p` = corrected circular-shift p (see
placebo section), `circ_pmin` = the smallest p that design can produce, `perm_p` = 2000-draw
permutation p. `*` = fewer than 10 block pairs. Bonferroni α = 0.05/44 = 0.00114, applied to
`perm_p` only.

| block | k | universe | est | blocks | pairs | ρ̄ | 95% CI (Fisher-z) | t | circ_p | circ_pmin | perm_p | sigBonf | < kill |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 2 | pit_dead_retained | pooled | 52/77 | 50 | +0.0768 | [−0.206,+0.348] | +0.53 | 0.737 | 0.026 | 0.6017 | False | True |
| 60 | 2 | pit_dead_retained | xsmean | 57/77 | 55 | **+0.1739** | [−0.096,+0.420] | +1.29 | 0.342 | 0.026 | 0.1689 | False | False |
| 60 | 2 | survivor_only | pooled | 52/77 | 50 | +0.0153 | [−0.264,+0.292] | +0.11 | 1.000 | 0.026 | 0.9215 | False | True |
| 60 | 2 | survivor_only | xsmean | 57/77 | 55 | **+0.1646** | [−0.105,+0.412] | +1.22 | 0.368 | 0.026 | 0.1909 | False | False |
| **90** | 3 | **pit_dead_retained** | **pooled** | 37/51 | **36** | **−0.2878** | **[−0.563,+0.045]** | −1.75 | 0.200 | 0.040 | 0.0665 | False | True |
| 90 | 3 | pit_dead_retained | xsmean | 38/51 | 37 | −0.2775 | [−0.552,+0.051] | −1.71 | 0.160 | 0.040 | 0.0820 | False | True |
| 90 | 3 | survivor_only | pooled | 37/51 | 36 | −0.2781 | [−0.556,+0.056] | −1.69 | 0.160 | 0.040 | 0.0745 | False | True |
| 90 | 3 | survivor_only | xsmean | 38/51 | 37 | −0.2730 | [−0.548,+0.056] | −1.68 | 0.160 | 0.040 | 0.0900 | False | True |
| 120 | 4 | pit_dead_retained | pooled | 28/38 | 27 | **+0.1062** | [−0.285,+0.467] | +0.53 | 0.579 | 0.053 | 0.5962 | False | False |
| 120 | 4 | pit_dead_retained | xsmean | 29/38 | 28 | **+0.2050** | [−0.182,+0.537] | +1.07 | 0.474 | 0.053 | 0.2509 | False | False |
| 120 | 4 | survivor_only | pooled | 28/38 | 27 | **+0.1259** | [−0.267,+0.483] | +0.63 | 0.474 | 0.053 | 0.5197 | False | False |
| 120 | 4 | survivor_only | xsmean | 29/38 | 28 | **+0.2133** | [−0.174,+0.543] | +1.11 | 0.474 | 0.053 | 0.2339 | False | False |
| 150 | 5 | pit_dead_retained | pooled | 24/31 | 23 | −0.1701 | [−0.544,+0.260] | −0.79 | 0.600 | 0.067 | 0.4013 | False | True |
| 150 | 5 | pit_dead_retained | xsmean | 24/31 | 23 | −0.1660 | [−0.541,+0.264] | −0.77 | 0.533 | 0.067 | 0.4208 | False | True |
| 150 | 5 | survivor_only | pooled | 24/31 | 23 | −0.1572 | [−0.535,+0.273] | −0.73 | 0.600 | 0.067 | 0.4393 | False | True |
| 150 | 5 | survivor_only | xsmean | 24/31 | 23 | −0.1668 | [−0.542,+0.264] | −0.78 | 0.533 | 0.067 | 0.4228 | False | True |
| 180 *(rev-1 headline)* | 6 | pit_dead_retained | pooled | 19/25 | 18 | −0.0315 | [−0.491,+0.442] | −0.13 | 1.000 | 0.083 | 0.8996 | False | True |
| 180 | 6 | pit_dead_retained | xsmean | 19/25 | 18 | +0.0771 | [−0.404,+0.525] | +0.31 | 0.667 | 0.083 | 0.7586 | False | True |
| 180 | 6 | survivor_only | pooled | 19/25 | 18 | −0.0949 | [−0.538,+0.389] | −0.38 | 0.833 | 0.083 | 0.7146 | False | True |
| 180 | 6 | survivor_only | xsmean | 19/25 | 18 | +0.0656 | [−0.414,+0.517] | +0.26 | 0.833 | 0.083 | 0.8011 | False | True |
| 210 | 7 | pit_dead_retained | pooled | 18/22 | 17 | −0.1258 | [−0.572,+0.378] | −0.49 | 0.636 | 0.091 | 0.6242 | False | True |
| 210 | 7 | pit_dead_retained | xsmean | 17/22 | 16 | −0.0120 | [−0.505,+0.487] | −0.04 | 0.909 | 0.091 | 0.9580 | False | True |
| 210 | 7 | survivor_only | pooled | 18/22 | 17 | −0.1475 | [−0.587,+0.359] | −0.58 | 0.636 | 0.091 | 0.5632 | False | True |
| 210 | 7 | survivor_only | xsmean | 17/22 | 16 | +0.0416 | [−0.464,+0.526] | +0.16 | 0.909 | 0.091 | 0.8671 | False | True |
| 240 | 8 | pit_dead_retained | pooled | 16/19 | 15 | −0.0207 | [−0.527,+0.497] | −0.07 | 0.889 | 0.111 | 0.9280 | False | True |
| 240 | 8 | pit_dead_retained | xsmean | 15/19 | 14 | **+0.2530** | [−0.321,+0.691] | +0.91 | 0.444 | 0.111 | 0.3338 | False | False |
| 240 | 8 | survivor_only | pooled | 16/19 | 15 | −0.0292 | [−0.533,+0.490] | −0.11 | 0.889 | 0.111 | 0.9125 | False | True |
| 240 | 8 | survivor_only | xsmean | 15/19 | 14 | **+0.3136** | [−0.260,+0.724] | +1.14 | 0.333 | 0.111 | 0.2314 | False | False |
| 270 | 9 | pit_dead_retained | pooled | 14/17 | 13 | +0.0432 | [−0.520,+0.580] | +0.14 | 1.000 | 0.125 | 0.8841 | False | True |
| 270 | 9 | pit_dead_retained | xsmean | 13/17 | 12 | **+0.3726** | [−0.256,+0.780] | +1.27 | 0.625 | 0.125 | 0.1934 | False | False |
| 270 | 9 | survivor_only | pooled | 14/17 | 13 | −0.0027 | [−0.553,+0.549] | −0.01 | 1.000 | 0.125 | 0.9930 | False | True |
| 270 | 9 | survivor_only | xsmean | 13/17 | 12 | **+0.3471** | [−0.283,+0.768] | +1.17 | 0.750 | 0.125 | 0.2229 | False | False |
| 300 | 10 | pit_dead_retained | pooled | 12/15 | 11 | −0.3125 | [−0.768,+0.354] | −0.99 | 0.429 | 0.143 | 0.2999 | False | True |
| 300 | 10 | pit_dead_retained | xsmean | 12/15 | 11 | −0.0720 | [−0.644,+0.552] | −0.22 | 0.571 | 0.143 | 0.8166 | False | True |
| 300 | 10 | survivor_only | pooled | 12/15 | 11 | −0.3420 | [−0.782,+0.324] | −1.09 | 0.429 | 0.143 | 0.2559 | False | True |
| 300 | 10 | survivor_only | xsmean | 12/15 | 11 | −0.0358 | [−0.622,+0.576] | −0.11 | 0.714 | 0.143 | 0.9055 | False | True |
| 330 | 11 | pit_dead_retained | pooled | 12/14 | 11 | −0.0397 | [−0.625,+0.574] | −0.12 | 1.000 | 0.143 | 0.8916 | False | True |
| 330 | 11 | pit_dead_retained | xsmean | 11/14 | 10 | **+0.4250** | [−0.279,+0.832] | +1.33 | 0.429 | 0.143 | 0.1944 | False | False |
| 330 | 11 | survivor_only | pooled | 12/14 | 11 | +0.0007 | [−0.599,+0.600] | +0.00 | 1.000 | 0.143 | 0.9985 | False | True |
| 330 | 11 | survivor_only | xsmean | 11/14 | 10 | **+0.4703** | [−0.226,+0.849] | +1.51 | 0.429 | 0.143 | 0.1479 | False | False |
| 360 | 12 | pit_dead_retained | pooled | 10/12 | 9* | −0.2251 | [−0.774,+0.516] | −0.61 | 0.500 | 0.167 | 0.4958 | False | True |
| 360 | 12 | pit_dead_retained | xsmean | 9/12 | 8* | +0.0280 | [−0.690,+0.719] | +0.07 | 1.000 | 0.167 | 0.9280 | False | True |
| 360 | 12 | survivor_only | pooled | 10/12 | 9* | −0.2254 | [−0.774,+0.516] | −0.61 | 0.500 | 0.167 | 0.5142 | False | True |
| 360 | 12 | survivor_only | xsmean | 9/12 | 8* | +0.0575 | [−0.675,+0.732] | +0.14 | 1.000 | 0.167 | 0.8596 | False | True |

`block=30` is excluded and the exclusion is argued from a property fixed in advance, not from
its answer: at block=30 a block holds exactly ONE 30-day sub-boundary, so the numerator variance
is purely cross-sectional dispersion across assets with zero within-block time dimension — a
different estimand, not a shorter-block version of the same one. 360d is the top end because
beyond it the panel yields fewer than ~8 block pairs.

**Revision 3 (J) — the grid is eleven DIFFERENT estimators, not eleven samples of one, and
revision 2 stated the block=30 exclusion as if it were binary.** The same argument that excludes
k=1 outright applies at reduced strength across the low end of the grid. Pooling the numerator
about a single grand mean over k = block_days/30 sub-window groups removes ≈1/k of the
market-factor variance, while the denominator's own small-sample penalty is only
1/(block_days−1). Measured directly, on the sign-flip null where the true VR is exactly 1,
E[VR_block] runs **0.759 → 0.897 across k = 2…12** (full table in the power-ceiling section
below) — biased low, and *rising with k*. So the k=2 cells carry ≈50% and the k=3 cells ≈33% of
the strength of the argument that excludes k=1, and **most of the otherwise-unexplained level
trend across the grid is this bias.** The consequence for R1 is direct: "every cell reads
ρ̄ < 0.10" demands unanimity across **eleven different estimators**.

**Revision 3 (J) — exact block accounting; "non-overlapping BLOCKS" was inexact.** The block
starting at b0 has b1 = b0 + block_days. The denominator takes `dr[b0+1 : b1]` —
*block_days − 1* daily returns. The numerator's sub-windows end at b0+30, …, **b1**, so its last
window ends on the **first day of the next block** and consumes `dr[b1]`, a return no denominator
includes. The numerator therefore covers `block_days` returns against the denominator's
`block_days − 1`, and the blocks are not literally non-overlapping in the daily-return index.
Checked consequence: adjacent blocks' *numerator* sets are still **disjoint** (block b's last
window ends at b1, block b+1's first ends at b1+30), so **this does not inflate ρ̄** — it is an
accounting and wording defect, not a bias. A sample-matched variant that removes it is now
shipped and reported for the level read. Relatedly, and wording only: the **blocks** are disjoint
in that sense, but adjacent **pairs** share a block (pair (b,b+1) and pair (b+1,b+2) both contain
block b+1). The Fisher-z SE 1/√(n_pairs−3) is still the right SE — it is the SE of a correlation
on n_pairs bivariate observations, and pair-sharing is the dependence structure every lag-1
autocorrelation has.

Rolled up by the free parameter that actually moves the answer:

| block | k | pairs | ρ̄ across the 4 cells | all < kill? | CI resolves the 0.10 line? |
|---|---|---|---|---|---|
| 60 | 2 | 50–55 | +0.077 +0.174 +0.015 +0.165 | **False** | no |
| **90** | 3 | 36–37 | −0.288 −0.278 −0.278 −0.273 | True | **yes (below)** |
| 120 | 4 | 27–28 | +0.106 +0.205 +0.126 +0.213 | **False** | no |
| 150 | 5 | 23 | −0.170 −0.166 −0.157 −0.167 | True | no |
| 180 | 6 | 18 | −0.032 +0.077 −0.095 +0.066 | True | no |
| 210 | 7 | 16–17 | −0.126 −0.012 −0.148 +0.042 | True | no |
| 240 | 8 | 14–15 | −0.021 +0.253 −0.029 +0.314 | **False** | no |
| 270 | 9 | 12–13 | +0.043 +0.373 −0.003 +0.347 | **False** | no |
| 300 | 10 | 11 | −0.312 −0.072 −0.342 −0.036 | True | no |
| 330 | 11 | 10–11 | −0.040 +0.425 +0.001 +0.470 | **False** | no |
| 360 | 12 | 8–9 | −0.225 +0.028 −0.225 +0.057 | True | no |

Summary: **32/44 cells below the 0.10 kill line, 12/44 at or above it.** Cells above the line
include 60d/xsmean (55 pairs) and 120d (27–28 pairs) — **more block pairs, i.e. more power, than
revision 1's 18-pair headline.** ρ̄ spans **[−0.342, +0.470]** across the sweep.

### Revision 3 (L) — the independence claim is REFUTED, and the 44→11 deflation goes with it

Revision 2 printed here, and in `p1_variance_ratio_persistence.py:488-492`, that "universe and
estimator move ρ̄ by ≤0.15 within a block length, block length moves it across the whole range",
and used it to deflate 44 cells to 11 independent readings. **That string was hardcoded; the code
never computed it.** It is now computed. Measured within-block-length spreads of ρ̄ (max − min
over the 4 cells at each block length):

| 60d | 90d | 120d | 150d | 180d | 210d | 240d | 270d | 300d | 330d | 360d |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.159 | 0.015 | 0.107 | 0.013 | 0.172 | 0.189 | 0.343 | 0.375 | 0.306 | **0.510** | 0.283 |

**8 of 11 block lengths violate the ≤0.15 claim** (criterion stated explicitly: spread strictly
> 0.15; the review panel quoted 9/11 from this same list of eleven spreads — a one-cell counting
difference that does not touch the refutation), with a **maximum spread of 0.510 at 330d, over
three times the asserted bound.** Worse for the claim's direction: at blocks ≥ 240d the
**estimator axis moves ρ̄ MORE than the block-length axis does** — mean |pooled − xsmean| at a
fixed block length = **0.331** vs mean |one 30-day step in block length| at a fixed
universe/estimator = **0.288**. "Block length is the free parameter that moves the answer" holds
at the short end only. **The claim is withdrawn, and the 44→11 multiplicity deflation that rested
on it is withdrawn with it.**

### The anchor, and the argument for it

**Anchor = block 90d / pit_dead_retained / pooled: ρ̄ = −0.2878, 95% CI [−0.5631, +0.0450],
t = −1.75, 36 pairs.**

**90d is INHERITED**, from `lore/harvest_6b32_appendix/mech_B.md:44,48` — the source panel's own
pre-registration ("cross-sectional persistence of 90d realised trend-efficiency across 30
non-overlapping windows", ρ̄ ≥ 0.10). **180d was TASK_917's own choice, inherited from nothing.**
No statistical-property argument was needed to pick 90d over 180d; inheritance settles it, and
inheritance is the only anchor rule that cannot be reverse-engineered from the answer. For the
record, the properties that would have been argued had inheritance been unavailable also favour
90d over 180d: twice the block pairs (36 vs 18), a proportionately narrower Fisher-z CI
(width 0.61 vs 0.93), and identical non-overlap/tiling status.

**Disclosure of the inheritance's LIMIT** (this matters, and revision 1 could not have made it
because it never read the provenance): what is inherited is the **90-day non-overlapping window
length and the 0.10 threshold**. mech_B's *statistic* was the persistence of 90d realised
**trend-efficiency**, tested against **1000 asset-label permutations** with **NW t ≥ 3**.
TASK_917 re-specified the statistic as market-level pooled VR(30) persistence, for which an
asset-label permutation is undefined (the series is a market-level scalar, one value per block).
That re-specification is the task file's, not the implementer's — but it means the anchor
inherits the *window convention only*, and the null shipped here is weaker than mech_B's.

**Revision 3 addendum on the threshold, not just the window.** The **0.10 kill line itself was
never calibrated to this estimator.** It is inherited from mech_B's *different* statistic — 90d
realised trend-efficiency persistence, 1000 asset-label permutations, NW t ≥ 3 — and nothing in
revision 1 or 2 checked whether 0.10 is even attainable by market-level pooled VR(30) persistence
on 36 block pairs. Measured below: it is not.

### Revision 3 (H) — the TILING ORIGIN: the free parameter revision 2 never disclosed

`_block_boundaries` (`p1_variance_ratio_persistence.py:141-142` in revision 2) hard-wires the
tiling phase to **day 0**. It was never named as a parameter, never swept, and never disclosed.
Sweeping all 90 one-day tiling origins of the **same history** at the anchor block length:

| statistic | value |
|---|---|
| ρ̄ mean over 90 origins | **−0.0687** |
| ρ̄ sd over 90 origins | **0.1662** |
| ρ̄ min / max | **−0.3778 / +0.3125** |
| origins putting ρ̄ **at or above** the 0.10 kill line | **14/90 = 15.6%** |
| the shipped origin (day 0) | **−0.2878**, i.e. **−1.32 sd** into the negative tail |

Coarse 30-day tiling phases, which show the same thing at every block length that admits them:

| block | ρ̄ by 30-day phase | spread |
|---|---|---|
| 90d | −0.288 / +0.042 / −0.039 | 0.330 |
| 180d | −0.032 / −0.132 / −0.321 / −0.249 / −0.276 / −0.040 | 0.289 |
| 270d | +0.043 / +0.096 / −0.199 / −0.016 / +0.146 / +0.043 / −0.417 / −0.574 / −0.435 | **0.720** |

A **single** 30-day origin shift changes the sign or the magnitude wholesale: 60d
+0.077 → **−0.059**, 90d −0.288 → **+0.042**, 120d +0.106 → **−0.169**.

**Consequence, stated as a correction and not a caveat: R2 IS NOT A POWER GUARD.** With
SE(ρ̄) ≈ 0.166 at the anchor, "the 95% CI excludes 0.10" is decided by where the point estimate
happened to land under an undisclosed free parameter. It is a lottery on the tiling phase, not
resolving power. The tiling-phase distribution is now reported wherever a cell is called
resolving.

### Revision 3 (I) — the POWER CEILING (the load-bearing correction)

**Null construction (model-free).** Multiply every **day's** cross-section of daily returns by a
common random sign. This destroys serial structure completely — so the **true variance ratio is
exactly 1** — while preserving every marginal variance *and* the entire cross-sectional
covariance (the market factor, which the k-bias above is a property of, so a per-asset-day flip
would destroy the thing being measured). Nothing is fitted: no market share, no volatility
process, no distributional assumption. 200 draws, seed 9171. The null's 30-day-return builder is
verified byte-identical to the shipped `dp.qday_return_nogap` on the real panel **before any draw
is taken**, and the run aborts if it is not.

| block | k | E[VR_block] \| true VR = 1 | obs sd(VR_block) | null sd(VR_block) | upper-tail p | implied signal-var share |
|---|---|---|---|---|---|---|
| 60 | 2 | 0.7589 | 0.3786 | 0.3620 | 0.363 | 0.048 |
| 90 | 3 | 0.8277 | 0.4552 | 0.3639 | 0.124 | 0.331 |
| 120 | 4 | 0.8536 | 0.3901 | 0.3280 | 0.204 | 0.260 |
| 150 | 5 | 0.8521 | 0.3783 | 0.3227 | 0.179 | 0.241 |
| 180 | 6 | 0.8656 | 0.3795 | 0.2949 | 0.129 | 0.359 |
| 210 | 7 | 0.8741 | 0.3314 | 0.2922 | 0.209 | 0.180 |
| 240 | 8 | 0.8659 | 0.2991 | 0.2963 | 0.403 | **0.000** |
| 270 | 9 | 0.8989 | 0.2549 | 0.2611 | 0.493 | **0.000** |
| 300 | 10 | 0.8647 | 0.3642 | 0.2757 | 0.095 | 0.396 |
| 330 | 11 | 0.8749 | 0.2919 | 0.2426 | 0.189 | 0.263 |
| 360 | 12 | 0.8966 | 0.2749 | 0.2396 | 0.284 | 0.163 |

**The observed block-VR dispersion is indistinguishable from pure estimator noise at all 11 block
lengths** (upper-tail p ∈ [0.095, 0.493]; implied signal-variance share 0.00–0.40, and exactly
zero at k = 8 and k = 9). The review panel's independent draws gave p = 0.387, 0.124, 0.182,
0.206, 0.134, 0.275, 0.449, 0.543, 0.112, 0.216, 0.295 and a share range of 0.00–0.36 — the same
picture from a different seed.

ρ̄ is **attenuated** by that share: an observed ρ̄ is at most the share times the true one. With
a share of at most ≈0.4 anywhere on the grid, **the largest |ρ̄| this instrument could report
against a *perfectly persistent* true VR is ≈0.4** — against a Fisher-z SE of **0.14** (most
pairs) to **0.45** (fewest), and **0.17** at the anchor by its own tiling-origin dispersion. *The
ceiling and the noise are the same size.* (Panel figure: ceiling ~0.3 with SE ≥ 0.14. Either way
the conclusion is identical and it is not close.) And the anchor's power to separate
ρ̄ = 0.10 from ρ̄ = 0 at α = 0.05 is **8.9% two-sided / 15.6% one-sided** on 36 pairs.

**Honest conclusion: P1 COULD NOT HAVE FIRED A DEFENSIBLE KILL AT ANY BLOCK LENGTH.** The
un-resolvability is a property of the instrument, not a close call about a free parameter.

**Caveat carried against this section, because the review panel raised it against itself:** the
panel's *parametric* simulations used a market variance share w ≈ 0.31, lower than the real panel
implies, so their simulated attenuation is **pessimistic**. The **model-free sign-flip share test
above is the load-bearing evidence**; the parametric numbers are corroboration only.

### Revision 3 (J) — the level-vs-breakeven read was WRONG (numerator/denominator sample mismatch)

`p1:158` took `den_vals = dr[b0+1:b1][dr_fin[...]]` — **every** finite daily return in the block —
while `p1:159-166` took the numerator only from asset-days inside **gap-free 30-day windows**
(same defect at `:365-373` for the whole-sample read). The excluded **3.8%** of denominator
asset-days (gap-adjacent days, listing/delisting edges) are systematically **more volatile**, and
they inflate the **denominator only**, biasing the level low. Revision 2 then compared that
biased level against the 1.024 break-even and reported a shortfall.

| estimator | observed level | its OWN null level (true VR = 1) | z vs its own null |
|---|---|---|---|
| shipped (denominator = all finite daily returns) | **0.9043** (n_num 15754, n_den 491401) | 0.9602 ± 0.073 | **−0.76** |
| sample-matched (denominator = asset-days feeding a valid numerator window) | **0.9398** (n_num 15754, n_den 472620) | 0.9978 ± 0.076 | **−0.76** |

Panel's independent figures: shipped null 0.9554 ± 0.073, matched null 0.9928 ± 0.076,
z = −0.70. **The corrected read: the observed level is ≈0.7–0.8 sd from a RANDOM WALK, not a
shortfall against the 1.024 break-even.** Revision 2's "0.9043 vs 1.024" compared a
sample-mismatched estimate against an uncalibrated bar.

**Rule adopted:** the estimator's own null level is now printed **whenever** the level is compared
to breakeven. A level read against 1.0 or against 1.024 without it is not interpretable. Note
that the shipped 44-cell ρ̄ sweep is **deliberately left on the unmatched estimator** so the
panel-reproduced table above stays bit-identical; the mismatch is a *level* bias that a
correlation is nearly invariant to, and the matched variant is reported for the level read where
it actually bites.

### The placebo, and the minimum achievable p

Revision 1's circular-shift null ran shifts `s = 1..N−1` and **contained bit-identical copies of
the statistic under test**:

- `s=1`: `roll(vr, −1)` pairs block *b* with *b+1* — the observed ρ̄ itself. Confirmed
  `c1 == rho_bar` → `True`.
- `s=N−1`: `roll(vr, −(N−1))` pairs *b* with *b−1* — the same unordered pair set, a second copy.
- `c(s) == c(N−s)` for all *s*, so every remaining draw was a duplicate: the headline cell's
  "24-value null" held only ~12 distinct values, 2 of them self-copies.

Consequence: **minimum achievable p ≈ 2/24 ≈ 0.08**, so **no cell could ever have reached
Bonferroni α = 0.00625**, and revision 1's "none Bonferroni-significant" was **guaranteed by
construction** — circular, not evidence. Revision 2:

- **Circular-shift null**: degenerate shifts excluded (`s = 0`, `s = 1`, `s = N−1`, and every
  `s > N/2` mirror). Kept: distinct lags `s = 2..⌊N/2⌋`. Draws = **5** (at 360d, where N = 12, so
  ⌊12/2⌋ − 1 = 5 — revision 2's text said 4, which contradicted its own `circ_pmin` of
  0.167 = 1/6; the table was right and the prose was wrong) to 37 (at 60d); plus-one two-sided p.
  **Minimum achievable p per cell is printed in the `circ_pmin` column above: 0.026 at 60d, 0.040
  at the anchor, up to 0.167 at 360d.** Every one of those is **above** the Bonferroni line
  (0.00114), so **no multiplicity claim is made against this null** and the "not
  Bonferroni-significant" language is dropped rather than restated.
  **Revision 3, and this is why `circ_p` is no longer tabled as if it were comparable to the
  observed statistic:** `circular_shift_null` recomputes the finite mask per shift
  (`p1:258`), so a rolled series pairs a *different and smaller* set of blocks than ρ̄ does —
  measured, the anchor's circular draws use a **median of 23 pairs against the observed 36**, and
  180d **13 against 18**. A null draw computed on ~2/3 of the pairs has a wider sampling
  distribution than the statistic it is compared with, which biases the comparison toward "not
  significant" for a reason unrelated to persistence. This is **not verdict-bearing** (no cell's
  reading turns on `circ_p`), but the comparison is not a fair one and is now flagged per cell
  (`circ_median_n_pairs`, `circ_n_pairs_comparable`). `permutation_null` (`p1:271-277`) does
  **not** have this defect — it holds the NaN pattern fixed, so every draw carries the identical
  `n_pairs` — which is the second reason it is the only null carrying a multiplicity claim.
- **Permutation null** (added, and the informative one): 2000 random reorderings of the finite
  block VRs among the finite block positions, NaN pattern held fixed so every draw carries the
  identical `n_pairs`, ρ̄ recomputed identically. **Minimum achievable p = 1/2001 = 0.00050**,
  which *is* below Bonferroni α = 0.00114, so the multiplicity claim is made here or nowhere.
  Result: **0/44 cells are Bonferroni-significant, and 0/44 reach even unadjusted p < 0.05.**
  The anchor's own permutation p is **0.0665** — ρ̄ = −0.2878 is not distinguishable from a
  random reordering of the same block VRs.

### Restated verdict

Pre-declared decision rule (in `p1_variance_ratio_persistence.py`'s docstring, argued from
decision cost and estimator power, never from the answer):

- **R1 — unanimity guard on a destructive verdict.** P1's kill does not fail a candidate, it
  permanently closes a family (gated trend/reversal, CPD switches, XS-AUTOCORR books). A reading
  that reverses under a free parameter with no principled setting is evidence about the
  parameter, not the venue. The kill fires only if **every** swept cell reads ρ̄ < 0.10.
  → **FAIL** (32/44) — **but see (K): this failure is uninformative and revision 2 reported it as
  though the data had produced it.**
- **R2 — power guard *as claimed*; not one in fact.** The anchor's Fisher-z 95% CI must exclude
  the 0.10 line. → **PASS** (anchor CI [−0.563, +0.045], upper end below 0.10) — **but see (H):
  this "PASS" is a tiling-phase lottery outcome, not resolving power.**

**VERDICT: INCONCLUSIVE — the KILL DOES NOT FIRE. No regime-gating idea is closed by this
instrument.**

### Revision 3 (H) — the pro-kill counterfactual is WITHDRAWN, and here is its refutation

Revision 2 printed, at `p1_variance_ratio_persistence.py:527-531` and here: "read on the anchor
cell alone, the kill condition IS met and IS resolvable — ρ̄ = −0.2878 with a CI whose upper end
(+0.045) lies below 0.10, and all four 90d cells agree to within 0.02. That is the strongest
pro-kill reading in the pack." **That claim is refuted, not softened:**

1. **The four 90d cells agree because they share one tiling origin**, hard-wired to day 0 at
   `_block_boundaries` and never disclosed as a free parameter. Their agreement measures the
   *universe* and *estimator* axes and says nothing about the axis that moves the answer.
2. Across all 90 tiling origins the anchor's ρ̄ has **mean −0.0687, sd 0.1652–0.1662, range
   [−0.3778, +0.3125], and 15.6% of origins land at or above the kill line**; the shipped origin
   sits **1.33 sd into the negative tail**.
3. Therefore **R2 is not a power guard.** With SE ≈ 0.167, "the CI excludes 0.10" is a lottery on
   where the point estimate landed. **There was no resolvable pro-kill reading in the pack.**

### Why the kill is withheld, restated on the corrected evidence

1. **The instrument lacked the power to fire it at any block length** (I). The sign-flip null
   cannot distinguish the observed block-VR dispersion from estimator noise anywhere on the grid;
   attainable |ρ̄| is capped at ≈0.4 against an SE of the same order, and the anchor's power to
   separate 0.10 from 0 is ≈9%. This, not the parameter sensitivity, is the reason.
2. The anchor's "resolving" CI is a **tiling-phase artefact** (H).
3. The anchor's own permutation null does not reject a random reordering of the same block VRs
   even unadjusted (p = 0.0665), let alone at Bonferroni 0.00114.
4. The **0.10 threshold was inherited from a different statistic** (mech_B's 90d
   trend-efficiency persistence, 1000 asset-label permutations, NW t ≥ 3) and **never calibrated
   to this estimator**.
5. **R1's failure is uninformative** (K, next).

### Revision 3 (K) — R1's failure carries no information, and 10 of its 12 vetoes are `xsmean`

Using this artifact's **own** Fisher-z SEs, on a venue **exactly as dead as the kill asserts**
(ρ = 0): the per-cell probability of reading ρ̂ ≥ 0.10 by chance alone runs **0.235 → 0.411**
across the 44 cells (the panel's figures: 0.246 → 0.403). Hence

- **P(R1 unanimity | ρ = 0) = 0.0094** on the conservative 11-reading dimension (7.3 × 10⁻⁹ if
  all 44 cells are treated as independent — the conservative figure is the headline because the
  cells share data and are positively correlated),
- **P(R2 resolves below | ρ = 0) = 0.083**,
- conjunction **≤ 0.0094 (0.94%)** — bounded by the smaller, because R1 and R2 are *not*
  independent (the anchor is one of R1's own cells).

**A dead venue fails R1 about 99.1% of the time.** So "32/44 below → FAIL" is what a dead venue
looks like; it is not evidence against one. R1 is **kept** — it is the correct *decision* rule
for a destructive verdict — but it is a decision rule, **not evidence**, and revision 2's
"→ FAIL (32/44)" wording implied the data had produced it.

And the **composition** of the 12 vetoing cells matters:

- **10 of 12 are `xsmean`**, an estimator this module's own docstring (`p1:177`) marks
  "multiplicity grid only" and which **TASK_917:20-21 explicitly excludes from resolving the
  sign** ("only market-level pooling can resolve the sign; do not report per-asset verdicts").
- **12 of 12** have a 95% CI containing **both 0.10 and 0** — not one of them resolves the line
  it vetoes.
- Under R2's own CI standard: **0/44 cells resolve the 0.10 line from above, 4/44 from below.**

### Revision 3 (M) — the k-parity question is SETTLED = sampling noise

Revision 2 listed as its fourth reason that "ρ̄ alternates sign with k (k=2 +, k=3 −, k=4 +,
k=5 −). This pattern is **unexplained** … flagged as an open instrument concern", and elsewhere
that the estimator "may be unsound at every block length". **That framing is withdrawn.**

White-noise ground truth (review panel: 2000 seeds, the real NaN mask, T = 4657 / A = 398) returns
ρ̄ means of **−0.020 … −0.093, monotone in block length, with zero parity structure** — textbook
−1/n_pairs bias. The null sd **matches this module's Fisher SE to 2 dp** (at 90d: 0.168 vs
0.174), so **the CIs printed above are honest.** The pattern does not replicate under any
perturbation: the origin shifts measured in (H) break it (bd=60/90/120 at origin 30 →
−0.059 / +0.042 / −0.169), Q = 20 and Q = 45 give different sequences, and non-multiple-of-30
block lengths scatter. Pattern statistics against the sign-flip null: **parity contrast p = 0.61,
range p = 0.89.** The estimator **recovers correct signs on ground truth; it is noisy, not
broken.**

**Symmetrically, and equally load-bearing: no persistence is claimed either.** **0/44 cells**
exclude the kill line on the *above* side — not one cell in the sweep is evidence *for*
persistence. The honest reading is **UNDERPOWERED BY CONSTRUCTION**: with 8–55 block pairs the
Fisher-z standard error 1/√(n−3) runs from **0.139** (55 pairs) to **0.447** (8 pairs), so a 0.10
threshold is inside the noise almost everywhere — and (I) shows that is not merely a small-sample
accident but a ceiling the estimator cannot get above at any block length. **P1 as specified
could not have resolved the question either way at this panel length.** What a future row would
need is stated plainly: either a statistic with a finer time resolution than one scalar per
90-day block (the non-overlap requirement is what caps `n_pairs` at ~36), or mech_B's original
per-asset formulation with its 1000-draw asset-label permutation null — which has an *N* the
market-level scalar version can never have.

Context statistics, still labelled context, never verdicts:

- Whole-sample pooled VR(30) level: see **(J)** above. The shipped read is **0.9043**; the
  sample-matched read is **0.9398**; both are ≈0.76 sd from their **own** null level (0.9602 and
  0.9978 respectively), i.e. **from a random walk — not a shortfall against the 1.024
  break-even.** Revision 2's naked "0.9043 vs 1.024" line is withdrawn.
- Per-asset whole-life cross-sectional sd(VR(30)−1) = **0.3268** (394 assets) — inside the task
  file's pre-registered 0.16–0.46 expectation. Reported as CONTEXT ONLY, per the task file's
  explicit instruction that per-asset verdicts are unusable. **Revision 3 caveat, added because
  this number is quoted as corroborating the methodology:** it inherits a *per-asset* version of
  the k-bias. A contributing asset needs only **8** sub-windows (measured: **median 40, 10th
  percentile 16**), so each per-asset VR is biased low by an asset-specific amount and the
  cross-asset sd carries cross-asset **bias** dispersion on top of genuine dispersion in VR. It
  is context with a stated defect, and it corroborates less than revision 2 implied.

---

## P2 — post-shock reversal-slope κ — **KILL FIRED** (verdict unchanged through revisions 2 and 3; the placebo is corrected and demoted to descriptive)

**Nothing in the estimator, the entry timing or the fee denomination changed.** Entry
(`entry_open = logopen[t+1,i]`, the first traded print after the event, never the shock-day
close) and the fee denomination (one-sided bp/side × 4 fills, using 916's real
`trader.backtest.BORROW_RATE_SCENARIOS`) were both independently verified correct and were
deliberately left alone. κ₃ is bit-for-bit what revision 1 reported.

Event definition (|z| ≥ 2.5 AND range/volume ≥ 2× trailing median, all standardisers trailing
and leak-free) fires on 14978 (t,i) pairs → 14959 usable on 2195 distinct dates over 12.75 yr =
172.2 dates/yr if counted as "any date with ≥1 qualifying event" — ~12× the task file's
"10–15/yr expected". Calibrated against the panel's own per-date event-count distribution,
requiring **≥15 SIMULTANEOUS** events on a date ("cascade cohort") yields **167 dates / 12.75 yr
= 13.1/yr**, inside the pre-registered band. The any-event-date reading is carried alongside as
a sensitivity disclosure throughout.

Hierarchical-Bayes pooled slope (DerSimonian–Laird random effects, empirical-Bayes closed form —
disclosed simplification vs full MCMC, appropriate to the ~2 CPU-h budget), cascade cohorts:

| h (days) | κ_pop | 95% CI | cohorts | events | any-event-date read |
|---|---|---|---|---|---|
| 1 | +0.0169 | [−0.024, +0.058] | 167 | 7357 | +0.0785 (2194 cohorts) |
| **3 (headline)** | **+0.0361** | **[−0.022, +0.094]** | 167 | 7360 | +0.1229 (2192 cohorts) |
| 5 | +0.0132 | [−0.055, +0.081] | 167 | 7354 | +0.1437 (2187 cohorts) |

### Cascade-threshold sweep (preserved — this is the evidence the verdict rests on)

The cohort definition has exactly one hand-set parameter. Swept 8 → 40 at h=3:

| min events/date | cohorts | cohorts/yr | events | κ₃ | 95% CI | reaches 0.18? |
|---|---|---|---|---|---|---|
| 8 | 430 | 33.8 | 10026 | +0.0845 | **[+0.048, +0.121]** | False |
| 10 | 302 | 23.8 | 8938 | +0.0492 | **[+0.006, +0.092]** | False |
| 12 | 226 | 17.8 | 8140 | +0.0307 | [−0.019, +0.080] | False |
| **15 (headline)** | 167 | 13.1 | 7366 | +0.0361 | [−0.022, +0.094] | False |
| 20 | 112 | 8.8 | 6461 | +0.0121 | [−0.058, +0.082] | False |
| 25 | 89 | 7.0 | 5960 | +0.0042 | [−0.075, +0.083] | False |
| 30 | 73 | 5.7 | 5525 | +0.0117 | [−0.079, +0.102] | False |
| 40 | 56 | 4.4 | 4937 | −0.0271 | [−0.133, +0.079] | False |

**Sweep max κ₃ = +0.0845, at the most permissive threshold. κ never approaches the 0.18–0.20
kill line anywhere.** The two most permissive settings are exactly the ones whose CI sits
entirely above zero — i.e. what the data contains is a **real but ~2–5× too small** reversal, not
a near miss that a threshold choice suppressed. The threshold that was chosen (15) is not
carrying the verdict.

### Placebo (added in revision 2; corrected and demoted to DESCRIPTIVE in revision 3)

**Design: circular event-date shift, returns held fixed.** Every event keeps its asset, its shock
magnitude x = |shock|, its direction s, and its **cohort membership**; only the date at which the
reversal is measured moves, by a common circular offset δ per draw, and y is re-measured at the
shifted date through the identical first-traded-print convention. Nothing is resampled or
simulated. Degenerate offsets are excluded — |δ| < 60 days from either end of the 4653-day
circular span — because a small shift leaves the placebo window inside the real shock's own
volatility cluster (the same degeneracy class as P1's s=1 / s=N−1 self-copies). 200 draws,
seed 917. The vectorised cohort estimator used for the draws is cross-checked against the
original `hierarchical_kappa` on the true assignment at run time: Δ = 1.39e−17.

### Revision 3 (C) — the placebo's draws were a SAMPLE-SIZE MIXTURE, not a null

`p2_post_shock_reversal.py:286-292` relocates each event with `t2 = (t + δ) % span`. The panel is
deliberately **raw and un-ffilled**, so a relocated event frequently lands on a date where its
asset had **no print at all**, and `m = np.isfinite(y)` then silently discards it. Measured over
revision 2's own 200 shipped draws:

| | min | median | max | observed |
|---|---|---|---|---|
| events per draw | **356** | **2124** | 7174 | **7360** |
| cohorts per draw | **82** | **139** | — | **167** |

**68% of draws use fewer than half the observed events**, and the `m.sum() < 50` floor happily
admits a draw at **4.8%** of observed count. A draw with a fifth of the events has a much wider
sampling distribution, so revision 2's placebo sd measured the *mixture*, not the estimator's
null.

**Correction: the operative placebo is now SIZE-MATCHED** — a draw counts only if it retains
≥ 90% of **both** the observed event count and the observed cohort count. The raw mixture is
still reported, labelled as the mixture it is.

| | raw mixture (revision 2) | **size-matched (operative)** |
|---|---|---|
| draws | 200 | **12 / 200** |
| mean | −0.0056 | +0.0048 |
| **sd** | 0.0485 | **0.0311** |
| max\|κ\| | 0.2057 | 0.0768 |
| **p** (two-sided, plus-one) | 0.4627 | **0.2308** |
| **minimum achievable p** | 0.0050 | **1/13 = 0.077** |
| draws reaching the 0.18 kill line | 1 | **0** |

**WITHDRAWN CLAIM.** Revision 2 stated: "1/200 placebo draws itself reaches the 0.18 line
(max 0.2057), so the null distribution of this estimator has ~0.5% mass at or above the kill
line." **That is untrue.** The draw in question is δ = 1812, computed on **426 events and 102
cohorts** — 5.8% of the observed event count. **Size-matched, 0/12 draws reach 0.18.** The claim
is deleted rather than qualified.

**Revision 3 (D) — the positive corollary, which is the useful half.** The size-matched placebo
sd is **0.0311** against the DerSimonian–Laird **analytic SE of 0.0296** — ratio **1.05**. The
analytic SE that carries the headline CI is therefore **validated by simulation**, and the
apparent excess dispersion in revision 2's placebo was **entirely** the sample-size mixture.

**Revision 3 (E) — the 60-day volatility-cluster exclusion is IDLE, not load-bearing.** Revision 2
justified `MIN_PLACEBO_SHIFT = 60` by analogy to P1's s=1 self-copies. Measured rather than
argued: κ₃ just outside it is **d=60: +0.0011, d=65: +0.0010, d=70: +0.0060, d=80: −0.0024**, and
the 55 offsets *inside* it (d = 1…55) have mean **+0.0019**, max |κ| 0.0805, and **none** reaching
0.18 — no inflation toward the observed value and no decay structure. The exclusion is harmless
but does no work. **The offset class that genuinely breaks this null is LIFETIME RELOCATION** —
moving an event to a date outside its asset's traded life — which is exactly what the size match
screens.

**Revision 3 (F) — the placebo is DESCRIPTIVE, not a gate, and is no longer counted as one.**
Both branches of the reading at `p2:455-464` conclude the kill stands: if the placebo cannot
distinguish κ ≈ 0.036 from what the estimator returns on arbitrary date sets, there is not even a
small event-specific effect to scale up; if it *can*, the effect is real but ~5× below the 0.18
line and below every fee tier. **A test whose every outcome gives the same verdict is
descriptive.** It is not tallied as an independent agreeing channel below. The size-matched
design reinforces this from the other side: its **minimum achievable p is 0.077**, so it could
not have produced a significant result even in principle.

**Revision 3 (G) — the twin-estimator check now REFUSES.** `p2:428` printed `MISMATCH` and carried
on, which is fail-open: the vectorised cohort estimator is licensed to produce the placebo draws
*only* because it reproduces `hierarchical_kappa` exactly, so a mismatch means every placebo
number would come from a different estimator than the headline it is compared against. It now
aborts. (On this run: Δ = 1.39e−17, OK.)

**Reading, unchanged in substance: κ ≈ 0.036 is not distinguishable from what this estimator
returns on arbitrary date sets of the same size and cohort shape.** It is generic post-large-move
behaviour, not something specific to the cascade-event definition — which strengthens the kill
rather than weakening it. What is *deleted* is the "~0.5% mass at the kill line" caveat, which
was an artefact of the mixture.

### Cost side (unchanged)

Bounce gap (close-entry vs first-print entry, per liquidity decile, h=3): **+0.0177** in the
lowest-liquidity decile decaying to ~0.0000 by the top decile — same sign and shape as the task
file's Cox–Peterson citation, and materially smaller than the any-event-date reading because
cascade-date assets skew more liquid.

Honest hedge round trip (4 fills: shocked asset + BTC hedge, entry+exit; real
`BORROW_RATE_SCENARIOS`, **not** legacy21):

| tier | fee×4 | + borrow(low) | + borrow(high) |
|---|---|---|---|
| legacy21 (cross-check only) | 84.0 bp | 102.0 bp | 120.0 bp |
| **base_taker80 (operative)** | 320.0 bp | **338.0 bp** | **356.0 bp** |
| base_maker40 | 160.0 bp | 178.0 bp | 196.0 bp |

Mean |shock| = 1739.8 bp; expected capture = κ_pop × mean|shock| = **+62.7 bp** at h=3 — clears
**no** tier, not even legacy21's 102 bp cross-check bar.

**VERDICT: κ₃ = +0.036 (`0.036056622631943466`, byte-identical to revision 2), CI straddling
zero, on 13.1 cohorts/yr (inside the pre-registered 10–15/yr band); ≤ +0.0845 at every threshold
from 8 to 40; and captured bp below every fee tier. The event-conditional liquidity-provision
family is CLOSED.**

**Three independent channels agree: magnitude, the 8→40 threshold sweep, and the cost
arithmetic.** Revision 2 counted the placebo as a fourth; it is not one, per (F) above — both of
its branches conclude the kill stands, so it discriminates nothing. It is retained as a
*descriptive* result: the small residual κ is not even event-specific.

---

## P3 — post-listing event study — **KILL DID NOT FIRE** (verdict unchanged; placebo DESIGN replaced in revision 3)

398 assets with a real listing row (full PIT panel, dead retained), median active life 1252 days.
Age-conditional IC (age ∈ [0,89] days post-listing, market-residual β=1 demeaning, h=1 forward):
**mean IC = +0.0304, NW-t(lag 10) = +4.15**, n = 946 dates, median 24 assets/date.

### `include_dead` sensitivity (NEW — the spec calls dead-coin retention load-bearing)

| panel | assets | mean IC | NW-t | n_dates |
|---|---|---|---|---|
| dead RETAINED (headline) | 398 | **+0.0304** | **+4.15** | 946 |
| dead EXCLUDED (survivor-only) | 363 | **+0.0279** | **+3.50** | 901 |

Same sign, same order of magnitude, still significant: the finding is **not** an artifact of
retaining dead coins. Retention does carry part of it — dropping dead names costs 8% of the IC
and 16% of the t-stat — which is the direction the spec predicted.

### Revision 3 (1) — revision 2's scaling argument is WRONG for this design, and is replaced

Revision 2 shipped `linspace(0.10, 0.90, 33)` of each asset's own active life as its placebo, and
justified retiring revision 1's "no same-signed draw at ≥50% magnitude" clause with this
argument: *`max|placebo|` grows monotonically with the number of draws B, so an "any draw" clause
becomes strictly harder as B grows and would eventually reject any finite effect.* That is true
of an **i.i.d. null** and **false of the null revision 2 actually shipped.** A deterministic
`linspace` over a bounded interval of **one scalar** does not sample a distribution; it densifies
a fixed curve, so `max|placebo|` **saturates**:

| B (grid size) | max\|placebo IC\| | argmax fraction | reported p = 1/(1+B) |
|---|---|---|---|
| 33 | **0.0181** | 0.875 | 0.0294 |
| 65 | **0.0181** | 0.875 | 0.0152 |
| 129 | **0.0181** | 0.875 | 0.0077 |
| 257 | **0.0181** | 0.875 | 0.0039 |

`max|placebo|` is **flat in B while the reported p falls ~8×**. The p-value was a pure function of
grid density — **a free parameter with no principled setting, the same defect class that made
P1's 180d block length indefensible.** The clause still had to go, but for the stronger reason in
(3) below, and the p-value it was replaced with had to be re-derived from a real null.

### Revision 3 (2) — the operative null is now an INDEPENDENT (i.i.d.) one

**200 draws, each asset drawing its OWN uniform shift fraction in [0.10, 0.90] independently**,
clipped into its admissible per-asset day band:

| | this run (seed 917) | review panel (independent draws) |
|---|---|---|
| mean | +0.0010 | −0.0006 |
| sd | 0.0053 | 0.0055 |
| **max\|IC\|** | **0.0163** | 0.0188 |
| n_ge (draws ≥ \|true IC\|) | **0** | **0** |
| **p** | **0.0050** (the design minimum, 1/201) | 0.0050 |
| parametric z | **+5.53** | +5.63 |
| true IC ÷ null 99th pct \|IC\| | **2.2×** (0.0137) | 2.4× (0.0125) |

**Max-based, quantile-based and parametric rules all pass.** The linspace grid is retained only
as a **labelled diagnostic** (it is what revision 2 shipped, and the saturation evidence lives on
it): its numbers are unchanged — mean +0.0011, sd 0.0074, range [−0.0142, +0.0181], 0/33 draws
at least as large, and the revision-1 fractions 0.25/0.50/0.75 still reproduced and flagged.

### Revision 3 (3) — the degenerate-region exclusion is now a PER-ASSET day-count condition

Revision 2 derived its 0.07/0.93 boundary from 90 days ÷ the **median** 1252-day life. That
argument does not bind, because **the shift is per-asset proportional**: measured,
**138/398 assets (34.7%) live < 720 days**, and for those even the grid's largest fraction
(0.875) leaves the placebo's scored window overlapping **that asset's own true window** — i.e.
the placebo partly re-scores the statistic under test. Measured contamination (share of placebo
scored mass whose *true* age is also in [0,89]):

| shift fraction | self-copy share | cells |
|---|---|---|
| 0.100 | **0.156** | 5543 / 35435 |
| 0.875 (the grid's argmax draw) | **0.093** | 3287 / 35470 |
| 0.900 | **0.158** | 5575 / 35363 |

Three of the top-four |IC| draws are in that group. The operative condition is now, per asset *i*:
**shift_days ≥ 90 AND L_i − shift_days ≥ 90** (shifts are clipped into that band; only 1 listed
asset has L_i < 180 and admits no valid shift at all). **Verified, not asserted: the worst
per-draw self-copy share across all 200 i.i.d. draws is 0.0000**, against 0.09–0.16 on the
diagnostic grid.

**And this is the stronger, previously unstated reason revision 1's clause had to go.** Removing
the self-copy cells drops the grid's argmax draw from **+0.0181 to +0.0167** — so the draw at
"59% of the true IC" that broke that clause **was itself partly a copy of the statistic under
test.** Revision 2 dropped the clause for a reason that was false; the clause deserved dropping
for this one.

**Pass criterion (operative, evaluated on the i.i.d. null): NW-significant AND |true IC| >
max|placebo IC| AND empirical p ≤ 0.05** → True, 0.0304 > 0.0163, p = 0.0050 →
`placebo_clean = True`. Under revision 1's literal clause this run would still read
`same_sign_as_any_placebo = True`, and that reading is printed in the run output rather than
suppressed — now alongside the measurement that shows why the draw producing it does not count.

Age-bucket profile (mean forward residual return/day, market-neutral): age 0–9d **−78.30 bp**,
10–19d −43.84, 20–29d −24.11, 30–44d −12.22, 45–59d −10.71, 60–74d −16.69, 75–89d −1.30 — a
broad-based decaying post-listing UNDERperformance. The **median** is negative in every bucket
too, and re-verified in this revision: −93.93, −75.77, −54.53, −50.67, −55.02, −51.91,
−47.05 bp — so this is not a handful of outliers. (Note the shape difference: the *mean* decays
sharply with age while the *median* is roughly flat at ≈−50 bp, so the steep mean profile in the
first 30 days is partly a tail effect even though the level is not.)

Standing caveat (unchanged in substance, corrected in number): the tail includes extreme
single-day moves that were not separately scrubbed — the worst forward residual return inside
the scored window is **−1.905 in log terms**, and the worst raw daily log return is −1.942
(≈ −85.7% simple) on a dead/failed listing. Revision 1 quoted "~−148% log-return" here; that
figure could not be reproduced against the panel and is replaced by the two measured values
above. The median-based check indicates the finding is broad-based rather than solely
tail-driven, but a fuller robustness pass (winsorising, per-asset weighting) was not run here.

**VERDICT: NO KILL — this instrument survives as a genuine, placebo-clean measurement**, now on a
200-draw **independent** null at the minimum achievable p (revision 2's 33-draw linspace grid is
demoted to a diagnostic, and its p was a function of grid density). The finding
also survived a full external leakage audit: the listing-date artifact was refuted (only 2/398
assets start near panel start, and excluding them changes nothing to 6 decimals), leave-one-out
market is identical, NW lags 0–90 are stable. Per the task's own discipline this is an
**instrument reading, not a promotion claim and not a book design**; any book/controller build on
it is explicitly out of scope for this row.

---

## P4 — hourly residual IC surface — **PREPARED, NOT RUN**

Full hourly-aggregation + skip-entry + stale-tape-screen + descriptive-IC-surface pipeline built
(`p4_hourly_residual_ic_surface.py`); wiring self-test PASSED against the fixture's 90-day minute
slice (T = 2257 hours, 66 IC cells computed, 24 with enough hours, no exceptions, plausible
small-sample shapes). The 816 residual-TSMOM re-read is **stubbed** (`reread_816_best_cell`)
rather than attempted from a summary. NOT invoked against `trading_data.db`.

**Universe accounting, corrected — revision 1's P4 paragraph said "8-asset minute slice" and "10
assets after universe filtering" in the same sentence, which read as a contradiction. Both
numbers are real and here is the reconciliation.** The fixture's minute table holds 11 distinct
`asset_id`s. P4 uses its own **local** `discover_universe()`, which — unlike the
`trader.data.load_db_backtest_panel` path P1–P3 use — applies **no `source_filter="kraken_rest"`**
and matches any `BASE/USD` symbol row in `assets`. It therefore admits the 8 real kraken_rest
crypto assets (ZEC, DASH, ETC, ETH, LTC, REP, XBT, XMR) **plus 2 non-crypto test rows with zero
`ohlcv_daily` coverage**, `TWX/USD` (id 90001) and `HOLE/USD` (id 90003) → **10 panel columns**.
A third test row, `TWXUSD` (id 90002), is dropped only incidentally, because its symbol has no
`/`. So: **8 real assets, carried as a 10-column panel.**

This is harmless in the wiring self-test (the two extra columns are all-NaN and contribute to no
IC cell) but it **must be reconciled before the `--full` run** — either add the source filter to
`discover_universe()` or route P4 through the same canon panel loader P1–P3 use, so P4 measures
the same universe object as the rest of the pack. Left unchanged in this revision because P4 is
prepared-not-run and altering its universe is a behaviour change outside this fix pass's scope;
the divergence is now documented in `discover_universe()`'s docstring and printed by the
self-test.

Estimated real-run cost: low tens of minutes to ~1–2 hours (streaming aggregation over ~254M
minute rows / ~273 assets is the smaller cost; the per-hour Python IC loop over ~65.7k hours × 66
cells is the larger and should be vectorised before a real run) — well inside the task's
≤2 CPU-day budget. Invoke via
`python p4_hourly_residual_ic_surface.py --full --db-path trading_data.db` as a scheduled
server/overnight one-shot; `--estimate-cost` prints the estimate without running anything.

---

## Row-level summary

| instrument | rev 1 verdict | rev 2 verdict | **rev 3 verdict** | what revision 3 changed (arguments only — no verdict moved) |
|---|---|---|---|---|
| P1 variance-ratio persistence | KILL FIRED (closes every regime-gating idea) | INCONCLUSIVE — kill does not fire | **INCONCLUSIVE — kill does not fire (unchanged)** | pro-kill counterfactual **refuted** (tiling origin was an undisclosed free parameter; 15.6% of origins land above the kill line); **power ceiling disclosed — P1 could not have fired a defensible kill at any block length**; level-vs-breakeven read corrected (sample mismatch, own-null level now printed); R1's failure shown uninformative (P = 0.0094 on a dead venue; 10/12 vetoes are `xsmean`); the ≤0.15 independence claim and the 44→11 deflation **withdrawn**; k-parity **settled = sampling noise** |
| P2 post-shock reversal κ | KILL FIRED | KILL FIRED (placebo-backed) | **KILL FIRED (unchanged; κ₃ byte-identical)** | placebo was a **sample-size mixture**, now size-matched (sd 0.0485 → **0.0311**, p 0.4627 → **0.2308**, 12 usable draws, p_min 0.077); the "~0.5% null mass at the kill line" claim **deleted** (that draw had 426 events / 102 cohorts; size-matched **0/12** reach 0.18); analytic SE **validated** (ratio 1.05); 60-day exclusion shown **idle**; placebo reframed as **descriptive, not a gate** — three channels, not four; twin-estimator check now **refuses** |
| P3 post-listing event study | no kill, placebo-clean | no kill, placebo-clean (33-draw grid) | **no kill, placebo-clean (unchanged)** | revision 2's `max\|placebo\|`-grows-with-B argument **refuted on its own null** (saturates at 0.0181 for B = 33…257 while p falls 8×); **i.i.d. 200-draw null now operative** (max\|IC\| 0.0163, 0/200, p = 0.0050, z = +5.53); degenerate-region exclusion made **per-asset** (contamination 0.09–0.16 → **0.0000**, verified); the argmax draw that broke revision 1's clause was **itself partly a self-copy** (0.0181 → 0.0167) |
| P4 hourly IC surface | prepared, not run | prepared, not run | **prepared, not run (unchanged)** | untouched in revision 3; universe accounting and the pre-`--full` fix stand as written in revision 2 |
