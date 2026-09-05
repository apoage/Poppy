> **STATUS (2026-09-06): SHIPPED (run 53)**
> Legs 1-4 shipped in `trader/`, every flag defaulting to legacy. **G1 (OFF = byte-identical)
> re-derived independently by two lenses**, each proving its own harness non-vacuous by planting
> poison first: `parity 93c7d2f7` · `backtest e5c47eee` · `windowed-2023 9b281243` ·
> `overlay 0f5a055d`. Selftest **12 -> 16 checks**. Three cycles (MAJOR -> MAJOR -> close).
>
> **The number this row existed to produce (TASK_820 gate-2, never written down before):** at
> `--net-bps 21` the ENTIRE overlay Sharpe cost is the **shadow-scale DD throttle**
> (+0.97 -> **+0.61**, 378 half / 53 deep). The 5% per-asset cap and 0.2% band are **inert at
> their configured defaults** — and that is a FINDING, not a wiring bug: the book's measured
> `max|pos|` is **0.03846** over all 1,673 decisions, genuinely below a 5% cap, and both
> mechanisms were proven live by forcing them to bite (cap at 0.03 -> 353/1673 capped,
> 0.967->0.931; band at 0.03 -> 0.967->0.893).
>
> **Two of this task file's own quoted magnitudes did NOT survive measurement** (implementer said
> so rather than building to them): (a) the log-vs-simple bias is quoted here as "≈ +0.3
> Sharpe-equivalent (≈5 bp/day)" but measures **-1.02 bp/day / Sharpe -0.043** on FULL net@8 —
> opposite sign, ~7x smaller (bear window: **-13.09 bp/day / -0.661** BEAR22 regime, or
> **-14.54 / -0.713** calendar-2022 — state which, they differ); (b) "+9-12x/yr" drift turnover is
> denomination-ambiguous — **1.24x as a multiple** (56.6 -> 70.2 ann_turn), **+13.6 as additive
> turns/yr**. **G2 tripwire did NOT fire**: net@21 +0.97 -> +0.92 (delta **-0.046**, inside -0.10),
> so no disclosure obligation attaches to prior 819/898 rows.
>
> **Defects found and closed by the panel, not by the author:** (1) `true_bh_series` had a
> **zero-share fixed point** — `n == 0` at a calendar re-form row zeroed the share vector
> permanently, silently collapsing the series to `log(1e-300)` (reproduced: `bh.min() = -691.25`,
> Sharpe -1.4358). Live on `--source db --include-dead`, the survivorship-honest path this row
> serves. Now REFUSES loudly. (2) Its sibling, found during the fix: `0 * NaN = NaN` for
> zero-quantity holdings, floored by `max(v, 1e-300)` into a plausible number — **absence read as
> a value**. Floor removed. (3) `--overlay` combined with any honesty flag **silently dropped the
> request and printed the mislabeled legacy `B&H` column at exit 0**, opposite in sign to the
> figure requested; `--overlay-only` was accepted without `--overlay`, which would have recorded
> the overlay-OFF Sharpe as leg 4's isolated number. Both now refuse with **exit 2, zero stdout**.
>
> **Tests that should have existed:** there were NONE for `true_bh_series`, `gross_simple_series`,
> `drift_retrim_turnover` or `_isolated_rcfg` — which is why (1) survived to the panel. ST13-ST16
> add them, each proven non-vacuous by planting the defect and watching RED. ST15 encodes the
> `_isolated_rcfg` proof as a **3x3 identity matrix** with an `all` row proving the probes live.
> ST13-16 sit ABOVE the ST8 JAX guard on purpose — below it they would vanish on a JAX-less host.
>
> **Remainders (planner lane, deliberately not taken):** G3 + the four record-fix riders edit
> `lore/` and `experiments/LEDGER.md`. **Two operator decisions seeded:** (i) the DELISTING
> CONVENTION — `_book_value` marks a retired holding at its frozen last price until the next
> annual re-form rather than writing it off; currently **3.55% of terminal book value** (3.77% max,
> 2025-08-02), unbounded for a concentrated or high-mortality window. Now DISCLOSED on every
> true-B&H line rather than silently applied; the alternative is an operator modelling call.
> (ii) wiring the honesty pack under `--overlay` (currently refused) is its own queue row.
>
> **Standing trap recorded:** "`parity` GATE 1a: PASS" is materially WEAKER than byte-identity —
> `PARITY_TARGETS` (`backtest.py:455-461`) checks six metrics and `gross` is not among them, and
> its 0.02 Sharpe tolerance absorbs a 1% fee-arithmetic error at 8 bp. Never substitute one for
> the other.
>
> **Surviving MINOR/NIT, named rather than swept (deciding lens, no BLOCKER/MAJOR):**
> (i) `cli.py:140-148` the delisting-disclosure PRINT PATH has no regression guard — planting a
> no-op there keeps selftest GREEN 16/16 while the convention silently vanishes from output. The
> same "guard cannot reach its own defect" shape this run hit repeatedly. Nothing wrong today;
> the remedy is simply unprotected against a later edit. (ii) `cli.py:91` `--overlay-only all`
> without `--overlay` is accepted silently at exit 0 (cap/dd/band correctly refuse) — the one
> value argparse cannot distinguish from its default; fix is a `default=None` sentinel.
> (iii) `backtest.py:632-633` `phantom_day_max` names a date when the phantom share is identically
> zero. (iv) `cli.py:129-131` the `rep is None` branch returns exit 0; verified UNREACHABLE today
> because canon bails first, latent only.
>
> **Verification weight:** the deciding pass ran a **360-invocation exhaustive enumeration** of the
> flag-combination space (248 refusals / 112 accepted) finding **zero** surviving silent-drops and
> **zero** over-broad refusals, re-planted **15 poisons**, and re-derived every published figure
> from scratch (true-B&H series matched to `maxabs 0.0`). Across three cycles the panel found what
> the authors did not, every time.

# TASK_920 — scorer honesty pack (the four corrections the panels verified)

**Status**: PROPOSED 2026-09-05 (planner; four record corrections surfaced by the tactics panel
and verified in primary files — `lore/fable_tactics_panel_2026-09-04.md` §"Record corrections"
+ magnitudes from `lore/harvest_6b32_appendix/tactics_gaps.md`).
**Size XS/S. Risk: MEDIUM-SENSITIVE surface (`trader/backtest.py`) — therefore strictly
flag-gated, OFF = bit-identical, pins mandatory.**

## The four corrections (receipts re-verified by two panels independently)

1. **The "B&H" column is not buy-and-hold** (`trader/backtest.py:225-227`): it is a
   daily-rebalanced equal-weight book charged ZERO turnover. A **true B&H book (fixed shares,
   annual re-form) has never been computed** — add `--bh-book {legacy-label, true-bh}` (B4).
2. **Log-return summation cannot see a rebalancing premium by construction**
   (`trader/backtest.py:221-223`): add `--accounting {log, simple}`; the bias is ≈ +0.3
   Sharpe-equivalent (≈5 bp/day; 1/N −0.40 → ~−0.1 on simple returns).
3. **Constant-dollar drift re-trims are uncharged** (turn counts target-weight changes only;
   estimate +9–12×/yr ≈ −0.03..−0.04 Sharpe at 21 bp on the +0.97 bar): add
   `--charge-drift-rebalance` with the measured decomposition printed.
4. **RiskConfig default-off rows never recorded** (`trader/config.py:108-130`: 5% per-asset
   cap, shadow-scale DD throttle, 0.2% band): one run each on the frozen trend book — closes
   TASK_820's never-written gate-2 overlay number. Expected FAIL on Sharpe (vol-target cuts
   exposure where the book earns); the yield is the missing number itself.

## Gates

- **G1: OFF = bit-identical** on the pinned trend-book read (SHAs recorded) — non-negotiable,
  this scorer grades every row.
- **G2: drift tripwire (pre-registered)**: if charging drift re-trims moves net@21 by
  > −0.10, every prior 819/898 number gets a dated footnote in LEDGER — disclosed, not re-run.
- **G3**: the two lore corrections land: tactics-panel §"Proposed rows" row-1 reversal-arm
  relabel (7d = time-series, only 21d is cross-sectional — already fixed in TASK_918's
  library), and the AS-correction (0.04–0.05, not 0.3–0.5 Sharpe) added to the tactics file's
  record-corrections list where it was dropped.

## Record-fix riders (notes only, no code)

- **819 TEST B2 is the correct receipt for "adaptive forecaster fails"** (per-regime refit
  lost 4/4 OOS regimes −2.34/−2.41/−1.60/−2.09 vs TSMOM; the commonly cited −0.16 is
  survivor-universe net@8, never run PIT) — cite this, not −0.16, going forward.
- **898's spot-legal variant was flattered by cash drag** (gross de-levered to 43%;
  leverage-matched FULL +0.321 → −1.308) — footnote the 898 row.
- **892's anti-edge concentrates in maintenance re-trims** (−0.060/−0.062/−0.050; entries/
  exits NULL) — the receipt any GP-style book's re-trim channel guard cites.
- **893's leading backward-fill** (7.38% of cells, 2021–23): any new featuriser reading
  `close_d` unprotected gets future-stamped prices — add the assert to the shared panel
  builder used by 915/917/918.

## Out of scope

`eval_vector.py` semantics (separate surface, 882-pinned), re-running dead experiments, any
change to a gate number.
