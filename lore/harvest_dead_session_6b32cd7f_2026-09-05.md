# Harvest of dead session 6b32cd7f ("Trader direction memo") — 2026-09-05

## What this is

Session `6b32cd7f` (opus, interactive, 2026-09-03 → 09-05) ran four research
efforts and died repeatedly to session limits. Three were distilled into lore
before dying; **the fourth — the mechanism-design panel (wf_80f2d139) — was
never synthesized**: the session pulled only 3 of its 12 surviving results
before its final death. This file is that missing synthesis plus a gap-audit of
the three distilled panels (their raw reports contain material the lore files
missed).

Primary artifacts preserved in `lore/harvest_6b32_appendix/`:
- `mech_A.md`, `mech_B.md` — digests of all 12 mechanism-panel results
  (designers + refuters), each re-graded under the margin-rollover fact below.
- `loops_gaps.md`, `tactics_gaps.md` — gap audits vs the existing lore files.
- `mining_gaps.md` — gap audit of the five Sep-3 mining subagents vs the
  direction memo.
- `raw/` — all 34 structured result JSONs + 2 dead-agent partials.

The dark-factory session(s) were NOT touched.

## 0. The late fact that re-prices everything (main session, never folded in)

**Kraken margin rollover is real and the record never charges it.** Shorting
spot goes through margin: opening fee + rollover 0.01–0.02%/4h on BTC,
0.02–0.04%/4h on alts (22–88%/yr on shorted notional) + 2% liquidation fee.
`trader/` has no margin cost; `borrow_rate_annual` defaults to 0.0
(`mcts_trader/env/jax_trading_env.py:115`); LEDGER/DECISIONS/pathway never
mention rollover.

Consequences, now verified against every surviving design:

1. **The benchmark itself is borrow-uncorrected.** The +0.97 net@21 trend book
   is ~50% short in BEAR22. The +0.97/+0.56 bars and every "BEAR-positive" read
   (including 819's BEAR22 +1.10) need re-grading with borrow charged on both
   sides. At alt rates and a 50% short leg the drag is ≈22%/yr ≈ twice the
   whole 21 bp trading-cost stack.
2. **Every continuous market-neutral / β-hedged design dies or is wounded on
   this arithmetic alone**: ARC-GP (borrow drag 1.3–6× its entire learned
   increment), Valerian hourly book (effective round trip on shorts 24–48 bp ⇒
   required IC +30–75%), Comfrey tactic-PnL book (6–12 bp/day borrow ≥ its
   entire required 6–10 bp/day gross edge), XS-AUTOCORR (its own text says
   "dead on financing"; only a long-biased arm survives at 2–3%), HCB fast band
   (0.4–1.8 Sharpe uncharged on a ~50%-short book).
3. **Related cost fact (loops panel, receipted)**: the record's "fee16" does
   not exist on Kraken's current ladder; schedule changed 2026-07-09. Below
   ~$100k/30d the honest taker is 22–38 bp/side — **every net@21 column in the
   record is optimistic**. The operator's actual tier is still unrecorded.

**Action 0 (before any new benchmark read): write down the operator's fee
tier; add a borrow charge to the scorer; restate the +0.97 benchmark net of
borrow; re-grade BEAR22 reads.** No new experiment is interpretable until
this lands.

## 1. Mechanism-design panel — the missing synthesis

Roster (from journals + digests): designers ARC-GP, Comfrey (tactic-PnL
learner), Valerian (hourly residual ladder), SCRR (post-shock liquidity
provision), HCB (hourly native-feature band), XS-AUTOCORR (cross-sectional
autocorrelation regime), ARC (bounded residual on GP anchor); refuters
completed for Valerian, Comfrey, SCRR. **Every completed refutation is
REFUTED; every designer's self-odds are 1–10%.** All die on arithmetic, not
taste — the recurring killers: break-even IC vs 21 bp (+borrow), bid-ask
bounce masquerading as reversal slope, and selection multiplicity.

What survives is measurement instruments, and **four of the six survivors are
TASK_915 appendices** — the panel independently converges on 915 as the next
node:

| Salvage | Cost | Gate it feeds |
|---|---|---|
| SCRR Stage-0: post-shock reversal-slope κ table, **first-traded-print entry** (close-entry is bounce: bias 0.04–0.16 vs 0.08 bar; Cox–Peterson 1994), date-clustered | 2 CPU-h | 915 event-conditional family |
| XS-AUTOCORR P1: cross-sectional VR persistence, zero fitting (needs ρ̄≥0.10; sd[VR(30)] noise 0.16–0.46/asset ⇒ only market-level pooling can resolve; break-even VR(30)≥1.024) | hours | decides any regime gating on this venue |
| Valerian: hourly residual IC surface h∈{1,2,4,6,12,24} with true skip entry + stale-tape screen (repo's only hourly read implies IC_1h≈0.008 vs 0.0145–0.037 required) | ≤2 CPU-d | 915 second panel; closes the untested 1h–6h band |
| Comfrey: gross-residual target cells h∈{7,21,63}, sign-interacted ridge, τ_IC≥5d gate; B0 strategy-momentum baseline (spurious-IC control: cost-in-target learns bookkeeping, IC 0.028–0.05 unless sign-flip placebo run) | ≤1 CPU-d | 915 target/spec lines |
| Tactic-library **oracle-selector hull** over 8 arms, scored with `score_window` (898's), NOT `cell_metrics`; stop if hull < +0.97 | <5 CPU-min, 85% to close | kills/permits ANY selector incl. MuZero |
| HCB: 48-cell IC surface + break-even overlay (own arithmetic: 4h needs gross Sharpe ≈16; no interior optimum in 1–6h) | ≤2 CPU-d | likely closes the hourly band decisively |

Two lore-integrity notes from the refuters: hourly-vs-daily GP saving is 0–4%
(not 22–25% as claimed vs a strawman); and 892's anti-edge concentrates in
**maintenance re-trims** (−0.060/−0.062/−0.050; entries/exits NULL) — the GP
book's re-trim channel needs an explicit guard, and the search-vs-argmax leg
was never measured (`trace.py:635` hard-codes argmax).

## 2. Tactics panel — what the distillation missed

- **The equal-weight blend, not the random selector, is the binding null** for
  the hull test; PASS conjuncts: ≥+0.97 CI lo>0, beats blend AND walk-forward
  best-single by ≥0.20 each, 2026H1>0, flat ≤50%, cadence sign agreement.
  Hindsight-luck bound +0.6–0.8 Sharpe. Block-CI certification first
  (block_len=5 under-covers monthly switches; effective n≈30 decisions). The
  899 ingest is a precondition (else the unseen window shrinks to 24
  decisions).
- **Execution/turnover module is the single most concrete positive-EV item in
  the whole harvest: 60–70% odds of +0.10–0.15 Sharpe on any book.** Stride
  table H=1/2/3/5 → 0.165/0.117/0.0957/0.0746 flips/day; hysteresis
  θ=0.5σ_S kills 40–50% of flips; per-bp slope 0.0085–0.0093; composed
  ceiling H=3 + 12 bp + θ → PIT ≈ +0.70. Unmeasured axes: hour-of-day
  (boundary to 13–17 UTC, zero signal cost), maker spec (post at touch, 24 h
  deadline, cross remainder) gated on 907-B2 median AS(1h) ≤ 2 bp.
- **Scorer fixes with magnitudes**: the log-return summation bias is ≈ +0.3
  Sharpe-equivalent (5 bp/day; 1/N −0.40 → ~−0.1 simple-return); a **true
  buy-and-hold book (fixed shares, annual re-form) has never been computed** —
  add a B4 book. Drift-accounting tripwire: if the fix moves net@21 by >−0.10,
  every prior 819/898 number needs a footnote.
- **Two corrections to the lore file itself**: (a) the 7d reversal leg is
  time-series −sign(7d) equal-weight, NOT cross-sectional (only the 21d tercile
  leg is); (b) the AS-drag correction (0.04–0.05 Sharpe, not 0.3–0.5) was
  dropped from the tactics distillation's record-corrections list — and
  TASK_915's summary still inherits the stale "nonlinear ≤ linear" overclaim.
- CPD caveat: Mason 2024 (read in full) found the CPD variant does NOT beat
  non-CPD — cuts against the kept "rule-based CPD switch".

## 3. Loops panel — what the distillation missed

- **Post-listing event study** (first 60–90 days = one shared-structure regime
  class; direct event-study IC with shifted-listing-date placebo) — CPU
  minutes, absent from 915's F-family. Entirely new angle.
- **Execution rails conflict needs an operator ruling**: mandatory market-cross
  and 15–60-min rests are forbidden in code (`phase2_rails.py:170-178`,
  D65/D78). Also 4 majors ≈ 1.5% of turnover; STEP 0 has only ~13 gap-free 1-h
  windows; a "limit then market after 60 s" baseline already exists
  (`real_trade_config.py:20,23`).
- **The correct receipt for "adaptive forecaster fails" is 819 TEST B2**:
  per-regime-refit ridge lost 4/4 OOS regimes (−2.34/−2.41/−1.60/−2.09 vs
  TSMOM); the commonly cited −0.16 is survivor-universe net@8, never run PIT.
- Multiplicity/selection arithmetic to carry beside every "best cell":
  oracle-null selection inflation ≈ +0.5–0.75 (fires ≥+0.97 30–45% by chance);
  τ power: 30 months → t≥2 needs ρ≥0.27–0.37.
- Placebo hygiene: reflection/time-reversal surrogates FAIL for sign tactics
  (PnL invariant) — correct null is circular shift + per-date permutation.
  Bella skeleton has drift knobs (`diff_teacher/skeleton/types.py:35,50`) —
  pin to zero at R4.
- Two lore claims rebutted in the raws: "re-estimation = re-trims" (892
  measured frozen-policy weight adjustments, no refit) and the continual-row
  detection-window argument (at the corrected σ≈0.06–0.07, w* is 19–100 days —
  adaptation may be above the floor).

## 4. Mining subagents — unexploited observations and data traps

- **806 frozen encoder IC 0.1226 at H=20 min beats linear (0.0598) ~2× — never
  followed up.** Persistence IC −0.0519 (minute reversal) never used either.
- **816 residual TSMOM best cell +1.53 CI[+0.32,+2.88]** — "under-powered
  lead, not a flat null"; mechanism note: turnover ≈1.39/rebalance invariant to
  H (the book fully re-ranks); daily cells gross-negative before cost.
- **898's spot-legal variant was flattered by cash drag**: gross de-levered to
  43%; leverage-matched FULL +0.321 → −1.308.
- Data traps for 915's design: minute panel ends 2026-03-31 (Apr–Aug 2026 not
  API-fetchable — the G-C `eval_vector` leg cannot reach 2026-06; 899 staged,
  RO02 drive fix pending); `ohlcv_daily` is dual-source (`agg1m` vs
  `kraken_rest` from 2024-07-15, 33/64 slots); 893's leading backward fill
  (7.38% of cells) future-stamps any featurizer that reads `close_d`
  unprotected; 907-B2's estimand-1 double-counts.
- Never-proposed anywhere (per the miners): rough volatility, options,
  market-making-as-strategy; the 1.45 GB L2 capture remains unmined as a
  signal source (framing is a pre-registration commitment, not a ruling).
- 836/E4 caveat: reference checkpoints likely have untrained (BC-only) value
  heads — treat eval absolutes as joint-decode artifacts.

## 5. Ranked actions (the harvest's "good ideas", in order)

1. **Cost truth pack** (Action 0): operator's fee tier on record; borrow in the
   scorer; benchmark restated net of borrow; BEAR22 re-grades. Decides whether
   the bar is +0.97 or something lower/harder.
2. **TASK_915 G-A** (the record's named next node) with the panel's appendices
   folded in: gross-residual targets + τ_IC gate + B0 baseline, hourly IC panel
   with skip entry, κ table with first-print entry, VR persistence P1,
   post-listing event study, circular-shift/placebo family, SR0(N)
   multiplicity row.
3. **Tactic-library hull test** (<5 CPU-min, 85% closure): 8 arms, equal-weight
   blend as binding null, `score_window`, block-CI certification, 899 ingest
   precondition. If hull < +0.97, every selector idea incl. MuZero is closed
   cheaply.
4. **Execution/turnover module** (60–70% for +0.10–0.15): H grid + hysteresis +
   accounting tripwire + hour-of-day boundary; maker spec after the 907-B2 AS
   gate; needs the D65/D78 rails ruling first.
5. **Scorer corrections**: B4 true B&H book; simple-return scoring option;
   re-trim channel guard (892 receipt); fix the two lore overclaims and the
   tactics-panel mislabel.
6. **Only then learners**: a selector over K bounded arms (MuZero-shaped) or an
   ARC-class bounded residual — each has pre-registered kill experiments in the
   appendix digests, and each is downstream of 1–3.

## 6. Honest bottom line

Nothing in the dead panels found the edge. Every mechanism died on arithmetic —
mostly the 21 bp stack, now doubled by the uncharged borrow. The record's live
hypotheses after this harvest are unchanged in kind but sharper in number:
(1) the daily conditional-mean IC ceiling (915, still the named node, now with
five cheap appendices aimed at it); (2) second-moment / tactic-PnL
predictability — still never directly measured, and the only surviving designs
touch it through event-conditional and persistence instruments rather than
continuous books; (3) execution/turnover as bounded, bankable Sharpe. The
margin-rollover fact is the most consequential single number surfaced this
week: it doesn't close a lane, it re-prices the scoreboard.
