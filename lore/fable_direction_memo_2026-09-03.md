# Direction memo — what stands on OHLC (consultation, 2026-09-03)

Author: Claude Fable 5.1, consulted read-only by the planner lane. Nothing in the tree was
changed; no run, no benchmark, no server contact. The dirty tree (907 B2 halt state) is as found.

Read first-hand: `CLAUDE.md`, `.claude/rules/*`, `docs/pathway.md`, `experiments/LEDGER.md`
(full, incl. FRAMEWORK EPOCHS preamble and post-table corrections), `tasks/DECISIONS.md` D60–D78,
both 2026-09-02 factory handoffs, `docs/pathfinder_audit_2026-08-19.md` §B–§D,
`lore/notes_2026-09-0{1,2}.md`, `docs/findings/{diversified_trend_shipcandidate,
modeling_surface_exhausted}.md`. Mined by subagents with `file:line` receipts (each receipt
below was either re-read in the primary file or is attributed to the mining pass):
`docs/findings/*` (26 files), `lore/notes_*` 2026-07-05→09-02, `tasks/TASK_8xx/9xx` banners,
`mcts_trader/` feature/env/reward source, `trader/` engine, `runs/819_single_asset/*`.
External: Gârleanu–Pedersen (JF 2013); Grinold–Kahn and Clarke–de Silva–Thorley (transfer
coefficient); Yang–Zhang (2000) range volatility; Levin–Lyons–Ni signature regression and
Futter–Horvath–Wiese "Signature Trading" (2023); Barroso–Santa-Clara (2015) and
Daniel–Moskowitz (2016) volatility-managed momentum plus the crypto follow-ups (2025);
Avellaneda–Lee statistical arbitrage and a 2025 negative crypto replication (abstract-level);
"Cryptocurrency anomalies and economic constraints" (2024, out-of-sample decay; abstract-level).
External sources were read at abstract/summary level only; none is load-bearing for a verdict.

Goalposts adopted verbatim from `docs/pathway.md:9-21`: unseen years, fee16/slip5, ≥3 seeds,
beat the 819 book's net@21 +0.97 (survivor-tilted; PIT +0.56, CI straddling 0), hold
positions, BEAR-positive, then the gate ladder. No softer criterion is used anywhere below.

---

## 0. Bottom line

1. **The record has closed the question it was built to answer, and the answer is negative
   for the current line.** Learned policies on the daily surface: real in-window fit
   (+0.9914, instrument-independent, `882/decisive-legA`), no OOS edge (`875`, 1/5 folds),
   OOS actions worse than a matched coin flip (`892`, Δ −0.043..−0.056, every CI < 0),
   self-play collapses to HOLD *adaptively* at zero cost (`887`), search subtracts (`879`,
   Δ 1.602), distillation subtracts (`879`), a better teacher does not transfer (`861`).
   `pathway.md:37`'s own H1 kill clause ("vector planner still collapses to HOLD with a
   verified action-conditional model") has fired: `834/E3-p1b` PASS + `884`/`887`.
2. **Those failures are one failure.** The line fused prediction and control into a single
   learned object (policy/value) and then could never say which half failed. In a market the
   price path is exogenous; the only controlled state is inventory, whose dynamics are known
   exactly. Tree search over known inventory dynamics adds nothing over closed-form control
   *given the same forecast*. The +5.29/+8.02 "perfect-model search" ceilings (`831/E1`,
   `E5-inversion3`) are the value of a 3-day lookahead, not of search. What the program
   measured, six ways, is that no learned object extracted a conditional-mean signal from
   daily OHLC that survives out of sample.
3. **The single load-bearing doubt is therefore not about models. It is whether daily OHLC on
   this venue carries a conditional-mean signal at all in 2024–2026**, at the information
   coefficient the goalposts require (order 0.02–0.03 per decision, market-residual,
   vol-standardised — §1.3). The record answers this only indirectly: the one direct IC
   ladder (`819 CAPACITY`) is a pooled time-series read on 20 assets with close-only
   features and a noise floor that is the max of four shuffles (§1.1 E-2); the one daily
   signal ever found (slow trend) is measured dead since 2024 (`898`). And the "OHLC-first"
   line has been close-only by construction: the vector encoder receives `highs=lows=close`,
   `volumes=ones`, with the cross-sectional channels inactive (§1.2 A-2). Daily
   predictability has never been measured on the PIT universe, with a market-residual target,
   across horizons, with range or volume in the feature set.
4. **Recommendation, ranked by (odds shift)/(cost to falsify): replace, not continue or
   restructure.** Kill the 828 training lane. Build no more policy machinery until a
   pre-registered *predictability-ceiling instrument* has said whether there is anything to
   plan over. On PASS, the trader is a solved problem: forecast → Gârleanu–Pedersen
   closed-form dynamic trading under the measured cost, graded on the existing `trader/`
   scorer and `eval_vector`, PIT. On FAIL, the operator's OHLC preference should be overruled
   and the memo pre-commits to saying so (§6).
5. **The 907 fill probe does not bear on this call.** Its estimands calibrate minute-horizon
   maker adverse selection; the daily question is decided by IC and turnover, where a 5 bp
   term is second-order to the 21 bp tier (§1.4). Finish it for D65 closure if the operator
   wants the number; do not let it gate the direction call.

---

## 1. Where is the edge? What the record establishes, what it assumes

### 1.1 Established (citable as closed, post-E-c, instrument-named)

- **E-1 Trend was the daily surface's only signal, and it is gone.** 819 Phase 3: PIT 276-column
  panel (273 pass inclusion), dead-coin-inclusive, 2018–2025: net@21 **+0.56 CI[−0.13,+1.23]**; survivorship was worth
  ≈+0.42 Sharpe (`docs/findings/diversified_trend_shipcandidate.md:29-36`). 898 decay
  (net@8): 2021 +1.782 → 2023 +1.068 → **2024 +0.041 → 2025 +0.663 → 2026H1 −1.659**, no
  recent CI excludes 0; spot-legal long-flat variant 2026H1 **−5.221 CI[−7.89,−2.09]**
  (`experiments/LEDGER.md` row `898/trend-decay`; `pathway.md:111-118` struck). 820 gate 2
  on truly unseen 2025-12-13→2026-07-04: net@8 **−1.31** (`TASK_820:107-117`). The book
  is 1/n-equal-weight sign-of-30-day-return, turnover 0.1551/day (`trader/config.py:89-105`,
  `trader/backtest.py:455-462`): a 21 bp tier costs ≈11.9%/yr of gross.
- **E-2 Daily-horizon capacity ladder: null, but a weak instrument.** Pooled daily panel, 20
  assets, strided from the minute cache (`runs/819_single_asset/cap_common.py:57-64`),
  2021-04→2025-08, test 2024-07→2025-08 (426 days × 20): OOS IC ridge −0.010/+0.011 · GBM
  +0.020/+0.037 · MLP +0.036/+0.016 · GRU +0.007/+0.033 at H=1/5 d, every net@21 CI including
  0 (`CAPACITY_RESULT.md` via the mining pass; `_cap_*_log.txt`). Construction, verified in
  the primary files: target = raw forward return standardised by the asset's own train-window
  vol, **not cross-sectionally demeaned** (`cap_common.py:140-141`); features **close-only**
  (lags, realised vol, momentum); IC = **pooled Spearman over flattened (day, asset) rows**
  (`cap_common.py:188`), a time-series read, not the per-date cross-sectional IC that 806/811
  use; the "|0.059| noise floor" is the **max of four single shuffled-label draws**
  (`cap_aggregate.py:46-57`), and the draw that set it tripped the pipeline's own leak
  assertion (`_cap_mlp_log.txt:57`: `IC=-0.0586 (leak-free<0.02: False)`). Verdict as
  history: nothing cleared. Verdict as a ceiling on daily predictability: not established.
- **E-3 The minute surface has a real, linear-optimal, cost-bound signal.** OOS IC
  0.151/0.141/0.060 at H=1/5/20 min, n≈217k strided (`runs/806_predictability_probe/result.md`
  via the mining pass); nonlinear gets 40–61% of linear (`811`); 897: **0/576 attainable-tier
  cells, 0/864 at any nonzero fee, best −5.045**; a flat 5 bp adverse-selection term alone
  takes fee-0 H=1 from +204 to −634 (`LEDGER` row `897/809-corrected`). Mean |forward
  return| at H=1 is 7 bp — smaller than one side of cost. The killer is immediacy.
- **E-4 The learned-policy line on daily bars is closed on evidence** (§0.1). Two of its
  earlier positives were instrument artifacts and were withdrawn (`857/860` bankrupt runs;
  the +0.898 OOS anchor). The only surviving positive learned signal is `836/E4-p1b`
  (value net ρ 0.21–0.30 vs realised k=3 return, n=95, 1/5 windows, pre-E-c, never
  re-measured — `pathfinder_audit:99-103`).
- **E-5 Instrument doctrine.** A number without instrument+epoch+pin is not a result
  (`LEDGER.md:28-41`); the +0.97 bar is survivor-tilted and no vector-line verdict ever ran
  PIT (`pathfinder_audit:71-79`); OOS seed spread on the imprint recipe is 1.0–2.0 vs 0.041
  in-window (`875`), so single-seed OOS numbers are uninterpretable.

### 1.2 Assumed, or measured only in a form that cannot settle it

- **A-1 That the failure lives in learning/search rather than in the signal.** The program's
  design cannot separate them: every 828 verdict is a policy verdict. The D47 gloss ("the
  aggregate exists — the trend teacher harvests it in-window at fee0") is true and
  irrelevant to 2024–2026, where the teacher itself reads ≈0 (E-1). The only direct IC
  measurement (E-2) is small-universe and single-target.
- **A-2 That "OHLC" has been used. It has not, anywhere in the vector line.** The daily
  panel is a 2-D close array built by **striding** the 1-minute closes
  (`prices_np[::bar_minutes]`, `mcts_trader/training/env_factory.py:417-418`; "samples ONE
  MINUTE rather than aggregating the bar", `experiments/835_stage4_eval/eval_vector.py:1738`).
  The encoder is called with `volumes=None, highs=None, lows=None`
  (`mcts_trader/env/jax_encoder.py:986-993`) and substitutes `highs = lows = price_window`,
  `volumes = ones` (`:124-134`); self-play pins `n_channels=27` over that close array
  (`mcts_trader/training/self_play.py:564, :633`), so the cross-sectional channels 27–31
  (ranks, β, idio-vol, xs-momentum) are inactive. The high/low channels are byte-identical to
  close, volume channels are constants. `trader/` aggregates true UTC-midnight daily bars
  (`trader/data.py:137-190`) but its signal reads closes only (`trader/config.py:91`). The one
  line that touched high/low was 813's breakout/ATR rules (dissolved on hold-out); volume was
  dropped at cache build. The one thing OHLC carries that a close series does not — the
  intraday range, a 5–8× more efficient volatility estimator (Parkinson; Garman–Klass;
  Yang–Zhang 2000) — has never entered a daily-surface signal or a sizing rule in this repo
  (Kelly/vol-targeting: zero hits; inverse-vol sizing on the trend book read −0.06,
  `trader/config.py:92-94`). An "OHLC-first" line has been a close-only line.
- **A-3 That BEAR-positive requires a short signal.** A market-neutral book is BEAR-positive
  in expectation if it has any edge; the requirement constrains the *target* (market-residual
  returns), not the model class. No market-neutral daily book has ever been measured: 816's
  "lower frequency to daily" cell kept the **63 lagged one-minute returns as features** and
  only strided the rebalance to H=1440 minutes (`runs/816_operating_point/probe.py:214, :254`);
  its task file asked for real daily/4h bars that were never sourced (`TASK_816:51-53`); the
  "0/54" is a stale printf constant over a 48-cell run (`PHASE1_RESULT.md:9`). TASK_817's
  cointegration/spread-reversion test is DRAFT, never run (`TASK_817:32-34`); TASK_825's
  effective-breadth measurement `(Σλ)²/Σλ²` is PARKED (`TASK_825:52`). Mining pass: zero
  statarb/OU/pairs/PCA/decile code anywhere in-tree.
- **A-4 That per-decision EV is the right unit.** `notes_2026-07-06.md:139-145` states the
  correct diagnosis for 821 — "the edge lives at the portfolio level, below the per-decision
  detection threshold" — and the vector rebuild was meant to fix it. It then trained a
  factored per-asset policy (`sampled_search.py:299-304`; marginal visit targets,
  `self_play.py:728-729`) whose deployable argmax rounds each asset to HOLD unless the
  marginal is sharply state-dependent (`E5-exit2`). The breadth thesis was never given an
  estimator that aggregates across assets *before* thresholding; a cross-sectional IC is
  exactly that estimator.

### 1.3 The load-bearing doubt, stated so it can be killed

> Does the daily OHLC panel on Kraken, point-in-time, carry a market-residual,
> vol-standardised forward-return signal with per-decision cross-sectional IC ≥ ~0.02–0.03 on
> 2024-01→2026-06, at any horizon 1–21 days, from any causal OHLC-derived feature family?

Why that number. Grinold–Kahn: IR ≈ IC · √BR · TC. BR = independent bets per year =
(decisions/yr at horizon h) × N_eff, where N_eff is the participation ratio
(Σλ)²/Σλ² of the residual-return correlation spectrum — crypto's single dominant factor makes
N_eff of 64 assets plausibly 5–15, to be measured, not 64. With h=1 d, N_eff=8: BR≈2900,
√BR≈54. TC under a cost-aware controller ≈0.5–0.7 (Clarke–de Silva–Thorley). Gross IR 1.0
then needs IC ≈ 0.03; the bar of +0.97 *net* at 21 bp with the controller's turnover pushes
it toward 0.03–0.04 at h=1, or a smaller IC at longer h where turnover falls but BR falls
too. The instrument in §4 measures every term of that product on the same panel rather than
assuming any of them. If no cell reaches the IC floor with a Newey–West t ≥ 3 after
multiplicity correction, no model class rescues the daily surface: the ceiling binds
ridge, GBM, signatures, TRM and MuZero alike.

### 1.4 What 907 changes: nothing about this

907's estimands (markout deciles at 1m–1h; touch-conditional fill rate as an optimistic
bound — `experiments/907_live_fill_probe/PREREGISTRATION.md:19-64`) calibrate a maker model
for a ~525k-decision/yr surface that 897 closed with the penalty at zero. A daily book
trades ~0.15/day of gross; at that turnover the difference between 5 bp and 10 bp adverse
selection is 0.155 × 365 × 5 bp ≈ 2.8%/yr ≈ 0.04–0.05 Sharpe-points of drag on a 60–66%-vol
book — real, second-order to whether IC exists. **[CORRECTION 2026-09-05: this sentence
originally read "≈0.3–0.5 Sharpe-points/yr", an arithmetic error ~8× too large, caught by the
execution-loop explorer of the self-learning-loops panel; the qualitative conclusion is
unchanged and is now stronger.]** The probe is worth finishing for the record and for execution *style*
(limit vs market on a daily rebalance), and it is not on the critical path of the direction
call. Its measured finding that BTC/USD's mid is frozen for 46.5% of book EVENTS (38.9% of
capture TIME: 138,449 s of 355,963 s, `handoff_run_50.md:72`; denomination corrected 2026-09-05)
(`notes_2026-09-01.md:17-19`, falsified as real stillness, `handoff_run_50.md:78-94`) is a
useful prior for any limit-order execution layer later.

---

## 2. Structural reading of the MuZero line

Stated once, because it decides "continue vs restructure vs replace":

- MuZero's search earns its cost in **controlled Markov processes** where the action changes
  the transition and the tree discovers consequences the policy prior cannot see. Here the
  price transition is exogenous. The controlled part of the state is inventory, whose
  transition under fees, deadbands and budget is known in closed form
  (`jax_trading_env.py:1621-1820`). Planning over known inventory dynamics with a return
  forecast is a convex (quadratic-cost) or piecewise (proportional-cost, no-trade-band)
  program with textbook solutions; no tree is required and none can beat the closed form on
  the same forecast.
- The record already contains this demonstration. With the true environment as the leaf
  (an R=3 lookahead) search scores +5.29/+8.02; with any learned leaf it scores negative;
  removing every learned component moves it monotonically toward the lookahead number
  (`E5-inversion2`, `E5-inversion3`, `849/leaf` v̂ corr +0.03). The wrapper is inert or
  harmful; the leaf is the forecast; the forecast is the whole question.
- Self-play could not manufacture a forecast because the targets it distils are its own
  visit distribution over a signal it cannot see; 887 shows the fixed point is the correct
  response to a negative-EV action set, at zero cost. Epsilon-mixing makes it trade more
  and lose more, monotonically (A −2.86% / E −9.11% / D −13.84%).

The operator likes MuZero as an algorithm (`notes_2026-08-19.md:17-18`) and D47 leaves the
line alive pending "a new structural hypothesis about reward, data, or edge". The honest
translation: the algorithm is not refuted, it is mis-applied to a problem with no
controllable transition. If it returns, it returns as the *controller* over inventory with a
forecast it did not have to learn — and at that point Gârleanu–Pedersen does the same job in
one matrix equation.

---

## 3. Continue, restructure, or replace

| | Odds of hitting the goalposts (unseen 2024–26, PIT, fee16/slip5, ≥3 seeds, BEAR-positive) | Cost to falsify | Ratio | Verdict |
|---|---|---|---|---|
| **A. Continue as-is** — 828 Stage-3 variants (ExIt from the imprint, K/pw/R sweeps, D-arm mixing) | ≈0–2%. Four independent convergences to the HOLD attractor (preflight, p1b, p1c, ExIt); 887 adaptive collapse at fee0; 892 anti-edge OOS; the record's own H1 kill clause fired. Further variants are parameter nudges by the repo's own rule (`pathway.md:30-31`). | Already paid: ≥3 GPU-days per battery, information now negative. | worst | **Kill the training lane.** Keep env + `eval_vector` as the cost-honest simulator. |
| **B. Restructure around the weakest link** — keep the MuZero shape; replace the learned leaf/targets: value trained as a supervised forward-return forecaster (the `836/E4` signal), proposals from a GP aim portfolio instead of book-dithering, prior weight 0 | 5–10%. Every component change is downstream of the same IC doubt; the wrapper is measured subtractive (`879` Δ1.602) and the deadband/ladder action grammar (±w_unit steps, w_max 6.25%, 16-position cap — `jax_trading_env.py:1623-1648, :2088`) caps expressible books. Re-inherits the survivor-tilted 64-slot frame unless re-cut PIT. | Medium: 1–2 GPU-days per battery plus a PIT frame re-cut. | low | Not first. Only worth doing if C passes and one wants search as the controller for the operator's reasons. |
| **C. Replace** — predict-then-optimise, instrument first: (i) a pre-registered *predictability-ceiling* sweep over target × horizon × feature family × estimator, PIT, by-year, converted through the fundamental law into a net-Sharpe bound under GP-optimal turnover; (ii) on PASS, the Gârleanu–Pedersen book at the winning cell, graded on `trader/` and `eval_vector` | If the ceiling clears: 30–40% to reach the PIT bar +0.56, 15–25% for +0.97 (crypto anomaly literature shows decay; realistic-execution studies cut reversal Sharpe 1.4 → 0.3). Unconditionally ≈10–15%. **But the FAIL branch is decisive** — it ends the OHLC question with the overwhelming case the honesty clause demands, which no amount of A or B can produce. | **CPU-days, zero GPU** (`trader/` scorer and PIT builder exist; ridge/GBM/signatures are library code). Kill date 2 weeks. | best | **Do this.** |

Ranking: C ≫ B > A. C dominates because its information is two-sided: PASS licenses a
cheap, closed-form trader; FAIL retires the daily OHLC substrate on evidence rather than on
fatigue.

Why C is structural, not a knob-turn, under the repo's own rule: it changes the estimation
target (market-residual, vol-standardised forward return instead of a policy), the decision
formalism (closed-form dynamic trading under costs instead of tree search over a learned
model), and the geometry of the inputs (range-based second moments and path signatures
instead of raw channels).

---

## 4. The pick: TASK_9xx — Predictability ceiling of the daily OHLC surface, then the GP book

### 4.1 Goal

Decide, on pre-registered bands, whether daily OHLC on the PIT Kraken universe carries a
conditional-mean signal that can clear the goalposts after costs; if yes, ship the smallest
trader that harvests it.

### 4.2 Mathematics

**Target.** For asset i, date t, horizon h ∈ {1, 3, 7, 21} days:
y_{i,t,h} = (r_{i,t→t+h} − β̂_{i,t} · r_{mkt,t→t+h}) / σ̂_{i,t}, with β̂ from a trailing 90-day
regression on the equal-weight market (causal), and σ̂_{i,t} the **Yang–Zhang** range
estimator on the trailing 30 true daily OHLC bars (σ²_YZ = σ²_overnight + k·σ²_open-close +
(1−k)·σ²_Rogers–Satchell). Absolute-return and close-to-close-σ variants run as control
arms so the memo's A-2/A-3 claims are themselves tested. The residual target makes the
book market-neutral by construction (BEAR-positive in expectation); a separate market-factor
timing arm (single series, low breadth) is reported, never pooled.

**Features (causal, standardised cross-sectionally per date).** F1 lagged returns and
trend/reversal at {1,3,7,21,63} d (includes the 819 signal as a control). F2 range family:
YZ vol, Parkinson/GK ratios, vol-of-vol, range-to-close ratios, HAR-RV terms. F3
cross-sectional: ranks, dispersion, β, idio-vol, dollar-volume rank. F4 truncated
log-signature (level ≤3) of the last 30-day path of (log-close, log-range, log-volume,
time) — a universal linear basis for path functionals (Lyons; Levin–Lyons–Ni), computed
with `iisignature`/`esig`, fitted by ridge; the one genuinely new geometry on the table.

**Estimators.** Ridge (closed form) and one GBM per cell, expanding-window refit each
quarter, strictly causal, purge ≥ h between train end and test start. No neural nets in
the ceiling sweep: 811 found nonlinear at 40–61% of linear on the minute surface; 819
CAPACITY's daily point estimates actually order nonlinear ABOVE linear (GBM +0.037 vs ridge
+0.011 at H=5) but all inside a weak noise band, so the daily evidence is inconclusive, not
"nonlinear ≤ linear" **[CORRECTION 2026-09-05 — this sentence originally claimed both surfaces
showed nonlinear ≤ linear; TASK_915's one-paragraph summary inherited the claim and should be
amended]**. GBM stays in the sweep as the conjunctive/nonlinear estimator; neural nets are
excluded as a pre-registered scope choice (variance and cost at ~550k asset-days), not on
evidence, and the point is the ceiling, not the model.

**Statistic.** Per-date cross-sectional Spearman IC between forecast and y; mean by year
2022…2026H1; Newey–West SE with lag h; Bonferroni over the cell grid (≈4 targets × 4
horizons × 5 feature sets × 2 estimators = 160 cells). Breadth measured, not assumed:
N_eff(year) from the residual correlation spectrum; BR = N_eff · 365/h.

**Conversion.** Gross IR bound = IC · √BR · TC. TC and turnover come from the
**Gârleanu–Pedersen** solution on the same panel: with signal decay φ_h estimated from the
IC autocorrelation, quadratic cost Λ calibrated so that a full rebalance costs 21 bp of
notional, and Σ from Ledoit–Wolf shrinkage of residual returns, the optimal book is
x_t = x_{t−1} + Λ⁻¹A (aim_t − x_{t−1}), aim_t = weighted average of the current and
expected future Markowitz portfolios with slower-decaying signals weighted more. Net Sharpe
bound = gross − turnover·21 bp/σ_book. (For proportional costs the no-trade band of
Davis–Norman is the alternative; GP's quadratic form is used first because it is closed
form and the repo's cost model is linear in turnover, `trader/backtest.py:188-190`.)

### 4.3 Instrument

- New small module `experiments/9xx_ic_ceiling/` (SPEC ≤1 page; runner imports shared
  code; token-size mandate `pathway.md:91-94`). Panel = `trader/data.py` true UTC daily
  aggregation from the minute table (not the stride), PIT universe via the 819 POWERED
  builder (each asset from its own listing, dead coins retained), liquidity filter as 819.
  Daily table extended by TASK_899's staged backfill before the run (`notes_2026-08-31.md:63-67`).
- Grading of the book: unmodified `trader.backtest.cell_metrics` (seed 819, n_boot 2000,
  tiers 8/21) for the PIT engine read, and `eval_vector.py --operator policy-argmax
  --vector-weight-rebalance` with the GP weights quantised to the ±w_unit ladder as the
  in-env read — the same two-instrument agreement discipline as `882/decisive-legA`.
- Epoch: post-E-d, `ohlcv_daily`/minute both current to the 899 seam; the row names its
  epoch and pins the PIT frame per `LEDGER.md:28-41`.
- Compute: CPU only, safe_run-wrapped, on gpu-server if the workstation rule bites;
  ≤2 CPU-days. No GPU.

### 4.4 Gates (binary, pre-registered before the first fit; SPEC sha256 recorded)

- **G-A Signal exists.** ≥1 cell with mean IC ≥ 0.02 on 2024-01→2026-06, NW t ≥ 3 after
  Bonferroni, and IC > 0 in each of 2024, 2025, 2026H1 separately.
- **G-B Ceiling clears.** For some G-A cell, the fundamental-law net-Sharpe bound at 21 bp
  ≥ +0.97 on 2024–2026H1; record MARGINAL if ≥ +0.56 only.
- **G-C Control realises it.** The GP book at that cell, PIT, 2024-01→2026-06, both
  instruments agreeing within 0.1: net@21 ≥ +0.97 with bootstrap CI lo > 0, positive on
  BEAR22 (in-sample, form check) and on 2026H1 (unseen), holds positions (gross ≥ 0.5
  median, no >95%-flat days), ≥3 seeds where randomness enters (GBM seeds; feature
  bootstrap for ridge). Passing G-C hands the book to the existing gate ladder
  (`docs/findings/deployment_gate_ladder.md`) — nothing promotes past it here.

### 4.5 Kill criteria and pre-committed consequences

- **G-A fails in every cell** → the daily OHLC surface has no measurable conditional-mean
  signal in 2024–2026. Consequence, pre-committed: the OHLC-only constraint (D4) is
  recommended for repeal; the next row is a *data* row (perpetual funding/basis, on-chain
  flow, or the operator's news-LLM hybrid — none is OHLC), not a model row. No further
  policy machinery on daily bars under any architecture, MuZero included.
- **G-A passes, G-B fails** → edge exists but breadth × cost caps it below the bar; record
  the measured IC/N_eff/turnover triple and stop the trader build. The only remaining
  lever is breadth (more venues/assets) or cost tier, both operator decisions with numbers.
- **G-B passes, G-C fails** → the controller, not the signal, is at fault; that is the first
  legitimate re-entry point for a search-based controller (Option B) — with the forecast
  already measured, so the comparison is clean.
- Kill date: 14 days from SPEC commit. No metric shopping: the cell grid is fixed in the
  SPEC; anything added later is a new row.

### 4.6 What to steal, and from where

- Gârleanu–Pedersen 2013: aim portfolio, partial trading, decay-weighted signals; their
  commodity-futures implementation is the template (replace futures with the PIT spot panel,
  quadratic cost calibrated to the 21 bp tier).
- Yang–Zhang 2000 for σ̂; HAR-RV (Corsi) for the vol features. This is the OHLC-native
  content the line never used.
- Levin–Lyons–Ni signature regression; Futter–Horvath–Wiese's Sig-Trading is the later,
  closed-form path-dependent mean-variance extension to reach for if signatures carry IC
  (it lacks transaction costs in the base paper; pair with GP rather than adopt whole).
- Barroso–Santa-Clara / Daniel–Moskowitz vol-managed momentum: the crypto replications
  (2025) report enhanced returns rather than crash protection; enters here as the
  "sign × vol-scaled" target arm, cheaply.
- Avellaneda–Lee residual OU as the interpretation of the market-residual target; note the
  2025 negative crypto replication of PCA statarb — a reason to measure IC first rather
  than build the book first.
- Grinold–Kahn / Clarke–de Silva–Thorley: the conversion. It is the instrument that turns an
  IC table into a Sharpe bound that the goalposts can be compared against.

### 4.7 Why the record does not already falsify this

- `819 CAPACITY` (E-2) is the closest prior measurement and differs on every axis that
  matters: 20 assets not the PIT ~273; a pooled time-series IC on a non-demeaned target, not
  a per-date cross-sectional IC on a market-residual one; close-only features; H ∈ {1,5} only;
  test window 2024-07→2025-08; pre-E-c epoch; a noise floor that is a max of four draws, one
  of which failed the pipeline's own leak check. It is also the reason the prior here is
  low, and the memo says so.
- `816` never fitted a daily predictor: its H=1440 cell used minute-lag features with a
  strided rebalance (A-3). `813` (vol-harvesting triggers) tested discrete rules on
  hold-outs of n=6–25, not IC.
- The vector line measured policies, never IC, and never saw a high, low or volume (A-2);
  its one forecast-shaped read (`836/E4-p1b`,
  ρ 0.21–0.30 at k=3, n=95) is positive and was never re-measured.
- No range-based volatility, signature feature, market-neutral target, or closed-form
  cost-aware controller appears anywhere in the tree (mining pass: zero hits for
  signatures, Kelly/vol-targeting, statarb/OU/pairs code; GARCH exists only as a Flax
  module in `physics_core.py:88`).

---

## 5. Alternatives considered and set aside (one line each, with the reason)

- **Neural SDE / diffusion world model (Bella) inside search.** Never tested as an ansatz
  (`encoder_bella_class_closed.md:29`), but a generative model calibrated to the data has no
  drift information beyond what a forecaster extracts; the bottleneck is the conditional
  mean, not scenario coverage. Keep as R4 stress generator (D5).
- **Online learning / universal portfolios (Cover; OLPS family).** Regret is against the
  best constant-rebalanced portfolio, which is not BEAR-positive; the reversal-exploiting
  variants (OLMAR/PAMR) are known to be cost-fragile. Could be a later arm inside C's
  controller comparison, not a direction.
- **Regime HMM / change-point on the market factor.** Single series, breadth ≈1; folds into
  C as the market-timing arm, reported separately.
- **Full Signature Trading.** Adopt only if F4 carries IC (G-A); its base form has no costs.
- **Perpetual funding / basis carry; on-chain flows; news-LLM hybrid (on record by name
  only, `notes_2026-09-01.md:43-44`, cross-reference broken).** Not OHLC. These are the
  pre-committed pivot targets if G-A fails; the literature's strongest post-2022 crypto
  Sharpe numbers sit in funding/basis arbitrage, not in OHLC anomalies.
- **Venue change / maker economics.** Fork-C stands (stay Kraken); 897 showed the killer
  is immediacy, which no venue removes.

---

## 6. Honesty clause

My prior, stated so it can be scored: P(daily OHLC, PIT, clears +0.56 net@21 on 2024–26H1
under honest costs) ≈ 25%; P(clears +0.97) ≈ 10–15%. Reasons for the low prior: E-1/E-2/E-3
above; crypto anomaly literature reports out-of-sample decay and large execution haircuts;
the market's single-factor structure caps breadth.

Why I nonetheless do not overrule the operator's OHLC preference today: the decisive
measurement has never been run in the form that would settle it, it costs about two CPU
days, and the repo's own doctrine forbids reading a verdict on one formulation as a law of
nature (`docs/findings/README.md:12-19`; `LEDGER.md:10-26`). Overruling now would repeat the
error the record warns about. Overruling after G-A fails is the pre-committed consequence in
§4.5, and this memo is the place it was pre-committed.

Why I do overrule the "continue" option: the evidence against the learned-policy line is
post-E-c, multi-instrument, pre-registered, and its own kill clause fired. Continuing would
be the parameter-nudge the pathway rule forbids, with a GPU bill.

One process note for the planner, not a finding: the pathfinder audit of 2026-08-19 laid
out a five-step falsification path; all five steps have reported and all were negative. That
is the record working. The next row should be the one that can end the question, not the
one that keeps a lane warm.

---

## Appendix — receipts index (primary file, line)

`docs/pathway.md:9-21` goalposts · `:37` H1 kill clause · `:111-118` fallback struck ·
`experiments/LEDGER.md:10-26` epochs · `:28-41` instrument discipline · rows `830/E0-REAL`,
`831/E1-REAL`, `828s3/preflight`, `828s3/env-wall`, `835/S4-p1c`, `836/E4-p1b`, `832/E5-*`,
`849/leaf`, `875/wf-generalization`, `879/rung3-distill-remeasure`, `882/decisive-legA`,
`884/rung1-smoke`, `885/hold-collapse-screen`, `887/collapse-confirm` (+ 2026-08-19
correction), `892/action-ev`, `898/trend-decay`, `897/809-corrected` ·
`tasks/DECISIONS.md` D47 (`:76`), D65 (`:37`), D76–D78 (`:11-13`) ·
`docs/pathfinder_audit_2026-08-19.md:48-109, 127-166` ·
`docs/findings/diversified_trend_shipcandidate.md:10-36` · `modeling_surface_exhausted.md:3-31` ·
`docs/findings/README.md:12-19` · `trader/config.py:89-105` · `trader/backtest.py:128-138,
188-190, 455-462` · `trader/data.py:137-190` · `mcts_trader/training/env_factory.py:417-418` ·
`experiments/835_stage4_eval/eval_vector.py:1738` · `mcts_trader/env/jax_trading_env.py:1621-1820,
1868-1872` · `mcts_trader/env/jax_encoder.py:48-70, 328-329` · `mcts_trader/jax_agent/sampled_search.py:299-304`
· `mcts_trader/training/self_play.py:728-729` · `experiments/907_live_fill_probe/PREREGISTRATION.md:11-64`
· `lore/notes_2026-07-06.md:139-145` · `lore/notes_2026-08-19.md:17-18` ·
`lore/notes_2026-09-01.md:17-19, 43-44` · `lore/notes_2026-08-31.md:63-67` ·
`lore/handoff_2026-09-02_factory_run_50.md:78-94` · `tasks/TASK_817:32-33` · `tasks/TASK_820:107-117`.
