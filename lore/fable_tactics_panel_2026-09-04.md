# Tactics-not-alpha panel — 2026-09-04 (consultation, read-only)

Companion to `lore/fable_direction_memo_2026-09-03.md` (the direction memo). Separate entry by
operator instruction. Nothing in the tree changed; no runs; no server contact.

## Question (operator, 2026-09-03, verbatim gist)

"Explore new and unusual ways to have better hold on policy and basically still get the edge — or
behave the way that brings profit — not hunting alpha but hunting tactics." Hardware note: 24 GB
GPUs now, 64 GB GPUs being prepared for the server.

Working interpretation: profit from HOW the policy behaves (sizing, rebalancing, regime-conditional
switching between bounded tactics, risk control, execution) rather than from forecasting returns,
and a policy whose behaviour is controllable and auditable.

## Method

Workflow `tactics-not-alpha-panel` (run `wf_9989c4be-8a4`): seven explorer agents, one per
tactic family, each briefed with the goalposts (`docs/pathway.md:9-21`), the record's closed
verdicts, the data and cost facts, and a no-free-lunch discipline (expected profit must be
attributed to exactly one of: forecast, risk premium, rebalancing premium, cost reduction).
Explorers completed (7/7, ~1.77M subagent tokens, 469 tool uses). The 14 adversarial refuters and
the synthesizer FAILED on an account session limit (reset 02:20 Europe/Prague). The refutation and
ranking below are therefore a SINGLE-REVIEWER pass by the consulting model over the seven reports,
with every load-bearing repo receipt re-read in the primary file. Explorer reports are in the
workflow journal (`~/.claude/projects/-home-lukas-Trader/<session>/subagents/workflows/wf_9989c4be-8a4/journal.jsonl`).
A panel-grade refutation can be obtained by resuming the run (cached explorers) after the reset.

## Result — the accounting all seven explorers converged on independently

| Family | Actual profit source | BEAR-positive without a forecast | Explorer's odds of clearing goalposts | Surviving claim |
|---|---|---|---|---|
| K-tactic selector (hierarchical; MuZero over K arms) | forecast of relative tactic returns, K dims | only through a bear-positive arm | 3–5% | scaffold + a minutes-long ceiling test (below) |
| Execution / turnover (hysteresis, H=3, maker fills, accounting) | cost reduction only | no; inherits the gross book's sign | 0–2% alone; 60–70% it adds +0.10..+0.15 to any book | fold into every book |
| Regime-gated trend/reversal (variance-ratio or changepoint rule) | forecast of autocorrelation persistence | conditional | 3–5% | VR-persistence diagnostic (30–40% nonzero) |
| Portfolio vol targeting / Yang-Zhang sigma / DD throttle | re-times a premium, creates none | no | 1–2% | one row closing a never-tested hole |
| Deep Momentum Networks / CPD networks | forecast; published break-even 2–3 bp/side vs 21 bp here | no | ~2% | only the rule-based CPD switch (merged with the row above) |
| Stochastic portfolio theory / rebalancing premium | log-growth term, market-relative; long-short = bet that BTC dominance does not rise | no (dominance rose 2018, 2022, 2023–25) | 1–2% | banded target-weight controller |
| Universal portfolios / Hedge over experts | none of its own; regret bounds vacuous at T ≤ 2000 | inherits experts | 2–4% | Hedge as the transparent selector baseline |

Structural fact under every row: the record's only bear-positive behaviour is the trend book's
short leg — BEAR22 baseline +1.095 vs long-flat variant −2.471 (LEDGER `898/trend-decay`). That is
a forecast. Tactics bound, re-time and cheapen a forecast's harvest; they do not replace it.

## Reviewer's refutation notes (single reviewer; receipts re-read)

- Tactic selector: decomposition E[R_sel] = Σ_k π_k μ_k + Σ_k Cov(1{k chosen}, R_k) − c·E[switch
  turnover] is correct; the first term is ≤ 0 on the measured arms for 2024–2026 (trend net@8
  2024 +0.041 / 2025 +0.663 / 2026H1 −1.659, no CI excluding 0; long-flat 2026H1 −5.221; equal-weight
  constant mix gross −0.29 BULL24 / −1.63 CHOP25 PIT, `runs/819_single_asset/POWERED_RESULT.md:55-56`),
  so the bar must come from the timing term over ~30 monthly decisions, where SE ≈ 0.18 σ_month.
  Not falsified by 813 (minute-scale triggers, n = 6–28 events) or by 831x/B (a 3-day true-env
  lookahead). Hedge with a one-hot flat arm converges to flat whenever all arms' trailing EV ≤ 0 —
  the 887 fixed point made explicit; a SPEC must pre-register ">50% flat = FAIL".
- DMN/CPD: explorer read Lim–Zohren–Roberts 2019, Wood et al. 2021/2022/2023-24 and Pollok–Robik
  2026 in full. Published cost slopes −0.6..−0.8 Sharpe/bp and break-even 2–3 bp/side put the
  family 7–10× past break-even at 21 bp; the in-repo continuous-sizing surrogate (tanh) read
  −0.09 vs sign (`PORTFOLIO_RESULT.md:70`). Dead for the goalposts; the rule-based changepoint
  switch is the only unmeasured object and is the same object as the VR gate.
- Convexity/VR: Dao et al. 2016 decomposition G_T = (λ/2)(S_T−S_0)² − (λ/2)ΣD_t²; expectation
  ∝ (VR(T) − 1). PIT gross +0.13 (BULL24) / +0.22 (CHOP25) says VR(30) ≈ 1 on 2024–25 — the
  whipsaw premium consumed the payoff, not the fee (cost drag 0.15–0.28/regime). A gate is a
  forecast of VR persistence; the persistence diagnostic is the load-bearing measurement and
  has never been run here (901's VR is a bootstrap block-length tool, `range_probe.py:47-51`).
- Execution: turnover 0.155/day is essentially all sign flips (analytic zero-crossing rate 0.165
  vs measured 0.152–0.162). Whole cost stack on PIT = gross +0.74 vs net@21 +0.56 = 0.18 Sharpe
  (`POWERED_RESULT.md:57`). Two unmeasured gross effects hide here: H=3d read +1.35 net@21 vs
  +0.97 on survivors with only ~0.07 attributable to cost (`PORTFOLIO_RESULT.md:68`, n = 1), and
  hold-shares vs constant-dollar moved BEAR22 gross +0.225 vs +1.218 (`experiments/830_corridor/SPEC.md:95-98`).
- Sizing: Cederburg et al. 2020 — significant vol-managed gains for 8/103 strategies; MOM is the
  exception (0.48 → 0.99). The 819 book's highest Sharpe is in its highest-vol regime (BEAR22), so
  vol targeting cuts exposure where the book earns. Per-asset inverse-vol measured −0.06 and
  raised turnover 0.155 → 0.181 (`PORTFOLIO_RESULT.md:49`). Portfolio-level vol target never
  tested; DD-throttle Sharpe on gate 2 never recorded (`TASK_820:111-115` reports maxDD −32.2%
  and turnover halved only). Yang–Zhang vs close-to-close: Baltas–Kosowski −17% turnover at equal
  Sharpe — a cost lever, not an edge.
- SPT: γ* = ½σ²(1−ρ)(1−1/N) = 3–25%/yr at crypto inputs; constant-mix turnover ~7×/yr so cost is
  not the problem. The premium is a growth-rate (variance) term, zero to first order vs
  buy-and-hold under iid, negative under momentum; long-only versions carry beta (equal-weight
  daily 1/N PIT gross −0.40 FULL / −2.54 BEAR22); long-diversity/short-market lost ~30% relative
  in 2022 (ex-stable BTC dominance 40.9% → 46.5%).
- OLPS: Cover bound (m−1)/2·ln(T+1) ≈ 15 nats for m = 6, T = 365 — vacuous. OLMAR/PAMR are
  aggressive reversal forecasts: NYSE(O) 3.68e16 at zero cost → 0.05 at 25 bp (Moon–Kim–Moon 2019).
  Convex-hull ceiling of the measured experts on 2024–26 is below the bar unless the unmeasured
  daily reversal book is strong.

## Record corrections surfaced (verified in the primary files)

1. `trader/backtest.py:225-227`: the "B&H" column is a daily-rebalanced equal-weight book over
   present assets, charged zero turnover — a uniform constant-rebalanced portfolio, not
   buy-and-hold. The long-only constant mix is therefore ALREADY measured (negative in both bears).
2. `trader/backtest.py:221-223`: gross = Σ pos·(log c_{t+H} − log c_t) sums constituent log
   returns, so the scorer cannot see a rebalancing premium by construction; any rebalancing-premium
   test must use simple portfolio returns. Second-order for the goalposts (arithmetic Sharpe).
3. Same scorer: turn counts target-weight changes only; the constant-dollar book's daily drift
   re-trims are uncharged (explorer estimate +9–12×/yr, ≈ −0.03..−0.04 Sharpe at 21 bp on the
   +0.97 bar). Unmeasured.
4. `RiskConfig` (`trader/config.py:108-130`): per-asset cap 5%, shadow-scale DD throttle, 0.2%
   rebalance band — default OFF; their Sharpe effect on the trend book has never been recorded.

## Proposed rows (all CPU, all on `trader/` with the PIT 273-asset true-daily panel, all pre-registered)

1. **Tactic library + convex-hull ceiling** (the decisive one). Build K ≈ 7–8 position matrices:
   flat; 819 trend; long-flat trend; equal-weight constant mix; beta-hedged constant mix;
   vol-targeted trend; daily 7d and 21d cross-sectional reversal L/S. Score each by year net@21
   (simple and log returns both). Compute on 2024-01→2026-06: best fixed mixture (hull), ex-post
   oracle selector, walk-forward best single, Hedge (theory η), random-selector null (200 seeds).
   STEP-0 kill: hull < +0.97 ⇒ no selector of any kind, MuZero included, can clear the bar with
   this library; record hull, oracle and reversal-book numbers and stop. Also the first direct H2
   read (`pathway.md:38`). Minutes of CPU.
2. **Execution/turnover module**: H ∈ {1,2,3,5} × hysteresis θ ∈ {0, .25, .5, 1}·σ̂ × accounting
   {target-weight, drift-inclusive, hold-shares}, scored at 5/8/12/21 bp, Bonferroni over cells;
   print the flip/renormalisation/drift turnover decomposition. Pre-registered band: if max uplift
   over the frozen rule ≤ +0.20 net@21 the family is capped as cost-only and folds into any book.
3. **VR-persistence diagnostic + gated trend/reversal rule** with circular-shift placebo and
   Bonferroni over 8 gated cells; PASS requires beating BOTH the ungated trend and ungated
   reversal books in each of 2024, 2025, 2026H1.
4. **Portfolio vol target (40%) + Yang–Zhang vs close-to-close σ + DD-throttle Sharpe** on the
   frozen trend book; expected FAIL on Sharpe, expected yield = turnover lever and the missing
   gate-2 number.

Dropped: long-only and long-short SPT books (measured / dominance-trend); OLPS mean-reversion
algorithms (cost-fragile); DMN/CPD networks at 21 bp (break-even 2–3 bp); pure sizing as edge.

## Composition with the direction memo

The tactic library is the CONTROL half; the memo's §4 IC-ceiling instrument is the PREDICTION half.
Any forecast arm the IC instrument finds drops into the library as a tactic, with Gârleanu–Pedersen
as its execution math (row 2's Λ/turnover module). Row 1 STEP-0 runs first because it is minutes
and, if the hull is below the bar, it says immediately that everything hinges on the IC instrument.

MuZero's place: the selector over K arms with switching costs — sensible only if the hull clears
the bar AND switching costs make the choice sequential; otherwise Hedge or a ridge selector is
the honest algorithm. 64 GB GPUs are irrelevant to every row here; they would buy larger world
models or self-play, which the record measures as not the bottleneck.

## Honest statement

Tactics do not manufacture expected return. What the tactic framing does is make the question
small enough to answer in minutes and the policy small enough to trust: one integer per period,
every tactic pre-registered and individually graded, turnover bounded by construction, flat as a
visible arm (887's attractor becomes a logged choice, and a >50%-flat selector is a pre-registered
FAIL), the deployable artifact a bill of materials rather than a checkpoint. Odds of the goalposts
are unchanged at ~3–5%; the cost of finding out and the deployability of whatever is found both
improve.
