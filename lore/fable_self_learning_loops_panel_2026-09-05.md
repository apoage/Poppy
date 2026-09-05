# Self-learning-loops panel — 2026-09-05 (consultation, read-only)

Third entry in the consultation series (`fable_direction_memo_2026-09-03.md`,
`fable_tactics_panel_2026-09-04.md`). Separate entry by operator instruction. Nothing in the
tree changed except the two lore corrections listed at the end; no runs; no server contact.

## Question (operator, 2026-09-04, verbatim)

"what about self learning algorithm loops .. something that learn pattern that serve to monte
carlo yet another whisper thing.." — is there a self-learning loop design, one that learns
patterns to serve Monte Carlo evaluation or search, that the record has not already falsified
and that could plausibly create edge. The operator half-expects the answer is another faint hope.

## Method

Workflow `self-learning-loops-panel` (run `wf_650304b3-151`, resumed once): five explorer agents,
one per loop design, each briefed with the goalposts, the record's closed verdicts on loops, the
data facts, and an amplifier-not-source discipline (state exactly what information the loop's
learned object carries that a direct estimator lacks, and what it is worth if daily
conditional-mean IC ≈ 0). Then two refuters per design (amplifier lens; repo-record lens).
Explorers 5/5 and refuters 10/10 completed (~4.3M subagent tokens, 789 tool uses over two
sessions); the synthesizer failed on the account session limit twice. The ranking below is the
consulting model's own synthesis over the fifteen reports, with every load-bearing repo receipt
re-read in the primary file. Reports: `~/.claude/projects/-home-lukas-Trader/<session>/subagents/workflows/wf_650304b3-151/journal.jsonl`.

## Answer

**No.** A self-learning loop is an amplifier of whatever information its training targets carry.
It cannot create a conditional mean that is not in the data, and every design examined either
(a) reduces to a forecaster estimated less efficiently than a regression aimed at the same
quantity, (b) converges to flat or 1/N when the plausibility set admits zero drift, or (c) is a
cost or risk tool with a ceiling of +0.10..+0.18 Sharpe on a book whose gross is measured at ≈0
on 2024–2026. All ten refuters returned refuted=true; every explorer put its own design at
0–5% odds of the goalposts. The record's six months of loops (812, 828s3, ExIt, 849, 857/860/879,
861, 884/885/887, 892) are exactly the empirical form of this result: the loop amplified a
signal that reads ≈0 out of sample and converged to the two attractors the theory predicts
(HOLD when the risk term dominates, noise-churn when it does not).

The one identity that decides it (iterated expectations): for any causal position process w_t,
E[gross] = Σ_t E[w_t · E[r_{t+1} | F_t]]. If the conditional mean is zero on the loop's
information set, expected gross is zero for every rule, however well the loop knows variance,
tails, dependence or regime persistence; net is −turnover × 21 bp. Second-moment knowledge moves
expected profit through one channel only: re-timing a premium that already exists.

## Design-by-design

| Design | Learned object | Information beyond a direct estimator | Worth if daily IC ≈ 0 | Refuted (2 lenses) | What survives |
|---|---|---|---|---|---|
| World model serving MC rollouts (Dreamer / neural SDE / Bella-in-the-tree) | conditional law P(r_{t+1:t+h} \| x_t) | joint Σ(x), tails, within-horizon dependence; the mean channel is a by-product estimated with MORE variance than ridge; MC leaf adds noise ~σ_w/√K (K=512: ~22 bp/leaf vs a ~7–20 bp/day signal) | zero as alpha; negative as a planner leaf | 2/2 | nothing that needs the generator: a block-Rademacher sign-flip null preserves dependence exactly with zero fitting variance |
| Execution-only loop (order placement for a pre-decided daily list) | queue-value surface P(fill \| state), E[markout \| fill] | none: under a martingale mid the paired saving of post-at-touch over taker is p·(spread + Δfee), a lookup; Nevmyvaka's DP gains live in the 5–10k-share impact regime | +0.06..+0.12 Sharpe at 21→8–12 bp on a 0.155/day book; whole stack +0.18 | 2/2 | a one-time PAIRED cost-tier measurement, after the operator's actual Kraken tier is written down |
| Adversarial / minimax self-play vs a market generator | worst-case reweighting of paths; robust fixed point | information about the POLICY's fragility, not the market; Blanchet–Chen–Zhou Prop. 1: worst-case mean = μ̂ᵀφ − √δ‖φ‖, flat once √δ ≥ ‖μ̂‖ | flat (long-short, with costs) or 1/N (long-only) — both already measured | 2/2 | Wasserstein-DRO shrinkage as a closed-form sizing/gating rule inside a GP book IF a forecast exists; R4 stress gate (D5 as written) |
| Continual / few-shot / meta-learning | a time path θ_t of parameters | only Var_t(IC_t)·φ (time-variation of the conditional mean), read with less variance by a by-year IC with placebo; detection window for IC 0.02–0.03 exceeds regime lifetimes | negative: re-estimation = re-trims; the 819 lookback ensemble already read −0.16 | 2/2 | one added statistic to TASK_915 (monthly-IC autocorrelation with an overlap-aware null); a lookback sub-arm of tactics row 1 |
| Evolutionary / GP rule search with MC fitness | archive of rule trees + the selection record | none about the market (a thresholded rule carries ≤ the IC of its terminals; TASK_915's GBM already is a conjunctive estimator); the selection record (N, E[max SR_N]) is the one thing no grid here has emitted | a noise maximiser: E[max SR_N] under H0 clears +0.97 in-sample at N≈10 on the 2.5-yr window, N≈200 on the 7.7-yr PIT span | 2/2 | the multiplicity arithmetic as a standing reference: report SR₀(N) beside every "best cell" |

## Points the refuters made that change how the record should be read

1. **Kraken fee ladder vs the record's "fee16".** The July-2026 Kraken Pro ladder (fetched
   2026-09-05 by the execution refuter; abstract-level, verify before relying): base tier
   0.40/0.80, $50k 0.15/0.30, $100k 0.12/0.25, $250k 0.10/0.22, $500k 0.08/0.20, $1M 0.06/0.18,
   $2.5M 0.04/0.15 (maker/taker, %). No tier has a 16 bp taker and no tier has a 16 bp maker.
   `POWERED_RESULT.md:7` calls 21 bp "Kraken taker16+slip5"; `TASK_897:284` records the
   attainable MAKER set as {8, 16} bp. The two are inconsistent, and the operator's actual tier
   is not on record. If the account sits below ~$1M/30d, every net@21 column is optimistic;
   if it sits at a maker tier, 907's phase-2 execution question is a schedule lookup. Writing
   the tier down is a zero-cost, load-bearing record fix.
2. **Denomination of the frozen-mid finding.** 46.5% is a share of book EVENTS; the time share
   is 138,449 s / 355,963 s = 38.9% (`handoff_run_50.md:72`). The memo said "of the time";
   corrected 2026-09-05. Same error class run 50 was about.
3. **819 CAPACITY does not show nonlinear ≤ linear.** Point estimates order GBM/MLP/GRU ABOVE
   ridge at both horizons (H=5: GBM +0.037 vs ridge +0.011), all inside a four-draw noise band.
   The memo's §4.2 justification and TASK_915's one-paragraph summary ("nonlinear ≤ linear is
   measured on these surfaces") inherited an overclaim; memo corrected 2026-09-05, TASK_915's
   sentence is the planner's to amend. GBM stays in the sweep; neural nets are excluded as a
   pre-registered scope choice, not on evidence.
4. **Per-date IC detection power.** Under an exchangeable null the per-date Spearman IC over
   N ≈ 240–270 live assets has SD ≈ 1/√(N−1) ≈ 0.06–0.07, not 1/√N_eff ≈ 0.35. N_eff governs
   the fundamental-law breadth (correlated bets), not the null SD of the IC statistic. TASK_915
   should state both.
5. **MC attenuation biases a generator-vs-ridge IC test toward KILL**: a rollout mean of K
   paths reads IC × 1/√(1 + 1/(K·IC²)); at K=512, IC=0.02 the measured IC is ~0.008. Any test of
   a generator's implied mean must use K ≥ ~1/(0.1·IC²) ≈ 25,000 or the closed-form mean.
6. **The world-model "escape" from 849 is real but not new**: making the leaf's action-dependence
   exact through known inventory/cost dynamics is the same mechanism that restored sensitivity
   when the true env replaced the learned leaf (E5-inversion2/3); it fixes the wrapper, not the
   numerator. The record's 2026-08-19 correction to 887 stands: 887 measured a learning/search
   failure, not absent edge; the daily conditional mean remains unmeasured in the form that
   settles it (TASK_915).

## What survives, and where it belongs (none of it a lane, none of it needs a GPU)

- A **block-Rademacher / circular-shift placebo family** for TASK_915's null (dependence-
  preserving, zero fitting variance) — replaces the E-2 four-draw floor. CPU minutes.
- **Monthly-IC autocorrelation τ** with an overlap-aware null, one added statistic in TASK_915.
- **Multiplicity accounting**: SR₀(N) = E[max SR_N \| H0] reported beside every best cell; a rule
  grid graded on 2024–26 must carry N ≤ ~50 (MinBTL). Belongs in TASK_915's SPEC and in
  tactics-panel row 1.
- **DRO shrinkage** μ̂ᵀφ − √δ‖φ‖ − cost > 0 as the GP book's position gate, if G-A passes.
- **A paired maker-vs-taker cost measurement** at the operator's recorded tier (D65/gate-3
  infrastructure), replacing the asserted fee16/slip5 with an observed number.
- **Bella at R4** as regime stress generator, as D5 already says — after rebuilding it for daily
  bars (it is a minute-bar generator trained on `is_active = 1` survivors only,
  `diff_teacher/data_loader.py:151-157`, whose paths measured ~7× too easy,
  `lore/notes_2026-05-18.md:87-96`).

## Compute

None of the surviving items needs a GPU. The only design where 64 GB cards matter is the
daily-surface world model (≈10⁷ parameters, ≤1 GPU-day on one 24 GB card for 20 quarterly
refits), and it is refuted as an alpha object before it is built. The cards would buy larger
generators or larger self-play; the record measures neither as the bottleneck.

## Corrections applied to earlier lore this session

- `fable_direction_memo_2026-09-03.md` §1.4: adverse-selection drag 0.3–0.5 → 0.04–0.05
  Sharpe (arithmetic error ~8×; conclusion unchanged, stronger).
- Same file §1.4: frozen-mid denomination (events vs time).
- Same file §4.2: "nonlinear ≤ linear on these surfaces" → 811 only; 819 inconclusive with
  nonlinear above linear numerically.
