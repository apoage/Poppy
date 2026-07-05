# What "Alpha" Actually Means — A Field Survey

> *Observation: the apothecary's scruple weighs differently in Nuremberg than in Bologna; the grain of one city is not the grain of another. I keep three sets of weights on the bench, each labeled in a different hand, and I do not begin a dose without naming which set I am about to use.*

---

## The word I have been using as if I knew what it meant

`workstation` is on. The desk fan is on. `nvidia-smi` is ticking through nothing in particular — there is no run in flight tonight, the 3090 is idle at 39°C, and the only thing the workstation is doing is keeping the apothecary in pixels. It is the evening of May 22, 2026, and I am reading back through the last fifteen articles of this series with a yellow highlighter and a knot in my stomach.

The knot is this. The word **Alpha** appears 263 times across the drafts in `bloom/drafts/`. The notation **Δα** appears another 187 times. Every result table I have published in this diary lives or dies by Alpha. Article 6 is built on **Δα = −18.44pp** for the TASK_738 legacy-encoder ablation. Article 7 is built on **+6.21% → +3.61%** for the TASK_737 champion that wasn't. Article 8 is the five-seed funeral of that same +3.61%. Article 10 is built on **+0.72pp → −1.20pp**, the recalibration that closed an entire campaign. Article 14 turns on a quarter-Alpha sign bit. Article 22, the year-end retrospective, is essentially a frequency table of negative Δα.

And in fifteen articles I have not, once, written down what Alpha *is*.

I have written down what Δα measures in this codebase — the paired difference of a variant's deployment Alpha against the baseline's, same seeds, same data, same operator, only the experimental mechanism flipped. I have been precise about *that*. But I have used "Alpha" as if it were a single, well-defined object — as if when I write **Δα = +3.61%** the reader and I share a definition. We do not. There are at least four definitions in active use in the literature, two of them sharply incompatible, and the one I have been using is not the textbook one.

This article is the audit. What does "Alpha" mean in the finance literature, what does it mean in this project, where do the two diverge, and what does that divergence cost when I write a result table.

I am going to start at the textbook and walk forward.

---

## The textbook Alpha (Jensen, 1968)

The word "alpha" enters quantitative finance through Michael Jensen's 1968 paper *The Performance of Mutual Funds in the Period 1945–1964*. Jensen wanted to measure whether mutual fund managers were *skilled* or merely *exposed* — whether their returns came from picking better assets or from holding more market.

He took the Capital Asset Pricing Model, which says an asset's expected excess return is proportional to its sensitivity to the market:

$$\mathbb{E}[R_i - R_f] = \beta_i \cdot \mathbb{E}[R_m - R_f]$$

where $R_i$ is the fund return, $R_f$ the risk-free rate, $R_m$ the market return, and $\beta_i$ the fund's sensitivity to market moves. He added a residual term:

$$R_i - R_f = \alpha_i + \beta_i (R_m - R_f) + \epsilon_i$$

and called the residual constant $\alpha_i$ — **Jensen's alpha** — the part of the fund's excess return that *cannot* be explained by its market exposure. Positive alpha means the manager added return beyond what their beta could have produced passively. Negative alpha means the manager destroyed value relative to the same exposure held passively.

The formula is a regression, not a difference of means. To compute it for a fund over a sample period, you regress the fund's excess returns on the market's excess returns, take the intercept, and that's the alpha. The slope is the beta. There is a standard error on the alpha. There is a t-statistic on whether it differs significantly from zero. There is an R² for how much variance the regression explains.

This matters because Jensen's alpha is *risk-adjusted*. A fund that returned +20% in a year when the market returned +15% has a raw outperformance of +5pp — but if the fund had a beta of 1.5, the CAPM-expected return given that exposure was +22.5%, and the alpha is *negative*. The fund underperformed *its risk-adjusted benchmark* by 2.5pp despite beating the index.

The intuition is clean. A leveraged index fund will beat the index whenever the market goes up — that's not skill, that's beta. To isolate skill, you have to net out the beta. Jensen's alpha does exactly this.

The reason this definition matters — and the reason it is still the textbook starting point in every CFA curriculum, every finance PhD program, every regulated-fund disclosure — is that it has *teeth*. It produces a number with a standard error. It can be tested against the null hypothesis $\alpha = 0$. It can be compared across funds. It has been the spine of mutual-fund performance evaluation for fifty-seven years.

This is not what my project measures.

---

## The variants — where Jensen's alpha goes when it grows up

Jensen's alpha is the original, but the literature has spent fifty-seven years generalizing it, and the generalizations matter for understanding what people mean when they say "alpha" in 2026.

The first major extension is the **Fama-French three-factor model** (1993), which replaces the single market factor with three: market excess return, small-minus-big (SMB) capitalization premium, and high-minus-low (HML) value premium. The regression becomes:

$$R_i - R_f = \alpha_i + \beta_{m,i}(R_m - R_f) + \beta_{s,i} \cdot \text{SMB} + \beta_{v,i} \cdot \text{HML} + \epsilon_i$$

A fund that beats the market by tilting toward small-cap value stocks does not get credit for alpha under this model — that's a known risk premium, not skill. The alpha here is *whatever's left after subtracting all three known factor exposures*. It is a strictly smaller number than Jensen's alpha for most funds.

**Carhart (1997)** added a fourth factor — momentum (UMD, up-minus-down) — turning the three-factor into a four-factor model, on the empirical observation that stocks which had recently outperformed continued to outperform on a one-year horizon. **Fama-French (2015)** added profitability (RMW) and investment (CMA) factors, producing the modern **five-factor model**. The alpha in the five-factor regression is the residual after subtracting market, size, value, profitability, and investment exposures.

The trend in academic finance is unidirectional: every factor that someone publishes as a "source of alpha" eventually gets folded into the benchmark model, and what used to be alpha becomes beta. This is sometimes called *the factor zoo* — there are now hundreds of published factors, most of which fail out-of-sample — and the working assumption among quantitative researchers is that anything that looks like alpha for long enough will turn out to be a previously-undiscovered beta.

A different axis of variation is **the holding horizon**. Daily alpha, monthly alpha, annual alpha, lifetime alpha — these are not the same number, and they do not all have the same statistical properties. A strategy that produces +5bps per day has an annualized Sharpe of roughly 8 (which, if true, would make it the most successful strategy in human history); the same strategy measured monthly might have a Sharpe of 0.6 and look ordinary. Horizon is not a footnote; it is the most-often-omitted disclosure in published alphas.

A third axis is **the benchmark choice**. Jensen's original alpha used a broad market index. Modern hedge fund evaluation often uses a *peer benchmark* — alpha relative to the median fund in a strategy bucket — which is fundamentally a different object. There are also *bespoke benchmarks* (the fund's own backtest, an equal-weight passive of the same investable universe, the SPY ETF, the global market portfolio) and the choice between them can flip the sign of the alpha. The compass artifact on optimal transport for rebalancing notes the same problem in a different domain: <cite>"the choice of ground metric on asset space—correlation distance, sector hierarchy, factor loadings, or liquidity scores—determines what 'closeness' means financially"</cite> — *and the choice of benchmark, in the alpha calculation, determines what "outperformance" means financially.*

A fourth axis is **whether alpha is gross or net of costs**. A gross alpha is the regression intercept on pre-transaction-cost returns. A net alpha subtracts the fund's actual implementation costs — slippage, fees, market impact, financing. These are not always small numbers. A strategy with a +4% gross alpha can have a +0.5% net alpha after a realistic execution model is applied. In published academic papers the alpha is usually gross. In an investor's brokerage statement it is unambiguously net. *Asking what the alpha is, in the field, is incomplete; you must also ask whose accounting.*

A fifth axis — and this one is the one that connects most directly to what I do — is **whether the alpha is computed in-sample or out-of-sample**. In-sample alpha (regression on the backtest period) is almost always inflated by selection bias, overfitting, and the silent winnowing of strategies that didn't work in the same period. Out-of-sample alpha — measured on data the strategy parameters were not chosen against — is the real number. The gap between in-sample and out-of-sample alpha is sometimes called the *p-hacking penalty* or *the haircut*, and on quantitative strategies it has been measured at 50–70% in published replications. *Half of your in-sample alpha is, on average, an artifact of having looked at the data first.*

This is enough variants for the article. There are more — risk-adjusted alpha, treynor-adjusted alpha, information ratio (which is a normalized cousin of alpha), alpha-at-risk (downside-conditioned alpha). The point is not the taxonomy. The point is that the word **alpha**, in any sentence longer than a tweet, has to come with at least one and usually several qualifiers — what model, what horizon, what benchmark, gross or net, in-sample or out — for the number to be meaningful.

---

## The hedge-fund definition (alpha as marketing)

Outside the academic factor-model literature, there is a second definition of alpha that dominates in practice — particularly in the hedge fund world, in commercial backtesting platforms, and in the way most quants talk on a podcast.

In this looser definition, **alpha is skill-based excess return**. The factor-model decomposition is not required. The benchmark is whatever the manager says it is — often the S&P 500, sometimes a custom blend, sometimes nothing more rigorous than "the return I would have gotten if I had not traded." The risk adjustment is informal, if it appears at all.

This is the alpha you see in pitch decks. It is the alpha you see in the title of trading books and YouTube videos. It is the alpha that the entire algorithmic-trading-content ecosystem refers to when it says "find your edge, capture alpha, beat the benchmark."

It is also — to be honest about it — closer to what *I* mean in this series when I write **Δα = +3.61%**. I am not running a CAPM regression. I am not netting out a momentum factor. I am taking a raw difference between agent return and a passive benchmark, averaged across periods, and calling the result alpha.

There is a reasonable critique of this definition, and there is a reasonable defense. The critique is that this kind of alpha is *not risk-adjusted* — if my agent loads heavily on momentum exposure and the test period happens to be a momentum-friendly regime, my measured outperformance is partially or entirely an artifact of that exposure, not of skill. The defense is that for research-stage work where the goal is to demonstrate that a learned policy can beat passive holding *at all*, the looser definition is operationally sufficient as long as the periods are diverse enough that no single factor exposure can explain the pattern.

The defense holds, conditionally. The condition is that *the diversity assumption be verified*, not assumed. This series has six benchmark periods spanning bull, bear, sideways, FTX crash, ETF approval, and a mixed regime. That diversity is real and was selected against exactly this critique. But it is not the same thing as a factor-model decomposition, and a regulator would not let me call this "alpha" without qualification in a fund disclosure.

The compass artifact on hindsight optimization frames it well — the academic framework distinguishes <cite>"skill from luck"</cite> through counterfactual reasoning, which is what factor models do statistically. The hedge-fund framework defines alpha as outperformance and then *argues* about whether the outperformance is skill or luck. The first frame builds the test into the metric. The second frame relies on out-of-sample validation to do the same work. Both can be defensible; they are not interchangeable.

---

## This project's Alpha — what the code actually computes

Let me show you, line for line, what Δα is in this codebase.

The alpha calculation lives in `mcts_trader/training/self_play.py`, lines 758–791. The audit document is `TASK_124_ALPHA_CALCULATION_AUDIT.md`, dated January 29, 2026 — the one I had to write when the random-init agent on epoch 1 was reporting 645% Alpha against a +5.2% market and I had to find out why. The relevant block, quoted from that audit:

> ```python
> # train_jax_fast.py:871-872
> alpha = trajectory.get('alpha', 0.0)
> market_return = trajectory.get('market_return', 0.0)
>
> # self_play.py:758, 764-767, 791
> final_return = float(final_states.cumulative_return[i])  # From JAX env
> start_prices = cpu_prices[start_idx]
> end_prices = cpu_prices[end_idx - 1]
> asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
> market_return = float(np.mean(asset_returns)) * 100
> alpha = final_return - market_return
> ```
> — `tasks/TASK_124_ALPHA_CALCULATION_AUDIT.md`

That is the definition. Five lines of Python. Let me translate.

For each benchmark period, the agent starts with an initial portfolio value and runs through the period. At the end, `final_states.cumulative_return[i]` holds the agent's percentage return — `(final_portfolio_value / initial_portfolio_value - 1) * 100`. Separately, the system computes the *equal-weight passive return* of the same investable universe over the same window: take the per-asset return of each asset from window start to window end, average across assets, multiply by 100. That's `market_return`. Alpha is the arithmetic difference, in percentage points.

Then **Δα** — the paired delta against a baseline — is the variant's Alpha minus the baseline's Alpha, computed seed-by-seed against a baseline run on the same seeds, and either reported as a mean across seeds (with the per-seed values disclosed) or per-period (with the seed reduction either named or applied).

Notice what is *not* in those five lines.

There is no factor model. There is no beta. There is no regression. There is no risk adjustment. There is no Sharpe normalization. There is no horizon adjustment. There is no significance test on whether the alpha differs from zero. There is no haircut for the fact that the benchmark periods were selected by hand. The benchmark is *equal-weight passive holding of the same universe*, which is itself an idiosyncratic choice — it is not the SPY, it is not the global market portfolio, it is the equal-weighted return of whatever assets the agent could have traded over that window.

What this metric *is*, with precision, is a **comparative deployment metric**: how much more (or less) did the agent return than a no-trade strategy on the same assets, over the same window, expressed in percentage points.

There is also the *operator* axis, which Article 10 raised explicitly. Until May 19, 2026, all my benchmarks were scored under **argmax of the policy logits**. From May 20 forward, the canonical scoreboard is **eval-MCTS sims=16 joint**, the same decoder used by the deployed agent. The numbers in Articles 6 and 7 are argmax-Δα. The numbers in Articles 10 and later are eval-MCTS-Δα. They are not the same metric. They are *Δα under a different action-selection operator on the same checkpoint family*, and as Article 10 showed, the difference between them can shift a +0.72pp result to −1.20pp without anything in the trader itself changing.

So the full disclosure is: **this project's Δα is the paired difference of equal-weight-passive-excess-return, computed across six benchmark periods, averaged over a stated number of seeds, under a named action-selection operator (argmax or eval-MCTS).** Five qualifiers. None of them are Jensen's alpha. All of them are in the code.

The audit history is shorter than I'd like to admit. The original calculation, in January 2026, did not have the same-step observation guard — the agent observed price\[t\] and traded at price\[t\], and reported +645% Alpha that was entirely an artifact of perfect-information trading. The fix lives in `jax_trading_env.py:1304-1358` and pushed observation windows back by one step. The original calculation did not deduct margin on limit-short fills — free-money bug, +76% spurious Alpha — fixed at line 615. The original calculation used a static `PRNGKey(42)` that let the agent memorize trajectories across epochs — fixed in `setup.py:254`. Each of those was a different Alpha. The **formula** stayed the same. The **numbers it produced** moved by hundreds of percentage points. *The formula was not the meaning; the formula plus the environment plus the operator plus the seeds plus the periods was the meaning.*

The point is not that the May 2026 Alpha is the right one and the January 2026 Alpha was wrong. The point is that *they are different metrics*. The diary entries I wrote in January, when I said "the agent achieved +645% Alpha," were not lying — they were honestly reporting a number whose definition I did not yet fully understand. The diary entries I write today, when I say "Δα = −1.20pp under eval-MCTS sims=16," are reporting a number whose definition I understand better, but not perfectly, and not in a way that maps cleanly onto the textbook.

---

## The gap

Let me name the gap explicitly.

This project's Δα is **not Jensen's alpha**. It is not a Fama-French residual, not a Carhart residual, not a five-factor residual. There is no regression involved in its computation. There is no factor decomposition. There is no risk adjustment. There is no statistical significance test. There is no out-of-sample protocol beyond the choice of which benchmark periods to include, and that choice was made by me, before the experiments ran, with no formal blinding.

It is also not the hedge-fund alpha — at least not the version that gets sold in pitch decks — because the benchmark is not a published market index, the periods are not the fund's live track record, and the returns are not net of realistic transaction costs at the size a real fund would trade. The transaction-cost model in the JAX env is a fixed-fraction fee on each trade plus a slippage assumption that is not calibrated to real exchange microstructure.

What it *is*, calibrated against itself across seeds, is a **comparative metric for ablations on a learned planning agent**. If I run the same agent with mechanism A and mechanism B, on the same six periods, the same five seeds, the same eval-MCTS decoder, and mechanism A produces a higher Δα than mechanism B, that is meaningful information *about the relative performance of A vs B inside this evaluation harness*. It is not a claim that mechanism A would beat the market in production. It is not a claim that A has alpha in the Jensen sense. It is a claim that A's policy, evaluated through this specific apparatus, captured more equal-weight-passive-excess-return than B's did, with a quantified seed spread.

This is OK for research. It is not OK for a fund.

The reason it is OK for research is that the question I am trying to answer with this metric is *does mechanism A produce a better policy than mechanism B*, and as long as the noise floor of the evaluation is small relative to the spread I am claiming, the answer is recoverable from a paired comparison. The reason it is not OK for a fund is that the question a regulated fund has to answer is *is this manager skilled*, which requires actually separating skill from beta — which requires a factor model. My metric does not separate skill from beta. If the agent happens to learn a momentum-loading policy and the test periods are momentum-friendly, my Δα looks great and the agent has, in the Jensen sense, captured zero alpha.

I want to be careful about this, because it is the kind of methodological asymmetry that is easy to handwave past. The series has been writing **Δα** in result tables for fifteen articles. Every one of those tables is calibrated against this specific equal-weight-passive operator on this specific universe under this specific MCTS decoder over these specific six periods. They are internally consistent. They are *not* commensurate with the alpha numbers a Fama-French regression would produce on the same checkpoints. If a finance professor read Article 10 and said "your campaign Alpha numbers are not alpha," they would be technically correct and I would have no rebuttal beyond *I have been careful to make the metric do the job I needed it to do, which is to discriminate mechanisms paired against a baseline.*

The honest accounting is: when I publish a result, I should be saying **paired Δ-equal-weight-passive-excess-return under eval-MCTS sims=16 joint**, not **Alpha**. The first is a mouthful and the second is wrong. The diary has been using the wrong word.

---

> *Observation: the apprentice asks the apothecary which scale is correct. The apothecary turns over each scale on the bench and notes which city's grain it weighs. The Bolognese scale weighs honestly in Bolognese grains. The Nuremberg scale weighs honestly in Nuremberg grains. There is no scale that weighs in grains-in-the-abstract; there is only the choice of city, and the discipline of naming it before pouring.*

---

## The kickback

There are three layers to this one, and they have to land in order.

**Layer one: Alpha is a category, not a number.** When I write **Δα = +3.61%** in Article 7, and **Δα = −18.44pp** in Article 6, and **Δα = −1.20pp** in Article 10, I am writing three numbers that *look like they live on the same axis* and *do not*. Article 7's number is averaged across five seeds under the argmax operator on the May 13 codebase. Article 6's is paired across three seeds under the argmax operator on the May 14 codebase, on a different ablation. Article 10's is paired across five seeds under the eval-MCTS sims=16 operator on the May 20 codebase, on yet a different ablation. The qualifiers matter so much that two of these three numbers (Article 7's +3.61% and Article 10's −1.20pp) describe checkpoints from *related experiments* and produce *opposite signs* of the same general claim about the same general intervention. *The unit has been shifting underneath the numbers.* I have been writing as if I knew what unit I was using. I have been writing as if Alpha were the metric. It is *a* metric, conditional on five qualifiers, several of which I have not been disclosing in the article bodies.

**Layer two: the metric has drifted across the project's lifetime.** TASK_124 in January 2026 fixed three bugs that each moved the alpha calculation by hundreds of percentage points without changing the formula. The TASK_118 work introduced a *separate* alpha computation — `compute_max_alpha_topk` in `mcts_trader/jax_agent/alpha_reward.py` — for the alpha-relative reward shaping mechanism, which is a *third* sense of the word alpha inside the same repository: not deployment Alpha, not Jensen's alpha, but a **per-step opportunity ceiling** used as a normalizer in the RL reward signal. The reward-shaping work documents itself this way:

> Top-4 = realistic position limit (can't hold all 64). Sum across steps = total opportunity ceiling. Averaging effect = implicit rebalancing assumption. No NP-hard optimization needed.
> — `tasks/TASK_118_ALPHA_RELATIVE_REWARD.md`, "SOLUTION" section

That is *a fourth definition of alpha* — the hindsight oracle's top-k cumulative return — used internally as a denominator in `compute_alpha_ratio()`. It is not the deployment Alpha. It is not Jensen's alpha. It is an upper bound on what the policy *could have captured* under top-k position constraints, used to normalize the reward signal so the agent's incentive is constant across regimes. The same repository, two files apart, uses the word **alpha** for two operationally different objects. Most readers — including past-me — do not track this. The metric drift is silent.

**Layer three** — and this is the one I have been dreading writing — **the campaign verdict is a function of the chosen metric**. Article 10's closing line was, paraphrased: *fourteen mechanisms, zero promotions, single-mechanism class exhausted, pivot to structural directions.* The unit of "promotion" in that sentence was paired Δα under eval-MCTS sims=16 joint, with a promotion bar of "mean ≥ +3pp across seeds, with no seed worse than −3pp." That is a specific operational rule. Change the rule and the verdict changes.

If I had kept the argmax operator, **746 oracle promotes** (+0.72pp before recalibration, the closest the campaign came to a positive signal under the old scoreboard). If I had lowered the promotion bar to "mean ≥ 0pp," **734 flow-matching promotes** (mean +3.38pp before recalibration, weakly positive after). If I had used a Jensen-style risk-adjusted alpha, several variants whose Δα was raw-positive would have *negative* Jensen alpha after netting out the higher beta exposures they learned, and several variants whose Δα was raw-negative would have *non-negative* Jensen alpha because their drawdowns came on smaller positions. If I had used a per-step opportunity-ratio (the TASK_118 alpha), the rankings reshuffle again — some variants that look bad on terminal Alpha look fine on per-step capture, because they over-trade on bad signals but still capture a respectable fraction of available opportunity.

The verdict I published in Article 10 — *fourteen mechanisms, zero promotions* — is true *under the metric and threshold I chose*. It is not true *in the abstract*. There is no Alpha-in-the-abstract on which the verdict could be checked. The metric is a choice and the verdict is downstream of the choice.

This is not nihilism. It is metric discipline. A campaign's findings are only as portable as the metric they were measured against. When the metric is well-defined and the threshold is explicit and both are documented, the findings are reproducible — anyone can run the same numbers and confirm or refute. When the metric is treated as transparent — as if "Alpha" were a thing rather than a definition — the findings *look* portable and are silently not. The fix is not to throw out the campaign findings. The fix is to name the metric, every time, in full.

---

## What's left, what I learned

The working definition, for the rest of this diary and for any article that comes after:

**Alpha**, in this series, means *paired Δ-equal-weight-passive-excess-return*, computed across six benchmark periods (bull, bear, sideways, ftx_crash, etf_approval, mixed), averaged over a named number of seeds (3 or 5), under a named action-selection operator (argmax for pre-May-20 results, eval-MCTS sims=16 joint for post-May-20), against a paired baseline (a1b1 unless otherwise specified) at the same seeds. I will write **Δα** when I mean that object. I will not write **Alpha** unqualified.

This is the metric I have actually been using. The honest move is to keep using it and to be specific about what it is, not to retroactively replace it with Jensen's alpha (which would require recomputing every regression with factor data I have not been collecting) or with the hedge-fund pitch-deck alpha (which is even less rigorous than what I have now).

What I learned, in priority order:

1. **The unit on a number is not optional**. Fifteen articles of writing **Δα = +3.61%** without naming the operator is fifteen articles of pretending the unit was implicit. It was not implicit. The May 20 recalibration was the project's most expensive lesson in unit discipline — a whole campaign of small-mechanism work got re-scored against a different operator and the verdict flipped. The cost of unit drift is paid by the future-self who tries to reconcile two diary entries and cannot.

2. **Risk adjustment is not a footnote**. This project's Alpha does not subtract beta, and if the agent's strategy loads heavily on a particular factor in the test periods, the Δα is partially a beta artifact. For research-stage ablation comparison this is acceptable; for any claim about deployment, it is not. The next time I write about deployment performance I will at minimum disclose the per-period beta of the agent's returns against the equal-weight passive, even if I cannot compute a full Fama-French residual.

3. **The same word can name different objects in the same repository**. `compute_max_alpha_topk` and `final_return - market_return` are both called "alpha" in this codebase and they are not the same metric. This is the kind of silent ambiguity that wastes a future agent's time. The fix is naming discipline at the function level — the reward-shaping alpha should probably be renamed `compute_opportunity_ceiling_topk` to remove the collision.

4. **A verdict is a function of its metric**. *Fourteen mechanisms, zero promotions* is true under eval-MCTS sims=16 joint with the +3pp/−3pp bar. The structural pivot is justified under that metric. If the metric changes, the verdict has to be re-read. This is not a license to fish for a friendlier metric — that would be p-hacking dressed in epistemic-humility costume. It is a license to keep the unit visible while doing the work.

5. **The textbook is still the textbook**. Jensen's alpha exists because it has teeth — standard errors, t-statistics, factor models. The reason I do not use it is that this project is research, not a fund. The day this project tries to be a fund is the day the metric has to be replaced with something that has teeth. I should not pretend the current metric has teeth it does not have.

The series carries on with the metric it has. The metric has a name now. The name is long and I will write it out in full at least once per article from here forward.

---

> *Observation: the apothecary writes the city's name on the bottom of every weight. The grain remains the grain. The discipline is the labeling.*

---

## Sources

- Jensen, M. (1968). *The Performance of Mutual Funds in the Period 1945–1964*. Journal of Finance, 23(2), 389–416. The original alpha paper.
- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds*. Journal of Financial Economics, 33(1), 3–56. Three-factor model.
- Carhart, M. M. (1997). *On Persistence in Mutual Fund Performance*. Journal of Finance, 52(1), 57–82. Four-factor extension.
- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model*. Journal of Financial Economics, 116(1), 1–22.
- `~/Trader/docs/Algorithmic Alpha/Algorithmic Alpha Advanced.txt` — the JAX-stack survey that frames hindsight as a teacher in modern RL trading.
- `~/Trader/docs/Algorithmic Alpha/compass_artifact_wf-2cbcc8e7-...md` — *Using hindsight as a training signal without memorizing the past*, on LUPI, HER, and the skill-vs-luck decomposition.
- `~/Trader/docs/Algorithmic Alpha/compass_artifact_wf-07d39a27-...md` — *Fastest JAX-compatible algorithms for combinatorial k-of-n selection*, on the operator-and-metric coupling in selection problems.
- `~/Trader/docs/Algorithmic Alpha/compass_artifact_wf-90679f6f-...md` — *Optimal transport for portfolio rebalancing*, on the choice of ground metric as a meaning-determining choice.
- `~/Trader/tasks/TASK_124_ALPHA_CALCULATION_AUDIT.md` — the January 2026 audit that pinned down the project's actual Alpha calculation and walked through the bugs that produced 645% spurious Alpha.
- `~/Trader/tasks/TASK_118_ALPHA_RELATIVE_REWARD.md` — the fourth-definition-of-alpha used as a reward normalizer (`compute_max_alpha_topk`).
- `~/Trader/bloom/drafts/06_classifier_alpha_anticorrelation.md` — Article 6, where Δα = −18.44pp first lands.
- `~/Trader/bloom/drafts/07_task737_champion_that_wasnt.md` — Article 7, the +6.21% → +3.61% melt.
- `~/Trader/bloom/drafts/10_eval_mcts_asymmetric_lift.md` — Article 10, the operator recalibration that closed the campaign.
- `~/Trader/lore/notes_2026-05-15.md` — the diary entry where the classifier-Alpha anti-correlation first gets named.
- `~/Trader/lore/notes_2026-05-20.md` — the eval-MCTS recalibration.

*Written on `workstation`, May 22, 2026. The 3090 idles at 39°C through the writing of this article. The next campaign has not begun.*
