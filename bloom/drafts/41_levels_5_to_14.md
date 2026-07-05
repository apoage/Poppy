> *Observation: the catalog sits in the cabinet drawer marked "future seasons." Pairs cointegrated like willow and alder. Triangles among three salts. The Ornstein-Uhlenbeck pulse of a salt dissolving and re-precipitating in its own jar. The page on volatility clustering, with its variance that breathes. The drawer is opened in spring and in autumn. The pages are updated, dated, re-filed. None of the plants are sown.*

---

**by Lukas — May 23, 2026**

# The Curriculum That Wasn't (Levels 5–14)

*On the eighteen synthetic regimes the December 2025 vision document promised, the math I actually wrote down for ten of them, and the substrate that moved underneath the whole plan before any of them could be planted.*

---

## What was on the page versus what got planted

The first article in this series — *I Thought I Was Going to Break Causality* — describes a Markdown file called `ALPHAZERO_TRADING_VISION.md`, dated December 19, 2025, in which I lay out a twenty-three-level curriculum for the agent. Levels 0 through 22. Each one a synthetic market regime, each one isolating a single mathematical tactic, each one ordered from trivial to fused. Sine waves at the bottom, multi-tactic fusion at the top. The phrase I used in the doc was "like teaching a Go player all the joseki before letting them play."

Article 0 in this series, and the chronology page, are honest about what actually got built. Levels 0 through 4 — the basic-mechanics block — were implemented. Sine wave, step function, GBM with drift, momentum-only, multi-asset rotation. They ran. The model passed them. By the time Foxglove existed as a phase, the curriculum had stalled at the boundary between "basic mechanics" and "statistical arbitrage." Of the twenty-three levels in the original plan, eighteen were never written. The other five rounded out the basic block and then the plan crossed a substrate transition and stopped.

This article is about ten of the eighteen unbuilt levels. Specifically, levels 5 through 14 — the middle stretch of the original curriculum, the one where the math gets interesting. Pairs trading, triangle arbitrage, index arbitrage, momentum, mean reversion, breakout, bid-ask bounce, order imbalance, market making, GARCH volatility clustering. None of them ran. All of them have a design document on disk — `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md`, 2920 words, with mathematical foundations, generator pseudocode, and pass criteria for every level.

It is May 23, 2026. `workstation` is on. The single 3090 is idle — the seed sweep I queued last night finished at 04:17 and `nvidia-smi` shows the card down to 19 W and 47 °C. The directory `mcts_trader/synthetic/` contains exactly five generator methods: `_level_0_sine`, `_level_1_step`, `_level_2_gbm_drift`, `_level_3_momentum_only`, `_level_4_multi_asset_rotation`. Nothing past `_level_4`. The advanced design doc has been opened, edited, and re-filed perhaps a dozen times across six months. I have read it more often than I have read anything I actually shipped.

What follows is a walk through the catalog. I want to do three things. First, lay out what each of the ten levels was supposed to teach — the level progression table from the design doc, quoted verbatim. Second, take three of them — pairs trading, mean reversion, volatility clustering — into the math, the way I would have if I had ever actually sat down to write the generator. Third, ask the harder question: would building them have changed any of the campaign verdicts that the last twenty articles have been about? I don't think the answer is obvious. The fact that I am writing this article in May 2026 instead of February 2026, with the catalog still in the drawer, is itself the answer to a question I never quite asked out loud — *do I think these levels would have made the model trade better than the real-data SegmentPool did?* — and the answer that emerged by default, by the shape of what got prioritized, was no. Or at least, not enough to delay the substrate change. This article is the place where I make that default decision explicit and ask whether I still agree with it.

## The structure of the missing curriculum

The level progression table, copied from `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md`:

| Level | Name | Tactic | Key Skill |
|-------|------|--------|-----------|
| 5 | Pairs Trading | Statistical Arbitrage | Cointegration detection |
| 6 | Triangle Arbitrage | Cross-pair | Mispricing exploitation |
| 7 | Index Arbitrage | Frontrunning | Index vs components |
| 8 | Momentum | Trend following | Ride winners |
| 9 | Mean Reversion | Fade extremes | OU process |
| 10 | Breakout | Range → Trend | Consolidation detection |
| 11 | Bid-Ask Bounce | Spread capture | Microstructure |
| 12 | Order Imbalance | Predict direction | Volume analysis |
| 13 | Market Making | Liquidity provision | Inventory management |
| 14 | Volatility Clustering | GARCH dynamics | Risk sizing |

Read it once and you will notice the curriculum has structure. There are clusters.

**Levels 5–7 are the statistical-arbitrage cluster.** They are about *relationships between assets*. Pairs trading asks the model to notice that two price series share a mean-reverting spread. Triangle arbitrage asks it to notice that three exchange rates encode a closed cycle that must, by no-arbitrage, multiply to one — and that when they don't, there is money on the table. Index arbitrage asks it to notice that a basket is the same thing as its weighted components, and that a lag between basket and components is exploitable. These three levels are all variations on the same skill: *cross-sectional structure*. The model that passes them has learned not to look at assets one at a time.

**Levels 8–10 are the regime-shape cluster.** They are about *time-series patterns*. Momentum is "the trend that just happened keeps happening." Mean reversion is "the deviation from the mean reverses." Breakout is the joint of the two: ranges that resolve into trends. These three levels do not require cross-sectional reasoning; each asset can be traded in isolation. What they require is *temporal recognition* — the model has to notice that the price's recent shape is informative about its next shape.

**Levels 11–13 are the microstructure cluster.** They live closest to the order book. Bid-ask bounce asks the model to recognize that the high-frequency price oscillation is not signal but the artifact of alternating buy- and sell-initiated trades, and to provide liquidity in the gap. Order imbalance asks it to read the volume field as a directional predictor. Market making asks it to actually quote both sides and manage inventory under adverse selection. These are the levels that go into territory the rest of the curriculum doesn't touch: *the agent as participant, not observer*. The price moves *because of* the agent's quotes, not in spite of them.

**Level 14, GARCH, is the singleton.** It teaches volatility clustering — that today's variance predicts tomorrow's variance. It is the foundation of every risk model in production finance. It is also the only level in the back half of the curriculum that the original design doc gives a single self-contained section. There is no GARCH cluster. GARCH stands alone.

What the doc explicitly promises, in its own prose:

> *"A model is considered 'fully trained' when it passes: 1. Level 14 with +0.5% reward AND 2. Max drawdown < 15% AND 3. No level regression (can still pass Levels 0-13). This represents mastery of: Basic mechanics (Levels 0-4), Statistical arbitrage (Levels 5-7), Momentum/reversal (Levels 8-10), Microstructure (Levels 11-13), Volatility dynamics (Level 14)."*
>
> — `~/Trader/lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md`

This is the criterion for graduation. A model that passes all fifteen levels has, on the design doc's account, *mastery*. None of the models trained in this project ever sat for that exam, because the exam was never written. But the rubric exists, and the rubric is what I want to walk through next, level by chosen level, in the math.

## Three levels, in the math they would have needed

I want to go into mathematical detail on three of the ten. Pairs trading, because it is the cleanest cross-sectional level. Mean reversion, because it is the only level in the entire curriculum for which a closed-form optimal policy is known. Volatility clustering, because it is the level closest to actual production risk management.

### Level 5: Pairs trading on a cointegrated spread

Two assets are cointegrated when their linear combination is mean-reverting even though each of them individually is not. This is the trader's version of a relationship between two random walks: the walks themselves drift, but the *distance between them* breathes around a fixed center.

The math the design doc writes down:

```
Spread(t) = Price_A(t) - β × Price_B(t)
Spread follows OU process: dS = -θ(S - μ)dt + σdW
```

`β` is the hedge ratio. `θ` is the speed at which the spread snaps back to its mean. `μ` is the mean it snaps back to. `σ` is the spread's volatility. `dW` is a Wiener increment — Gaussian noise. The OU process is what you get when you take a random walk and add a restoring force that pulls it toward `μ` with strength proportional to its current displacement.

The trading rule, also from the doc:

> *"Trade when spread deviates > 2σ from mean: Spread > μ + 2σ: Short A, Long B (expect convergence); Spread < μ - 2σ: Long A, Short B."*

The model passes this level when it learns to (a) detect that two assets are cointegrated, (b) wait for the spread to deviate by two standard deviations, and (c) execute the paired trade in the convergence direction. That is three distinct capabilities, and only one of them — the directional trade — looks like anything the basic-mechanics levels teach. The other two — *pair detection* and *patience under deviation* — are new.

The generator that would have produced this level, in the doc, is about a hundred lines of Python. The interesting middle:

```python
THETA = 0.15        # Mean reversion speed
MU = 0.0            # Long-run mean
SIGMA_SPREAD = 0.05 # Spread volatility

common_factor = self._generate_gbm(drift=0.0001, volatility=0.01)

for pair_idx in range(N_PAIRS):
    spread = np.zeros(self.n_steps)
    spread[0] = self.rng.normal(0, SIGMA_SPREAD)

    for t in range(1, self.n_steps):
        dW = self.rng.normal(0, 1) * np.sqrt(1)
        dS = -THETA * (spread[t-1] - MU) + SIGMA_SPREAD * dW
        spread[t] = spread[t-1] + dS

    price_a = common_factor * (1 + spread / 2)
    price_b = common_factor * (1 - spread / (2 * beta))
```

There is a single shared GBM, the *common factor* — call it the underlying market — and then thirty-two pairs sitting on top of it, each pair driven by an independent OU spread. The two assets in each pair share most of their variance (the common factor) and disagree only on a small mean-reverting wobble (the spread). The design specifies thirty-two pairs out of sixty-four assets per episode, so half the universe is paired and the model is forced to discover which asset goes with which.

What would tell me the model had learned this? The pass criteria in the doc are explicit: *"Average reward: > +2.0%; Trade frequency: 10-30% (selective trading); Pair correlation awareness: Model should trade correlated pairs together."* The third criterion is the one that matters. Trade frequency you can hit with any contrarian strategy. Reward you can hit with luck. *Trading pairs together* — taking opposite-sign positions in `SYN_07` and `SYN_39` at the same step, repeatedly, when their spread blows out — is the signature that the cross-sectional structure has been learned.

I never built the readout that would have measured pair-coordination. The metric does not exist in `tools/DS24/`. If I had built the level, I would also have had to build the metric. That is the kind of work the design doc does not budget for and that always takes longer than the generator.

### Level 9: Mean reversion, and the one level with a known optimum

The level I find most interesting in the entire catalog — and the one that I think, in retrospect, I should have built first — is Level 9. Pure Ornstein-Uhlenbeck dynamics. No pairs, no cross-sectional structure, no microstructure noise. Each asset, alone, reverts to a mean.

The math, from the doc:

```
dP = -θ(P - μ)dt + σdW

Where:
- θ = mean reversion speed (0.1-0.3)
- μ = long-run mean (VWAP = 100)
- σ = volatility

Trade when: |P - μ| > 2σ
Direction: Toward μ
```

This is the same OU process as the spread in Level 5, but the price itself is the mean-reverting variable. There is no hidden cointegration to discover. The structure is on the surface. What makes this level special is the result that, for an OU price under a simple linear-quadratic cost on inventory, the *optimal trading policy is known in closed form*. Bertsekas's dynamic programming textbook works it out. The optimal position at each step is a known linear function of the displacement from the mean. The optimal expected utility is a known function of `θ`, `σ`, and the discount rate.

Which means Level 9 is the one level in the entire curriculum where you can compare the agent to *analytically derived ground truth*. Not to a beam-search oracle, not to a hand-tuned baseline, not to "buy and hold." To the actual mathematical optimum, computed in closed form.

If I had built Level 9, the experiment I would have run is the following: train the agent until it passes, then measure the gap between its realized expected return and the Bertsekas optimum. That gap is *exactly the amount of value left on the table by the function approximator and the planner*. It is the cleanest measurement of agent suboptimality the curriculum can produce. Every other level admits the excuse "the oracle is imperfect" or "the baseline is weak." Level 9 admits no such excuse. The optimum is known. The gap is the gap.

I want to be careful not to claim too much, because I never ran it. But I will say this: every other test in this project — the campaign Articles 5 through 22 are about — compares mechanisms against each other in a noisy benchmark with a noisy oracle and a noisy seed distribution. The classifier-Alpha anti-correlation discovered in Article 6 was visible only because we ran paired three-seed tests and looked at the *delta* between conditions. Absolute performance was almost never the metric. We did not, in any campaign, have a single experiment where we could say "the agent attained X percent of its theoretical maximum."

Level 9 would have given me that. And I never built it.

### Level 14: GARCH and the variance that remembers

The last math level I want to walk is the singleton at the top of the curriculum — volatility clustering, GARCH(1,1).

The math, from the doc:

```
Return_t = σ_t × ε_t, where ε_t ~ N(0,1)
σ_t² = α + β × σ_{t-1}² + γ × Return_{t-1}²

Parameters: α=0.05, β=0.85, γ=0.10
```

Each return is a unit-variance Gaussian shock scaled by the current volatility. The current volatility is itself a recursive function of the previous volatility (`β` controls persistence — 0.85 is high) and the previous shock squared (`γ` controls how much a big move feeds the next period's variance). The constant `α` is the baseline variance the system relaxes toward. The hallmark of GARCH is *clustering*: big moves predict big moves, calm predicts calm, and a single large shock kicks variance up and then it decays back on a timescale set by `β`.

The pass criterion is mild: *"Average reward: > +0.5%; Must show vol-aware position sizing; Max drawdown < 15%."* What the level is really teaching is risk control, not direction. The expected return of a pure GARCH series is zero. There is no edge in *direction*. The edge is in *sizing* — taking smaller positions when variance is elevated, larger ones when it is compressed, and the proof that the model has learned this is that its drawdown stays bounded under the highest-variance regimes.

This is the level closest to production trading. The deployed Trader uses a Sharpe-style position-size scaling that, in spirit, does what a GARCH-aware policy would do. But that scaling is hand-engineered. It sits outside the network as a post-processing layer. A model that has *internalized* GARCH would not need the external scaler; it would size correctly because its policy head had learned to do so. None of the trained models in this project have been measured for that. It is one of the things the Level 14 generator would have let me test.

The generator itself is short — twenty lines — but contains the only piece of curriculum math that is recursive on its own output. Every other level's price series is a function of exogenous noise. GARCH's variance is a function of its own history. That makes the generator's wall-clock cost per episode higher than any other level, because the variance update cannot be vectorized across time (each step depends on the previous step's squared return). I am not sure the synthetic curriculum, as built around the basic levels, had a code path that handled per-step recursion efficiently. It would have needed one.

## Market making: the level that probably should not have been built

I want to spend a moment on Level 13, market making, because it is the level in the catalog I am least confident *should* have been built — even in a counterfactual where the rest of the synthetic curriculum was completed.

The design doc proposes a generator that approximates the Avellaneda-Stoikov model:

> *"Quote bid at: mid - spread/2 - inventory_penalty; Quote ask at: mid + spread/2 + inventory_penalty; Where inventory_penalty = γ × inventory × σ²."*

And the structure the generator produces is a price process where 95 percent of the time small noise comes through and 5 percent of the time an "informed" order moves the price by a percent. The model's job is to quote both sides, capture the spread on the 95 percent of trades that are noise, and avoid being run over by the 5 percent that are informed.

The issue is that market making is fundamentally *interactive*. It requires a counterparty model. A market maker quotes prices; the rest of the market either trades against those quotes or doesn't, and the decision of whether to trade is informed by what the quotes themselves reveal. The generator described in the doc has no such feedback loop. It generates a price process *as if* the agent's quotes had no effect on the trades that arrive — which is precisely the assumption a real market maker cannot make. The agent that learns to "trade" this Level 13 has not learned to make markets. It has learned to predict a particular kind of noise-plus-jump process and to size positions accordingly. The skills are adjacent. They are not the same.

The other nine levels of the back half are all *passive-environment* levels — the agent observes prices and decides whether to trade; the prices do not know the agent exists. That is fine for the other nine, because the tactics they teach (cointegration, mean reversion, momentum, GARCH) are tactics on series that are themselves indifferent to the trader. Market making is the one level whose subject matter requires the environment to react. The synthetic-generator format the curriculum is built on does not provide reaction. Level 13 was, on reflection, the wrong level to write a generator for. It was the right level to *flag as needing a different framework* — a small order-book simulator, or a paired-agent setup — and then to defer.

The design doc does not flag this. The design doc proposes a generator and pass criteria as if Level 13 were the same kind of thing as Levels 5–14. The fact that I wrote it that way is a small piece of evidence that, in December 2025, I did not have the taxonomy of *passive synthetic levels* versus *interactive levels* clearly enough articulated to draw the line. Reading the doc in May 2026 I can see exactly where the line should have gone. At the time, I could not.

> *Observation: the pages on cointegration and on volatility clustering are kept together because the substances they describe both breathe. The page on market making is filed in the same drawer but the gardener notes, in the margin, that this one would require a second specimen — a counter-gardener, watering against me — and the second specimen has not been bred.*

## The kickback: three reasons the catalog stayed in the drawer

The honest question this article has to answer is: *if these levels were designed in December and never built by May, what did the project do instead, and was the substitution a good trade?*

The answer is in three layers.

**Layer one: the substrate moved.** Article 20 in this series — *The Dojo* — describes the moment the training environment stopped being the synthetic curriculum and started being something else. The replacement is a piece of code called `SegmentPool`, inside `train_easy.py`. It loads eighteen historical date ranges into RAM, cycles through them every batch, and serves the model windows of real market history — the actual 2018-10 through 2025-11 record at one-minute resolution, 27 channels per asset, 64 assets per episode. Bull regimes, bear regimes, crash, recovery, the post-2022 chop. Every gradient update sees a mixture of all eighteen.

Real bull regimes contain real momentum patterns. Real sideways regimes contain real mean reversion. Real co-moving assets — pairs of large-cap tokens correlated through shared sector exposure — contain something that *is*, statistically, cointegration. The synthetic curriculum was designed to teach tactics *in isolation*: Level 5 is *only* pairs trading, Level 8 is *only* momentum, the noise floor is whatever the generator's `volatility` parameter says it is. The real-data SegmentPool teaches tactics *in superposition*: every batch contains all of them, mixed in whatever proportion history happens to provide, with the noise floor being the actual market's noise floor.

These two are not equivalent. The synthetic approach gives you *clean signal* — you know exactly which tactic is relevant in any given episode, because the generator only put one there. The real-data approach gives you *coverage* — you know the agent has seen what the deployment world looks like, in approximately the proportions it sees in deployment. We chose coverage. The choice happened around the time TASK_710 and TASK_712 went in. The SegmentPool's per-batch all-regime mixture was so much more effective at stabilizing the value head than any per-level training schedule we had tried that the question "should we go back and build Levels 5–14?" was not so much answered as silently dropped.

**Layer two: the choice of coverage-over-isolation is a research-style decision, and I do not yet know if it was correct.** Article 6 in this series — the classifier-Alpha anti-correlation — is about a finding that has held across three independent experiments now: classifier-side improvements (lower loss, higher balanced accuracy) anti-correlate with trader-side improvements (higher Alpha). Better classifier, worse trader. The mechanism, on the cleanest reading, is that a sharper classifier makes the policy more confident in its directional bets, which makes it trade more, which makes it pay more friction, which dominates whatever edge the sharper classification provided.

It is possible — I want to be careful here, because I cannot prove this counterfactually — that the classifier-Alpha anti-correlation is partly a consequence of *too much coverage and not enough isolation*. The model has been trained on real-data SegmentPool from the beginning of Foxglove. It has never seen an isolated pairs-trading episode where the policy that pays off is *exactly* the OU-spread-fade. It has seen instead a mixture of regimes where pairs-trading-like opportunities are smeared together with momentum-like opportunities and with the directional bull-rally pattern that all the campaign Articles 7–14 are about being unable to capture. The classifier, in such a regime, learns to recognize bull and bear *patterns*. It does not learn the clean *policy response* to pairs trading because it has never seen pairs trading in isolation.

An alternative campaign — and this is the campaign the original December 2025 vision document described — would have run Levels 5–14 to graduation, where each level was a clean, isolated, mathematically-grounded tactic with known optimal policy and measurable per-tactic agent performance. Only after passing Level 14 would the model see real data. The hypothesis under that campaign is that by the time the model reaches real markets, its policy head has *already* internalized the ten tactics as separate response shapes, and the real-data phase becomes fine-tuning rather than from-scratch discovery.

I did not run that campaign. I cannot know what would have happened. The honest statement is that the *coverage* path has produced a model whose Alpha sits in single-digit-percent territory and whose campaign mechanisms have, by Article 22's count, all been null. The *isolation* path was never tested. I do not know if it would have done better, because the work to find out — build ten generators, build ten per-level evaluation harnesses, run ten level-by-level Stage-A campaigns, and only then move to Stage-B on real data — was the work I declined to do when the SegmentPool started working well enough to defer.

**Layer three: the catalog survives even though the catalog wasn't run.** `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md` is on disk. The math is in there. The pseudocode is in there. The pass criteria are in there. The version control history shows the file has been touched lightly across the months — small edits to clean up parameter names, one pass to align the level numbering with `MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`, a note added at the top about prerequisites. The file is alive in the sense that it is *maintained*. It is dead in the sense that it has not produced a single training episode.

There is one path under which the catalog gets opened. `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the document that closes out the single-mechanism class of experiments — lists four research directions for the next phase. One of them is *incompatible dataset*: training the agent on data drawn from a distribution materially different from the deployment distribution, as a control on whether the deployment data itself is the limiting factor. The synthetic curriculum, if it were ever built, would be one obvious source of incompatible data. Levels 5–14 are not the deployment distribution. They are, by construction, controlled distributions whose statistics are known. A future campaign that wanted to ask the question "is the bottleneck in the *data*, not the *model*?" could revisit the design doc and build the generators it specifies.

If that happens, future-me opens `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md`, reads the math, copies the pseudocode into `mcts_trader/synthetic/`, and the levels go live. The work the December 2025 me did — writing down the spread formula, the OU dynamics, the GARCH update, the triangle-arbitrage cycle, the breakout phase machine — that work is not lost. It is filed. The catalog drawer is, perhaps surprisingly, the right place for it.

## What is left

The score sheet, as of today:

Levels 0–4 are implemented. Sine, step, GBM with drift, momentum-only, multi-asset rotation. They live in `mcts_trader/synthetic/synthetic_market.py`, they pass the pass criteria, and any agent built after January 2026 can be sent back through them as a smoke test. They are the small portion of the curriculum that survived the substrate transition because they were already shipping when SegmentPool replaced the broader plan.

Levels 5–14 are designed-not-built. The math is in `lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md`. The pseudocode is in the same file. Each level has pass criteria. None of them have been used to train anything. The work to make them live is, by my estimate from the doc, roughly two weeks of generator implementation plus another two weeks of evaluation harness and dashboard work — call it a month for a single engineer with the rest of the codebase already in place. That month has been available, in principle, for five of the last six. It has been spent elsewhere.

Levels 15–22 are partially designed. The companion doc `lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md` sketches the volatility-and-regime cluster: vol carry, regime-shift detection, multi-tactic fusion. The sketches are less complete than `ADVANCED_SYNTHETIC_LEVELS_DESIGN.md` — closer to outlines than to specifications. There is no equivalent of the pairs-trading generator pseudocode for any of them. If the catalog ever does get opened in earnest, levels 15–22 would need a design pass before they would need an implementation pass.

The aspiration is preserved. The implementation is gated on a campaign that does not yet exist. I do not know — and I want to be honest about this rather than pretend I do — whether that campaign will ever exist on this project, or whether the work that would have gone into the synthetic curriculum will end up reallocated permanently to real-data interventions, planning-side mechanisms, and the structural pivots that Article 22 and `TASK_761` are about. Either outcome is consistent with what the project has been. The catalog stays in the drawer either way. The drawer is opened periodically. The catalog is updated. None of the plants are sown.

> *Observation: the seasons turn. The catalog is opened in spring. The pages are dated, re-filed. The gardener moves to other beds.*

---

## Sources

- `~/Trader/lore/ADVANCED_SYNTHETIC_LEVELS_DESIGN.md` — primary, the catalog of Levels 5–14.
- `~/Trader/lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md` — companion, sketches Levels 15–22.
- `~/Trader/lore/SYNTHETIC_CURRICULUM_DESIGN.md` — broader curriculum design (Levels 0–22 framing).
- `~/Trader/bloom/drafts/01_break_causality.md` — the December 2025 vision document and the twenty-three-level promise.
- `~/Trader/bloom/drafts/06_classifier_alpha_anticorrelation.md` — the classifier-vs-Alpha finding referenced in the kickback.
- `~/Trader/bloom/drafts/20_the_dojo.md` — the SegmentPool that replaced the synthetic curriculum.
- `~/Trader/bloom/drafts/22_year_of_negative_results.md` — the campaign count that this article writes against.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the *incompatible dataset* direction that could, in principle, revive the catalog.
