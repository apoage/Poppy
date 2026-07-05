# Better Classifier, Worse Trader

*On three independent experiments that broke the most natural assumption in supervised-then-policy training: that a sharper directional head must be the foundation of a better trader. On the moment I stopped proposing encoder mechanisms. On a research report I commissioned only after the bug was already named.*

> *Observation: a specimen in the south bed flowers more brightly each season and yields less fruit each season. I weigh the harvest. The petals have multiplied. The seed-count has halved. I record the inverse, in two columns, on the same page. I do not, at first, believe my own scale; I weigh the harvest again. The scale is correct. The specimen is not a contradiction. The specimen is information.*

---

## The morning I read the row

`workstation` is on. The desk fan is on. `nvidia-smi` is ticking through the post-eval validator in a tmux pane I forgot I opened. The 3090 is at 41°C, idle. It is the morning of 2026-05-15 and on the second monitor I have the result table for TASK_738 — the legacy 27-channel encoder ablation, three seeds, paired against the canonical Stage-B baseline.

The row I am looking at says, in the column it calls **Δ Bal.Acc**, **+7.48pp**. Balanced accuracy on the directional head — the classifier that predicts the sign of the next-period return for each asset — has gone up by seven and a half points. That is enormous. On a stack where seven months of work has been spent trying to break a 0.574 directional-accuracy ceiling, *seven and a half points of balanced accuracy* is the kind of number you stop and reread, because surely it is a logging bug.

It is not a logging bug. The next column, **Δα** — paired Alpha against the same seeds of the same baseline — says **−18.44pp**. The column after that, **Trade count (sample, bull period, seed 7)**, says **136 → 13,246**. The column after that, **Sharpe (same sample)**, says **+8.68 → −27.20**.

I sit there with my coffee and I do the thing I always do when a number is impossible, which is to walk through the pipeline imagining where a sign could have flipped. There is no sign flip. The encoder is genuinely classifying better. The trader, using that genuinely-better classifier, is genuinely trading itself into the ground. Fifty times the trade rate of baseline. Sharpe sign-flipped from a respectable plus eight to a catastrophic minus twenty-seven. Six out of six periods negative. All three seeds negative.

This is the third time in two weeks that this exact shape has appeared, and that is the thing that makes me put the coffee down. The first time, in TASK_737, it had been confounded with a single-seed outlier and I had not yet seen the shape. The second time, in TASK_741 Mint, it had been small enough and ambiguous enough that I could file it under *interesting failure*. The third time — TASK_738, the catastrophic one, fifty-times over-trading and the Sharpe in the basement — was when the pattern became a property of the system instead of a series of accidents.

The article you are reading is about the property.

---

## What I thought was true

Let me restate the naive theory, because it had been the working theory for the entire previous research program.

The trader has, at the top of its stack, an **encoder**. The encoder takes a window of market state — currently 39 channels, originally 27, augmented somewhere around April with the cross-sectional features that the TASK_732 work added — and produces a per-asset latent vector that the rest of the network reads from. Attached to the encoder during Stage-B (the supervised-then-RL warm-start phase that Foxglove was specifically built for) are several heads: a **directional classifier** that predicts the sign of the next-period return, a **rank head** that predicts cross-sectional ordering, a **value head** that estimates expected discounted return under the current policy, and a **policy head** that produces an action distribution which the MCTS planner uses as its prior.

The classifier is the cheapest signal to read during training. It produces **Bal.Acc** — balanced accuracy across the long/short/flat partition of next-period returns — and it produces a training **loss** that is the dominant component of the Stage-B objective. Loss goes down, Bal.Acc goes up, and that has been the proxy by which every Stage-B experiment for six months has been graded.

The naive theory is: a sharper directional classifier is a better foundation for the policy head sitting next to it. The policy reads from the same shared encoder; if the encoder learns features that classify direction better, those features are more informative for the policy, the policy bets more accurately, Alpha goes up. *Better classifier, better trader.* This is the theory most ML people would write down without thinking about it. It is the theory under which Foxglove was designed as a *supervised imprint phase* — the entire architectural premise is that classification quality bootstraps planning quality. It is the theory that explained, at least to my satisfaction, why every research subtask in Phase 2 had been an encoder-side intervention: encoder redundancy (TASK_136), regime mixing (TASK_710 and TASK_712), return-weighted trade loss (TASK_713), the dense-sign auxiliary objective (TASK_733), the cross-sectional channel debate (TASK_732), per-asset signal injection (TASK_741 Mint). Every one of those touched the encoder, the heads attached to the encoder, or the loss landscape *of those heads*.

The theory held until April 2026. Then it started cracking, in a way that — looking back now — was visible in TASK_737 if I had known how to read it. The crack opened wide in TASK_741. The crack closed over the head in TASK_738.

What I want to do in the rest of this piece is walk the three confirmations, show the math of why a sharper classifier makes the policy over-trade, quote the research report I commissioned only after the fact, and then describe what happens to a six-month research program when you discover its central premise was working in the wrong direction.

---

## What "Alpha" means here, and why it can hate Bal.Acc

Quick glossary, because the words have to mean the same thing for the rest of the piece to work.

**Bal.Acc** (balanced accuracy) is the classifier head's accuracy on the three-way next-period sign label — long, flat, short — averaged across classes to correct for label imbalance. A random model scores around 0.33. The Stage-B baseline, after 200 epochs, sits around 0.57 — the famous **0.574 ceiling** that an entire family of TASK_733-shaped experiments tried to break.

**Δα** (Alpha) is the deployment metric. After Stage-B training finishes, the checkpoint is taken into eval on six held-out benchmark periods (bull, bear, sideways, ftx_crash, etf_approval, mixed-regime) and the trader is run with its production decision operator. Returns are compared against a passive equal-weight baseline of the same assets across the same window. The difference, in percent, is the Alpha. **Δα** is the paired difference between a variant's Alpha and the baseline's Alpha — same seeds, same data, same decision operator, only the experimental mechanism flipped.

**Paired multi-seed Δα** is what we trust. After TASK_743 demolished the TASK_737 single-seed champion in May (see the previous article in this series), the rule has been: nothing is real on this stack until you have three seeds, paired against the baseline at the same three seeds, with all three deltas in the same sign or with a mean magnitude that dwarfs the spread. Single-seed gaps under about 5pp are noise.

Trade count, Sharpe, fees, and per-period decomposition are diagnostic columns that explain *why* a Δα is what it is.

The naive theory says **Δ Bal.Acc** and **Δα** are positively correlated. The empirical finding of May 2026 is that on this stack, in the experimental regime we have actually been able to probe, they are *anti-correlated, statistically reliable, all-seeds*. That is the entire claim of this article. The rest is the evidence and the mechanism.

---

## Confirmation one: TASK_737, retroactively legible

TASK_737 is the experiment from the previous article in this series — the four-cell ablation, rank-head crossed with auxiliary classifier, on 2026-05-13. The headline number was `no_aux_solo` (rank ON, auxiliary classifier OFF) scoring **+6.21% Alpha** against a paired `a1b1` baseline at **−1.99%**. A thirteen-point spread on a single seed. The diary entry called it a champion. Then TASK_743 reseeded `no_aux_solo` across seeds 7, 13, 23, 31, 42 and the mean came back at **+3.61%** with a range **−4.38 to +7.82** — a 12.2pp spread, the original +6.21% identifiable as the upward outlier. The champion melted.

The point of revisiting TASK_737 in *this* article isn't the seed variance — Article 7 covers that — it's what `no_aux_solo` actually was, mechanically, on the classifier-vs-Alpha axis. The variable that flipped between cells was the **auxiliary classifier loss**. Turning the auxiliary head OFF removes one of the two pressures sharpening the directional probabilities during Stage-B. The model that came out was, on training-side metrics, a *worse classifier* — its training loss was higher, its Bal.Acc was lower. And in the single-seed read, it was a *better trader*.

From the workstation diary on 2026-05-13:

> The two heads interact NEGATIVELY (−13pp combined effect). aux ON costs Alpha; rank ON earns Alpha; their combination is worse than the parts.
>
> — `lore/notes_2026-05-13.md` §95

That entry has the texture of someone proposing a mechanism three minutes after seeing a number, which is precisely what was happening. The mechanism I invented to explain *why* aux-off was better was that the two heads were fighting for representational capacity in the encoder, and that removing one freed the other to bet harder. That mechanism is wrong. The real mechanism — the one I would not have words for until two days later — is that the auxiliary head was *successfully sharpening the classifier*, and the sharpened classifier was hurting the policy. Turning the aux head off *broke the classifier slightly, which let the trader breathe*.

I did not see it. I wrote a victory writeup. Then the second seed loaded and the second seed said −4.38% and the writeup got rewritten with the word *unreliable* in it twice. (`runs/737_eval/result.md` is now historical, not canonical.) The single-seed outlier hid the structural pattern; the structural pattern is the one I am writing about now.

---

## Confirmation two: TASK_741 Mint, the first clean signal

TASK_741 Mint is the experiment that ran on 2026-05-14, twenty-four hours after the TASK_737 retraction. It is also the experiment that, more than any other, made the anti-correlation visible *in a single, properly-paired, multi-seed run*.

The mechanism: a small auxiliary injection — a FiLM-style modulation of the post-temporal-pool encoder surface — using five per-asset cross-sectional scalars (ranks, beta, idiosyncratic volatility, cross-sectional momentum) lifted from channels 27–31 of the encoder input. The intuition was clean and, on its own terms, defensible: the cross-sectional scalars carry per-asset information that gets diluted when the temporal pool aggregates across timesteps; re-injecting them after the pool restores the per-asset signal where the classifier needs it most. The design doc was `lore/adr/0007-film-injection-post-temporal-pool.md`. The implementation was done by a subagent on 2026-05-13 and committed as `1957a6c`.

The training-side result was a clean victory:

| Metric | Δ AVG (mint − base) | Per-seed Δ | Verdict |
|---|---:|---|---|
| Bal.Acc | **+3.29pp** | +7.78, −0.25, +2.34 | Mint = better classifier |
| Training loss | **−0.8** every seed | — | Mint clearly fits the loss better |

A real and reliable improvement in the classifier — multi-seed, paired, large enough to clear noise. On the training dashboard, Mint looked like the cleanest win of the entire Phase-2 mechanism sweep.

Then the deploy-time table loaded.

| Metric | Δ AVG | Per-seed Δ | Verdict |
|---|---:|---|---|
| Alpha | **−6.98pp** | **−14.62, −1.22, −5.08** | Mint = worse trader (all 3 negative) |

From the result-card the subagent wrote:

> Same failure mode as TASK_737: adding a head/mechanism that sharpens the classifier *can hurt deployment metrics* if it makes the policy over-confident in the wrong direction during trending regimes. The training loss / Val Agree is NOT a reliable proxy for deployment Alpha on this setup.
>
> — `project_task741_result.md`

And, the sentence that ended up in the lore notes as the shorthand for the whole class of failure:

> **Better classifier, worse trader.**
>
> — `project_task741_result.md`

Per-period decomposition told me where the damage lived: bull regime, ftx_crash, etf_approval — the rally and trend regimes — each lost 8 to 11 points of Alpha. Sideways and bear were less affected. The classifier got sharper everywhere; the trader got worse specifically where directional signal was *strongest*. That is a fingerprint, not a coincidence. The sharper the classifier, the more decisively the trader bet, the more transaction costs ate the per-bet edge, the worse it did. The variance came from the regime; the mechanism was uniform.

I want to make this concrete: TASK_741 is the first time, on this project, that the anti-correlation was visible as a *clean, properly-paired, multi-seed effect*. TASK_737 had been a single seed; the multi-seed reanalysis under TASK_743 had washed the gap out into noise. TASK_741 was three seeds. All three classifier deltas positive. All three Alpha deltas negative. The directionality was identical. The magnitude was a paired 6.98 points against. It was, in the literal sense, *the experimental result that should have ended the encoder research program.* I knew it was important. I had not yet seen the third confirmation that would make me say it out loud.

---

## Confirmation three: TASK_738, the catastrophic one

TASK_738 ran on 2026-05-15 as part of the Tier-2 mechanism sweep — one of three variants (738 legacy_27ch, 739 dense_sign, 734 flow_matching) batched together to interrogate three different angles of the same question. TASK_738 was the encoder-side variant: revert to the legacy 27-channel encoder input (drop channels 27 through 38, the cross-sectional augmentation introduced in TASK_732), retrain Stage-B from the legacy 27-channel Phase-1 checkpoint, evaluate paired against canonical baseline. The hypothesis being tested: were the cross-sectional channels actually load-bearing for Alpha, or had they been making things worse all along?

The training-side result, again, looked clean:

| Metric | Value |
|---|---:|
| Δ Bal.Acc | **+7.48pp** |
| Δ training loss | **−0.98** |
| Δα (paired, 3 seeds) | **−18.44pp** |
| Per-seed Δα sign | **0/3 positive** |
| Per-period Δα sign | **0/6 positive** |

From the lore entry:

> Reverting cross-sectional encoder input (drop channels 27-38, use legacy 27-ch P1 ckpt) trains a "better classifier" (Bal.Acc +7.48pp, loss −0.98) but trades **catastrophically worse** (Δα −18.44pp, all 3 seeds, all 6 periods negative). Bear regime worst hit (−30.09pp).
>
> — `lore/notes_2026-05-15.md` §107

And the diagnostic that made it impossible to interpret as anything other than the same failure mode at twice the magnitude:

> Mechanism (sample, bull period, seed 7): trade count 136 → 13,246 (50× over-trading), Sharpe +8.68 → −27.20. Same signature as TASK_741_mint_xs.
>
> — `lore/notes_2026-05-15.md` §107

Fifty times the trade rate of baseline, on a single bull-regime sample. Sharpe sign-flipped from +8.68 to −27.20 in the same sample — which is to say the trader was no longer mis-priced, it was *actively dumping value into the order book*. The 27-channel ablation had produced a classifier that was the closest thing to clairvoyant we had ever measured on this stack, and that clairvoyance had been translated, by the policy stack downstream, into a strategy whose realized return was indistinguishable from setting money on fire on every bar.

This is the moment the anti-correlation stopped being a phenomenon I was tracking and became a *property of the system I had to design around.* From `lore/notes_2026-05-15.md` §110 — the page that ended up linked from MEMORY.md as the canonical pointer for the whole class of failure:

> Better classifier (loss DOWN, Bal.Acc UP) → worse trader (Δα DOWN, all seeds negative, over-trading collapse).
>
> Trio of regressions (737 no_aux_solo single-seed outlier, 741 mint_xs, 738 legacy_27ch) all share signature: training loss improves significantly while Alpha collapses with trade-rate explosion. Three independent experimental setups, three confirmations — this is the dominant failure mode of "make the encoder/classifier sharper" interventions on this stack.
>
> **The bottleneck is downstream of the encoder.**
>
> — `lore/notes_2026-05-15.md` §110

I wrote that paragraph on the morning of 2026-05-15 with the same cold feeling I imagine the gardener has when the inverse relationship has been weighed three times and is no longer dismissable. Three setups. Three independent mechanisms — auxiliary-head removal, FiLM injection, channel reversion. Three signatures, all the same shape: loss down, Bal.Acc up, trade rate explodes, Alpha collapses. The notion that this was variance — that three different mechanisms had each independently rolled the same bad seed — was statistically absurd. Three different gates of the same kind had been kicked, and the same thing had come out from behind each one.

Something else came out of that morning too, which I want to state plainly because it's the part the dashboard doesn't show: every one of the three variants *passed its training-side gate*. Stage-B was scored on classifier loss and Bal.Acc. Each variant lowered the loss; each variant raised the Bal.Acc. By the criteria under which the previous six months of work had been graded, all three variants were improvements. The gates were the wrong gates. We had been promoting checkpoints into eval on a signal that was, on this stack, *anti-correlated with the deployment metric we actually cared about*. The fact that it took three converging confirmations to see this is, I think, the embarrassing part. The fact that I am writing this now and not in November is the part I have to live with.

---

## The math of why this happens

The mechanism is not mysterious once you write it down, and I should — because by the time the third confirmation had landed I needed a story for why three different upstream tweaks were all producing the same downstream failure.

Start with the policy. At decision time, the production trader doesn't sample directly from the policy head. It runs MCTS rooted at the current state, expanding child nodes proportionally to a PUCT score that is a function of the policy prior P(s,a), the visit count N(s,a), and the value estimate Q(s,a). After the simulation budget is spent (32 to 128 sims, depending on the run), the final action distribution is the visit-count distribution at the root. The trader then picks an action — argmax in benchmark, sampled-with-temperature in some deploy configurations.

Two things determine whether a trade actually happens. First, the **sharpness of the value function across actions** — how much better, in expected Q-value, the top-ranked action is than the second-ranked at the current root. Second, the **temperature** of the final action selection — how much that Q-value gap has to dominate to push the policy past its hold-by-default prior.

Now consider what happens when the upstream classifier gets sharper. The classifier shares parameters with the encoder backbone that produces the latent state. The value head reads from the same latent. A sharper directional classifier means the encoder's latent is more directionally informative *per asset, per timestep*. The value head, reading from a more discriminative latent, produces value estimates with a wider per-action spread. Q(s, long) and Q(s, short) become more confidently separated from Q(s, hold). MCTS visits concentrate. The visit-count distribution at the root spikes. The argmax decision becomes near-deterministic at any bar where the sharpened logits cross a small threshold.

The Alpha that results is, schematically,

```
Alpha = gross_return − (trade_rate × fee_per_trade) − slippage
```

For small classifier improvements, the marginal **gross_return** per additional trade is small — the classifier is sharper but it isn't *correct*, especially in trending regimes where the per-asset cross-sectional signals it's leaning on (ranks, beta, idio-vol) decorrelate from forward returns. The marginal **trade_rate**, however, is large — the value-head spread is now driving every borderline bar across the threshold. Fees scale linearly with trade rate. Slippage scales worse. The trader bets more often, isn't enough sharper to make those bets net-positive after fees, and ends up Alpha-negative *exactly when its classifier looks best.*

The trade-count numbers from TASK_738 are this math made empirical. 136 trades at baseline, in a bull period where the optimal strategy is approximately "buy and hold and ignore noise." 13,246 trades under the better-classifier variant in the same period, an essentially-continuous churn of micro-positions. The fee bleed alone, at the canonical 5bp round-trip, eats more than the entire baseline Alpha for that period. The Sharpe collapse from +8.68 to −27.20 is what happens when the realized return drops by an order of magnitude *and* its variance goes up at the same time. There is no positive interpretation. The model has become certain enough about each bar's local direction to bet on it, but the bets are not, in aggregate, more informed than baseline — they are just more numerous.

The naive theory expected the curve. The empirical finding is that on this stack, in the configuration we have actually trained, the curve has the *opposite slope*. Sharpen the classifier, lose Alpha. Three confirmations. It is not subtle.

---

## The Apothecary, briefly

> *The specimen flowered. The blossoms doubled. The fruit halved. I weighed the harvest a third time. I marked the contradiction on the chart, in the second column, in a different ink. I will not increase the dose of fertilizer. I will look, instead, at where the fruit is taken to market.*

---

## A research report that explained it after the fact

In early May I commissioned a deep-research report. The brief was something like *"in a MuZero-style agent with a Neural-ODE prediction head, what are the documented failure modes when policy collapses to HOLD and the world model collapses to identity?"* I had been chasing a tangentially-related pathology in some earlier runs — Bella's Wasserstein collapse, an unrelated symptom from a different ablation — and I wanted a literature scan to ground the diagnostic vocabulary.

The report came back as a forty-page synthesis titled *"Coupled Policy-Prediction Collapse in Model-Based Reinforcement Learning: A Comprehensive Analysis of Dual Failure Modes in Stochastic Trading Environments."* Its central claim is that policy collapse and prediction collapse are not independent pathologies but two halves of a single feedback loop. The report names the components — Entropy Collapse, Dormant Neuron Phenomenon, Trivial Solution Trap, Posterior Collapse — and walks the coupling.

I want to quote two passages, because — and this is the part I want to be honest about — the report did not predict the anti-correlation I was about to find. It explained it after I had already named it. The asymmetry is part of the lesson.

The first passage is on the trivial-solution attractor in the prediction head:

> "The Trivial Solution: In many physical and financial systems, the function f(z,t)=0 is a strong local minimum for the optimizer. [...] In financial time series, the 'no change' prediction is often statistically superior to a random direction prediction (due to the Martingale property of efficient markets) and yields a low MSE, dominated only by the aleatoric noise (volatility)."
>
> — Gemini Deep Research report, commissioned 2026 (`docs/AI_bleed research/big gemini.txt`)

The second passage is on the coupling, on the *exact* feedback loop that the TASK_738 trade-count number is the empirical fingerprint of:

> "The interaction between Policy Collapse and Prediction Collapse creates a stable equilibrium of failure. [...] The Neural ODE trains on this data. Since the action is always HOLD, the model never learns the causal consequence of trading. [...] MCTS uses this atrophied model. When the search tries to expand a BUY node, the model (trained only on HOLD) returns a trivial next state or an inaccurate one. The value function V(s) evaluates this state as no better than the current state."
>
> — Gemini Deep Research report, commissioned 2026

The first quote names *prediction collapse* — the world model converging on identity because identity is a strong local minimum. The second names the coupling that turns prediction collapse into policy collapse, or — and this is the part directly relevant to TASK_738 — the inverse, where a *sharpened* prediction surface forces the policy past its hold-by-default prior into over-trading. The trading-side failure mode the report describes most directly is the over-cautious HOLD collapse; the dual to that, which my experiments were producing, is the over-confident over-trade collapse. Same coupling, opposite sign. The math the report sets up — value-head saturation, policy entropy as the gating quantity, the feedback loop between data distribution and model confidence — is exactly the math under which "better classifier, worse trader" is a *prediction* of the framework, not an anomaly.

I want to be careful not to overclaim what the report bought me. It did not predict the May 13–15 results. It would not have, even if I had read it in April, because the surface symptom in my experiments was the opposite of the symptom the report was written around. What the report did was *give the failure mode a vocabulary*. Coupled collapse. Entropy floor. Value-head saturation. After 2026-05-15 I had a name for what I was looking at, and the name belonged to a documented class of pathologies in deep RL that long predates my project. That is a different and lesser thing than predictive insight, but it is not nothing. It means the bug is not exotic. It means the answer probably lives in the documented mitigation literature.

The deeper lesson — the uncomfortable one — is the temporal order. The model that explained my bug arrived in my inbox *after* my own experiments had named it. If I had read the literature first, I might have designed the gate differently. I did not. I designed the gate on the naive theory, ran six months of experiments through it, and only after the third confirmation went looking for the framework that would have predicted the first. That is a real failure of research discipline, and I want it on the page in this article and not hidden in a methods footnote.

---

## The kickback, in two parts

The kickback has two faces and I owe both of them.

The first face is the research program. Every encoder-side intervention from January through April had been pointed at the wrong part of the stack. TASK_136 (encoder redundancy compression, 22 GB → 338 MB, beautiful piece of engineering that improved nothing downstream). TASK_710 and TASK_712 (SegmentPool, all seventeen regimes mixed simultaneously, an architecturally elegant solution to a problem that wasn't the bottleneck). TASK_713 (return-weighted trade loss, designed to sharpen asset-selection precision in the classifier head, which on the May 2026 evidence would have *cost* Alpha if it had worked). TASK_733 (the dense-sign auxiliary, the breakthrough-the-dir_acc-ceiling family of mechanisms, all of which were designed to do *more of the thing that anti-correlates with Alpha*). TASK_732 (cross-sectional channels, which TASK_738 then proved are load-bearing in a non-obvious way — keeping them costs Alpha when paired with the sharpened heads, removing them costs more Alpha because the policy depends on them, the local optimum of "keep them and live with the trade-rate explosion" is the canonical baseline). TASK_741 Mint (per-asset post-pool injection, the cleanest classifier improvement of the whole program, and the cleanest Alpha regression).

Six months of work, almost all of it on the same shared backbone, almost all of it pushing on the metric that — under multi-seed deployment evaluation — is *negatively correlated with the metric the project exists to optimize*. The bottleneck was never the encoder. It was the policy's reaction to a sharper encoder. Specifically: the value head's effective temperature, the MCTS's visit-count concentration under a sharper value surface, the trade-gating threshold, the policy entropy regularization. Every one of those is downstream of the encoder. None of them were in the active research queue as of 2026-05-14. As of 2026-05-15, in the document that became `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`, all four directions are explicit lanes of the structural pivot. The downstream-of-encoder thesis is the spine of every experiment that has run since.

The second face is the metric. Bal.Acc had been the *eval-time signal we trusted*. When TASK_737 first looked like a champion, the reason it looked like a champion *to us* — to me, in the diary entry I wrote on the morning of 2026-05-13 — was partly the Alpha number and partly the fact that the training trajectory looked clean: loss curve down, Bal.Acc curve up, all the visible-on-the-dashboard signals saying *this one is working*. The Alpha number turned out to be a single-seed outlier; the training metrics turned out to be the *opposite* of a confirmation. Both of the things I had been treating as positive evidence had been pointing in the wrong direction. From the memory entry that ended up canonical for this class of failure:

> Trust **paired Δα over 3 seeds** as the success metric going forward. Don't promote on Bal.Acc or loss alone.
>
> — `project_task741_result.md`

The uncomfortable form of this lesson is that in deployed RL — at least, in the deployed-RL regime my single-GPU workstation has actually been able to probe — intermediate training metrics and the deployment metric can be correlated, anti-correlated, or orthogonal. There is no general theorem that tells you which. You have to *measure both, on every experiment, before you trust either.* I had been measuring both, but I had been gating promotion on the wrong one. The rule that came out of 2026-05-15 — paired Δα over three seeds is the only trustable signal, training metrics are diagnostic only — is the rule under which every experiment since has been graded. It is a slower rule. It costs more compute. It makes the campaign honest in a way it hadn't been.

---

## What's load-bearing now

It is 2026-05-21 as I write this. The 3090 is running a sims=64 sweep on `workstation`, scoring an old checkpoint under the new eval operator; the desk fan is going; `nvidia-smi` says 67°C and 318W. The anti-correlation finding is six days old and is already the most-referenced lore page of the project. The phrase *"better classifier, worse trader"* is in three memory entries and at least five task SPECs as a one-line shorthand for the failure mode. The Gemini deep-research report is the reference text I open when I want to remember why the policy and the predictor are coupled in the first place.

The encoder hasn't changed since 2026-05-15. The 39-channel canonical Stage-B is the baseline against which everything is measured. The active research queue is downstream: TASK_734 flow-matching (a dynamics-side, not classifier-side, mechanism, which on three seeds produced a paired Δα of +3.38pp with all six periods positive — the first credible Tier-2 candidate of the whole program); the TASK_759 eval-MCTS recalibration sweep; reward shaping, entropy regularization, value-head temperature, richer oracle depth. None of these touch the directional classifier. By design.

Article 10 in this series is going to be about the structural pivot — how four Tier-3 directions got proposed after the single-mechanism class as a whole exhausted itself. Article 7 is the TASK_737 single-seed story in detail. Article 8 is TASK_743, the five seeds and the funeral. This article — the one you are reading — is the connective tissue between them, the moment the bottleneck moved.

I want to end on the thing I said three paragraphs in and want to say again: the property of the system is not that the encoder is uninteresting. The property is that *on this particular stack, the policy's reaction to a sharper encoder is to over-bet, and over-betting at this fee structure costs more Alpha than the sharper signal earns*. That is not a universal claim about MuZero traders. It is a specific claim about *this* trader on *this* hardware against *this* fee model with *this* MCTS budget. Change the fee structure, change the planning depth, change the policy entropy floor, and the sign of the anti-correlation might flip back. The whole point of the structural pivot is to *change those things on purpose* and see whether the sign returns.

> *The gardener does not blame the flowers. The gardener finds the road to the market.*

---

## Sources

- `lore/notes_2026-05-15.md` §107 (TASK_738 legacy_27ch), §110 ("pattern reinforced") — primary citation, the three-way confirmation page.
- `lore/notes_2026-05-13.md` §95 (TASK_737 four-cell ablation), §99 (TASK_743 multi-seed reversal), §106 (TASK_741 Mint narrative).
- `~/.claude/projects/-home-lukas-Trader/memory/project_classifier_alpha_anti_correlation.md` — MEMORY.md-linked canonical entry for the failure class.
- `~/.claude/projects/-home-lukas-Trader/memory/project_task741_result.md` — TASK_741 Mint paired three-seed result card.
- `runs/738_legacy_27ch/result.md`, `runs/741_mint/result.md`, `runs/737_eval/result.md` (historical), `runs/734_flow_matching/result.md` — per-experiment result cards.
- `tasks/TASK_733_*.md`, `tasks/TASK_737_*.md`, `tasks/TASK_738_*.md`, `tasks/TASK_741_*.md`, `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — task specs.
- `lore/adr/0007-film-injection-post-temporal-pool.md` — TASK_741 Mint design.
- `docs/AI_bleed research/big gemini.txt` — *"Coupled Policy-Prediction Collapse in Model-Based Reinforcement Learning,"* Gemini Deep Research report commissioned 2026; quoted §3.1 (Trivial Solution) and §4.1 (Cycle of Stagnation).
