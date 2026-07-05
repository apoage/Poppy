# 39 — The 98.9% Hold Ratio: Oracle Trade Density as a Hard Ceiling

> *Observation: the apprentice weighs the powder on a scale calibrated for grams. The dose is measured in micrograms. For nearly every patient the scale reads zero, and the gardener — reading only the scale — concludes the medicine has no effect. I file the scale with the other instruments that flatter their operators.*

---

By Lukas. Workstation: `workstation`. Single 3090. The training run that frames this piece logged 620 epochs followed by another 210, plateaued, and refused to move.

## The graph that wouldn't go up

`workstation` is on. `nvidia-smi` shows the card warm at 320 W, oscillating gently as the data thread keeps feeding it. The DS24 dashboard at `localhost:8050` has been parked on the same plot for weeks: `balanced_agreement` against `epoch`, a curve that climbs through the 40s, breaks into the 50s, and then — somewhere around epoch 200 — *settles*. Not crashes. Not oscillates. Settles. Like a stone reaching the floor of a pond.

The number it settled at was **52.5%**. Two full training runs — 620 epochs, then a fresh restart for 210 more — both ended up there. I had a list of six prior interventions stacked into the loss path, each one a small bugfix or a targeted lever I had argued (correctly, in isolation) should move the metric. None of them did. The plot stayed flat.

When the same dashboard reads the same number for the third week running, you stop trusting your interventions and start suspecting the metric. That suspicion is what this article is about.

The metric below the metric was `oracle_hold_ratio = 0.989`. The oracle — the supervised teacher that produces the labels Foxglove imitates — was telling the model to HOLD in 98.9% of all positions, across all 64 assets, across all 200 steps in every training window. The model, being a competent classifier, had learned to do exactly that. **52.5% balanced agreement is what you get when the model has nailed HOLD and is guessing on everything else.**

This piece is the upstream sibling to Article 29 ("Bull Blindness"). Article 29 documented the *consequence*: every Foxglove imprint we trained failed in bull/rally regimes because it couldn't be coaxed to take a position when the market was rising. The class-imbalance math there is the same math here, but Article 29 looked at it from the regime side — *why does this break in bull markets?* This article looks at it from the construction side — *why does the label distribution have this shape to begin with?* The label distribution didn't drift into 98.9% HOLD. It was *defined* into 98.9% HOLD by a deliberate design choice that we hadn't yet recognized as load-bearing.

## The number that wouldn't move

Let me lay out the sequence honestly, because the shape of the sequence is half the lesson.

TASK_674 closed six concrete plateau bugs. One of them was a sample-counter mistake that caused the experience buffer to refresh roughly 17× less often than intended — *"Each visit takes 17 samples but counter only increments by 1. An episode with `SAMPLES_PER_EPISODE=200` gets sampled `200 × 17 = 3400` times before exhaustion instead of 200"* — from `TASK_674_TRAINING_AGREEMENT_PLATEAU_BUGS.md`. Another was an optimizer warmup that reset every merge, eating 2.5 epochs of usable LR after each model swap. These were real bugs. Fixing them was real progress. The number didn't move.

TASK_675 cleaned fifteen technical-debt items — the LR-schedule `decay_steps` was 5× wrong (cosine decay configured for 2.5 million steps when the run only had 500 k), the validation path was re-encoding through the TRM every batch instead of using the latent cache, a fistful of similar quiet inefficiencies. These were also real. The number didn't move.

TASK_677 — the Turtle Breaker, which Article 31 covered in its own right — applied a post-averaging temperature sharpener to the multipass oracle's policy targets. The intuition was that averaging ten oracle passes was washing out the signal: *"if 7/10 say BUY and 3/10 say HOLD, target is `[0, 0.3, 0.7, 0, 0]` … this 70/30 split makes the 'correct' action look ambiguous to the model."* Sharpen post-average, the argument went, and the targets would regain their bite. The sharpening worked. The targets did look more like the original peaked ones. The number didn't move.

TASK_679 went after the threshold directly. The oracle accepted a trade only if forward cumulative return cleared `min_alpha_threshold = 0.001` over a 20-step lookahead — a 2% bar. I halved it to `0.0005` — a 1% bar — under the explicit hypothesis that the trade-event count was the binding constraint and a lower threshold would simply let more of the existing positive moves cross it. The number didn't move.

TASK_680 was the last attempt before I stepped back. Coin-flip trade-biased sampling inside `OnTheFlyGenerator`: pre-compute the timesteps in each episode where the oracle traded, then 15% of the time sample from that list instead of uniformly. The model sees trade contexts roughly 6× more often per batch with the bias on. From `TASK_680_TRADE_BIASED_SAMPLING.md`: *"Bias timestep sampling toward windows where the oracle actually traded. Use a coin-flip approach (not weighted np.random.choice) for O(1) sampling."* It is a clean, cheap, well-targeted intervention. We validated it didn't introduce leakage and didn't blow up gradient variance. We ran it. The number didn't move.

Five distinct interventions, each with a defensible mechanism, each implemented correctly. `oracle_hold_ratio` came in at 0.989 every single time. `agree_trade` — the agreement metric restricted to non-HOLD ground-truth actions — was **8.9%**. Random would be roughly 20% (one of five trade-class buckets). The model wasn't underperforming on trade decisions; it was *anti-performing* on them.

When five orthogonal interventions all fail to move a metric in the same direction, the metric isn't *stuck*. It's *fixed*. There is a constraint upstream of every lever you've been pulling, and the levers can't reach it. I had spent the better part of a month optimizing inside a box whose walls I hadn't drawn.

## The physics-vs-portfolio confusion

The wall, when I finally went looking for it, was sitting on line one of the oracle's design spec. The oracle answers the wrong question.

Stated plainly: the oracle answers *"what should the agent buy this step?"* It does not answer *"which assets are going up?"* Those are different questions. The first is a portfolio-management question — given a capital constraint and a top-k execution policy (with `k=1` in our case), which single asset is the *best* one to concentrate on? The second is a market-physics question — for each asset independently, is its forward return above or below some threshold?

For most of Foxglove's life I had been treating these as the same question, because in a one-asset world they collapse. They are not the same in a 64-asset world.

Here is the picture in motion. The training generator samples a 200-step window across 64 synthetic assets. At step `t`, the oracle's `greedy_lookahead_path(returns, k=1, lookahead=20)` looks 20 steps ahead and asks: across all 64 assets, which one gives the highest cumulative return over the next 20 steps that also clears the threshold? It picks one. It labels that asset's action BUY (or SELL, depending on sign). It labels the other 63 assets HOLD. Then it advances one step and repeats.

Suppose 30 of the 64 assets are in rallies that morning. The oracle picks the strongest one. The other 29 *also* have positive forward returns, *also* clear the threshold, *also* would be profitable trades. The oracle labels all 29 HOLD. Why? Because the oracle is also the value oracle, and the value head is trained on the realized PnL of the actual k=1 path. If the policy is allowed to say BUY on 30 assets but the value is computed assuming you only get to take one, the policy-value loop breaks. So the policy is *forced into* the k=1 mask to stay coupled to the value.

Now look at what the encoder sees. The cross-asset self-attention layer in `trm_encoder_v2.py` consumes all 64 assets simultaneously. It sees 30 assets with structurally similar bullish features. It is given labels that say: *one of these 30 is BUY, the other 29 are HOLD.* The encoder's only way to fit this is to learn that bullish features are *inconclusive about action*. The information that says "this is going up" is real, present, and consistent across many assets — and the loss is telling the encoder it doesn't matter.

From `TASK_681_ORACLE_TRADE_DENSITY_CEILING.md`, in the language I had to find before I could see what I was looking at:

> *"Current training conflates Market Physics (is an asset going up?) with Portfolio Management (is it the best asset out of 64?). By enforcing k=1 at the label level: 63 assets might have identical bullish structures, but the oracle picks only one (due to the k constraint). The model sees 63 bullish patterns labeled 'HOLD' — forcing the encoder to ignore valid market physics."*

That paragraph is the heart of the article. Everything else is bookkeeping around it.

The encoder *can* in principle learn cross-asset ranking — the architecture supports it. The TRM bottleneck (`trm_flax.py:138`) compresses to a 64-dim global state and projects out to 320 logits in one joint operation; the assets are not independent inside the network. So the model has the capacity to learn "asset 12 is the *most* bullish of these 30 bullish ones." But cross-asset ranking is a strictly harder learning problem than per-asset directional prediction. The latter learns a feature → label mapping; the former learns a feature → *order statistic over peers* mapping. The order statistic depends on the entire context, has higher sample complexity, and is more sensitive to label noise. Asking the model to skip past per-asset physics straight into the harder problem — without giving it the easier problem as scaffolding — is, in retrospect, exactly the kind of thing that produces a 52.5% plateau.

What the model does instead is the lazy fixed point: HOLD-everywhere. From the same task:

> *"The model learns all-HOLD because it correctly classifies 98% of data by doing nothing."*

I started calling this the Turtle Collapse. Article 31 covered the Turtle Breaker — the warmup-and-sharpening attempt to escape the collapse from inside the loss. This article is about the realization that the collapse is not a training pathology; the collapse is the *optimum* of the loss as specified, and you cannot escape an optimum by adjusting the optimizer. You have to change what you're optimizing.

## The math

It is worth doing the arithmetic explicitly, because the picture clicks the instant you write it down.

At each step, with `k=1` and `n_assets=64`, the oracle emits exactly one trade label (BUY or SELL) and 63 HOLD labels. Per step the class distribution over (asset, action) pairs is:

```
P(HOLD)            = 63/64 = 0.984
P(any trade action) = 1/64  = 0.0156
```

That trade-action mass is then split across the four trade buckets (BUY, SELL, LimitBuy, LimitSell), so each individual trade class lands somewhere around 0.4% of the per-step (asset, action) mass — and in practice LimitBuy/LimitSell are far rarer than BUY/SELL, so the marginal class probability for those is in the low single digits per ten thousand.

The threshold gate makes things slightly worse: the oracle only emits a trade if the best forward return clears `min_alpha_threshold`. In quiet curriculum levels, the gate fires on roughly 70% of steps, so the empirical trade density observed in our logs sits at:

```
Observed trade density = 1 - 0.989 = 0.011 (1.1%)
Maximum theoretical    = k / n_assets = 1/64 = 0.0156 (1.56%)
Headroom remaining     = 0.0046 (0.46pp)
```

Just stating that out loud is the diagnosis. The oracle is already at 70% of its *structural ceiling*. There is no threshold tweak, no sampling bias, no temperature trick that can buy more than 0.46 percentage points of additional trade density without rewriting the oracle. Every lever we had been pulling was inside that 0.46pp box.

Per asset, per training window, expected trades follow a near-deterministic count. With 200 steps in a window and `k=1`:

```
Expected trades per window per asset = 200 / 64 ≈ 3.1
```

Three trades per window per asset. With four trade classes, that's *under one example per trade class per asset per window*. From `TASK_681`:

> *"1024 × 0.011 / 4 classes ≈ 2-3 examples per batch"*

Two to three examples per trade class per asset per batch is — and this is the careful phrasing the task uses — *"sparse enough to explain the plateau"*, though not strictly unlearnable. Curriculum volatility variation, asset clustering, and correlation across assets can paper over some of it. But the loss landscape with a 0.984 marginal is brutal. The cross-entropy of the all-HOLD policy is

```
H_all_hold = -0.984·log(0.984) - 0.016·log(0.016)
           ≈ 0.0156 + 0.0660
           ≈ 0.082 nats per (asset, step)
```

Compare to the entropy of *uniform-over-5-classes* at `log(5) ≈ 1.61 nats`. The all-HOLD policy achieves loss roughly 20× lower than uniform by doing nothing. To improve on all-HOLD, the model has to pay calibration on the dominant class — push HOLD probability below 0.984 — to buy *correctness* on a tiny minority class. The gradient mass favors HOLD by **63:1** at the level of the marginal. Article 29 worked through the same number from the regime side and called it bull blindness; here it is the structural multiplier sitting under every gradient step.

And the `agree_trade = 8.9%` figure becomes legible. When the model is forced to trade — when the ground-truth oracle says BUY on asset 12 — the model *almost never* gets the right asset. Why would it? It hasn't been taught which of the 30 bullish-looking assets in the batch is *the* top-1; it's been taught that bullish features are not predictive of action. So when forced to commit, it commits noisily. Below random, because the 8.9% is conditional on the oracle having identified a strong signal, and a model trained to ignore strong signals will reliably mis-rank them.

There is one more arithmetic point worth making, because it caught me late and changed the urgency of the fix. The per-step trade label mass is 1.56% by construction. But that's *per (asset, step)* — collapse across the asset axis and the per-step probability that *anything trades at all* is roughly 70% (the gate-firing rate at typical curriculum levels). So you can't even console yourself with "rare but consistent" — the rare-trade events are concentrated in 70% of steps, leaving 30% of steps with the oracle producing 64 HOLDs and zero trades. Those steps contribute *exactly nothing* to the trade-side gradient. The model spends roughly a third of its parameter updates being told that *the entire universe of 64 assets is in HOLD-land right now*. That is the most efficient possible regime in which to learn the all-HOLD fixed point.

Worse: per-asset per-class density is even uglier than the per-asset trade-class density. The four trade-classes (BUY, SELL, LimitBuy, LimitSell) are not balanced inside the 1.56% trade mass. BUY and SELL dominate; LimitBuy and LimitSell are vanishingly rare in the synthetic data Bella produces. So when `TASK_681` reports *"~2-3 examples per class per asset per batch"*, that's the average over trade-classes — the limit-order classes are routinely at zero. The model has zero gradient for the limit-order classes for entire epochs at a time. Whatever it learns about LimitBuy is learned at the prior, with no posterior update.

> *Observation: the apprentice asks whether the patient has 1.56% of the disease. The apothecary cannot answer in fractions; either you treat or you do not. The apprentice records "no diagnosis possible" and the patient is sent home with the powder unweighed.*

## The kickback

There are three layers to the kickback here, and I want to name them in order because they are not the same lesson.

**One: I diagnosed the constraint only after six attempted fixes.** This is embarrassing to write down. The 98.9% number was in every CSV log from the beginning. I had eyes on it. I labelled experiments by it. And I still spent five tasks pulling levers that, in retrospect, *could not have moved it* — the levers were all downstream of the ceiling. TASK_679 lowered a threshold that was no longer the binding constraint. TASK_680 oversampled trade windows that were the same windows the oracle had already produced. TASK_677 sharpened policy targets whose distribution was fixed by the oracle's k=1 mask. Each lever had a coherent local argument; none of them touched the actual constraint. The pattern was *try, fail, try, fail, try, fail* — and the diagnostic move, the move that actually mattered, was *stopping* and asking why nothing was moving. The first five tasks weren't wrong, exactly. They were *misaddressed*. They were debugging code inside a function whose calling convention was the bug.

This is the lesson I keep needing to relearn: when you find yourself proposing the third intervention for the same plateau, the third intervention is not the answer. The answer is to stop and ask *what is making this metric structurally bounded*.

**Two: the k=1 oracle wasn't wrong — it was a choice made for sparsity, and the sparsity was a feature.** This is the part I want to be careful not to over-claim. The original oracle design used `k=1` for an entirely defensible reason. In real trading, capital is finite. You can only meaningfully concentrate position in a handful of assets per step. Sparse trading mirrors actual portfolio behavior. From the historical context in `TASK_681`: *"k=10 caused value overestimation: oracle picks 10 assets but model executes 1 trade/step."* The original TASK_666 design specified `oracle_k = 4`; an experiment in the TASK_143/145 timeframe tried `k=10`, blew up the value head, and the codebase reverted — but reverted past `k=4` all the way down to `k=1`. The revert overshot the design.

So the k=1 setting was a deliberate sparsity prior, layered with an implementation overshoot, justified at the time by realistic capital constraints. What nobody modeled was the interaction between the sparsity prior and the cross-entropy loss landscape. A correct intent — sparse trading is realistic — produced a wrong outcome through an unmodeled interaction with the loss surface. **This is the most common shape of an architectural mistake in this codebase.** Not "someone made a bad call," but "two locally-justified choices interacted in a way nobody traced end-to-end."

**Three: the fix shape is k=4 or k=12, and it has already partly failed.** The upstream attack is to loosen the oracle's portfolio constraint. Let the oracle label more than one asset BUY per step. Accept that some of those BUYs are not the literal optimal trade — accept *more "wrong" labels by the original metric* — in exchange for a label distribution the loss can actually learn from. TASK_739, the Dense Sign-of-Forward-Return Oracle, is the most aggressive version: every asset, every step, gets a per-asset directional label, not a portfolio-conditional one. From `TASK_739_DENSE_SIGN_ORACLE.md`:

> *"A1 reframes the oracle target: every asset, every step, gets a directional label derived from the sign of forward log-return with an ε dead-zone for low-magnitude moves. HOLD becomes a learned class (not a marginal artifact), and the effective label count jumps from ~4 → 64 per step."*

That phrase — *HOLD becomes a learned class, not a marginal artifact* — is the entire reformulation in nine words. The current oracle's HOLD label is a default: most assets get HOLD because the portfolio constraint had to put 63 of them somewhere. The dense-sign oracle's HOLD label is a *prediction*: this asset is, by physics, not moving enough to trade. The two HOLDs look identical in the label tensor. They mean entirely different things.

TASK_739 ran. Article 6 covered the result inside the broader "classifier improvements anti-correlate with Alpha" pattern, so I'll only restate the verdict here: paired three-seed evaluation came in at **Δα = −2.63pp**. Inconclusive, leaning negative. The reformulation is *structurally correct* in the sense that it does break the 98.9% ceiling; the dense oracle produces a label distribution with healthy per-class density. But the downstream Alpha didn't follow. That outcome is now part of a broader pattern (Articles 6 and 10) showing that single-mechanism encoder-side fixes don't reliably lift live performance on this stack. The dense-sign oracle is, as of now, in the box marked "fixed the diagnosed problem, didn't fix the deployment problem."

Which is its own kickback. The plateau-at-52.5% finding is, I still believe, *correct* — the oracle's k=1 construction is a hard ceiling on agreement, the math says so, and breaking the ceiling does break the agreement plateau. But breaking the agreement plateau didn't, by itself, give us a better trader. That is the Article 6 lesson under a different mask: improving the supervised signal does not monotonically improve the deployed agent. The constraint we found was real and load-bearing, and removing it was *necessary*; it was just not sufficient.

## What's left

Sitting on `workstation` right now, the active line in `tasks/QUEUE.md` for the oracle question is TASK_764 — one of the four structural directions from TASK_761's pivot roadmap. The plan there is to sweep `k` more carefully: k=4 (the original TASK_666 design), k=8, and k=12, paired-seed evaluated under the recalibrated `eval-MCTS sims=16` joint protocol that Article 10 documented. The recalibration matters because Article 10's finding was that benchmark lift didn't match live lift; we now know that any agreement-side improvement has to be validated under the operator the deployed agent actually uses, not the benchmark we used to use.

I expect, modally, that one of those k values will land. The math says the loss landscape becomes learnable above roughly 5% trade density, which puts k in the 4–8 range as the credible band. I also expect, given Articles 6 and 10, that the lift will be smaller than the agreement gain suggests, and that we'll need at least one more orthogonal lever — probably reward shaping or planning-side entropy regularization, from the other branches of TASK_761 — to convert the cleaner labels into deployment Alpha.

The narrower claim I'm comfortable making, today, May 23 2026, is this: **the 52.5% balanced-agreement plateau across Foxglove training runs is structurally caused by the k=1 oracle's 98.4% HOLD construction, and it is not movable by any intervention downstream of the oracle.** That claim is the artifact. It is what TASK_681 is for. The fix is forthcoming.

I am writing this in the same Python environment the next oracle-k sweep is queued in. The dashboard at `localhost:8050` is still parked on the 52.5% plot. The desk fan is on; the second SSD where the git objects live is mounted; the card is idle for the moment, between runs. The plot will be different by the time this article publishes, or it won't be, and either way the article was about the diagnosis, not the cure.

The lesson I want to keep is the one I keep needing to keep: when five interventions in a row don't move a metric, the metric isn't broken. The model you have of the system is.

> *Observation: I retire the scale calibrated for grams. I am building a new one — calibrated for the dose, not for the patient. The first weighings are uneven. I expect them to be.*

---

## Sources

- `~/Trader/tasks/TASK_681_ORACLE_TRADE_DENSITY_CEILING.md` — primary source. Root-cause analysis, adversarial review across three rounds, fix-option matrix (A–F), and the "Turtle Collapse" framing quoted throughout.
- `~/Trader/tasks/TASK_674_TRAINING_AGREEMENT_PLATEAU_BUGS.md` — six plateau bug fixes, including the buffer sample-counter bug and the optimizer-warmup-on-merge reset.
- `~/Trader/tasks/TASK_675_TECHNICAL_DEBT_AND_STABILITY.md` — fifteen tech-debt items including the 5× `decay_steps` miscalibration and the latent-cache miss in `evaluate_batch`.
- `~/Trader/tasks/TASK_677_TURTLE_BREAKER.md` — post-averaging temperature sharpening on multipass oracle policy targets; covered in Article 31.
- `~/Trader/tasks/TASK_679_TRADE_SIGNAL_DENSITY.md` — threshold lowering (0.001 → 0.0005) and the original phase plan for trade-biased sampling.
- `~/Trader/tasks/TASK_680_TRADE_BIASED_SAMPLING.md` — coin-flip sampling implementation, adversarial review of latent overfitting, sampling efficiency, lookback alignment, and hold-accuracy degradation.
- `~/Trader/tasks/TASK_739_DENSE_SIGN_ORACLE.md` — the dense-sign reformulation. Paired-seed result documented in Article 6.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four structural directions; oracle-k reformulation lives under TASK_764.
- `~/Trader/mcts_trader/jax_agent/path_solver.py` — `greedy_lookahead_path`, the oracle whose `k` parameter is the constraint discussed here.

Related articles in this series:

- **Article 6** — *Classifier-Side Improvements Anti-Correlate With Alpha.* The 0-for-3 verdict that includes TASK_739's dense-sign result.
- **Article 8** — *Five Seeds and a Funeral.* The paired-seed protocol that demoted TASK_739 from "promising" to "inconclusive".
- **Article 10** — *Eval-MCTS Asymmetric Lift.* The operator-mismatch finding that reset what "improvement" means on this stack.
- **Article 17** — *Five Oracles.* The taxonomy of oracle designs Foxglove has trained against; the k=1 greedy path is one of the five.
- **Article 29** — *Bull Blindness.* The regime-side consequence of the same 98.4% HOLD construction discussed here from the construction side.
- **Article 31** — *The Turtle Breaker.* The targeted attack on Turtle Collapse from inside the loss path.
