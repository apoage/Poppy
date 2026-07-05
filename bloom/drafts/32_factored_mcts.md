# Factored MCTS

*The Apothecary surveys sixty-four pots in a single tray. The old habit is to ask, of the whole tray at once: which one cure should I draw today? But sixty-four pots have sixty-four answers, and the tray will not yield them to a single question. I learn instead to ask, of each plant in turn, the smaller question. Then I read the tray.*

— *The Apothecary's Lab Notebook, Article 32 — Lukas, 2026-05-22*

---

`workstation` is on. The desk fan is on. `nvidia-smi` is ticking through epoch 38 of a checkpoint validation in the next tmux pane, and the single 3090 is sitting at 71 °C with the FTW3's outer fan barely above idle because the workload is inference-only. The git objects for this repo live on the second internal SSD (`/dev/sdb1`, `storage disk`), the symlink dance I set up last winter because the system drive isn't big enough for both my models and my software. The CMP 90HX is gone — pulled on 2026-05-17 — so there is exactly one CUDA device on this machine, exactly one number that matters when I plan compute budgets, and exactly one slot for everything to fit in.

That single-3090 constraint is the whole reason this article exists. Everything I am about to describe — factored search, per-asset trees, the difference between a 320-way branching MCTS and sixty-four 5-way fans — is what you do when you cannot brute-force the problem. You cannot run a thousand simulations per decision on a single 3090 while also training the model and rolling the environment. You get thirty-two simulations. You spend them carefully. Or you change what you are spending them on.

That is the kickback this article is built around, and we'll get to it. But first the elephant.

## The elephant: 320

The action space in Poppy is 320. Sixty-four assets, five per-asset actions (`SELL`, `HOLD`, `BUY`, `LIMIT_SELL`, `LIMIT_BUY`), encoded as a flat index `j = asset * 5 + action`. Every decision the agent makes at every step of every self-play game is a single integer drawn from `[0, 320)`. The MCTS that selects that integer runs **thirty-two simulations** per decision in production. Sometimes sixty-four if I'm feeling generous and have the wall-clock to burn (TASK_760). The default is thirty-two.

Think about that for a moment. Thirty-two visits, three hundred and twenty children at the root. If the search distributed visits uniformly — which it never does, because that would be insane — every action gets a tenth of a visit. The search is not exploring the action space. It is sampling it, and sampling badly.

The reason this works at all in practice is the policy prior. UCB1 selects the child that maximises

```
UCB(s,a) = Q(s,a) + c · π(a|s) · √N(s) / (1 + N(s,a))
```

— the standard PUCT formulation that AlphaZero and MuZero inherit. With thirty-two visits split across three hundred and twenty children, what actually happens is that the prior `π(a|s)` from the policy head dominates the second term, the visits collapse onto the top-K predicted actions, and the other three hundred children get zero visits each. MCTS is mostly amplifying the prior. If the prior is good, the search is good; if the prior is bad, no amount of thirty-two-sim MCTS is going to find a better action by luck.

This is fine when you accept it. It is *not* fine when you mistake it for a search. For most of 2025 I was reading MCTS visit counts as if they meant something on their own. They don't. They mean what the prior already said, sharpened by one level of value backup. The search adds maybe one bit of refinement on top of the prior. Sometimes less.

The number-of-children-versus-budget problem is structural. The branching factor is 320 because there are 64 assets, each independently capable of one of five things. The simulation budget is 32 because anything larger eats my training throughput in half. The ratio 320/32 = 10 says: ten children per visit. Bayesian coverage is not happening. Something else has to.

## Standard MCTS in this stack — the joint-action tree

Let me be concrete about what the joint tree looks like, because the factored variant only makes sense in contrast.

In `mcts_trader/jax_agent/mctx_search.py`, `MuZeroMCTXSearch` wraps Google's `mctx` library. It builds a search tree where every node is a latent state from MuZero's representation network, every edge is one of 320 actions, and every leaf is evaluated by the value head. The Gumbel-MuZero variant is the default — it uses Gumbel noise plus a sequential-halving schedule to pick which action to expand next, which is more sample-efficient than vanilla PUCT when simulation budgets are small. `max_num_considered_actions=64` further restricts the root to the top-64 prior-favoured children, so really we are running 32 simulations across 64 candidates, not 320. The ratio improves to 1:2.

Even at 1:2 it is thin. Look at it from the per-asset angle. Sixty-four assets. If the search picks "BUY asset 7" as one of its 64 considered actions, it has spent one of its considered-action slots on one of the five things asset 7 might do. The other four actions for asset 7 — `SELL`, `HOLD`, `LIMIT_SELL`, `LIMIT_BUY` — are not in the consideration set unless they also rank in the global top 64. They almost certainly don't. Most assets will appear once or twice in the consideration set, on the action the prior thinks is most likely. The search never compares "BUY asset 7" to "SELL asset 7"; it compares "BUY asset 7" to "BUY asset 23" and "HOLD asset 41."

This is the joint-action tree's pathology in one sentence: it treats actions on different assets as competing alternatives in the same distribution. They are not. They are not even in the same decision. Whether asset 7 should be bought has very little to do with whether asset 23 should be sold. The decisions co-vary, sure — capital constraint, correlation, basket risk — but the *first-order* signal for "should I buy asset 7?" is whether the model thinks asset 7's price is going up, and that signal lives entirely in asset 7's prior column.

The math of why this is structurally under-sampled is worth writing out. If the per-asset decision is approximately independent, the joint distribution `P(a_1, …, a_64)` factorises as `∏_i P(a_i)`. The joint action space has size `5^64 ≈ 7 × 10^44`, but the *information content* of the optimal joint policy is `64 × log2(5) ≈ 148` bits — only 148 bits, because each asset is an independent 5-way classification once you accept the factorisation. The joint tree is paying compute against the `5^64` view; the structure of the problem says you only need 148 bits to specify the answer.

MCTS doesn't know this. It searches the joint space.

There is a finer point about UCB worth pausing on. The classical PUCT bound says that the regret of the search after N visits is roughly `O(√(log N · K / N))` where K is the branching factor. With K=320 and N=32, the bound is not a bound; it is a wish. `√(log 32 · 320 / 32) ≈ √(5.5 · 10) ≈ 7.4` — the regret bound says, in essence, "the search could be selecting almost anything." The reason MCTS works in practice on big-K problems is that the policy prior `π(a|s)` enters the UCB exploration term multiplicatively, suppressing children whose prior is low by orders of magnitude. With a sharp prior, the *effective* branching factor is something like `1 / max_a π(a|s)`, which for our policy head with its trained softmax is closer to 10 than to 320. That recovers a sensible regret. But it also reveals what the search is doing: it is *not* exploring 320 actions; it is *re-ranking the top ~10 prior actions* using one level of value backup. Saying "MCTS" out loud overstates the case.

> *MCTS treats "BUY asset 7" and "SELL asset 23" as competing alternatives in a single distribution. But per-asset decisions are mostly independent — the oracle selects k=4 assets to trade, with each asset's action chosen independently.* — TASK_724, problem statement, verbatim.

I read that paragraph at the end of February 2026 and immediately wrote `factored_search.py`. The diagnosis felt obvious in retrospect. The cure was less obvious than I thought it would be.

## The factorisation

What TASK_724 proposes — and what now sits in `mcts_trader/jax_agent/factored_search.py` as a 188-line drop-in replacement for `MuZeroMCTXSearch` — is to stop running an MCTS over 320 joint actions at all. Replace it with a per-asset one-step lookahead.

The procedure:

1. Encode the observation once. Get the latent state `z_t` from the representation network. Same as before.
2. For each of the 320 (asset, action) pairs, run one step of the dynamics model:
   ```
   z_{t+1}, r̂(asset, action) = dynamics(z_t, j)        where j = asset · 5 + action
   v̂(asset, action) = value_head(z_{t+1})
   Q(asset, action) = r̂(asset, action) + γ · v̂(asset, action)
   ```
3. Reshape `Q` to `(64, 5)`. Per asset, take the softmax across the 5 actions with temperature τ:
   ```
   π̃(a | asset) = softmax(Q(asset, :) / τ)
   ```
4. The training target is the per-asset policy `π̃` of shape `(B, 64, 5)`, flattened to `(B, 320)`. Each group of 5 sums to 1.0; the total sums to 64.0, *not* to 1.0.
5. The selected action is the argmax composition: each asset's best action, then pick the asset with the largest advantage over its own `HOLD` to actually emit. The other 63 assets implicitly hold.

The verbatim spec:

> *For each (asset, action) pair: 1. Compose joint action index: j = asset * 5 + action; 2. Run dynamics: next_latent, reward = dynamics(latent, j); 3. Run value prediction on next_latent; 4. Q(asset, action) = reward + discount * value(next_latent).* — TASK_724, architecture section.

A few things are worth noticing about this design.

**There is no tree.** The factored "search" is not a search in the MCTS sense — there is no selection-expansion-backup loop, no UCB, no Gumbel noise, no `mctx` call. It is a single batched forward pass through the dynamics model evaluated at every (asset, action) pair, followed by a per-asset softmax. The depth of the rollout is 1. If you want to call it MCTS, you're calling it the degenerate one-simulation-per-arm version where the prior gets bypassed and the Q-estimate is used directly.

That sounds like a downgrade. In one sense it is. In another sense it is a massive upgrade in coverage: instead of evaluating 64 of 320 actions (20%, joint-tree budget), you evaluate 320 of 320 (100%). Every action gets a Q-value. Every asset's per-action ordering is fully resolved.

The compute trade is worth being precise about. The joint MCTS, with B=256 and 64 simulations, does `256 × 64 = 16,384` recurrent inference calls per step. The factored version does `256 × 320 = 81,920` — five times more. On the 3090 this is the difference between ~95 ms per training step and ~200 ms per training step; epoch wall-clock roughly doubles (38 s → 80 s). That cost is real and it has to be eaten. The implementation sub-batches into 4096-element chunks to keep VRAM bounded; with the MuZero networks at `state_dim=128`, `hidden_dim=2048`, H=6 reasoning cycles and L=8 layers, peak activation memory per sub-batch is ~200 MB, comfortably inside the 3090's 24 GB.

The effective per-asset simulation budget under factored search is **exactly one full evaluation per action**, deterministically. Under joint MCTS with 32 simulations and 64 assets, the *expected* number of simulations per asset is 32/64 = 0.5 — and that is the average; the variance is large because the prior concentrates visits onto a handful of "hot" assets and starves the rest entirely. Factored search trades that biased, sparse coverage for uniform 1.0-per-action coverage at the cost of search depth (one step instead of up to four).

One more design subtlety: the entropy weighting I sketched in my notebook before reading the task spec — *"give each asset a sims-budget proportional to its prior entropy, so high-uncertainty assets get more sims"* — does not appear in the implemented version. The implemented factored search is uniform: every action gets exactly one evaluation, regardless of how confident the prior already is. The reason is mostly that uniform is JIT-friendly and entropy-weighted isn't — the moment you make the per-asset budget dynamic, you lose the static-shape XLA fusion that makes the whole batched-recurrent-inference trick cheap. I will probably re-visit entropy weighting in a multi-step factored variant. For now: uniform.

There is also a numerical-stability detail in `_q_to_policy` that took me a re-read of the code to appreciate. Before the per-asset softmax, the Q-values are *centered per asset*: `Q_centered(asset, :) = Q(asset, :) - mean_a Q(asset, :)`. This is a standard trick — softmax is shift-invariant, so the centering does not change the output mathematically. What it does is make the softmax operate on the *advantage* of each action relative to its asset's mean, not on the absolute Q-value. That matters because Q-values across assets are not on the same scale: a high-volatility asset has higher absolute Q-magnitudes than a low-vol one, and without centering, the temperature τ would interact differently with each asset's scale. Centering decouples them. Each asset's policy distribution depends only on its own Q-advantage structure. This is the kind of detail you would skip past as "obvious" if you wrote it yourself, but I copied it from a notebook six months earlier and only later realised it was load-bearing.

The training-side change is one I want to flag explicitly because it almost broke me when I first wired it up. The factored target has shape `(B, 320)` but the *normalisation* is different from MCTS visits: each group of 5 sums to 1.0, total to 64.0. If you naïvely keep the old global softmax CE loss — which assumes the target sums to 1.0 globally — you get a loss value 64× too large and a gradient that points in approximately the right direction but with completely wrong magnitude. The training loop appears to work, the loss curve appears to decrease, and the model learns something but not what you intended. I caught this only because the loss was an order of magnitude higher than I expected on the first epoch, which prompted a re-read of the loss code, which prompted the fix to per-asset softmax CE: reshape to `(B, 64, 5)`, take softmax-CE per asset, mean across assets. This is also why the BC (oracle-imitation) loss already used per-asset CE — that's how the oracle target was always structured, I had just inherited the joint-CE from the original MuZero implementation and never re-questioned it.

## The catch

Factored search assumes per-asset actions are approximately independent given the state. Let me state when that assumption is fine and when it is not.

It is fine when:

- The portfolio is sparse. Most assets are holding most of the time. Cross-asset capital pressure is not the binding constraint.
- The correlation structure is low. BTC moving up and ETH moving up are partially independent signals; the model's value head can score "buy BTC" and "buy ETH" as separately good without double-counting beta exposure.
- The oracle that supervises Foxglove also picks per-asset actions roughly independently (it does — see Article 17 — it selects the top-k by per-asset expected return, which is exactly the structure factored search assumes).

It is **not** fine when:

- The portfolio is concentrated and capital constraints bite. If you have 95% of your book in BTC and the model wants to add ETH, the joint action "BUY ETH" depends entirely on whether you also "SELL BTC" to fund it. Factored search cannot represent that conditional.
- Correlations are high (drawdown regimes). When everything goes down together, "BUY BTC" and "BUY ETH" are not independent positive bets; they are the same bet, taken twice. Factored search will double up exposure exactly when the market is punishing concentration.
- The optimal action requires hedging — long one asset, short a correlated one, to net out a market-direction bet and keep a basis bet. The joint tree can represent that as a single action selection. Factored search can represent it only as two argmaxes that happen to align, which is not the same thing because the value head is asked to evaluate each leg in isolation, and the value of half a hedge is not half the value of the hedge.

The factorisation is an approximation that trades joint-action expressiveness for sample efficiency. The interesting question is not *whether* it is an approximation — it obviously is — but whether the marginal expressiveness of joint MCTS is worth the cost in coverage. On a 32-simulation budget, almost always no: the joint tree's expressiveness is a fiction it doesn't have the visits to redeem. On a 320-simulation budget (per decision), the answer flips. We don't have 320 simulations per decision. We have 32. So factored wins.

That is the *a priori* argument, anyway. What actually happened when I ran the experiment is more interesting.

There is a second catch that I should mention even though it didn't bite me until later. The factored search's Q-value for `(asset i, action a)` is computed by running one step of the dynamics model on the *joint* latent state with the joint action `j = i*5 + a` injected. The other 63 assets are assumed to hold, but the dynamics model was *trained* on trajectories where typically 4 assets traded simultaneously. Inputs of the form "one asset trades, all others hold" are slightly off-distribution for the dynamics. The bias is small — the dynamics model is robust enough to generalise — but it is a real source of systematic error in the per-asset Q-values, and it is invisible until you go looking for it. In a properly factored MDP this would be no issue; the dynamics would factorise too. In our setup the dynamics are joint, the search is factored, and the seam between them is where the bias lives.

> *The specimen flowered, then withered overnight. The labels read "convergence"; the produce read "loss." I record the contradiction and continue.*

## The kickback, in three layers

I ran factored search at the end of February and watched the metrics roll in over the next 50 epochs. Three things happened simultaneously, and they took me about a week to disentangle.

**Layer 1: Bal.Acc went up. Alpha did not.**

The per-asset balanced accuracy — the metric that says "given an asset, how often does the model predict the right direction-of-trade?" — improved by about three percentage points. Policy loss came down. Per-asset CE was lower than joint CE had been, which makes sense, because the factored target has only 5-way structure to fit instead of 320-way. The classifier was happier.

Alpha did not move. In paired three-seed evaluation (seeds 7, 13, 23 — same data, same hyperparameters, only the search variant flipped), the mean Δα was **+0.4 pp**, with a standard error of 1.8 pp. Statistically indistinguishable from zero. Three of six benchmark periods went up, three went down, no period moved more than 4 pp in either direction. This is the canonical noise-band signature from Article 6 — and Article 8, and Article 22 — the *classifier-Alpha anti-correlation* striking for the fourth or fifth time depending on how you count.

I had read the diagnosis in Article 6 already by this point. I had written, in my own lore notes for 2026-04-29, the warning: *"Bal.Acc gains do not predict Alpha gains. Stop using Bal.Acc as a promotion signal."* I read that line. I ran the experiment anyway, because factored search was clearly a *structural* improvement, not a classifier-sharpening one — it changed the search, not the loss, not the encoder, not the head. Surely a structural fix would translate?

It didn't. Or rather: it did translate, just not into Alpha. It translated into sharper per-asset confidence on the same incorrect-for-Alpha gradient. The per-asset trade rate ticked up. Fees ticked up to match. Net Alpha sat in the noise band.

— Claude Code, third paired-seed run, when the second negative-Alpha seed came back: *"this is the Bal.Acc/Alpha anti-correlation again, the fix is structural but the bottleneck isn't here."*

**Layer 2: the factorisation assumption hides a class of trades.**

I spent most of the week of March 3 staring at trade traces from factored vs joint to figure out what the joint variant was doing that factored wasn't. The answer was small but real: the joint MCTS occasionally produced *coordinated* trades — sell one asset, buy a correlated one, in the same step — that the factored version could not represent.

This is the hedging case. In the joint tree, "SELL asset 7 AND BUY asset 23" is a single trajectory: the search expanded the SELL-7 root, then on the next visit, the same trajectory's continuation expanded the BUY-23 leg, and the backed-up value reflected the *combined* effect on the portfolio. In factored search, the SELL-7 advantage is computed assuming all other assets hold; the BUY-23 advantage is computed assuming all other assets hold; if the model decides both, it is because each is individually above the per-asset HOLD baseline, not because together they hedge.

In sparse-trading regimes this doesn't matter. The model trades 4 assets at a time, the other 60 are in HOLD, and assuming-the-others-hold is a true assumption. In coordinated-trading regimes it matters a lot. Joint MCTS encoded the coordination as part of the search trajectory. Factored search lost it.

I went back to look at the period breakdowns. Sure enough: factored search underperformed joint search by about 1.5 pp on the two periods with the highest cross-asset return correlation (the 2024 broad-rally period and the 2025 March-correlated-drawdown). It outperformed by about 1 pp on the four low-correlation periods. Net effect: noise. But the *mechanism* of the noise was not "factored search is randomly worse"; it was "factored search is structurally worse at coordinated trades and structurally better at independent ones, and the benchmark suite is a mix of both."

**Layer 3: factorisation is a compromise with the simulation budget, not a triumph over it.**

This is the real lesson, and it took me longer to internalise than the first two.

The factored variant looks like a *win* on a 32-simulation budget. It coverage-dominates joint MCTS. Per-action Q-values are deterministic and uniform. The structural mismatch between "320-way branching" and "32 simulations" is dissolved by re-cutting the branching factor into 64 independent 5-ways.

But the *only reason* factored search dominates is that I cannot afford more simulations. If the simulation budget were 320 — one per joint action, at the root — the joint tree's coverage problem goes away. Coverage equalises. The joint tree's *expressiveness* (it can represent hedges) becomes a pure asset, not a liability. Factored search's coverage advantage evaporates. Joint wins.

When I ran TASK_760 a month later — raising MCTS simulations from 32 to 64, then to 128, on the joint tree — the gap between joint and factored shrank by about half. Joint MCTS at 128 sims was within 0.5 pp of factored MCTS, and the variance overlapped completely. The factored variant's "victory" was a victory against an under-resourced opponent.

The right question, the one I should have been asking from the start, is not *factored or joint?* It is: *how many simulations does each decision deserve, and how should I allocate them?* That is the proper formulation. The answer involves per-step entropy of the prior, per-step expected value variance, and per-step opportunity cost in wall-clock — all of which I now have the instrumentation for, none of which were the framing of TASK_724.

> *Causality kicks back. The labels said "search improvement"; the produce said "compromise with the simulation budget I couldn't afford to lift."*

## What's left

Factored search is in the repo. It is the default in some configs and a `--factored` CLI flag in others; joint MCTS via `MuZeroMCTXSearch` is preserved as the alternative. The training-step machinery in `train_phase2.py` switches the policy loss between global softmax CE (joint) and per-asset softmax CE (factored) based on `config.use_factored_search`, and the BC loss already used per-asset CE so it lines up without modification. Both paths use the same PopArt scaling and the same per-epoch value calibration; the only differences are which Q-values get computed and how the policy loss reduces them.

The choice between factored and joint is now a per-experiment hyperparameter, not a fixed architecture decision. For low-budget, fast-iteration training I use factored — it converges faster on Bal.Acc and gives interpretable per-asset Q-values I can stare at in the dashboard. For end-of-experiment validation runs with sim=128 budget I use joint, because at that budget the joint tree's expressiveness is no longer wasted. Two tools, one rack.

What I would do differently: I would not have framed TASK_724 as "factored vs joint" at all. I would have framed it as "what is the simulation-budget allocation policy?", solved that, and let the factored / joint question fall out as a special case (factored = uniform-allocation, one-sim-per-action; joint with budget B = PUCT-allocation, B simulations total). The per-asset entropy-weighted budget I sketched and then abandoned is, in that framing, the missing middle: not factored, not joint, but joint-with-allocation-shaped-by-prior. I have not built that. It is in the queue.

The other thing I learned, which is now a series-wide rule: a *structural* improvement that pushes a classifier metric without pushing Alpha is not exempt from the Article 6 anti-correlation. I had told myself it would be, because the change was at the search level not the head level. The mechanism doesn't care which layer you touched. If the gradient signal is wrong (and Article 6 says it is — the bottleneck is downstream of any per-asset sharpening), then making any part of the model more confident in that signal just over-trades faster.

Factored search did exactly what it was designed to do. The thing it was designed to do was not, as it turned out, the thing the model needed.

The single 3090 outside in the next pane has finished the validation epoch. The fan ramps back down. The Apothecary closes the tray.

> *Sixty-four pots, sixty-four answers, one tray. The question was never which to ask; it was how many breaths each plant required before answering. I record the new question and continue.*

---

## Sources

- `~/Trader/tasks/TASK_724_FACTORED_MCTS.md` — implementation spec, status IMPLEMENTED, quoted verbatim twice above.
- `~/Trader/mcts_trader/jax_agent/factored_search.py` — the 188-line `FactoredPerAssetSearch` class, drop-in for `MuZeroMCTXSearch`.
- `~/Trader/mcts_trader/jax_agent/mctx_search.py` — joint Gumbel-MuZero MCTS via Google `mctx`, the comparison baseline.
- Article 06 (classifier–Alpha anti-correlation), Article 17 (five oracles), Article 20 (the dojo), Article 22 (year of negative results) — supporting lineage.
- Paired three-seed evaluation under TASK_724, seeds 7 / 13 / 23, executed on `workstation` over the week of 2026-03-02. Wall-clock per epoch: 38 s joint, 80 s factored.
- TASK_760 simulation-budget sweep (32 → 64 → 128 on joint MCTS), April 2026 — the data point that retroactively reframed TASK_724 as a budget-compromise rather than an architecture win.
