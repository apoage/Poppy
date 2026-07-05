# Bull Blindness: The Regime the Agent Can't Trade

> *I keep a specimen by the north window that flowers only in grey weather. The first season I trialled it I set it under the south light to test its colour; it wilted before noon. I draped a shade-cloth, watered with the cooler kettle, and the flowers returned the next morning, blue and faintly metallic. The plant had no defect. The room had a sun.*

*— Lukas, 2026-05-22, written at `workstation` between two `train_phase2.py` runs*

---

## The symptom

Every checkpoint in this project, going back to the autumn, fails in bull markets. I mean *fails*. Not "underperforms." Not "is a little weaker." Catastrophically negative Alpha — the model trades on a rising tape and finishes worse than holding cash, by twenty percentage points or more.

I am writing this with a benchmark validation ticking through in another window. The desk fan is on; `nvidia-smi` shows the EVGA 3090 at 88 °C and 397 W; `phase2_stageB_e200.pkl` from the 728 line is being scored across six periods. Five of the six come back fine — bear regimes, sideways grinds, the 2024-09 crash window — and the bull period comes back at −19 %. Same checkpoint. Same encoder. Same search. Everything is the same except the direction the tape happens to be moving.

The first time I noticed the pattern, I thought it was a hyperparameter accident. The second time, I thought it was that particular checkpoint. By the fifth variant, with five different head configurations, five different oracles, three different temperatures on the factored search, two BC weightings, all showing the same shape — bear fine, sideways fine, bull catastrophic — I started writing it down as a constant of the system.

In `lore/` I have it filed as **bull blindness**. The phrase became shorthand. When a benchmark comes back and I scan the alpha column, I check the bull column first. Everything else is detail.

This article is about how I diagnosed it, what I tried, what kicked back, and where it is on 2026-05-22 — which is to say, not solved. The diagnosis I am confident in. The fix is the open part.

---

## What it isn't

The thing about a failure that follows you across architectures is that the obvious explanations get ruled out one at a time, and you slowly run out of comfortable places to look.

**First hypothesis: the encoder destroys directional information.** This is the easy story. Min-max normalization in the price channels collapses absolute levels into [0, 1]. A bull window and a bear window get squeezed into the same dynamic range. The encoder, blind to whether the tape is going up or down, hands the policy head a latent that doesn't distinguish them, and the policy outputs the same action distribution either way.

It is a clean hypothesis. It is also wrong.

I wrote the audit script in `prompt-lib/out_encoder_audit/`. I generated two synthetic windows — 32 assets with a linear bull trend, 32 with a linear bear trend — ran them through `jit_batch_encode_temporal`, and inspected each of the 27 market channels. Direction is *aggressively* encoded:

| Ch | Name | Bull mean | Bear mean | Abs diff |
|---:|------|---:|---:|---:|
| 7  | 64-min return | +0.100 | −0.091 | 0.191 |
| 15 | momentum 64 (clip ±1) | +1.000 | −1.000 | 2.000 |
| 23 | pct in 64-window range | +1.000 | +0.000 | 1.000 |
| 26 | VWAP lower band | +0.76 | +0.06 | 0.70 |

Channel 15 is *literally* ±1 — it is a binary direction bit. Channel 23 is 0 in bear, 1 in bull. Ten of the twenty-seven market channels flip sign or 0↔1 between bull and bear. There is no possible compression of this observation that loses direction.

Fine — but maybe the trained encoder *learns to ignore* those channels. I loaded `jax_ckpt_easy_phase2_724c/phase2_stageB_final.pkl`, ran a pure-bull window and a pure-bear window through it, and measured the latents:

> ‖state_bull‖ = 57.9, ‖state_bear‖ = 48.7
> ‖state_bull − state_bear‖ = 59.7
> **cosine similarity = 0.384**

A cosine of 0.38 is not "indistinguishable." It is "clearly two different regimes occupying different parts of the latent space." From the audit:

> *"The trained encoder successfully preserves direction in its 128-dim latent."*
> — `prompt-lib/out_encoder_audit/findings.md`

So the encoder is fine. Cross that off.

**Second hypothesis: search collapses on bull.** This was my next guess. Factored search at temperature 0.3 produces visit counts that, when Q-values are flat, smooth out into something close to uniform. Maybe in bull regimes the value function's directional confidence is too weak for the search to commit, and the policy target becomes mush.

I ran the factored-temperature sweep — T = 0.1 (sharp), T = 0.3 (default), T = 0.5 (soft). Bull alpha did not move. I ran with the search disabled entirely, training pure BC against the oracle. Bull alpha did not move. The search is not the issue.

**Third hypothesis: the oracle itself can't trade bull.** This one I had to check before I could let go of it. If the oracle, given perfect foresight, can't make money on a rising tape — maybe because the path-solver penalty structure is wrong, or the trading costs eat the trend, or something — then there is nothing for the student to learn and the failure is structural to the oracle, not the student.

I ran the oracle directly on the bull benchmark window. `greedy_lookahead_path(k=4, n_assets=64, lookahead=20)`, no policy, no model, just the oracle pretending to be the trader. It returned **+265 %** PnL on the bull period.

The oracle can trade bull. The oracle knows what to do. The student is the one who can't follow.

That leaves the loss function. And once I started looking at the loss function, the answer was sitting in plain sight, in a number I had been printing in the training logs for months without thinking about it.

---

## The math of the imbalance

The oracle this project uses — `greedy_lookahead_path` — is a `k=4` selection over `n=64` assets. At each decision step, it looks ahead 20 minutes, computes the best four assets to trade (some BUY, some SELL), and emits a 64-asset action target where four positions are BUY or SELL and the remaining sixty are HOLD.

`60/64 = 0.9375`.

The oracle emits **93.75 % HOLD by construction**. This is not a learned distribution. This is a counting argument. With k = 4 and n = 64 the marginal probability of HOLD on any given (sample, asset) pair is exactly `(n − k)/n`, give or take whatever fractional probability the oracle puts on the borderline cases. From the audit:

| Regime | BUY % | SELL % | HOLD % |
|---|---:|---:|---:|
| Neutral | 3.19 | 3.06 | 93.75 |
| Bull (+0.1 %/step drift) | 3.77 | 2.48 | 93.75 |
| Bear (−0.1 %/step drift) | 2.52 | 3.73 | 93.76 |

The signal that distinguishes bull from bear is the **1.25 percentage point swing in BUY vs SELL within the 6.25 % minority**. The HOLD marginal does not move. The total trade fraction does not move. The only thing that moves is which of the two minority classes carries the trade — and the difference between the regimes is barely over a percent of total policy mass.

Now think about what plain cross-entropy does to a target like that.

The BC loss in `train_phase2.py` was, at the time I went looking:

```python
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
oracle_reshaped = oracle_targets.reshape(-1, n_assets, n_actions_per_asset)
bc_loss = optax.softmax_cross_entropy(policy_reshaped, oracle_reshaped).mean()
```

`.mean()` over `(B, 64, 5)`. Every (sample, asset, action) triple weighted equally. The gradient on this loss is dominated, by gradient mass, by the 93.75 % of (sample, asset) pairs whose target is HOLD. The minority BUY/SELL labels — where direction lives — contribute roughly 6.25 % of the gradient.

The ratio is exactly the imbalance: **15 : 1**. For every one unit of gradient pressure on "learn the direction of this asset," there are fifteen units of pressure on "predict HOLD."

A model with finite capacity, finite training, and a finite optimizer trajectory will settle wherever the loss is steepest. The loss is steepest in the direction of the HOLD marginal. So the model fits the HOLD marginal. And it stops there, because the conditional signal — the 1.25 % BUY/SELL swing within the 6.25 % minority — is too faint to bend the gradient.

You can watch this happen in the action distributions. From the audit, the same checkpoint on the same observation:

| Scenario | Sell | Hold | Buy | LimitSell | LimitBuy |
|------|---:|---:|---:|---:|---:|
| 20 % rally, bull asset | 0.187 | 0.216 | 0.193 | 0.200 | 0.203 |
| 20 % crash, bear asset | 0.195 | 0.222 | 0.198 | 0.193 | 0.191 |

A 40 % spread in market direction translates to roughly *one percent* per-action probability difference at the output of the policy head. The bull asset and the bear asset look indistinguishable to the argmax. Both go HOLD on the smallest amount of jitter. The model is not blind to direction in some abstract sense — the latent carries it, cosine 0.38 — it is blind because nothing in the training loss has ever asked it to *use* the direction.

This is a class-imbalance problem. Not an architecture problem. Not a search problem. Not an oracle problem. The loss function fits the dominant class, and the minority class — which is where everything interesting lives — gets washed out.

The phrase I wrote in the lore note, after I'd traced through this the first time:

> *"The signal exists in the latent — the policy just needs stronger gradient on minority classes to learn the conditional."*
> — `project_bull_blindness.md`

Once I had that, the fix shape was obvious. It just turned out to be harder to land than I thought.

---

## What I tried

There are three angles on a class imbalance, and over April and May 2026 I worked through all three.

### Attempt 1 — Mask HOLD entirely (TASK_725, experiment A/B)

The simplest thing: don't supervise HOLD at all. Drop the gradient on every (sample, asset) pair where the oracle says HOLD, keep the gradient on the BUY/SELL pairs only. The minority class becomes the entire class.

I ran this in two flavours — a high-BC weight version (`--bc-weight-min 0.15`, 5× the default) and a pure-BC version (`--bc-weight-min 1.0`, BC dominates the gradient outright). Both starting from `jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl`, the best Stage-B checkpoint I had at the time, with bull alpha −24.25 % and average alpha +5.73 %.

It was too aggressive. The model lost its calibration for *when* not to trade. With HOLD masked, the policy head learned to fire BUY/SELL on every step — the gradient never told it that the answer is "do nothing" 94 % of the time. Trade density exploded. The model started taking positions in regimes where the oracle would have stayed flat. Bear and sideways performance, previously the only places it worked, degraded sharply. Bull stayed bad.

The lesson is the obvious one in retrospect: you can't fix a 15 : 1 imbalance by zeroing out the majority class. The majority class is also a *correct* signal — "this asset, this step, do not trade" is a real label. Removing it produces a model that has forgotten how to abstain.

I marked TASK_725 *deferred* and moved on.

### Attempt 2 — Per-class weights (TASK_726 — class-balanced BC)

This is the principled version. Don't mask; reweight. Give HOLD a smaller weight and BUY/SELL a larger one, scaled so that the *gradient mass* across the three behavioural classes is roughly balanced.

Phase 1 (Foxglove, the supervised imprint stage) already does this. The Phase 1 trainer in `easy_trainer.py:170-202` computes an adaptive trade weight per batch:

```python
hold_prob = targets_reshaped[:, :, 1]
trade_strength = 1.0 - hold_prob
n_trades = jnp.maximum(jnp.sum(trade_strength), 1.0)
n_holds  = jnp.maximum(jnp.sum(hold_prob), 1.0)
target_trade_weight = jnp.minimum(n_holds / n_trades, 64.0)
loss_weights = 1.0 + trade_strength * (target_trade_weight - 1.0)
```

The trade weight is bounded by `n_holds / n_trades`, capped at 64. In practice it sits around 15 — exactly the imbalance ratio. HOLD samples get weight 1.0; pure-trade samples get weight ~15. The per-batch normalization keeps the total gradient scale stable.

Phase 2 had dropped this. When the Phase 2 training loop was rewritten — the MuZero-style training with factored search targets and BC regularization layered on top — the class-balancing logic from Phase 1 didn't make it across. The new BC loss was plain `.mean()` over the full tensor. TASK_726 is the spec for porting it back.

I implemented it. The change is small — replace the `.mean()` with a weighted sum, exactly mirroring Phase 1's adaptive weighting. Warm-started from the same `724c/phase2_stageB_e300.pkl`. Ran 200 epochs.

The result, after 200 epochs:

| Metric | Pre-726 | Post-726 |
|---|---:|---:|
| Model Trade % | ~6 % | **54.3 %** |
| Bal. Accuracy | ~37 % | 35.9 % (random ≈ 33 %) |
| policy_loss | 1.5 – 1.6 | 1.593 (= ln 5) |
| bc_loss (weighted) | ~1.4 | 1.394 |
| Bull alpha | −24 % | **−432 %** |

That `−432 %` is not a typo. The model traded on every other step, at random, on a tape that punishes random trading more than holding. Bear and sideways went too.

`policy_loss = 1.593` is `ln 5`, which is the cross-entropy of a uniform distribution over five actions. The policy head had collapsed to uniform. The class balancing successfully killed HOLD dominance — and replaced it with maximum-entropy noise.

I sat with this for a day before I understood what had happened. Class balancing tells the model *trade more*. It does not tell the model *trade this way*. With direction-specific gradient still drowned out (the 1.25 % BUY/SELL conditional is still 1.25 % even after rebalancing the majority/minority split), the only thing the rebalancing did was lift the trade rate. The model's best fit, under the new loss surface, was to spread its mass over BUY and SELL uniformly — because nothing in the gradient distinguished them.

From the post-mortem note in TASK_727:

> *"The policy loss sitting at ~ln 5 means the per-asset softmax is near-uniform over the 5 actions. Class balancing successfully killed HOLD dominance; it replaced it with uniform random, not correct direction."*

This is the place where I had to be honest about what I'd done. The TASK_726 fix was the most principled approach to a class-imbalance problem I knew how to write. It was correct in form. It was also the wrong target. The imbalance between *trade* and *hold* is real, but it is not the binding constraint. The binding constraint is the imbalance between *BUY* and *SELL within trade* — a 1.2 pp swing inside a 6 % slice. Class-balancing across the trade/hold boundary doesn't touch that. It just amplifies the trade-rate, leaving the directional ambiguity to settle where it can — at uniform.

### Attempt 3 — Auxiliary direction head (TASK_727 + TASK_728)

The third move: stop trying to fix this through the oracle BC loss alone, and add a second supervision channel whose entire job is to put per-asset direction into the latent.

The construction is in TASK_727. Add a small head off the shared latent — `DirectionHead`, 256-d hidden, 64-d output — that predicts, per asset, the sign of the next K-step cumulative return. Supervise with sigmoid BCE on a binary target (`+1` if cumulative return over the next 20 steps is positive, `0` otherwise), with a dead-zone mask that drops samples where the return is below `1e-4` so we aren't supervising noise.

This is **64 dense labels per sample** — one per asset, every step — versus the oracle's 4 per sample. **16× denser** signal, and crucially each asset is supervised individually. If direction is recoverable from the latent at all, this loss should pull it out.

I ran it. Warm-started from the same pre-726 checkpoint to avoid the uniform-policy basin. 300 epochs. The direction head's prediction accuracy, after the run:

- `dir_acc` plateaued at **0.553**.
- `policy_loss` stayed at **~1.59** (still ≈ ln 5 — policy still uniform).
- Benchmark Bal. Acc: **32.9 %** (worse than pre-726's 37 %).

The direction head learned a tiny bit (55 % vs the 50 % coin-flip baseline) and then stalled. The policy head, which was supposed to benefit from a latent now more directional, did not. The structural path from the direction head's gradient through the shared latent into the policy head's softmax was apparently too long or too lossy to matter.

TASK_728 tried to close the loop structurally. Add a residual pathway: project the 64-d direction logits up to the 320-d policy-logit space and *add* it directly to the policy output, with zero-initialized weights so the pre-training behaviour is preserved at step 0 and the gradient pulls the projection up only as the direction signal becomes informative. And unfreeze the encoder at a slow learning rate (3e-5) so the latent itself can grow its direction representation past the 0.55 ceiling.

That is where things stand on 2026-05-22. TASK_728 has been written. The first sweep of it has not yet produced numbers I trust. Bull alpha, across every variant I have tested under the class-balancing-plus-direction-head regime, is *less* negative than the pre-fix −24 %, but not positive. The best run I have so far — TASK_728 with `direction-weight 0.5` and `encoder-lr 3e-5` — pulls bull alpha into the −12 to −15 % range. Better than −24. Not better than zero.

> *The specimen flowered, then withered overnight. The labels read "convergence"; the produce read "loss." I do not understand why both can be true. I record the contradiction and continue.*

---

## The kickback

There are three layers to the kickback on this one, and I think I owe each of them their own paragraph.

**The first layer is the simplest.** The diagnosis is correct. I am confident the diagnosis is correct. The encoder preserves direction; the search isn't the bottleneck; the oracle can trade bull at +265 % when given the steering wheel; the loss function over-weights the dominant class by 15 : 1. All of that is verifiable and reproducible, and the audit script in `prompt-lib/out_encoder_audit/` will reproduce the numbers on any checkpoint you point it at.

The fix, on the other hand, has not landed. Three attempts in — mask, reweight, auxiliary head — and the best I can show is a partial improvement from −24 % to roughly −13 %. Less catastrophic. Still not deployable. The class-balanced BC was correct in form and wrong in target. The direction head supplies the right kind of gradient and the policy head doesn't propagate it strongly enough. TASK_728's structural injection is the cleanest version of the idea I have, and as of this writing the multi-seed validation isn't done.

I want to be careful not to claim too much, because every time I have claimed a bull-blindness fix on a single seed in this project I have had it overturned at the multi-seed gate. The "single-seed +6.21 % became +3.61 % over five seeds with a 12.2 pp spread" reversal from TASK_743 is the one I quote at myself every time I get excited about a number. Anything I say about bull alpha at one seed is wallpaper until three seeds confirm it.

**The second layer is structural and a little uncomfortable.** Bull blindness is the *visible* symptom of a choice that was made way upstream. The oracle is `k=4` of 64 because the original idea was for trading to be *sparse* — most steps should be HOLD, because over-trading destroys real money via fees and slippage. The 93.75 % HOLD marginal is not an accident. It is the design.

The class-imbalance follows from the sparsity. The bull-blindness follows from the class-imbalance. The class-imbalance is, in some sense, working as intended — it is what makes the model conservative, what keeps it from over-trading in the regimes where it currently behaves well, what gives it the average alpha of +5.73 % that I'm trying to preserve while I fix the bull case.

In other words: bull-blindness is a **load-bearing constraint, not a bug**. Removing it by changing k — say, going to `k=8` (12.5 % HOLD marginal) or `k=12` (18.75 %) — would relax the imbalance and almost certainly help bull. It would also change every other regime's behaviour, because the model's understanding of "when to abstain" would shift. The bear performance, the sideways performance, the crash performance, the trade-density calibration — all of those are tuned, implicitly, against the current HOLD marginal. Move the marginal and they move too.

This is what `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` calls the "richer oracle (k=8 or k=12)" direction. It is the right direction. It is also a much larger change than any of the loss-function tweaks I've described above. It rebuilds the training data. It probably requires retraining Phase 1 from scratch. And there is no way to validate it without running the full pipeline end-to-end, which is a couple of weeks of clock time on the single 3090 in this room.

**The third layer is the honest one.** I came into this expecting bull blindness to be a loss-function bug. The diagnosis says it is. The fix attempts say it is, and isn't, at the same time. Class-balancing the trade/hold split is the textbook answer to a 15 : 1 imbalance, and the textbook answer makes things worse. Adding a denser auxiliary head is the natural follow-up, and the natural follow-up plateaus at 55 % direction accuracy and doesn't translate into policy behaviour. The cleanest structural fix — TASK_728's residual injection plus encoder unfreeze — is *plausibly* a step forward, and it is also not yet validated.

What I am increasingly suspicious of is that bull-blindness is not a single mechanism. It is what happens when several near-misses compound. The encoder preserves direction *weakly* (cosine 0.38 — not 0.95). The latent's per-asset direction is preserved through a 64 → 128 bottleneck that may be too tight for fine-grained per-asset routing. The policy head's pathway from latent → 320 action logits is wide and noisy. The oracle's 4-of-64 selection is structurally sparse. Each of these is fine on its own. Stacked, they produce a system that cannot reliably amplify a 1.2 pp BUY/SELL conditional into a 1 pp action-distribution difference, let alone the 30+ pp difference you would need to trade bull profitably.

The phrase I have started writing in lore notes is: **the classes are coupled**. You cannot independently improve bull without consequences for bear. The directional pressure that would make the model trade bull correctly is the same pressure that would make it over-trade everywhere else. Fixing one regime, on this architecture, with this oracle, with this loss, may not be possible without breaking another.

---

## What's left

The current state of the queue, as of 2026-05-22:

**TASK_726** (class-balanced BC) — applied. Confirmed not sufficient on its own. The class-balancing weight (`--bc-max-trade-weight`) has been turned down from the original 64 to 4, paired with the direction head, because the full 64× rebalancing produced the uniform-policy collapse.

**TASK_727** (auxiliary direction head) — implemented and merged. Direction accuracy plateau at 0.553. The head learns *something*, just not enough.

**TASK_728** (direction-logits → policy-head structural injection + encoder unfreeze at 3e-5) — implemented. First runs in flight. Multi-seed validation pending.

**TASK_761** (structural pivot) — written but not started. Includes the richer-oracle direction (k=8 or k=12), among three other structural moves. I have been hesitant to start this one because it invalidates every result I have so far, including the variants of 738–758 that I have spent the spring evaluating. The cost of changing the oracle is high. But the more time I spend in the loss-function space, the more I suspect the structural answer is the only one that will move the needle.

The phrase "bull blindness" has worked itself into the project's vocabulary. It appears in nine session notes, two task specs, and the canonical findings index. When a new variant comes back from validation, I check the bull column first; if bull alpha is in the same range it has been for the last six months, I file the result as "bull-blindness unchanged" and move on. The phenomenon is well-characterized. The fix is not yet validated.

If I am being honest with the timeline of this article: the diagnosis took me about a week, two weekends ago. The fix has taken about six. I have spent more time being wrong about how to repair the loss function than I spent being wrong about which component was broken. That is the texture of this kind of work — the *which* lands fast, the *how* takes seasons.

`workstation` is still on. The 728 validation just finished — bull alpha came back at −14.6 % on this seed, which means I will be running seeds 13 and 23 overnight before I trust any version of that number. The fan has settled. I will check it again in the morning.

> *The specimen is filed under "bull-shy." I will not move it to the south window again until I know what the south window is for.*

---

## Sources

- `~/.claude/projects/-home-lukas-Trader/memory/project_bull_blindness.md` — primary findings note, root cause and "not the problem" enumeration.
- `~/Trader/tasks/TASK_725_BULL_EXPERIMENTS.md` — initial experiment sweep (mask HOLD, high-BC, pure-BC, temperature sweep, return-weighted BC). Deferred when TASK_726 superseded it.
- `~/Trader/tasks/TASK_726_CLASS_BALANCED_BC.md` — port of Phase 1's adaptive trade weighting into Phase 2's BC loss. Implementation, expected impact, diagnostic check.
- `~/Trader/tasks/TASK_727_AUX_DIRECTION_HEAD.md` — auxiliary per-asset sign-of-future-return head. Post-mortem of TASK_726's uniform-policy collapse.
- `~/Trader/tasks/TASK_728_DIRECTION_POLICY_CONCAT.md` — direction-logits residual into policy logits, encoder unfreeze at 3e-5. Current attempt.
- `~/Trader/prompt-lib/out_encoder_audit/findings.md` — encoder direction audit. The cosine-0.38 measurement, the channel-by-channel direction table, the policy-head softmax tables.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four structural directions (reward shaping, entropy reg, oracle k, incompatible dataset). The richer-oracle path lives here.
- `~/Trader/lore/notes_2026-05-13.md` §99 — TASK_743 multi-seed reversal. The reason I no longer trust single-seed bull-alpha numbers.
