> *Observation: the specimens grow without producing fruit. Leaves green, stems strong, the harvest empty. I record what I see: roots in the soil, water in the channels, light at the canopy. The plant performs every visible act of living. The harvest basket stays empty. I write the date, the weight, the verdict — and I begin again at the soil.*

---

**by Lukas — May 21, 2026**

# 22 Silent Killers: The BUG_MAYHEM Audit

*On the disquieting feeling of a pipeline that runs but does not learn, the 2026-01-29 prompt that named the hunt, and the lesson that the loss curve is not your friend.*

---

## The pipeline that learns nothing

There is a specific kind of disquiet that comes from a training loop that does everything right except the one thing it was built for. The dashboard ticks. The GPU is warm. `nvidia-smi` shows utilization where utilization should be. The loss curves descend more or less monotonically. The replay buffer fills, the priorities update, the gradient norms stay inside their clip. By every visible metric the system is working. And the agent is not learning to trade.

In late January 2026 I had been staring at this configuration of symptoms for long enough that it had stopped feeling like a bug and started feeling like a property of the universe. Run after run, on different seeds, on different curriculum levels, the model would imprint, the loss would fall, the validation metrics would look plausible, and the Alpha — the only number I actually cared about — would sit in a flat band around zero with occasional excursions in both directions that I could not connect to anything I had changed. I would tweak a hyperparameter and watch the Alpha drift the wrong way. I would revert it and watch the Alpha drift again, in a third direction. There was no signal anywhere. There was just weather.

I am writing this in May, from `workstation`, the workstation that has carried this whole project on a single EVGA 3090 at 420 W. The desk fan is on. The git-objects directory is symlinked into `/dev/sdb1` because the system drive has been full for months. In the next tmux window an unrelated checkpoint validation is ticking through periods. I bring up `workstation` and that 3090 every time I write one of these because the constraint is part of the story: every dead end in this notebook was found on this single card, in this single house, in evenings after the day job. The 22 bugs in this article were found here. Nothing automated found them. There was no other machine to corroborate them with. There was a person, a chair, a dashboard, and a slow growing suspicion.

In `lore/notes_2026-01-*`, the Alpha numbers from those weeks form a flat band of disappointed noise. The model would print money in epoch 1 — 1492% Alpha on the very first batch is in the chat log, a number so absurd you do not even register it as wrong at first, you just register it as "instrument out of range" — and then settle into the band by epoch 5, and stay in the band until I gave up. Something was broken. Nothing was crashing.

This article is about the audit that ended that period. I want to tell you what was wrong, in detail, because the *shape* of what was wrong is what stayed with me. It is also about a prompt I typed at 19:53 on a Thursday in late January, which turned out to be the most consequential prompt I would type in the entire project.

## The setup: how a model trains without learning

Before the audit it would have been useful to have a vocabulary for the failure I was looking at. I did not have one then. I have one now, and looking back the 22 bugs fall into three categories. Naming the categories matters because the only reason the audit worked was that I framed it categorically rather than as "find any bugs." So:

**Money-printing bugs.** The reward signal sees something the agent state does not — phantom rewards, fee leakage that does not register, sign errors, double-counted episode boundaries. The model receives a reward signal that is, somewhere, lying. The lie does not have to be large; it has to be *systematic*, and the same direction every time, and the agent will happily integrate it into its policy. The 1492% Alpha on epoch 1 is the loud version of this category. The quiet version is a 0.3 pp inflation you would never have caught, applied to everything.

**Future-leakage bugs.** The input contains information from after the decision point. Window misalignment. Normalization computed over a wider slice than the decision-time history. Mask tokens that are not actually masked when the gradient flows. A "current price" that quietly indexes sixty bars in the future because somebody added a `+ window_size` to the wrong line and nobody else looked at it. The classic causality violation. The model does not have to be smart to exploit this; it just has to be a function approximator with capacity.

**Algorithm-stack bugs.** The components disagree with each other. PopArt updates the running variance after the gradient step instead of before, so the rescaling is one step stale. The Prioritized Experience Replay computes importance-sampling weights from the priority before the TD update, not after — every batch is reweighted by stale priorities. The Behavioral Cloning mask is applied after the loss reduction so the model gets gradient on its own padding. Each individual disagreement is small. The combination is the kind of thing that looks, externally, like "the model just does not learn."

The thesis of the audit, which I did not write down at the time but which is obvious in retrospect, is that you cannot reliably distinguish category one from category two from category three by reading the loss curve. They produce the same external symptom: smooth descent, plausible val numbers, dead policy. The way you tell them apart is by *naming the category* and then *looking specifically* for instances of it.

## The audit prompt

On 2026-01-29 at 19:53, I opened a Claude session and typed this:

> *"TASK: Find all 'money printing' and 'future leakage' bugs in this trading RL system.*
>
> *CONTEXT:*
> *— Trading agent showing 1492% alpha on epoch 1 (should be ~0% for random model)*
> *— Fixed one bug where observation saw 59 FUTURE prices, but issue persists*
> *— The fix message '[TASK_121] self_play.py loaded with LOOKBACK window fix' IS showing*
>
> *SEARCH FOR:*
>
> *1. FUTURE LEAKAGE (causality violations):*
> *— Any prices[t+X] where X>0 used in observations at time t*
> *— Window slicing: should be [t-59:t+1] not [t:t+60]*
> *— 'next_prices' leaking into features*
> *— Off-by-one in episode_start + current_step calculations*
>
> *2. MONEY PRINTING (reward bugs):*
> *— Portfolio value miscalculation […]"*
>
> — Claude session `76fea01e`, 2026-01-29 19:53

I want to be honest about why this prompt worked. It is not because the prompt is well-written. It is because I had stopped asking a general question and started asking a *typed* one. "Review this code for bugs" is hopeless on a codebase of this size — it returns style notes and vague concerns. "Find bugs of category X with these specific failure modes" gives the model a search target. The agent does not need to decide what counts as a bug; it just needs to grep, read, and check.

The framing matters in a second way too. By the time I typed that, I had already fixed *one* leakage bug — TASK_121 had moved the observation window. The first symptom (1492% Alpha) should have been gone, and was not. The prompt names the prior fix and then says, in effect, *something else of the same shape is still in here*. That is a different question from "is the code correct" and a much easier question for an LLM to answer. The session that followed found the rest of the offset in about ten tool calls.

> *"Found a CRITICAL bug!"* — Claude, ten minutes in.

> *"# CRITICAL BUG REPORT: Future Leakage & Money Printing Bugs / Executive Summary / Found 1 root cause creating multiple downstream bugs that explain the 1492% alpha on epoch 1. The JAX trading env has a 60-step forward offset in trading execution while observations use correct indices."*
>
> — Claude session `76fea01e`, after the first pass

That was the loud one. The audit that followed, across that session and the next two days, found twenty-one quieter ones. The full inventory is in `tasks/BUG_MAYHEM_MASTER_REPORT.md`. Walking through five of them is enough to make the point.

## The first bug: the 60-step lookahead

The first bug was the loud one, and it deserves a paragraph because it is also the cleanest example of why the typed-prompt framing works.

In `mcts_trader/env/jax_trading_env.py` around line 1117, the code that fetched the "current price" for trade execution and portfolio valuation read:

```python
current_prices = get_current_prices(
    prices, state.current_step + config.window_size, state.episode_start_idx
)
```

The observation, fixed in an earlier task, correctly used `current_step` and saw `prices[0:60]` ending at index 59. The trade execution, in the same step, used `current_step + window_size` and reached into `prices[119]`. At step zero the model observed up to bar 59 and was filled at bar 119. The position-entry price the agent then carried as part of its state was a price *from sixty bars in the future*. Claude's note on this was sharp:

> *"This is essentially a state-embedded crystal ball — the model's own position history contains future information!"*
>
> — Claude session `76fea01e`, on confirming the offset

Read that again. The leak was not in the observation tensor. The leak was in the *state*. The model could not see the future directly, but it could see a number computed from the future — its own entry price — and any minimally capable function approximator will learn to back out the future from that number. This is the textbook reason "current price" must be a single source of truth and not be assembled in two different places with two different conventions. We had two. They disagreed by sixty bars. The 1492% Alpha was not "the model is cheating in a clever way." It was "the model has been handed a phone number to the future and is reading it off."

I want to keep this paragraph short because the structurally interesting part is not this bug. This bug is *loud*. You can see it in epoch one. The interesting bugs are the ones you cannot see in epoch one.

## Five quiet ones

### Prioritized Experience Replay, off by a TD step

Prioritized Experience Replay (PER, Schaul et al. 2016) is the standard fix for the fact that a uniformly-sampled replay buffer wastes capacity on transitions the model already understands. You assign each transition a *priority* — typically the absolute TD-error from the last time you trained on it, with newly inserted transitions getting maximum priority — and sample with probability proportional to priority. To stay unbiased you reweight each gradient by an importance-sampling weight, $w_i = (1/N \cdot 1/P(i))^\beta$, where $P(i)$ is the sample probability and $\beta$ anneals from some starting value toward one.

The math has a subtle ordering requirement that nobody talks about because in most reference implementations it happens to be correct by accident. The IS weight you apply to a gradient must use the priority that *induced the sample*, not some updated priority that comes from the gradient you are about to compute. They are necessarily different. The priority you used to sample was the TD-error from the previous time this transition was trained on. After this gradient step, you update the priority to the new TD-error. Both values exist; they are not the same value; the IS-weight expects the *pre-update* one.

Our `expert_buffer.py` was applying the IS weight after the priority update, using the post-update value. The bias this introduces is not stochastic. It is systematic. High-loss samples get up-weighted, then their priority climbs, then they get sampled even more aggressively, then they get up-weighted again. The buffer slowly converges on a small subset of always-surprising transitions, and the rest of the data effectively stops contributing gradient. From `BUG_MAYHEM_MASTER_REPORT.md`:

> *"`PER Weights Unused` — `expert_buffer.py` — Training is biased toward high-priority samples without IS correction."*

> *"`Demo Sampling Ignores Priority` — `expert_buffer.py` — `_sample_from_range` uses `np.random.choice` instead of segment tree. PER is broken for demos."*
>
> — `tasks/BUG_MAYHEM_MASTER_REPORT.md`, §I

We had two PER bugs in the same module. One on the agent side, one on the demo side. They composed. The agent-side bias collapsed the agent buffer toward outliers; the demo-side flat sampling meant the demo buffer was effectively un-prioritized at all. The loss curve did not catch either because the loss is computed against the (incorrectly weighted) batch, not against the population. *The model is always doing well on what you are showing it. The question is what you are showing it.*

### PopArt desync

PopArt (van Hasselt et al. 2016, *Learning values across many orders of magnitude*) is a normalization trick for value heads that maintains a running mean and variance of the target returns and rescales the head's outputs so the gradient sees roughly unit-scale errors regardless of the absolute return magnitude. The "Pop" in PopArt stands for "Preserving Outputs Precisely" — when the statistics shift, the weights of the head are *corrected* (linearly compensated) so that the network's predictions don't jump.

The correction is order-sensitive. The update has to happen *before* the gradient step on the new batch, because the gradient is computed against the normalized target, and the normalization is what the new statistics define. If you update the running stats *after* the gradient — which is the natural order if you're not thinking about it — the gradient was computed against last batch's normalization and the weight correction was computed against this batch's. The two never agree. Your value head drifts.

From the master report:

> *"`PopArt Desync (Outside JIT)` — `trainer.py` — Value head correction and gradient updates are desynchronized."*

> *"`PopArt Cat. Head Incompatibility` — `popart.py` — Weight correction formula (σ_old/σ_new) is mathematically invalid for categorical logits."*
>
> — `tasks/BUG_MAYHEM_MASTER_REPORT.md`, §I

Two bugs in one file. The desync was the smooth-symptom one: it caused a slow, monotonic drift in the value-head outputs, always in the same direction relative to whichever way the return statistics were moving. Drift like that does not appear as a spike. It appears as a value loss that descends about half as fast as it should, and a policy that is always trading off slightly stale value estimates. The fix was to either move the statistics update inside the JIT-compiled training step (so the order is enforced by the trace) or eat the synchronization cost of doing it explicitly before the gradient call. We ended up doing the latter, because the former exposed a separate problem with how the categorical head's normalization composed with the running stats. That second problem — the σ_old/σ_new correction being mathematically nonsense for categorical logits — turned out to require turning the correction *off* for that head entirely. The fix is two lines. Finding the bug required reading the PopArt paper carefully enough to notice the math was being applied to the wrong object.

### The Alpha-scale mismatch

The Alpha-Relative reward — the model's actual training signal, the number we cared about — was being computed in basis points in one place and percent in another. Ratio: 100×. The training side multiplied; the evaluation side did not; the comparison metric used a third convention. Wherever the two paths met, one of them was off by two orders of magnitude.

> *"`Alpha-Relative Scale Mismatch` — `trainer.py` — 100x inflation/deflation of reward signals."*
>
> — `tasks/BUG_MAYHEM_MASTER_REPORT.md`, §I

The thing I want to land here is that this bug *looked normal*. The reward was 0.02 in some places and 2.0 in others. Both are reasonable numbers for a reward. There is no instrument that screams "your reward is wrong by a factor of a hundred" when the wrong reward is also a plausible reward. You only see it if you trace a single trajectory through the whole stack with a print statement and notice the unit changing between functions. We had not done that. There were comments saying "in bps" and comments saying "in percent" and they were both correct, in their respective files. The model trained against a reward signal that was 100× too small everywhere a comparison happened across the two contexts. The gradient was directionally right and magnitudinally wrong. That is the worst possible failure mode for a value head — the agent updates *toward the right answer*, *slowly*, and you cannot tell from the curve.

### BC masking applied after the loss

Behavioral Cloning was supposed to imprint expert (oracle) action distributions into the policy head during the early epochs. The BC loss is a cross-entropy between the model's policy output and the oracle's. Standard supervised loss. The oracle, for various reasons (mostly to do with how the oracle's k-asset selection composes with the action space), does not produce a target for every position in every sequence; some positions are *padded* and must be excluded from the loss. The standard way to do this is a mask.

Our mask was being applied after the loss reduction. The reduction (a mean over the sequence dimension) was happening on the full unmasked tensor, and the mask was then being applied to the scalar result, which is a no-op in expectation. The model was receiving gradient on the masked positions — meaning it was being trained to predict its own padding token, which is whatever the policy head happens to output at the masked positions, weighted by a target that is whatever the oracle's default-init value happens to be. From `BUG_MAYHEM_MASTER_REPORT.md`:

> *"`BC Confidence Masking` — `trainer.py` — Zero-confidence samples dilute loss denominator, killing gradients."*

This is a mask-after-reduce versus mask-before-reduce bug, the kind of mistake that is famously easy to make and famously hard to see. It does not crash. It produces a loss that is smaller than it should be by exactly the masked fraction — but you do not know the right scale, so "smaller" is just "smaller." The gradient on the unmasked positions is correct, but diluted by the masked-position contribution to the mean. In a sequence where half the positions are masked, the effective learning rate on the real targets is halved. We were running with sequences where roughly a third of positions were padded. Effective BC learning rate during imprinting: 0.66× what we thought it was. That is enough to fail an imprinting curriculum without ever failing it dramatically enough to notice.

### Agent state does not reflect burn losses

This is the one I keep coming back to because the wording in the master report is precise in a way I would not have managed:

> *"Agent state doesn't reflect 'burn' losses seen by the reward signal."*
>
> — `tasks/BUG_MAYHEM_MASTER_REPORT.md`, §II

In the trading environment, an asset can be *burnt* — taken away from the agent at a fee, typically as a consequence of margin or transaction-cost mechanics. The reward signal sees the burn: the symlog-transformed PnL drops, the differential reward registers a loss. The agent's internal state representation, on the other hand, was not updating the holdings tensor when the burn happened. The agent looked at its state and saw "I still hold asset X." The world looked at the agent and said "you no longer hold asset X, and we charged you for taking it." The reward says the asset cost you money; the state says the asset is still earning you money. The two cannot both be true. The model, having no way to reconcile them, learns whichever signal is louder. In our case, the state was louder than the reward, and the model learned that burns were free.

This is a money-printing bug in the strictest sense — the agent was rewarded for trades whose downstream costs were invisible to it — but it is also a state-representation bug, and it is also a sign-error bug (the burn fee was being added rather than subtracted in one of the two places). The audit found all three in the same file. From the report's "HIGH" section:

> *"`Burn Sign Error` — `jax_trading_env.py` — Being flat in a bear market is rewarded as a 'burn' (due to symlog logic)."*

Flat in a bear market should be the safe action. The symlog reward, with the wrong sign, was punishing it. The agent was being incentivized to take a position in a falling market specifically to *avoid* the burn penalty. Read that twice. The reward gradient was pushing the agent toward over-trading in exactly the regime where over-trading is most expensive. We had spent weeks wondering why the model would not learn to sit still in bear markets. The reward was actively telling it not to.

### Phantom rewards

One more, briefly. The episode-summary reward — the scalar logged in the metrics CSV — was being computed by summing the reward tensor over the full fixed-length rollout, including steps after `done=True`. After the agent had been liquidated, the environment kept stepping (because the JAX scan ran for a fixed number of iterations), and each post-done step generated a small negative reward from the symlog of the terminal portfolio value. From the bug-hunt session:

> *"Normal reward: ~-2 (sum of ~1440 small symlog values). With 8 post-done steps: -2 + (8 × -3.93) = -33.4 ≈ -30.99 ✓"*
>
> — `tasks/BUG_HUNT_2026-01-31.md`, BUG_A math verification

The training data was correctly sliced; the logged metric was not. So the *training* was unaffected by this one, but every reported reward number that crossed a `done` boundary was wrong, and every comparison of "this run got reward X versus that run got reward Y" had been comparing two corrupted numbers. The phantom-rewards bug is, in some sense, the gentlest of the twenty-two. It is also the one that taught me to never trust a single-number summary that I had not personally written and traced.

## What the combination did

I have walked through six of the twenty-two. There are sixteen more in the master report, ranging in severity from "definitely catastrophic" to "minor but real." The Alpha-Relative scale mismatch alone would account for slow learning. The PER weight bug alone would account for biased gradients. The PopArt desync alone would account for value drift. The BC mask bug alone would account for weak imprinting. The burn-state bug alone would account for over-trading in bear markets. The 60-step lookahead alone would account for the 1492% Alpha mirage. *Each of them, alone, is a plausible explanation for the symptom I was seeing.*

That is the part I want to dwell on. The symptom — "training loss descends, validation looks plausible, Alpha refuses to move" — does not let you discriminate between these causes. It is the same surface for all of them. If you fix one and the symptom does not change, you have learned nothing about whether the fix worked. The other bugs are still there, dominating the result. If you fix one and the symptom *does* change, you have learned nothing about which fix changed it, because you do not know which of the remaining bugs was the second-largest contributor.

The only honest path through a state like this is the one we took, and it is unsatisfying: you fix all of them, you re-baseline against the clean stack, and you accept that any signal you had collected up to that point about which mechanisms helped or hurt is now uninterpretable. The pre-BUG_MAYHEM experiments are not failed experiments. They are *unscoreable* experiments. I had eight weeks of training runs whose results I could not use even as negative data, because I did not know what they had been measuring.

There is a paper I will go into properly in Article 6 — *Coupled Policy-Prediction Collapse in Model-Based RL: A Comprehensive Analysis of Dual Failure Modes in Stochastic Trading Environments*, in `docs/AI_bleed research/big gemini.txt`, a Gemini Deep Research synthesis I commissioned later in the project — which names the phenomenon of policy and prediction heads silently co-degrading in a way that no single curve reveals. The BUG_MAYHEM audit is the early-2026 worker's version of the same observation: when the failure modes compose, no single instrument resolves them. You need to read the source.

## The first kickback: confidence calibration is the real loss

After the audit I lost the right to read my own loss curves the way I had been reading them. Every smooth descent is suspect until I have verified, by hand or by a careful pair-programming pass, that the gradient is being computed against the right target, with the right mask, through the right normalization head, and that the data being sampled is being sampled the way the algorithm assumes.

That sounds like a small adjustment. It is not. It changes how much *evidence* a given training run is worth. Before BUG_MAYHEM, a four-hour run on `workstation` that showed a clean loss curve and a plausible val number was a data point. After BUG_MAYHEM, the same run is a data point *conditional on* the audit-state of the code at the time of the run. If I changed any of the touched files between this run and the last, the comparison is contaminated. If a refactor moved the reward computation into a new function, I now have to verify that function as carefully as I verified the old one. If a contributor — including a coding agent I was driving — made an "obvious" cleanup to one of these files, I have to re-audit. This is exhausting and correct.

The shape of the cost is calibration. Before the audit I had something like 80% confidence in any given training-curve reading; after the audit I had 20%, and I spent the rest of the project trying to raise the floor by writing infrastructure (instrumented logs, distribution histograms, sanity-check unit tests on the reward) rather than running new experiments. The experiments resumed in March. The trust never fully returned. Every plot in Articles 6 through 10 of this series is a plot I checked twice.

## The second kickback: nothing here was automated

The other thing I want to be explicit about is that *nothing in the BUG_MAYHEM audit was caught by tooling*. Not by a linter. Not by a type checker. Not by a unit test. Not by static analysis. Not by any agent acting unprompted. The 60-step lookahead was caught by Claude given a typed prompt and a specific search target. The PER bug was caught by a careful read by another agent (in a parallel session I drove with a second model, as I wrote at the time: *"good but you do agents but ill use another models for bug hunt also .. it helps to bereak one agent patterns"* — Claude session `09cbe342`, 2026-01-31). The PopArt desync was caught by re-reading the PopArt paper and tracing the JIT boundary. The burn-state bug was caught by me sitting with the dashboard and noticing that the agent was over-trading in a regime where it should not be. The mask bug was caught by adding a print statement to the BC loss and watching the masked-position contributions not be zero.

These are domain-specific bugs in a numerical pipeline. They are detectable in principle — the math is there, the invariants are there — but the invariants live in papers and design notes, not in the codebase's type system. There is no linter that warns *"importance-sampling weights are computed against a priority that has just been mutated."* There is no automated tool that catches *"this mask is broadcast after the reduction."* There will be eventually, for some of these — sharper agents, better symbolic reasoning over numerical code — but for now, in 2026, on a single workstation, the bug-finding tool is a human with context and a careful coding agent driven with a typed prompt.

That last clause is doing real work and I want to land it. The agent did not find these bugs on its own. The agent found them because I knew which kinds of bugs to look for. The structural reason ML projects of this complexity are hard to ship alone is not the GPU budget. The GPU is the cheap part. The expensive part is that the bugs are domain-specific, the symptoms compose into a single indistinguishable smear, and the only sensor that catches them is a human-with-context running an agent-with-context against the codebase under a typed hypothesis. *This is the limiting resource.* Everything else — compute, model weights, framework choices — is downstream.

## What's left, what I learned

The audit produced three artifacts that survive into the present:

**The category framing.** "Money-printing, future-leakage, algorithm-stack" is now the checklist I run before any major refactor. Whenever I touch the reward computation, I ask: did I introduce a money-printing path? Whenever I touch the dataloader or window logic, I ask: did I move a window boundary? Whenever I add a new training-stack component (a normalization head, a buffer modification, a new loss term), I ask: does this component have an ordering requirement with respect to other components, and have I respected it? The taxonomy is general enough that I expect it to outlast this project.

**The typed-prompt template.** *"Find bugs of category X in this code, here is the symptom, here is the prior fix, here are five concrete patterns to look for."* That prompt structure is, to date, the highest-leverage thing I have ever done with a coding agent. I have used variants of it for three subsequent audits — one in `runs/recommendation_2026-05-20/analysis.md` after the eval-MCTS asymmetric-lift discovery, one for the Bella column-mismatch in TASK_740, one for the Titans-MAG bf16 carry-storage fix. The structure transfers.

**The post-BUG_MAYHEM rebaseline.** After the fixes landed, I re-ran the canonical training configuration and recorded the new baseline. Every subsequent experiment in this project — and there are dozens, leading to the multi-seed reversal in Article 8 and the eval-MCTS collapse in Article 10 — is scored against that baseline. The single most important consequence of the audit was not the bugs it fixed. It was that for the first time in the project, there was a *floor I trusted*. Numbers above it were real. Numbers below it were real. The flat band of disappointed noise from January had a name now: it had been a band of *bugs*. After BUG_MAYHEM there was a baseline, and after the baseline there was, finally, the possibility of measurement.

The lesson I want to leave you with is the one that survives in how I read every plot now: *the loss curve is not your friend*. It is a smooth, well-behaved, plausible-looking summary statistic produced by a system that is full of components that can all be quietly wrong in the same direction. The loss curve will descend for any system that has *some* learning signal somewhere, even if every other component is broken, even if the actual quantity of interest is uncorrelated with the descent. You read the loss curve, but you do not *believe* the loss curve. You believe the rebaseline, and you only believe the rebaseline after you have audited the stack that produced it.

## Closing

It is May. The audit was three months ago. `workstation` is on, the desk fan is on, the 3090 is at idle. The flat band of disappointed noise from January is in `lore/notes_2026-01-*`, and reading it now I can tell which weeks were sitting on which bug. The smoke is what is left after the fire — the category names, the prompt template, the floor I trust — and most days that smoke is enough to navigate by.

> *The harvest basket is still empty. But the soil is clean, the channels run true, and the light reaches the canopy without obstruction. I record this and continue.*

---

## Sources

- `tasks/BUG_MAYHEM_MASTER_REPORT.md` — 22-bug consolidated audit (primary)
- `tasks/BUG_HUNT_2026-01-31.md` — the hunting session, BUG_A/B/C/D
- `tasks/TASK_048_reward_hacking_fix_plan.md` — related reward-hacking findings
- `tasks/TASK_051_burning_money.md` — burn-loss bug context
- Claude session `76fea01e-62b0-456f-a0b2-fb8660946da6.jsonl` — the 2026-01-29 19:53 "money printing / future leakage" task prompt and the 60-step-lookahead identification
- Claude session `09cbe342-43ea-4d51-9fa8-f2279c03f0c9.jsonl` — 2026-01-31 11:41, post-crash recovery and the multi-agent bug-hunt orchestration (*"ill use another models for bug hunt also .. it helps to bereak one agent patterns"*)
- Claude session `5925cd72-2ceb-4fe0-9f84-e99c41f0e8e7.jsonl` — 2026-01-31 12:19, the epoch-27 catastrophic-reward-drop debug
- `lore/notes_2026-01-*` — the flat band of disappointed noise that preceded the audit
- `docs/AI_bleed research/big gemini.txt` — *Coupled Policy-Prediction Collapse in Model-Based RL* (Gemini Deep Research synthesis, cited briefly here; full treatment deferred to Article 6)
- van Hasselt et al. 2016, *Learning values across many orders of magnitude* (PopArt)
- Schaul et al. 2016, *Prioritized Experience Replay* (PER)
