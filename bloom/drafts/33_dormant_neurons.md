> *Observation: at dawn, two-thirds of the bed unfurls toward the light. The remainder stays folded. I mark the unwaking specimens with red thread, prune them back to the bud, and water as before. By the next dawn, some have rejoined the canopy. The rest hold their shape, green, motionless, drawing nothing. I record the count of unwakers and continue.*

---

**by Lukas — May 22, 2026**

# Dormant Neurons

*On the units that stop firing, the paper that named them, the fix that recycles them, the modest lift it produced, and the harder question of whether dormancy was ever the disease in the first place.*

---

## The histogram with the long left tail

There is a plot I add to the dashboard early in 2026 that I keep returning to even now. It is one of the least elegant figures in `tools/DS24` — a stacked histogram, log-y, one panel per hidden layer, of per-unit activation norms collected over a moving window of training batches. The horizontal axis is the absolute mean activation of a single hidden unit, normalized by the layer's mean. The vertical axis is the count of units that fall in that bin. A healthy layer looks roughly log-normal: a hump near 1.0, tails decaying on both sides, nothing piling up at zero.

Almost none of my layers look healthy.

The pattern, run after run, is a hump near 1.0 *and* a spike at zero. Sometimes the zero-spike is 5% of the units. Sometimes 10%. In a couple of bad seeds, on the value head specifically, I have watched it cross 30% inside the first thousand updates and then sit there, immovable, while the rest of the network trains around it. Those units have stopped firing. Their pre-activations have drifted negative; their ReLU outputs are zero on every batch; their gradient is identically zero; the optimizer is no longer touching them. They are dormant.

I am writing this in late May 2026, from `workstation` — single EVGA 3090 FTW3 Ultra, 420 W, desk fan on, `nvidia-smi` showing a checkpoint validation in an adjacent tmux. The DS24 dashboard is up in a browser tab. On its leftmost panel, the dormancy fraction for the current run sits at 3.1% for the encoder MLP, 4.4% for the value head, 6.8% for the policy head. Those numbers used to be much worse. This article is about the period when I found out they were a known thing with a name, implemented the fix the literature pointed me at, found that the fix worked exactly as advertised, and then discovered that "worked exactly as advertised" and "moved the metric I actually cared about" were two different statements.

The literature gave me the diagnosis. The implementation took a week. The kickback took a month.

## The phenomenon

A neuron in a deep network is *dormant* when its activation, normalized by the typical scale of the layer it lives in, sits below some small threshold τ across the entire data distribution, for the duration of training. That is the Sokar et al. (2023) definition — *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML — and the load-bearing word in it is *duration*. A unit that happens to be quiet on one batch is not dormant. A unit that has been quiet on every batch for the last N optimizer steps, *and stays quiet after gradient updates*, is. The empirical claim of the paper is that this state is *absorbing*: once a unit enters it, the gradient signal to it is too weak to revive it, and ordinary training will not bring it back.

Sokar's paper is specifically about RL, and the reason it is specifically about RL is that dormancy rates in RL are dramatically higher than in supervised learning. The paper's headline numbers are not subtle — on DQN/Atari, after a few hundred thousand environment steps, *more than half* of some networks are dormant. In supervised vision baselines on the same architecture, the rate is a few percent. Something about RL training is uniquely hostile to keeping units alive.

The mechanism is not exotic. It is the dying-ReLU problem, amplified by three specific RL pathologies stacked on top of each other.

**Non-stationarity.** The target distribution in RL changes as the policy changes. A unit that was usefully selective for a feature in early training can become irrelevant once the policy contracts onto a smaller part of state space. Its inputs drift; its pre-activation distribution drifts with them; eventually the pre-activation lives mostly below zero. The unit goes dark. In supervised learning the target is stationary and a useful unit *stays* useful; in RL it does not.

**Sparse rewards.** Most steps return uninformative reward. The gradient signal that *would* revive a unit comes from the rare informative steps, and those signals have to fight through every other source of variance in the optimizer. In a trading agent where 90% of timesteps are `HOLD` and the reward is dominated by transaction-cost noise, the revival budget is tiny.

**Bootstrapped value targets.** This is the one that, for me, made the phenomenon feel structural rather than incidental. The value head trains against a target that is itself a function of the value head — `target = r + γ · V_θ⁻(s′)`. When the policy collapses, the value head sees a stream of near-constant targets — typically near zero, since the collapsed policy holds and the per-step reward is near zero. A near-constant target is a near-zero gradient. A near-zero gradient does not revive dormant units. The collapse and the dormancy reinforce each other: the dormant units cannot represent the dynamics that would un-collapse the policy; the collapsed policy produces targets so flat they cannot revive the dormant units.

The math of why this is self-stable is depressingly short. Take a single ReLU unit with weights `w`, bias `b`, input `x`, pre-activation `z = w·x + b`. The unit's output is `max(0, z)`; its derivative is `1` when `z > 0` and `0` when `z ≤ 0`. The gradient of any downstream loss with respect to `w` flows through that derivative. If `z ≤ 0` on every sample in a batch, the gradient with respect to `w` from that batch is the zero vector. The optimizer step is zero. `w` and `b` do not move. On the next batch, the same statement holds. The unit cannot revive itself from inside the optimization process; it can only be revived if some *other* unit upstream changes its output distribution enough to push this unit's pre-activation back into the positive regime. In a network where the upstream units are themselves being trained against near-constant targets, that push does not arrive.

The dual collapse picture comes from `docs/AI_bleed research/big gemini.txt`, which a Gemini Deep Research run produced for me in early 2026 when I asked it to survey the literature on MuZero training pathologies. The phrasing that hit hardest was:

> *"Research indicates that once neurons become dormant, they lose their ability to adapt to new gradients, even if the reward landscape changes (e.g., via entropy boosting). This results in a loss of network plasticity. The network effectively loses capacity, collapsing its representational power to a simple linear discriminator that outputs HOLD regardless of the input state."*
>
> — `docs/AI_bleed research/big gemini.txt` §2.2

That is the loss of plasticity claim. It is the reason the phenomenon matters. Capacity that has gone dormant is not just unused; it is *unreachable*. A 256-unit hidden layer with 30% dormancy is, behaviorally, a 179-unit hidden layer that you cannot get back to 256 by training harder or by raising entropy.

## Detection

You cannot fix what you cannot see. Before any thought of recycling, the first thing I needed was a number on the dashboard.

The implementation lives in `mcts_trader/jax_agent/redo.py`. The interface is unromantic — a `DormancyMonitor` class that accumulates per-unit activation statistics across batches, then on request produces a score per unit and a boolean mask. The scoring choice is the one place the design matters. Sokar et al. score units by their *normalized mean absolute activation*: `S_i^l = E_x [|φ_i^l(x)|]` divided by the layer's mean. A unit whose mean output sits at less than τ × (layer mean) is dormant. Their reference τ is 0.025; my project's spec in `tasks/TASK_042_REDO_DORMANT_NEURONS.md` picks 0.01 as the dormancy threshold, with the dormancy target *fraction* held below 10%.

I had a small internal argument early on about whether to score by mean activation magnitude or by the *coefficient of variation* — variance divided by mean. The TASK_042 file shows the result of that argument; the comment in the code keeps the receipt:

> *"def compute_dormancy_score(activations, axis=0):*
> *    'Compute normalized activation variance.'*
> *    mean = jnp.mean(activations, axis=axis)*
> *    var = jnp.mean((activations - mean) ** 2, axis=axis)*
> *    # Normalize by mean magnitude to get coefficient of variation*
> *    score = var / (jnp.abs(mean) + 1e-8)*
> *    return score"*
>
> — `tasks/TASK_042_REDO_DORMANT_NEURONS.md`

The coefficient-of-variation framing catches a failure mode the pure-magnitude score misses: a unit that outputs a stable nonzero *bias* but never modulates with input. Its mean absolute activation is large; its information content is zero. The CV score scolds it correctly. I keep both available; the dashboard uses CV by default.

The capture itself is the dull part. Hooking activations in Flax is not zero-effort — you have to either thread an `activations` dict through the call site or use `nn.Module`'s `sow` collection mechanism. I used `sow`. The forward pass during the monitor's measurement window writes into a `'intermediates'` collection that gets unpacked at the call site. It costs maybe 2% overhead on a forward pass, and it only runs during the measurement window, which is one batch in fifty.

The output is the per-layer histogram on DS24 that I described at the top of this article, plus a single scalar — *dormancy fraction* — that gets logged to the run timeline. The scalar is the thing I actually watch. A run whose value-head dormancy fraction crosses 15% in the first 5,000 updates is a run I now kill immediately, because the relationship between that crossing and subsequent Alpha collapse is the most reliable predictive correlation I have in this codebase. Not the loss curve. Not the gradient norm. Dormancy fraction.

That sentence — *"the most reliable predictive correlation I have in this codebase"* — is one of the few unambiguous wins from TASK_042, and you can read the rest of this article without it disappearing.

A small note on the *per-layer* breakdown, which is the thing I underestimated before building it. The aggregate dormancy fraction across the whole network is a number that hides where the rot is. In practice the dormancy distribution is wildly non-uniform: the encoder MLP holds 2–4% steadily; the policy head fluctuates between 5% and 10%; the value head is the one that, when a run is going to collapse, leads the rest of the network into the basement. The first time I plotted the three series on the same panel I realized I had been chasing the wrong quantity. The aggregate number is a lagging indicator of value-head dormancy. The value-head series is the leading indicator. Now the dashboard shows them stacked, with the value-head line drawn slightly heavier, and the early-kill rule is value-head-specific: above 15% inside epoch 3, the run dies.

The other thing the per-layer breakdown made visible is that *recycling itself has a per-layer signature*. The encoder MLP rarely needs recycling; when it does, the units come back quickly. The value head needs recycling most often and the recycled units regrow into dormancy fastest. The policy head sits in between. This pattern is consistent with the bootstrap-targets story — the value head trains against the flattest, least-informative targets, so its units have the weakest revival budget, so they fall dormant fastest and recover slowest. It is also evidence for the symptom-not-cause framing that the kickback section is going to make explicit.

## The fix: Redo

The Sokar et al. proposal is conceptually simple. Every N optimizer steps, score every unit, mask the dormant ones, and **recycle** them — re-inject plasticity by partially re-initializing them, and reset the optimizer state for the affected parameters so the new weights do not get steamrolled by stale momentum.

The non-obvious detail, which the paper is careful about and which I missed on first reading, is *which weights you reset*. There are three knobs per unit: incoming weights (the row of the previous layer's output matrix feeding into this unit), outgoing weights (the column of *this* layer's matrix feeding the next layer), and bias.

The Sokar prescription, reproduced in the Gemini summary almost word-for-word, is:

> *"Reset: Reinitialize the incoming weights W_in of dormant neurons using the original initialization scheme (e.g., Kaiming Uniform). Zero Out: Set the outgoing weights W_out to zero. This ensures the reset does not immediately disrupt the network's output, allowing the neuron to 'wake up' gradually via gradient descent."*
>
> — `docs/AI_bleed research/big gemini.txt` §5.1

The order matters. If you reinitialize the *incoming* weights without zeroing the outgoing weights, you have just injected random noise into the downstream layer's input — a non-trivial perturbation of an already-trained network. The unit will fire on the next batch with a random output, and that output will land on the next layer's weights with the old learned magnitudes. The network's behavior on the recycled batch changes discontinuously. Bad.

If you zero the outgoing weights, the recycled unit's output cannot reach the next layer at all on the first forward pass after recycling — the network's behavior is preserved exactly. But now the unit *has* a fresh, random row of incoming weights, so its pre-activation will vary across batches; some samples will produce positive pre-activation; the unit will fire; the gradient on its outgoing weights (which are currently zero) will be nonzero; the optimizer will start writing into the outgoing weights from zero. The unit *wakes up* into the network gradually, at a rate set by gradient descent on its outgoing weights, rather than rejoining it as a discontinuity.

My implementation in TASK_042 is, embarrassingly, simpler than the canonical recipe. I read the paper's main figure and not the appendix, and I implemented the *incoming-reset, bias-reset* version without the outgoing-zero step:

> *"def recycle_layer(layer_params, mask, rng):*
> *    'Recycle single layer's dormant neurons.'*
> *    kernel = layer_params['kernel']  # (in_features, out_features)*
> *    bias = layer_params.get('bias', None)*
> *    # Generate new weights for dormant neurons*
> *    new_kernel = jax.random.normal(rng, kernel.shape) * init_scale*
> *    # Replace only dormant columns*
> *    # mask shape: (out_features,) -> expand to (1, out_features)*
> *    mask_expanded = mask[None, :]*
> *    recycled_kernel = jnp.where(mask_expanded, new_kernel, kernel)*
> *    [...]"*
>
> — `tasks/TASK_042_REDO_DORMANT_NEURONS.md`, the `recycle_layer` helper

There is a subtle naming inversion in that snippet that I want to flag because it tripped me up for a day. The kernel has shape `(in_features, out_features)`. "Recycling a neuron" means re-initializing the weights that feed *into* it — but if the neuron we are recycling is one of the `out_features`, those weights are the *column* of this layer's kernel, not the row. The mask is over `out_features` and gets expanded along axis 0. That is the incoming-weight reset for the units in *this* layer. The outgoing weights — which feed the *next* layer — are the rows of *that* layer's kernel, indexed by the same dormancy mask. My current implementation does not zero them. It probably should, and that is on the list.

Beyond the weight reset, the optimizer state needs the same treatment. Adam's first and second moments are per-parameter scalars; they remember the gradient history that drove the unit dormant. If I reinitialize the weights but leave the moments alone, the next step pulls the new weights right back toward the old, dormant configuration. The recycle pass therefore zeros the relevant slices of `opt_state.mu` and `opt_state.nu` — Adam's momentum and variance — for the recycled units. The optimizer treats those parameters as if they had just been initialized, which is what we want.

The cadence is the last knob. Sokar recommends checking every 1,000 environment steps, which on my training loop translates to roughly every 10 epochs. I set `--redo-interval 10` and left it there for the duration of the experiments. Every tenth epoch, the monitor flushes its accumulated statistics, the recycler runs over the masked units, the dashboard logs the count of recycled neurons, and the loop continues.

In practice the *first* recycle pass after warm-up recycled the most units. A typical run, on the value head specifically, would recycle 30–80 units on the first pass, then 5–15 on each subsequent pass, then settle into a steady state of 0–5 per pass once the policy had warmed up. The histogram's zero-spike would shrink visibly within two passes. Dormancy fraction would drop from ~12% to ~3% in mid-training. Loss curves would get noticeably smoother — fewer little stair-steps in the value loss, less variance batch-to-batch.

> *The specimen wakes. I count the unfolded leaves and find more than yesterday. The harvest weight is unchanged.*

## The kickback

There are three layers to the kickback on this one, and I want to take them in order because the order is the lesson.

**Layer one: it worked, but it did not move Alpha.**

I want to be careful not to claim too much. Redo did exactly what the paper said it would do. Dormancy fraction dropped. Recycling pulled the network back from the kind of slow-rot capacity loss that, before TASK_042, I would discover only when looking at a dead training run in retrospect. The histogram's left tail thinned. The value loss curve was smoother. On a couple of runs where, pre-Redo, the value head had drifted into a 25–30% dormancy regime and then never recovered, Redo held it under 10% for the full duration. By every internal metric the fix touched, the fix succeeded.

And Alpha was within noise. A paired three-seed comparison, Redo vs no-Redo, seeds 7/13/23, on the standard Stage-B baseline at the time. Mean ΔAlpha was +0.4 pp, range −2.1 to +1.9, with two seeds positive and one negative. *Statistically nothing.* By the bar this project had already set for itself — paired Δα over at least three seeds, with the mean clear of about ±3 pp — Redo did not move the deployment metric.

This is the same shape as Articles 6 and 8. An upstream pathology gets diagnosed, named, fixed; the downstream metric does not move. The first time this happened I treated it as a bug in my evaluation pipeline. The second time, as a bug in my mechanism. By the third time — Redo — I had to take seriously that this was *the shape of the project*, not the shape of a single intervention. The pathologies are real. The fixes for them are also real. The deployment metric, Alpha, is somehow downstream of a layer of causality that none of these individually-correct fixes are touching.

**Layer two: was dormancy ever the disease?**

Sokar et al.'s framing is that dormancy *causes* capacity loss, which causes worse policy, which causes worse return. Recycle the dormant units, restore the capacity, restore the policy, restore the return. The causal arrow runs from network state to outcome.

There is an alternative framing in which the causal arrow runs the other way. Dormancy is a *consequence* of an under-constrained loss landscape — when the targets are flat, the gradient signal is weak, and weak gradient signal is *exactly* the condition that makes units fall dormant and stay there. Under this framing, reviving units in the same loss landscape produces dormancy *again*, because the structural cause is the landscape and not the units. The fix treats the symptom, the landscape regenerates it.

I have evidence for both framings in this stack. On the Sokar side: Redo demonstrably reduces dormancy, demonstrably smooths the loss curve, demonstrably prevents the cliff-edge collapse runs in which dormancy crosses 25% and never returns. The intervention works on its declared target. On the landscape side: in *every* run, dormancy fraction creeps back up between Redo passes. The recycled units enter the network, train for ten epochs, and a non-trivial fraction of them return to dormancy by the next measurement window. The slope of regrowth depends on how flat the value targets are during that window. On runs where the policy has contracted hard onto HOLD, the regrowth is fast. On runs where the policy is still exploring, slow. The dormancy rate is *informative about* the loss landscape; it is not *separate from* it.

Sokar's paper does not resolve which framing is correct. It demonstrates that recycling helps on the benchmark suite they study, which is enough for the paper's claim but not enough to settle the causal question. I do not resolve it either. What I will say is that on this stack, the second framing — dormancy as symptom of an under-constrained loss — is the one that has been holding up under repeated probing. The recycled-units-go-dormant-again result is not a failure of Redo; it is a finding about the landscape.

That finding redirects where to look next. If dormancy is a symptom of flat targets, then *changing the targets* — value-target sharpening, reward shaping, policy-entropy regularization, oracle reformulation — is the structural intervention, and recycling is local cosmetic surgery on top of it. TASK_761's roadmap reflects that reordering. Redo stays in the codebase; it is not the thing.

**Layer three: instrumentation paid for itself even when the fix did not.**

The most valuable artifact from TASK_042 is not `recycle_dormant_neurons`. It is `DormancyMonitor`. Once dormancy fraction was a number on the dashboard, two things became possible.

First, run triage. A run whose value-head dormancy crosses 15% in the first 5,000 updates is now an *automatic* kill — it correlates so reliably with later Alpha collapse that I do not even let those runs finish. Saved GPU-hours-per-month, conservatively, the equivalent of one or two full Phase-2 trainings. On a single 3090, that matters.

Second, regression detection. In early March, a misconfigured run pushed value-head dormancy to 40% inside epoch 3 — I noticed it on DS24 within an hour. The misconfiguration was a stale learning-rate setting that had been merged in and not validated; the dormancy spike made it visible faster than any of the standard loss/accuracy metrics would have. Caught, reverted, no Alpha cost.

The lesson is the unromantic one. The fix did not move the metric I cared about. The *instrument I built in order to validate the fix* is now load-bearing in how I operate the project. The instrumentation is more valuable than the fix.

## What's left

Redo is in the codebase. It is wired to `--redo`, `--redo-interval`, `--redo-threshold`, and `--redo-init-scale`. It runs by default on Stage-B training. It costs about 2% throughput and prevents the cliff-edge collapse mode that, before it existed, used to eat a run a week. Worth keeping.

Dormancy fraction is on the dashboard. It is the third metric I look at on a new run — after loss and after gradient norm, before Alpha. It is the metric I trust most for early-kill decisions. It is also the metric I trust *least* for predicting Alpha gains, because TASK_042 itself showed that improving it does not improve the deployment number.

What I would do differently is small but real. The outgoing-weight zero-out from Sokar's actual prescription is on the list to add — my current implementation re-injects recycled units as a discontinuity in network behavior rather than as a gradient-paced wake-up, and I suspect (without having tested) that the per-recycle Alpha noise I see in the paired comparison includes some signal from that discontinuity. Cheap to fix, worth a measurement.

The next research direction goes after the same problem from the loss-landscape side. TASK_761's policy-entropy regularization is the headline; the structural claim is that *constraining the policy distribution from above* gives the value head a richer target distribution, which gives gradient signal that can revive units without requiring an explicit recycler. Same disease, different layer of the stack. If it works — and the bar for "works" is, as ever, ≥+3 pp paired Δα over three seeds — then dormancy fraction should fall *as a side effect* of the policy-side intervention, which would be the cleanest possible evidence for the symptom-not-cause framing.

The name stays. Dormant neurons are dormant neurons. The phenomenon is real, the paper is right that it exists and right that it can be intervened on, and the project's vocabulary is better for having it. Some of the unwakers are still there at dawn. The gardener still marks them in red. The harvest basket is still the metric that matters.

## Sources

- Sokar et al., *The Dormant Neuron Phenomenon in Deep Reinforcement Learning*, ICML 2023.
- `tasks/TASK_042_REDO_DORMANT_NEURONS.md` — spec, scoring choice, recycling helper, CLI flags.
- `mcts_trader/jax_agent/redo.py` — `DormancyMonitor`, `recycle_dormant_neurons`, optimizer reset.
- `docs/AI_bleed research/big gemini.txt` §2.2 (phenomenon), §5.1 (ReDo prescription) — Gemini Deep Research synthesis.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — policy-entropy direction.
- DS24 dormancy panel — `tools/DS24/`.

> *Some specimens fold for the season and unfold the next. Some fold and stay. I record the count, mark the bed, and tend the soil.*
