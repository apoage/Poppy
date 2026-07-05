# Phase 2: When the Imprint Wakes Up

*On the architectural transition from imitation to planning, on a dynamics network that has been a blank slate for nine thousand epochs and is finally meant to learn something, on why the dream phase has not yet successfully dreamt.*

> *Observation: the seedling has learned the shape of the gardener's hand. It holds the shape well. The trellis is to be moved tomorrow, and the stem is asked to find sun without the wire. I record that the leaves are healthy, the roots are sound, and the question of whether the plant will grow upward on its own is not yet answerable. Both outcomes are documented in the catalogue.*

---

## The ceiling I can read off the dashboard

`workstation` is on. Single 3090, the EVGA FTW3 Ultra, 420 W TDP, in the case that also holds the `/dev/sdb1` SSD I had to symlink `.git/objects` into because the system drive was never big enough. The desk fan is on. `nvidia-smi` is ticking through a validation pass on a Foxglove checkpoint near epoch 8800 in one tmux pane. In the next pane the training dashboard at `localhost:8050` is showing the line that this article is about — the line that does not move.

The line is `agree_trade`. It is the per-bar fraction of the time the model and the oracle both decide to *trade*, conditioned on either having decided to. It is the metric I have been watching for a thousand epochs while it does, essentially, nothing. The dashboard records it at 0.427. A thousand epochs ago it was at 0.421. Two thousand epochs ago it was at 0.418. The balanced agreement number above it, which counts agreement on the easier `HOLD/HOLD` cells, is at 0.85 and looks like a finished line. The trade agreement number is at 0.43 and looks like a wall.

This is the imitation ceiling. It is real. It is structural. And it is the reason there is a `TASK_716` open in the queue called *Phase 2: Dynamics Training + MCTS Integration*, with its priority set to 0, depending only on the direct-sample pipeline that was finished a few weeks ago. The task spec is unambiguous about what's happening:

> "Phase 1 (The Cheat) is complete: 85% balanced agreement on consistent full-period oracle targets, 42.7% benchmark trade agreement (held-out data), model trades 18-22% vs oracle's 2-3% — identifies opportunities but can't filter to top-k. **Imitation ceiling reached**: agree_trade flat at ~42% for 1000+ epochs."
>
> — `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md`, lines 11-15

The numbers in that paragraph are not surprising once you stare at them. The oracle is `k=1` — it commits one trade per decision, the best of 64 candidates, in hindsight. The model is `k=64` — it emits an action distribution over every asset at every bar, and any asset whose softmax exceeds the threshold becomes a trade. The two objects are not the same object. The model and the oracle, on the *trade* sub-decision, *will always disagree at some structural rate* because they're answering different questions. Article 15 walked through why. Article 17 walked through the consequence — five oracles, one per `k`, none of them admitting they should have been five. Article 30 walked through what the state channels were doing in the meantime. Article 39 finished the autopsy: the agreement plateau at 0.525 on the cleaner setup, the same shape, the same wall.

The model has learned the oracle as completely as the oracle's representation will allow it to. The next move is not to imitate harder. The next move is to stop imitating.

This article is about that move. It is about Phase 2.

---

## The Cheat, the Dream, the Thought

The three-phase plan has a name in this codebase. `docs/easy_mode.md` is the design document, written in early 2026 when I was still trying to get the supervised phase to do anything at all on a single 3090, and the framing it uses is the one I keep coming back to:

> "End-to-end MuZero (learning representation, dynamics, policy, and value simultaneously from a cold start) is 'Hard Mode.' It requires massive batch sizes to stabilize gradients and often leads to non-stationary 'death spirals' (bad model → bad search → bad data). **'Easy Mode'** decouples these problems into three stable, solvable steps using **Expert Iteration (ExIt)**. This fits perfectly within the memory constraints of a single RTX 3090 (24GB VRAM) and provides clear debugging checkpoints."
>
> — `docs/easy_mode.md`, §0

The three steps have names. I gave them names because I needed handles on them when the dashboard was on fire and I had to say to someone — usually a chat agent, occasionally the wall — which part was broken.

**Phase 1: The Cheat.** Supervised learning. The oracle computes the optimal action by lookahead, we treat the answer as a label, and we minimize cross-entropy. *"Perfect 'System 1' Intuition"*, the table calls it. The training is stable because the targets are ground truth, the batch sizes can be large because there is no MCTS in the loop, and the result is an agent that is — the design doc says, optimistically — *"80-90% optimal purely on 'intuition.'"* In practice on this stack it is `85% balanced agreement / 42.7% trade agreement / 18-22% trade rate`. The first number is fine. The other two are why this article exists.

What Phase 1 explicitly does *not* do: it does not plan. It produces an action for the current bar by reading the current observation. It has no notion of "what happens next." It is a one-shot reflex.

**Phase 2: The Dream.** Dynamics training. The model gains a second function alongside the encoder and the policy head — a *dynamics function* `g(s, a) → s', r` — that predicts the next latent state and the immediate reward given the current latent and a candidate action. With a working dynamics function, an MCTS procedure can *roll out* counterfactual trajectories. Plant a candidate action, simulate the next state, ask the value head what that state is worth, repeat for thirty-two simulations across sixteen candidate actions per node, and the search produces a distribution over actions that reflects *multi-step* value, not single-step intuition. The agent stops being a reflex and starts being a planner.

What Phase 2 explicitly does *not* do: it does not do self-play. The data is still historical. The oracle is still around as a soft regularizer. The model is not generating its own episodes from scratch. It is *re-evaluating* historical states with a deeper procedure than the policy head alone could do.

**Phase 3: The Thought.** Self-play. The model plays full episodes against itself (or against Bella, the market generator), uses MCTS at every decision, accumulates the visit-count distributions as new training targets, and improves by a true ExIt loop with no external oracle. The design doc calls this *"'System 2' Reasoning"* and pins it to *"Low Compute / High VRAM (Buffer)"* — the search is the cost, the gradients are not. This phase is currently aspirational on this stack. There is a `real_trade_jax.py` that does MCTS at inference. There is no production loop that trains under self-play yet. Article 7 covered the campaign that *was* run; that campaign was Phase 1 work dressed in MCTS clothes (sims=16 at eval, sims=0 at train), and it is the reason I now distinguish carefully between *eval-MCTS* and *train-MCTS*.

The thing the diary articles glossed over is the middle one. Articles 4, 5, 6, 15 are Phase 1. Articles 7, 8, 9, 10 are mostly Phase-1-evaluated-with-MCTS, sometimes called Phase-2-ish but never actually crossing the architectural line. Phase 2 — the real one, with a dynamics function learning state transitions and the policy distilled from MCTS visit counts — has been an open task with checkpoints labelled `phase2_stageA` that ran a handful of times and a deployment-Alpha number that, when measured fairly, is indistinguishable from extending Phase 1. The Dream has not yet been dreamt productively. This article is the architectural spec of what it *would* look like, and a careful statement of why it hasn't worked yet.

---

## The architecture of Phase 2

The Phase 2 model has the same skeleton as the Phase 1 model. It has to — Phase 2 starts from the Phase 1 checkpoint and adds nothing topologically. What changes is which weights are non-zero and what objective they are trained against.

`TASK_716` gives the dimensions verbatim:

> ```
> state_dim:          128
> action_space_size:  320 (5 actions × 64 assets)
> value_support:      401 bins over [-10, 10]
> value_scale:        2.0 (Phase 1)
> hidden_dim:         512 (TRM)
> trm_hidden_dim:     2048
>
> Dynamics layers (all 100% sparse, ~3.3M params):
>   action_embedding/embedding:     (320, 128)
>   dynamics_dense1/kernel:         (256, 1024)   ← concat(latent_128, action_emb_128) → 1024
>   dynamics_dense2/kernel:         (1024, 529)   ← 1024 → state_128 + reward_401
>   dynamics_ln:                    (1024,)
>   dynamics_state_norm:            (128,)
> ```
>
> — `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md`, lines 24-38

Three and a third million parameters that have been initialized to zero and have stayed at zero through nine thousand epochs of Phase 1 training, because Phase 1 never touched the recurrent inference path. Phase 2 is the first time those weights see a gradient. The phrase *100% sparse* in the task spec is precisely literal — the dynamics layers are not under-trained, they are *not yet started*.

There are four moving parts that have to coordinate:

**The encoder + TRM.** This is the Phase 1 product. It maps an observation (27 channels × window × 64 assets) into a 128-dimensional latent. In Phase 2 Stage A it is **frozen**. The latent space is treated as fixed. The dynamics function will learn to operate inside the geometry the encoder produces; the encoder is not asked to make room.

**The dynamics function `g(s, a) → (s', r)`.** Two dense layers and a layer norm. Concatenates the 128-dim latent with the 128-dim action embedding, projects through a 1024-dim hidden, splits the output into a 128-dim next-latent plus a 401-bin categorical over reward. Trained from cold (He-normal init on the kernels, normal(0, 0.02) on the action embedding). Its loss is two terms:

```
consistency_loss = mean((g_state(s, a) - encode(o_{t+1}))^2)
reward_loss      = softmax_xentropy(g_reward(s, a),  categorical(r_t * 100))
```

The first term says "the next latent your dynamics predicts should equal the next latent the *frozen encoder* would have produced if it had seen the next observation." The second term says "the reward bin your dynamics predicts should match the actual realised per-step PnL." There is a subtlety: the consistency target uses `obs_{t+1}` run through the encoder, but `obs_{t+1}` includes the seven state channels that the oracle computed on its own trajectory. So the dynamics target has information the dynamics input doesn't. The task spec is honest about it:

> "consistency_loss will have irreducible floor because target latent includes oracle state channel information that dynamics can't access. Target: < 0.05, not 0.0"
>
> — `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md`, lines 84-85

This is one of the choices that I expect to revisit. Article 30 already showed that the state channels carry oracle leakage; the Phase 2 dynamics inherits a piece of that leakage as an irreducible loss floor. It does not break the training — `0.05` is a fine consistency floor for what comes next — but it does mean that the dynamics is not learning the *full* state transition, only the part of it that survives in the projection.

**The value head `v(s)`.** Inherited from Phase 1. In Stage A it is frozen. In Stage B it is unfrozen and gets a new target: instead of the oracle's remaining-PnL scalar, it gets the root value of an MCTS search rooted at the current observation. The search itself uses the freshly-trained dynamics and the current value head to score rollouts, so the new value target is bootstrapped from the very network being trained. This is the standard MuZero target and it is the standard place deep RL gets unstable. We will return to it.

**The policy head `π(a|s)`.** Inherited from Phase 1. In Stage A frozen, in Stage B unfrozen with a new target: the visit-count distribution produced by the MCTS search at the root. The policy stops being trained against the oracle's `a*`; it is trained against the search's visit-weighted distribution over actions. The oracle is kept around as a *decaying regularizer* — `bc_weight` falls from 0.05 to 0.005 over five thousand steps — so the Phase 1 trade intuition doesn't get unlearned overnight. After five thousand steps it is effectively gone and the model is policy-distilling itself.

The orchestrator is MCTS. The task spec pins the numbers:

> "MCTS sims | 32 — max_considered_actions | 16 — epsilon_greedy | 0.1 — batch_size | 256 (MCTS bottleneck) — REWARD_SCALE | 100"
>
> — `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md`, lines 322-330

Thirty-two simulations, restricted to the top-sixteen candidate actions by policy prior, ten percent of the time taking a random branch instead. The batch is cut from 1024 (Phase 1) to 256 (Phase 2 Stage B) because the MCTS forward passes are the bottleneck and 256 fits the timing budget — about eight seconds per training step, fifty-five minutes per epoch on the 3090. That is a noticeable slowdown but a survivable one; the direct-sample pipeline from `TASK_715` is what makes it survivable at all, because before that pipeline existed the data side of each step was already at four point six seconds, and Stage B would have run at thirteen seconds per step instead of eight.

The transition, stated cleanly: the policy that was *"imitate the oracle"* becomes *"imitate the MCTS-of-self"*. The targets that were oracle-derived become bootstrapped from the same network being trained. The Phase-1 frozen pieces — dynamics, action embedding — get unfrozen and learn for the first time. The Phase-1 trained pieces — encoder, policy, value — switch from teacher targets to self-derived targets. This is the architectural pivot the diary kept pointing at and didn't reach.

The two-stage execution is in the task spec at lines 290-308 and reads almost like a recipe:

> ```
> Stage A: Dynamics Warmup (~200-500 epochs, ~3-8 hours)
>   1. Create phase2_trainer.py with dynamics_loss
>   2. Add --phase 2 mode to train_easy.py
>   3. Modify data pipeline for consecutive window pairs (obs_t, obs_t1)
>   4. Precompute per-step rewards in _make_slot
>   5. Initialize dynamics weights (fresh Flax init, merge with Phase 1 params)
>   6. Train: freeze encoder+policy+value, train dynamics only
>   7. Monitor consistency_loss → plateau = done
>
> Stage B: MCTS ExIt (~1000+ epochs, ongoing)
>   8. Integrate mctx_search into training loop
>   9. Add ExIt loss (policy→visit_counts, value→root_value)
>   10. Add oracle BC regularizer (decaying weight)
>   11. Unfreeze policy + value heads
>   12. Monitor benchmark alpha → should turn positive
>   13. Gradually reduce oracle BC weight
>   14. Optionally unfreeze encoder at 10x lower LR
> ```

The success criteria are also stated. Stage A succeeds if consistency loss drops below `0.01` and reward MSE below `0.5`. Stage B succeeds if **the benchmark alpha turns positive on even one of six periods** and the model's trade rate drops from its current 18-22% toward 5-10%. The second condition is the one I care about. The Phase 1 ceiling produced a trade rate eight times the oracle's; if Phase 2 cuts it in half, that is the first structural argument that *MCTS top-k filtering is doing its job*. The first condition, positive alpha on one of six periods, is the milestone the checkpoint naming convention is reserved for: `phase2_alpha_positive.pkl`. That file does not yet exist on this disk.

---

## The four ways this can fail

Phase 2 is not subtle about its failure modes. The deep-RL literature has been documenting them for a decade. The task spec enumerates four of them at lines 350-360 and they map cleanly to four ways the dashboard in the next tmux pane could turn red.

**Value drift.** The Phase 1 value head was trained against the oracle's remaining-PnL scalar. In Phase 2 it gets retargeted against the MCTS root value, which is computed by the same network. If the Phase 1 value head was *miscalibrated* — and there is at least suggestive evidence from Article 6 and Article 39 that it is, that the classifier-side metrics improved while the deployment Alpha collapsed — then bootstrapping new targets from that head amplifies the miscalibration. Stage B's value loss term is `softmax_xentropy(value_logits, scalar_to_categorical(stats['root_value']))`. The `stats['root_value']` comes from the search which uses the same value head. The loop is exactly the kind of thing that, in deep RL, produces value functions that monotonically converge to numbers that are wrong by a constant.

**Policy collapse.** If MCTS visits, for whatever reason, concentrate on a single action class — and the failure mode I have most often seen on this stack is *all-HOLD*, which is what Article 39 ended up describing in detail — then the policy distillation step trains the policy to emit all-HOLD with high confidence. There is no recovery in the standard ExIt loop because the next round of MCTS rollouts will be even more peaked on HOLD (the priors are sharper, the visits concentrate harder), and the policy gets even more peaked, and so on. The `epsilon_greedy=0.1` term in the search is the supposed antidote but ten percent random branches do not, in practice, save a search whose root value head has decided that no trade is worth taking.

**Dynamics divergence.** The dynamics function has been a blank slate. In Stage A it gets two or five hundred epochs to learn its consistency loss to under 0.01. That's the *one-step* consistency floor. MCTS unrolls *five to sixty* steps. Small one-step errors compound: a 0.01 MSE in latent space at step one becomes who-knows-what after sixty applications of a non-linear map. The task spec recommends starting with `max_unroll=5` and increasing only as consistency stabilizes, and that's prudent, but it doesn't make the compound-error problem go away — it only delays the manifestation. If the dynamics hallucinates a state that doesn't correspond to any encoder-reachable latent, the value head's evaluation of that state is undefined.

**Bootstrap instability.** This is the meta-version of the value drift problem. The TD target uses the same network that is being trained. The Phase 1 supervised loop had ground-truth targets — Phase 2 does not. The literature's standard mitigations (target networks, double-Q, slow updates) exist and are not yet wired into this stack. The `popart.py` normalization that lives in `mcts_trader/jax_agent/` is the closest thing to a stability tool here and it has not been turned on for any Phase 2 attempt.

I want to say one more thing about these failure modes. They are not theoretical. The single time I let a Stage B run continue for two hundred epochs with `bc_weight` decaying as scheduled, what I observed was a trade rate that dropped — exactly as the success criterion specifies — toward five percent, accompanied by a benchmark Alpha that *also* dropped, toward worse than the Phase 1 starting point. The model became selective and became wrong in the same window. This is the policy-collapse + value-drift co-failure. I did not run the experiment to its end because the trajectory was unmistakable. I rolled back to Phase 1 and the task ticket has been open since.

---

> *The trellis was moved. The plant flowered, but inward. The flowers faced the soil and not the sun. I record that the flowering is a sign of vigor and that the direction is a sign of the trellis. Both observations are about the same plant.*

---

## The kickback, in three layers

There are three things I want to put on the page about why Phase 2 has not yet worked on this stack. They are not the same thing. They compound.

**One: Phase 2 has been attempted; the deployment Alpha hasn't moved.** The repository has checkpoints labelled `phase2_stageB_epoch*.pkl` in disk archives I now skip past when I'm `ls`-ing through `runs/`. The Stage A consistency loss converged on those runs — that part of the task spec is mechanically straightforward, it is a supervised problem with a clear target. The Stage B benchmark numbers, run honestly with the eval-MCTS recalibration from Article 10, showed Alpha gains that were statistically indistinguishable from extending Phase 1 for the same wall-clock time. The Dream, on this stack, has so far been a slightly different shape of the same Cheat. The architectural pivot happened in the code; the deployment metric did not notice.

**Two: Phase 2 needs a good Phase 1 to bootstrap from, and the Phase 1 we have is the constrained one.** Article 15 was about the imprint and what it costs to be a perfect imitator of an imperfect expert. Article 30 was about the state channels carrying oracle leakage that pumped up the agreement numbers without pumping up Alpha. Article 39 was about the autopsy of the agreement plateau at 0.525, where the model and the oracle, on the cleaner setup with the leakage removed, settled on a number that wasn't going to climb. The pattern across those three articles is that *Phase 1 has been hitting a ceiling that is set by the oracle's k=1 formulation and not by the model's capacity*. If Phase 2 is supposed to start from a Phase 1 that is at 85% balanced agreement and 42.7% trade agreement and 18-22% trade rate, but the *underlying constraint that produced those numbers* is a structural mismatch between oracle and policy, then Phase 2 inherits the mismatch. The dynamics function gets trained to predict transitions in a latent space that the encoder learned under the wrong objective. The value head gets bootstrapped against root values from an MCTS search whose priors come from a policy that was trained to imitate a k=1 oracle. Every piece is loaded with the same constraint.

**Three: the architectural sequencing was supposed to unlock the phases in order, and the empirical reality is that the Phase 1 ceiling reached so far is *below* where Phase 2 can productively start.** The design document was written assuming Phase 1 would saturate near 90% optimality. If that had happened, the dynamics would have a clean latent space to learn transitions in, the policy would have a strong prior for MCTS to refine, the value head would have a calibration that bootstrapping wouldn't ruin. What actually happened is that Phase 1 saturated at the 42.7% trade-agreement number — a different kind of saturation, on a different metric, *below* the design point. Phase 2 launched from that point inherits not a strong intuition but a constrained one. The fix to this is not in Phase 2's hyperparameters. *It is upstream.*

`TASK_761`, which sits in the queue under the heading **Structural Pivot Roadmap**, opens with the first direction being a richer oracle — replacing the k=1 hindsight solver with a multi-k formulation that produces policy targets the model's k=64 emission head can actually match in shape. The expected consequence is a Phase 1 that *doesn't plateau at 0.43 trade agreement* — that can climb past 0.5, past 0.6, into the range where the design doc thought it would land. If that lift happens, the Phase 2 architectural transition has, *for the first time*, a productive base to launch from.

I want to say this carefully because it's the kind of sentence that sounds like a victory lap before any victory has happened: **the reason Phase 2 hasn't worked isn't that Phase 2 is broken**. The architecture is in the code. The training loop runs. The dynamics learns its consistency loss. The MCTS produces visit counts. The policy distills. The value head bootstraps. *Each individual piece does what the spec says it should do.* What doesn't happen is the deployment Alpha gain, and the reason it doesn't happen is — I now believe, after the May 2026 single-mechanism-exhausted lessons recorded in Article 8 and the post-eval-MCTS recalibration in Article 10 — *upstream of Phase 2 itself*. The Phase 1 imprint is what's bottle-necking it. Phase 2 cannot rescue a constrained Phase 1.

I should not present this as a clean conclusion either. It is the current best hypothesis. I have been wrong before. The "single-mechanism-exhausted" framing has its own kickback waiting to happen — perhaps the Phase 2 attempts had a different bug, perhaps the value head retargeting was happening too fast, perhaps the bc_weight decay schedule was off by an order of magnitude. The honest position is that **Phase 2, as specified in `TASK_716`, has been built and run; the result has been null; the most plausible cause is the Phase 1 ceiling; the test of that hypothesis is the TASK_761 oracle reformulation**. That test is in the queue. It has not yet been run.

---

## What's left to do

The Phase 2 architecture is in the codebase. `train_phase2.py` exists as a separate entry point. `phase2_config.py` extends `EasyConfig` with the relevant flags. `SegmentPool._make_slot()` precomputes the per-step rewards the dynamics needs. The Flax checkpoint surgery — load Phase 1, replace zero dynamics keys with He init, train — is two functions in `train_phase2.py` that I wrote in an afternoon. The MCTS integration uses `mctx_search.py` from the Poppy subtree without modification. The pipeline runs. The deployment Alpha gain has not yet been observed.

The next campaign is the TASK_761 oracle reformulation. The first deliverable on that ticket is a multi-k path solver — `k ∈ {1, 2, 4, 8}` running in parallel, with the policy targets aggregated across horizons rather than committed to k=1. That deliverable is being scoped now. The Phase 1 retraining under the new oracle will take roughly four weeks of single-3090 time at the current direct-sample throughput, by my estimate — the same wall clock as the original Phase 1 run, because the structural cost is encoding, not the oracle. At the end of that retraining, the question I am most interested in is whether the trade-agreement number that has been pinned at 0.43 for a thousand epochs actually moves. If it moves to 0.55 or higher, Phase 2 gets relaunched on top of it. If it doesn't move, the bottleneck is somewhere else and the diagnosis has to start over.

The Dream is on the table. The architecture for it is specified, implemented, and dormant. The reason it is dormant is not that the implementation is broken — it is that the seedling, the part of it that already knows the gardener's hand, has not yet been allowed to grow under sun strong enough to make planning matter. The trellis has been moved once and the plant did not find the sun on its own. The next time the trellis moves, the plant will have been raised differently. That is the campaign.

---

## Closing

I am writing this on the morning of 2026-05-23, on `workstation`, with the same Python environment the training loop is running in. The dashboard pane is still open. `agree_trade` is still at 0.427. The dynamics layers on the current checkpoint are still 100% sparse. The Phase 2 task is still TODO in the queue, the way it has been since the Phase 1 ceiling first showed up cleanly in the metrics — except now the reason it's still TODO has a name.

The Dream has not yet been dreamt. The architecture for it is here, and waiting.

> *Observation: the trellis is to be replaced, not moved. The plant will be raised again, on the assumption that the second cultivation will produce a different stem. The catalogue records the first attempt without prejudice. The second begins when the seed is changed.*

---

## Sources

- `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md` — primary spec, architecture, two-stage execution plan, success criteria, risks.
- `tasks/TASK_715_DIRECT_SAMPLE_PIPELINE.md` — the data pipeline Phase 2 depends on for survivable wall-clock at batch=256.
- `docs/easy_mode.md` — the three-phase ExIt framing (The Cheat → The Dream → The Thought).
- Article 6 — classifier/Alpha anti-correlation, the eval-side metric mismatch.
- Article 15 — Causal Expert Imprinting, the Phase 1 training math.
- Article 17 — five oracles, k=1 vs the policy's k=64.
- Article 30 — state channels, the leakage that inflates agreement.
- Article 39 — the autopsy of the trade-agreement plateau on the cleaner setup.
- `mcts_trader/jax_agent/muzero_model.py` — `recurrent_inference`, dynamics implementation.
- `mcts_trader/jax_agent/mctx_search.py` — MCTS configured for 32 sims, top-16 candidates, ε=0.1.
- `easy_trainer.py` — Phase 1 trainer, the reference for `phase2_trainer.py`'s structure.
