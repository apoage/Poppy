# State Channels: Training on the Diagonal

> *Observation: I take a reading in still water and the needle settles at 83.6. I lift the instrument, carry it to the stream, and lower it into the current. The same instrument. The same calibration. The needle reads 33.8. The water is not the same water. The patient is not the same patient.*

By Lukas. The Apothecary's Lab Notebook, Article 30. Drafted on `workstation`, 2026-05-22.

---

## The instrument reads what it reads

Here is the symptom, plain. The model — Foxglove, the supervised-imprint stage of the pipeline — closes a long training run with **83.6% balanced accuracy** on the validation set. It is the best balanced accuracy I have ever seen out of this stage. The classifier-side of the trader has learned what the oracle did. I am pleased. I write a one-line note in the day's lore that says *"this is the new ceiling."*

Then I run the benchmark. Real candles, the held-out window, the rotation the trainer never saw. The same model. The same weights. The same input shape, the same encoder, the same head.

**33.8% balanced accuracy.** Below random. A coin flip would do better.

I sit with that number for a minute because the first reaction to a result this bad is to look for a benchmark bug — wrong rotation, off-by-one on the timestamps, normalization drift between dojo and bench. I check those. They are fine. The benchmark harness has been stable for months; Article 22 already worked its way through one calibration mismatch and this isn't that mismatch.

Then I run the same benchmark on the **previous** checkpoint — the one *without* state channels. The one trained on 27 input channels instead of 34. The one I had been calling the old baseline because the state-channel version had just dethroned it on training metrics.

The old model, without state channels, scores **58.6%** on the same benchmark.

So the truth of the situation is: *I added a feature that, on the validation set used to choose checkpoints, looked like the largest single improvement in the project's history. On real candles it made the model worse than the version without the feature.* The state-channel version is below random. The state-channel-less version is twenty-five points better than the version with the feature.

`workstation` is on. The desk fan is running. `nvidia-smi` is ticking through a benchmark validation in the next window — different model, this one is paused. I write a TASK file. The title of the file, in a moment of frustration I will not pretend didn't happen, reads *"FIX MEMORY THAT CLAUDE FUCKED UP."* I keep the filename. It's not a kind title and it isn't fair — the bug is mine, written by me, ratified by me, deployed by me, and noticed by me only after a benchmark forced the issue. But the file is named what it's named, and I am not going to retitle history to make myself look better.

This article is about what was in the file.

## What state channels are

State channels are auxiliary input features. They live alongside the market observation in the input tensor. Their job is to tell the model **what it owns** so it can decide **what to do next**.

The intuition is older than this project. A trader who looks only at price action — without knowing whether they are currently long, short, or flat — is not really a trader; they are a labeller. They can say "this looks like a buy moment" all day, but they cannot decide whether *this particular agent* should buy now, because the decision depends on what the agent currently holds. Buying when you are already long is different from buying when you are short. The agent needs context that is *about itself*, not about the market.

TASK_689 designed this. Six per-asset channels plus one global channel, tiled across the window dimension, concatenated with the 27-channel market observation. The full input becomes a `(34, window_size, n_assets)` tensor — 27 channels of market data, then:

| Ch# | Name | Range | What it tells the model |
|-----|------|-------|--------------------------|
| 27 | `position_state` | {−1, 0, +1} | Short / flat / long, current |
| 28 | `unrealized_pnl` | [−1, 1] | Signed return since entry |
| 29 | `hold_duration` | [0, 1] | Steps since last trade, normalized |
| 30 | `heat_fast` | [0, 1] | Trade recency, half-life ~3 steps |
| 31 | `heat_slow` | [0, 1] | Trade recency, half-life ~70 steps |
| 32 | `capacity_usage` | [0, 1] | n active positions / k slots |
| 33 | `confidence` | [0.2, 1.0] | Softmax max per asset |

`heat` is the only one that needs unpacking — it's an exponentially decaying counter of how recently each asset traded, two timescales so the model can learn what "recently" means for itself instead of being given a magic threshold. `capacity_usage` is global: the oracle uses at most `k=4` simultaneous positions, and the model needs to know when all four slots are full so it doesn't propose a fifth.

All channels are **tiled across the window axis** rather than written out as a history. The window is 64 timesteps wide, but the state channels carry only the *current* state, repeated 64 times. The choice was deliberate: a state-history channel would leak the trade pattern through the window and the model would shortcut to it. Tiling forces the model to use the current state as a context cue, not as a trajectory.

That was the design. The first version had a temporal leak — state at time `t` was computed from the oracle's action at `t`, so the model could read the answer off its own input — and that was caught in TASK_689 Phase A and fixed by shifting to `state[t-1]`. After the fix, validation balanced accuracy dropped from 98.5% to 50%, then climbed back to 90% over training. *"Honest but oracle-inflated"* I wrote in `lore/notes_2026-02-21.md`. Honest in the sense that the model was no longer reading the answer. Oracle-inflated in a sense I would not understand for another four months.

This is the part where, looking back, I can see the warning sign and I missed it. The TASK_689 design doc had the right words in it. *"Oracle-independent: State channels must work from the model's own actions at inference."* I wrote those words. I quoted them in three reviews. And then I did not implement them.

## The bug

Here is the bug, stated as plainly as I can state it.

**Training**: at each timestep, the state channels were computed from the **oracle's** trajectory. The oracle's position. The oracle's running PnL. The oracle's heat counter, decayed across the oracle's trade pattern. The oracle's capacity usage as the oracle filled its four slots.

**Inference**: at each timestep, the state channels were computed from the **model's** trajectory. The model's position. The model's PnL. The model's heat, the model's capacity.

To quote TASK_692 verbatim:

> *State channels are auxiliary context inputs that give the model information about its trading state (positions, PnL, heat, duration, capacity). During training they were computed from the **oracle's** trajectory. At inference they are computed from the **model's** trajectory. The distributions are fundamentally different.*

These two state distributions are not the same object. They share a shape and a dtype. They share nothing else.

The oracle is path-optimal. Its PnL trajectory is monotone-up by construction — the oracle never opens a position it doesn't close at a profit, because the oracle solves for the path-optimal sequence of trades over a known-in-advance return window. The oracle's heat counter tracks 3.9% trade rate, smoothly. The oracle's capacity is mostly empty most of the time, occasionally full when the regime supports four good co-incident bets.

The model is none of those things. The model's PnL meanders. It opens positions that go red before they go green. It opens positions that never go green. Its heat counter, at inference, sits at 27.7% trade rate — seven times the oracle, because the model over-trades. Its capacity usage spends more time near the cap. Its positions, viewed as a sequence, look like a noisy version of optimal *plus* a bias toward over-trading *plus* a long tail of mistakes the oracle would never make.

So at training time the model sees, as channel 27, a position trajectory that is always optimal. As channel 28, a PnL trajectory that climbs cleanly. As channel 30, a heat curve that is smooth and low. The model learns the joint distribution `P_train(market_obs, state_channels, target)` — and that joint distribution has, on the state-channels axis, lived inside a narrow tube around the oracle path.

At inference the state channels are computed from the model's own actions. Channel 27 is whatever position the model has stumbled into. Channel 28 is whatever the model's actual unrealized PnL has been. Channel 30 is a heat counter that spikes whenever the model over-trades, which it does.

The model is reading inputs it has **never seen during training**.

TASK_692 names this exactly:

> *The model only trained on the **diagonal** of the (state, target) space — where state matches oracle. At inference, every model mistake pushes it **off-diagonal** into states it has zero training signal for.*

That is the bug. The validation set had the same problem — it was generated the same way, with oracle-derived state — so the validation balanced accuracy was measuring how well the model predicted the oracle's next action *given the oracle's own state history*. Which is not the task. The task at inference is *predict your next action given your own state history*, and the model had no idea how to do that because it had never been asked.

Below random. Twenty-five points below the version without the feature. The feature was load-bearing in the wrong direction — the more the model leaned on the state channels during training (and a high-capacity model learns to lean on every informative channel it has access to), the worse it did at inference, because the state channels at inference were a different statistical object than the state channels at training.

This is the same class of problem that *Article 6* described in a different language: classifier-side improvements that don't transfer to deployment because deployment lives in a different distribution. There I had three independent mechanisms all making the classifier better and Alpha worse. Here I had one mechanism making the *labeller* dramatically better and the *trader* catastrophically worse. The shape is the same. The cause is the same: a metric measured on a distribution the model never has to face is not a measure of anything you care about.

The bug had been live since TASK_689 Phase A — months. We had added state channels, watched the benchmark numbers get worse, and concluded *"state channels don't help, maybe remove them."* That was the diagnosis on the table for weeks. It was the wrong diagnosis. The state channels weren't useless; they were computed from the wrong distribution. The direction of the effect didn't suggest distribution shift because the validation set was complicit — it lied in the same direction.

## The math

Let me write down what this is, in the language the literature uses, because if I had searched on the right words a year ago I would have found this answer waiting.

Let π* be the oracle policy and π̂ be the learned policy. Let `s_t` be the agent state at time `t` (position, PnL, heat, all the rest). Let `d_π(s)` be the state-visitation distribution induced by following policy π in the market — the probability of finding yourself in state `s` if you have been playing policy π up to time `t`.

The supervised imprint trains under:

E_{s ~ d_π*(s)} [ loss( π̂(s), π*(s) ) ]

i.e., **expected loss under the oracle's state distribution**. The minimizer of this objective is π̂ that copies π* well on states the oracle actually visits.

The deployment objective is:

E_{s ~ d_π̂(s)} [ loss( π̂(s), π*(s) ) ]

i.e., **expected loss under the model's own state distribution**. These are not the same expectation. They differ by the KL divergence between `d_π*` and `d_π̂` — and that divergence is non-zero whenever π̂ ≠ π*, which is to say always.

There is a standard result about how badly this hurts. If the per-step probability that π̂ disagrees with π* is ε, then under naive imitation the expected number of mistakes over an episode of length T is **O(ε T²)**, not O(ε T). The square is the compounding-error term: a mistake at step `t` pushes the state distribution off-diagonal, which makes the next mistake more likely, which pushes the state distribution further off-diagonal, and so on. The mistakes do not stay where they are dropped. They cascade.

This is the central theorem of *imitation learning under distribution shift*. The paper that names it cleanly is Stéphane Ross and Drew Bagnell's *"A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"* — AISTATS 2011, the paper that introduced **DAgger** (Dataset Aggregation). Fifteen years old at the time I am writing this. The literature has had this answer since I was in my mid-twenties.

DAgger's fix, in one line: train the model to imitate the oracle on states the *model itself* visits. Sample trajectories with the current learner, label those trajectories with the oracle, aggregate into the training set, retrain. Each iteration the model sees more of its own state distribution; the disagreement compounds less; the quadratic in T softens toward linear.

I did not call what I built DAgger. I did not search for "covariate shift" or "imitation learning" or "behavioral cloning compounds quadratically." I did the obvious thing — compute state channels at training time — and the obvious thing was the wrong thing for the same reason it has been the wrong thing for fifteen years of robotics and autonomous driving literature.

The right thing, when stated, sounds tautological: *to train under the inference distribution, train under the inference distribution*. The hard part is figuring out how to do that without spending the entire training budget on rollouts.

## The fix

`_build_model_state_trajectory()`. `train_easy.py:174-305`. A JIT'd `lax.scan` over T timesteps. The function takes the pre-encoded market observations, the per-step returns, and the model's current parameters, and rolls the model through the episode one timestep at a time, computing state channels from **the model's own actions** at each step.

The scan structure, in skeleton:

```python
def scan_step(carry, idx):
    pos, pnl, steps, h_fast, h_slow, prev_conf = carry

    # Build state channels from current carry
    state_ch = jnp.stack([
        pos, pnl,
        jnp.clip(steps / window_size, 0.0, 1.0),
        jnp.clip(h_fast, 0.0, 1.0),
        jnp.clip(h_slow, 0.0, 1.0),
        jnp.full(n_assets, capacity),
    ])  # (6, n_assets)

    # Tile state across window, splice with market obs
    state_tiled = jnp.tile(state_ch[:, None, :], (1, ws, 1))
    conf_tiled  = jnp.tile(prev_conf[None, None, :], (1, ws, 1))
    obs = jnp.concatenate([market_obs[idx], state_tiled, conf_tiled], axis=0)

    # Forward through the model — STOP GRADIENT
    output = jax.lax.stop_gradient(
        model_apply_fn({'params': model_params}, observation=obs[None, ...],
                       mode='initial_inference')
    )
    policy_logits = output['policy_logits'][0].reshape(n_assets, 5)

    # Action = argmax, confidence = softmax max
    action_types = jnp.argmax(policy_logits, axis=-1)
    model_confidence = jnp.max(jax.nn.softmax(policy_logits, axis=-1), axis=-1)

    # Incremental position step (TASK_693 — matches benchmark semantics)
    buy  = (action_types == 2) | (action_types == 4)
    sell = (action_types == 0) | (action_types == 3)
    new_pos = jnp.where(buy,
        jnp.minimum(pos + 1.0, 1.0),
        jnp.where(sell, jnp.maximum(pos - 1.0, -1.0), pos))

    # Update PnL, heat, hold duration from the model's own trajectory
    ...

    return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow,
            model_confidence), state_ch
```

Four things to notice.

First, the `stop_gradient`. The inner rollout must not backprop through itself. If it did, we would be doing policy-gradient through the rollout — a fundamentally different research direction with its own variance problems. The rollout is **only** there to compute the state distribution. Gradients flow only through the **outer** loss, where the model's prediction at each timestep is compared to the oracle's target *at that timestep, on the model's own state context*. That is the whole trick.

Second, the position-update rule. The first version of this code (TASK_692, before TASK_693) used **absolute** position mapping — `Buy → +1, Sell → −1` — and the *benchmark* used **incremental** stepping — `Buy: short→flat, flat→long, long→no-op`. So even after the rollout was correctly model-driven, the rollout was running a different *trader* than the benchmark was. A model that was short and chose Buy would, in the training rollout, flip directly to +1; the same model in the benchmark would step to flat. PnL diverges, heat diverges, every downstream channel diverges. The fix, TASK_693, replaces the absolute mapping with `jnp.minimum(pos + 1, 1)` / `jnp.maximum(pos − 1, −1)` — pure `where`, no conditionals, JIT-safe, semantically identical to the benchmark. Position trajectories now match step-by-step on the same inputs.

Third, the confidence channel. The original code hard-coded channel 33 to zero during training; the benchmark filled it with the model's softmax max. Another small mismatch hiding in the same shape. Fixed during TASK_693: confidence comes from the rollout's own softmax, carried in the scan carry as `prev_conf` so each step's input sees the previous step's confidence.

Fourth — and this is the part that matters for efficiency — the rollout is **JIT'd as a single XLA program**. `lax.scan` lets the entire 400-step trajectory compile into one fused kernel. The per-episode overhead, measured against the old oracle-state pipeline, came out around 36% — which is real but not killing. The cost of *not* doing this, the original TASK_689 design considered, was 400 sequential Python forward passes per episode at ~30–60 seconds per buffer refill. The `lax.scan` version is roughly two orders of magnitude faster than that naive approach and made the fix tractable on a single 3090.

There is one more piece, smaller but worth naming. The generator-trainer coupling is a **partial FIFO buffer refresh** — `set_model_params()` is called at epoch start, the oldest 20% of buffered episodes are marked exhausted, and they get regenerated next pass with the current model. This is the DAgger pattern: aggregate a moving window of (state, target) pairs sampled under recent learners. The model never trains on a completely stale state distribution and never has to throw out the entire buffer between epochs.

> *The roots have learned what is asked of them. The instrument is being recalibrated in the stream where it will be used. I record the new readings and continue.*

## The kickback

Three layers, going outward.

**Layer one — the bug existed for months.** State channels went into the pipeline in TASK_689 Phase A on February 21. The benchmark accuracy after they were added was worse than before they were added. That was visible. That was the *first* metric I looked at after a training run, every run. The shape of the regression — adding the feature makes the deployed model worse — was right there on the screen, every benchmark, for weeks. And the diagnosis on the table was *"state channels don't help, maybe remove them."* The diagnosis was wrong. The state channels were necessary. They were computed from the wrong distribution.

The reason the misdiagnosis stuck is mundane and worth being honest about: the **direction of the effect didn't suggest distribution shift**. Distribution shift, as a diagnostic, usually presents as *"train metric good, test metric not as good"* — a gap, but in the right direction. Here the gap was so extreme it crossed random (50%) and kept going. Below random feels like a *bug*, not like a distribution problem. So I went looking for benchmark bugs. There weren't any. The bug was in the *training pipeline*, in the line that built the state channels from `oracle_target_position` instead of from a rollout. That line looked obviously correct because in February it had been written deliberately, with reviews from three different models, all of whom had agreed the design was sound. They had agreed because the design *was* sound — the *implementation* skipped the part where the state had to be the model's own state at inference.

**Layer two — the fix is a 2010 paper.** Ross and Bagnell. AISTATS 2011. *"A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning."* DAgger. Dataset Aggregation. Fifteen years of robotics and autonomous-driving literature uses this. There is a textbook chapter about it. There are six pedagogical YouTube videos about it I could have watched in an afternoon. If I had typed `"covariate shift in imitation learning"` into a search engine at any point during the months the state channels were live, I would have landed on the right paper inside ten minutes, and the rest would have been implementation.

I did not type that. I typed `"behavior cloning trade execution"` and `"state augmentation neural trading"` and various phrasings that found me nothing because I was using my own vocabulary, not the vocabulary the field had agreed on. The lesson that took the longest to land in this whole project — and I will keep relearning it — is that **the bottleneck is often a missing word**. Once I had the word *DAgger*, the fix wrote itself. Until I had the word, the fix was invisible.

The reviewing models — Claude, GLM, Gemini — all named it correctly during the post-mortem reviews of February 22. Gemini's framing of *"off-diagonal coverage"* is the formulation I quoted earlier. GLM mentioned DAgger by name in the engineering review. Claude did the performance math on the rollout overhead. The texture of the back-and-forth was: *here is the correct framing, here is why your current implementation is exactly the problem this framing addresses, here is what the standard fix looks like.* Three independent models converged on the same answer in the same session because the answer is in the public corpus and has been in the public corpus for a decade and a half.

> *— Gemini, in the TASK_692 review thread:* *"The problem is off-diagonal coverage. The model is only ever scored against itself on the diagonal, where state matches oracle. The off-diagonal is the operating regime. The off-diagonal has zero training signal."*

I did not need to invent the phrase. I needed to look it up.

**Layer three — the fix did not save the campaign.** This is the third kickback and the most honest one. After `_build_model_state_trajectory()` was deployed, after position semantics were aligned with the benchmark, after the confidence channel was wired correctly — the state-channel version of the model recovered. Benchmark accuracy climbed off the floor. The 33.8% became something defensible, a number I could put next to the no-state-channel 58.6% without flinching.

But it did not become an alpha-positive model. The classifier got *better*. The trader, downstream, sat where it had been sitting for a different reason — the one Article 6 calls the *classifier-Alpha bottleneck*. Foxglove's job is to be a good labeller, but a better labeller is not, on this stack, a better trader. The state-channel fix bought us back the points we had been losing to the bug. It did not buy us the points we had been losing to the deeper structural issue.

It is the same lesson Article 4 carried about the 64× encoder bug. Fixing an obvious wrong does not, by itself, produce a right. It produces *the absence of an obvious wrong*, which is where you had thought you were the whole time. The work after the bug-fix looks identical to the work before it, just from a different starting altitude. The mountains in front of you are the same mountains.

## What's left

`_build_model_state_trajectory()` is canonical now. It is the function future state-channel work has to extend. `easy_config.py` carries `autoregressive_state: bool = True` as a config flag, not because it should ever be flipped off, but because keeping the oracle-state path alive as the `False` branch makes the next version of this bug catch itself — anyone who adds a channel to one path has to wire it into the other. The DAgger framing is in the codebase. The TASK_692 spec, including the off-diagonal language and the explicit reference to the imitation-learning literature, is in `tasks/`, where future me or future collaborator can read it before adding a channel.

The discipline that came out of this — and it's the part I expect to keep paying for — is one rule, sized for index cards: **training distribution = inference distribution, always check.** Not as a guideline. As a thing to verify before claiming any improvement on a feature that touches the model's input. The check is the cheap part: run one episode in training mode, run the same market data through the inference loop, compare the per-step state-channel tensors element-wise, assert they match. TASK_693's *Verification* section spells out exactly this protocol. If the trajectories diverge, the feature is broken; if they match, the feature might be working. It's a necessary, not sufficient, test, and we now run it.

The other half of what's left is the rule about *naming*. I did not have the word *covariate shift* in my mouth when I added state channels. I have it now. I have *off-diagonal coverage* and *DAgger* and *compounding error* and *Ross & Bagnell 2011* and the textbook reference for all of them. The reading I should have done in October I did in February, and the cost of doing it in February instead of October was four months of confused benchmark numbers and a task file with an angry title. The price of not knowing the term was paid in calendar time. Next feature I add to this pipeline, I am going to spend an hour looking up what the existing literature calls it before I write the first line of code. One hour against four months is a trade I do not need to think about.

The state channels work. The trader, on this stack, on this Foxglove stage, still has the classifier-Alpha gap Article 6 named. The two facts coexist. The bug-fix was real and necessary; it did not make the deeper problem go away. That's the diary entry. Tomorrow's work is downstream — planning side, MCTS, reward shaping, the things Article 6's roadmap pointed at when the encoder-side mechanism class was closed. State channels are no longer the suspect. They are infrastructure now. They get to fade into the background and stay there.

> *The instrument is calibrated where it is used. The reading is honest. The patient is, as ever, the patient — and the patient is still ill, but for reasons the instrument can no longer be blamed for.*

---

## Sources

- `~/Trader/tasks/TASK_692_FIX_MEMORY_THAT_CLAUDE_FUCKED_UP.md` — primary spec for the model-state rollout. The 33.8% / 83.6% / 58.6% numbers and the *"diagonal of the (state, target) space"* framing are quoted from this file.
- `~/Trader/tasks/TASK_689_AGENT_STATE_CHANNELS.md` — original state-channel design. The seven-channel layout, the heat decay constants, and the *"Oracle-independent"* design principle that the implementation initially failed to honor.
- `~/Trader/tasks/TASK_693_FIX_POSITION_SEMANTICS_IN_ROLLOUT.md` — the second-order fix: incremental position stepping, confidence channel, and the verification protocol for training-vs-inference trajectory match.
- `~/Trader/train_easy.py:174-305` — `_build_model_state_trajectory()`, the JIT'd `lax.scan` autoregressive rollout. Read in full while writing this article.
- Ross, S. & Bagnell, J. A. (2011). *"A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning."* AISTATS. The paper introducing DAgger. The fifteen-years-of-prior-art I should have read in October.
- Article 6, *Classifier-Alpha Anti-Correlation*, this notebook — the deeper bottleneck the state-channel fix did not resolve.
- Article 4, *The Encoder That Was Off By 64×*, this notebook — the previous instance of *necessary, insufficient* in the same campaign.
- Conversation logs, TASK_692 review session, 2026-02-22 (Claude adversarial / GLM engineering / Gemini architectural). Gemini's *"off-diagonal coverage"* line is from this session.
