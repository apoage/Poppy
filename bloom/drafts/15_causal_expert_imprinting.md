# What Foxglove Actually Does

*On the supervised phase of a model-based trader, on imprinting an oracle into a network's roots, on the precise reason a perfect imprint can still produce a bad trader. The training-side companion to Article 6.*

> *Observation: the seedling raises a true leaf in the eighth week. It has not yet seen rain it did not also see in the propagation tray. I press the soil with two fingers in the shape the gardener taught it, and lift my hands. The seedling will hold the shape for some time. Whether it holds it after the wind reaches it is a different observation, on a different page.*

---

## The question I keep coming back to

`workstation` is on. The desk fan is on. `nvidia-smi` says the 3090 is at 54°C and 312 W, running a Foxglove smoke-test on a checkpoint I want to retire by Monday. The training dashboard in the next tmux pane shows the familiar shape: balanced agreement climbing from a starting 0.34 toward a 0.74 ceiling, value loss falling, oracle-hold ratio settling into its narrow band. It is the morning of 2026-05-22 and what I am watching, in the literal sense, is the supervised pre-training phase that the entire rest of the trading stack is bootstrapped from. **Foxglove**. The careful-dosing herb. Digitalis: the right amount stabilizes the heart; the wrong amount stops it.

I want, in this article, to answer the question I have been asked at least four times this season by friends who only half-follow the trading project: *why bother with a supervised phase at all? You're building an RL agent. Why not just let it learn from the market?*

The short answer is that pure RL on financial data is a sample-efficiency catastrophe. The long answer is that there is a technique with a name — **Causal Expert Imprinting** — that turns the unique property of financial data (you can compute the optimal trade *in hindsight*, exactly) into a warm-start objective that is well-defined, stable, fast to converge, and *only as good as the expert you imprint*. The whole season's worth of kickbacks distill, on the training side, into that last clause. The imprint can be perfect and the trader can still trade badly.

This article is the math of the imprint, the implementation in this stack, and the precise way it fails when it fails. Article 6 was the eval-side observation: better classifier, worse trader. This is the training-side companion: *here is what the classifier is being trained on, and here is why training it to perfection is not the same as building a profitable agent.*

The Apothecary's frame for it: I press the soil into the shape the gardener taught me. The seedling holds the shape. That part works. Whether the shape *is the right shape for the weather it is about to meet* is the part the gardener does not yet know.

---

## The expert, and what it is exactly

The expert in this project is not a human trader. It is a function. Specifically: `path_solver.py` in `mcts_trader/jax_agent/`, which takes a window of historical (or synthetic) prices and returns, for each timestep, the sequence of actions that would have produced the maximum return *given perfect hindsight* over a short forward window.

The mechanism is a constrained dynamic-programming search. At each timestep `t`, with k assets in scope, the solver expands a small lookahead horizon (currently four or eight bars) and finds, by enumeration plus pruning, the action assignment that maximizes terminal portfolio value subject to transaction-cost penalties. The output for a 64-asset trader at a single bar is a 5-way per-asset action distribution — `BUY_STRONG`, `BUY_WEAK`, `HOLD`, `SELL_WEAK`, `SELL_STRONG` — which when read together describes the optimal trade for that bar. Concatenated across the episode, this is the **expert trajectory** τ\*.

The expert is computed *offline*, after the price window is fully observed. From its own point of view it is omniscient. From the agent's point of view it is unreachable: the agent at decision time has only the past, not the future. The Causal Expert Imprinting paper in `docs/imprinting/` calls this the **Oracle Problem in Financial Control** and frames the asymmetry as the central trick of the whole technique:

> "Unlike robotics or board games, where the optimal policy is unknown, finance possesses a unique characteristic: the existence of a Hindsight Oracle. Given a historical price series, one can calculate the mathematically optimal sequence of trades that maximizes profit. This 'Perfect Expert' creates an alluring opportunity for Imitation Learning."
>
> — `docs/imprinting/Causal Expert Imprinting.txt` §1

The seductive thing about this expert is that it is *exactly correct on the data it was computed on*. Unlike a human expert, who is biased, slow, and sometimes wrong, the oracle is provably optimal under its own objective. Unlike a reward signal in RL, which is sparse and noisy, the oracle action label is dense — there is one per asset per bar of the entire training window. And unlike a policy-gradient signal, which requires the agent to actually *act* in the environment, the oracle target is available *before* any rollout happens. You can compute the expert once, store the labels, and treat the whole thing as supervised learning.

This is what Foxglove is. Not behavior cloning of a human, not behavior cloning of a previous agent. Behavior cloning of a **constructed expert**, where the construction is mechanical and exact and produces a label for every state the agent will ever see during pre-training. The closest analogy in the literature is **DAgger**'s aggregation against a synthetic oracle, or **AlphaZero**'s self-play targets, except here the oracle isn't a search procedure over a game tree — it is a closed-form lookahead solver on a known price path.

A small but important corollary: because the oracle is constructed, its quality is bounded by the construction. A lookahead of `k=4` bars produces a different expert than `k=8` or `k=12`. The notion of "optimal" is parameterized by horizon. We chose k=4 in early 2026 because that was what `path_solver.py` could solve in JAX-batched form fast enough to keep the GPU saturated. The choice was tactical, not principled. The TASK_761 structural pivot, six months later, is going to revisit it explicitly — *the expert is the bottleneck*, and the question is whether a richer expert lifts the ceiling. We'll get to that.

For now: the expert is `path_solver.compute_optimal_path()`. It produces τ\* = (a₀\*, a₁\*, ..., a_T\*). It is computed offline. The agent will learn to imitate it. The imitation is what Foxglove does.

---

## The math of Causal Expert Imprinting

The published technique (`docs/imprinting/Causal Expert Imprinting.txt`, a Gemini Deep Research synthesis I had commissioned in early 2026 to ground the design vocabulary) frames the problem as **Learning Using Privileged Information** — LUPI in the literature. The expert sees a "privileged" space X\* containing the future trajectory; the student sees only X, the current observation. The training task is to find a mapping f(x) that approximates the expert's decision rule f\*(x\*), forcing the student to identify *latent precursors* in the present that are causally linked to the outcomes in the future.

Stated cleanly:

```
Expert:  a* = f*(x*)         where x* contains future prices p_{t+1..t+k}
Student: a  = f_θ(x)          where x contains only present + past
Loss:    L(θ) = E_{x ~ D} [ d(f_θ(x), f*(x*)) ]
```

The distance `d` here is doing the work. The MuZero design used in this stack puts the policy through a per-asset categorical head — one 5-way softmax per asset — and the natural loss is the per-asset cross-entropy between the predicted action distribution and the oracle's one-hot (or soft) target. Concretely, for a batch of B observations across 64 assets:

```
L_policy = (1/B) Σ_b Σ_i CE( π_θ(s_b)[i],  a*_b[i] )
```

Where `π_θ(s_b)[i]` is the student's predicted 5-way distribution for asset *i* at timestep *b*, and `a*_b[i]` is the oracle's target for that asset at that timestep. This is the cross-entropy that lives in `easy_trainer.py` at line 170, and it is the dominant term of the Stage-1 Foxglove loss. In the codebase the same loss has a second term — a value-head cross-entropy against an oracle-derived discounted return target, scaled into a categorical support — and a third optional term for the oracle-mask auxiliary head (TASK_695), but the load-bearing object is the per-asset action cross-entropy. Everything else attaches to it.

### The "causal" part

The word *causal* in Causal Expert Imprinting carries two meanings and the second one is the one that almost killed the technique in this project.

The first meaning is the LUPI sense: the imprint should force the encoder to learn features *in the present* that are causally upstream of the expert's decision *in the future*. The paper frames this through a **Causal Information Bottleneck** — a constraint on the representation z = h_θ(x) that maximizes mutual information with the target action while minimizing it with the raw input, stripping non-causal noise.

```
L_VIB = I(Z, Y) − β · I(Z, X)
```

In practice we did not implement an explicit VIB term. The bottleneck in this stack is structural — the encoder pools over a 64-timestep window into a fixed-dim per-asset latent, which acts as an implicit information bottleneck — not adversarial. The TCN-and-attention encoder family (TASK_136, TASK_732) handles the information geometry by architecture. That is a *choice*, and a defensible one, but I want to flag that we are not capturing the full benefit the published technique advertises. The richer regularization is on the shelf for the structural pivot.

The second meaning of *causal* is the operational one, and it is the meaning TASK_125 had to be designed around: **the imprint loss must not use any information the agent would not have at decision time**. This sounds trivially obvious. It is trivially obvious. And the first version of the AMAX (Optimal Action Imprint) feature in TASK_119 violated it anyway, by reading the future prices from the same buffer the agent's observation came from and computing the optimal path *during* the same training step, using those future prices. The agent's loss was being computed against a target whose construction had touched data the agent's observation didn't have access to — fine for the *target*, since the target is allowed to know the future, but the path of construction itself was time-mixed with the agent's training step in a way that, on the wrong implementation, would let gradients flow through the future prices into the agent's encoder via shared parameters. TASK_121 caught a related future-leakage bug; TASK_125 rewrote the imprint construction to be strictly **retrospective**:

> "The trick is **temporal separation**: During Episode N: NO imprint loss (agent explores freely). After Episode N ends: Compute optimal path (hindsight is NOW valid). During Episode N+1: BC loss samples from buffer (past episodes only)."
>
> — `tasks/TASK_125_CAUSAL_EXPERT_IMPRINTING.md`

In the Foxglove (non-RL) configuration this is even cleaner: the oracle is computed entirely offline, before training begins, on a fixed dataset of synthetic episodes or historical windows. The agent never sees the future prices at all. The oracle sees them once, produces the labels, and is then discarded for that window. The agent's observation tensor and the oracle's target tensor never share any computation, only the buffer entry that pairs them. **No future leakage, by construction.**

This sounds like a footnote and it is not. The reason it isn't is that the *training-side metric improvements* an imprint produces — Bal.Acc climbing toward 0.74 — are the result of the imprint working as designed, exactly transmitting the structure of an oracle that is (locally) optimal. If the imprint were leaking, you'd see suspiciously fast convergence and very high agreement that wouldn't generalize. We don't see that signature; we see honest convergence to a 0.74 ceiling that does generalize across train/val/heldout splits. The technique is doing what it says on the tin.

The kickback is that doing what it says on the tin turns out to be insufficient.

### The DQfD margin loss

The published Causal Expert Imprinting technique recommends a **Large Margin Classification Loss** from DQfD (Hester et al., 2018) rather than vanilla cross-entropy. Its form is:

```
L_BC(s, a*) = max_a [ Q(s, a) + l(a*, a) ] − Q(s, a*)
```

Where `l(a*, a) = 0` if `a = a*`, else a positive margin constant `δ`. The intent: force the Q-value of the expert's action to *strictly dominate* every other action by at least `δ`, creating a basin of attraction around optimal behavior rather than a mere preference.

In the Foxglove implementation we use cross-entropy, not large-margin. The reason is partly engineering convenience (cross-entropy is the JAX default and integrates cleanly with the per-asset softmax structure of TASK_145) and partly a deliberate choice — we *don't* want the imprint to be a strict dominance constraint, because Stage 1 is a warm-start, not a final policy. We want the model to leave some probability mass on the non-oracle actions so that Stage 2 RL has room to explore. The margin loss would prematurely close that off. The decision is documented in TASK_125's discussion sections, and on the May 2026 evidence it was the right call: the failure mode we encountered was the *opposite* of "policy can't escape the imprint" — the policy escaped it just fine and overshot in the wrong direction.

The MuZero paper's recommendation for the analogous case is a soft target — λ-weighted mixture of MCTS visit counts and expert one-hot — which `mctx.muzero_loss` supports via its `policy_targets` argument. In our case the MCTS isn't run during Stage 1 (Foxglove is pure supervised), so the target is just the oracle's soft per-asset distribution, sometimes with a small label-smoothing.

### The stability argument

The reason imprinted-then-finetuned outperforms tabula-rasa-RL in low-data regimes is not subtle. It is the cold-start problem.

A tabula rasa RL agent in a financial environment has to discover, from scratch, that any action other than hold ever produces useful gradient. The reward landscape is sparse (most timesteps produce near-zero PnL); the action space is large (320 actions in our case: 64 assets × 5 actions); the credit-assignment horizon is long; and the exploration noise required to find profitable actions costs simulated capital that scales linearly with the noise budget. PPO and DQN can in principle solve this; in practice on financial data they need millions of episodes and produce policies that are extremely sensitive to the exploration schedule.

The imprint short-circuits this. After 2,000 epochs of Foxglove (~3 hours on the 3090), the policy has Bal.Acc ≈ 0.74 — meaning at ~74% of timesteps the policy's argmax action matches the oracle's. The Q-values across actions have the right rank order. The value head is calibrated against oracle returns. The encoder has learned features that *are*, by the imprint's construction, predictive of the directional sign of the next return. This is not a deployable policy yet — it has never made a trade for real, never paid a fee, never encountered a market state outside the curriculum — but it is a *warm initialization* for Stage 2 RL that starts at, conservatively, six orders of magnitude better than random. The performance bound from the published synthesis:

> "The proposed architecture combines the stability of BC with the optimization potential of RL and the robustness of Causal methods. By using the historical expert not as a template for action but as a source of Privileged Causal Information, we solve the 'Cold Start' problem inherent in financial RL."
>
> — `docs/imprinting/Causal Expert Imprinting.txt` §10

The *performance bound* part is also explicit. The Causal Expert Imprinting note cites the standard offline-RL gap:

```
J(π*) − J(π̂) ≤  C / (1 − γ)²  · ε_shift
```

Where `ε_shift` is the distribution shift between the expert's data distribution and the deployment distribution. Imprinting tightens this bound by ensuring the agent stays inside the expert's support during Stage 1. The bound is asymptotic and not particularly tight in practice, but the direction of the inequality is the load-bearing part: *the imprint determines the worst case*. If the expert is good, the bound is tight; if the expert is bad, no amount of downstream RL recovers what the imprint missed.

This is the math under which the kickback eventually lands. Note the bound for later.

---

## How Foxglove actually does it

Implementation. The training entrypoint is `train_easy.py` at the repo root, 3525 lines. It is wired to the `e-dojo_v1.py` orchestrator (`tools/orchestration/CLAUDE.md`), which runs CPU-only and launches per-branch training subprocesses on the 3090. The dual-branch pattern is the Foxglove canonical: Branch A trains on historical windows (or synthetic, depending on configuration), Branch B trains on Bella's generated markets to prevent catastrophic forgetting on real distributions. Branches merge periodically using `merge_utils.fed_avg` or `fisher_merge`, and the merged checkpoint is benchmarked after each merge.

The trainer itself is `easy_trainer.py`, 573 lines. The core loop is the standard JAX-Optax pattern: `value_and_grad` on a `loss_fn` closure that takes a batch and returns `(total_loss, metrics)`. The batch contains:

- `observations` — `(B, 27, 64, 64)` temporal price tensors (channels × assets × timesteps; 27 channels in the canonical config, 39 in the cross-sectional-augmented configs, depending on the variant)
- `actions` — `(B, 320)` per-asset oracle action distributions, reshapable to `(B, 64, 5)`
- `returns` — `(B,)` oracle-derived discounted episode returns
- `oracle_mask` — `(B, 64)` binary mask of which assets the oracle traded (TASK_695 auxiliary head)
- `per_asset_returns` — `(B, 64)` per-asset realized returns under the oracle (used for TASK_713 return-weighting)

The loss reads:

```python
# easy_trainer.py:165-205 (paraphrased)
policy_reshaped  = policy_logits.reshape(-1, 64, 5)
targets_reshaped = batch['actions'].reshape(-1, 64, 5)
per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)

# Adaptive trade/hold class balance (TASK_149)
hold_prob       = targets_reshaped[:, :, 1]
trade_strength  = 1.0 - hold_prob
loss_weights    = 1.0 + trade_strength * (adaptive_trade_weight - 1.0)

# Return-weighted trades (TASK_713)
if 'per_asset_returns' in batch:
    abs_ret      = jnp.abs(batch['per_asset_returns'])
    return_boost = 1.0 + trade_strength * jnp.minimum(abs_ret / abs_ret.mean(), 10.0)
    loss_weights = loss_weights * return_boost

policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

A few things worth pointing out. First, the per-asset softmax is **multi-discrete** (TASK_145) — each of 64 assets has its own 5-way decision, independent. This was an early-2026 design pivot from a single 320-way softmax that had a pathological coupling: the joint softmax forced the model to spend probability mass on at-most-one-asset-at-a-time configurations, which is wrong for portfolio trading where simultaneous actions across assets are the whole point. Decoupling the per-asset heads removed the coupling and gave the policy structural permission to fire multi-asset trades.

Second, the loss has **adaptive class balancing** (TASK_149) because the oracle, at k=4 lookahead, produces ~63 holds and ~1 trade per bar per asset on average. Without reweighting, the model would collapse into a trivial all-hold policy and hit ~95% raw agreement by never trading. The TASK_670B notes call this the **Turtle Local Optimum** — defended against by upweighting trade actions until their gradient mass matches holds. The cap at MAX_TRADE_WEIGHT = 64.0 (raised from 10.0 in TASK_675) is what makes balanced accuracy a more honest metric than raw agreement; balanced accuracy is what we report.

Third, the **return-weighted trade loss** (TASK_713) is a finer-grained version of the same idea: not only does a trade gradient mass equal hold gradient mass, but a *high-return* trade gets more weight than a *low-return* trade. The intent was to teach the model *which* assets matter, not just that *a* trade happened. On the May 2026 evidence (Article 6) this was one of the encoder-side mechanisms that improved Bal.Acc but, in conjunction with the policy stack downstream, contributed to over-trading at deployment. The mechanism is still in the loss because it improves training-side metrics; the deployment-side cost it sometimes pays is documented and parameterized.

Fourth, the **data side**. `train_easy.py` carries an `OnTheFlyGenerator` that, per the CLAUDE.md memory-hierarchy contract, generates training episodes *in VRAM* — no pre-generation to disk. The generator has four branches: Bella (latent-diffusion synthetic markets, `--bella`), GPU-synth (parametric synthetic prices), CPU-synth (legacy), and historical (TASK_698, sampling random windows from a RAM-cached SQLite slice). Each branch produces `(T, n_assets)` close prices, runs the oracle on them, encodes the resulting state, and writes an episode into the buffer. The encoder runs as part of episode construction; per `easy_trainer.set_encoder()` it can be optionally cached for ~30% speedup, with the trade-off that the cached latents drift as the encoder updates (TASK_670B latent-drift fix: disable cache for stability, accept the speed cost).

Fifth, the **SegmentPool** (TASK_710 / TASK_712, mentioned in MEMORY.md). The synthetic curriculum has 17 distinct market regimes; the SegmentPool cycles all 17 simultaneously rather than running them in serial L0→L14 progression. This means each training batch sees a mixture of regimes and the encoder never gets to overfit to a single regime's idiosyncrasies. The pool runs indefinitely; the agent gets a uniform sample across the regime space.

That is what Foxglove is. Oracle labels precomputed per bar per asset, multi-discrete cross-entropy with class-balanced and return-weighted weights, value head against scaled categorical returns, optional auxiliary heads (oracle-mask, rank, dense-sign), dual-branch synthetic + historical with periodic merging, benchmark after each merge. The whole pipeline is supervised, runs at ~2400 epochs/hour on the 3090, and produces a checkpoint that scores Bal.Acc ≈ 0.74 reliably.

---

## The Apothecary, briefly

> *The shape is pressed into the soil. The seedling rises into the shape. The shape was correct in the propagation tray, where the light was uniform and the wind did not reach. I record the rising. I do not record yet whether the shape will hold against a sky it has not met.*

---

## The kickback, in two layers

### Layer one: the imprint is only as good as the expert

The first kickback is the one the season learned the hard way and named in Article 14: *the technique inherits the bugs of the oracle*. There was a stretch in early-to-mid 2026 — the path-solver sign-bug episode that Article 14 covers in detail — where the oracle was, on certain market configurations, returning the *negative* of the optimal action. The bug was subtle, the sign was flipped only on a specific branch of the lookahead enumeration, and the consequence was that the imprint phase was diligently training the agent to do the opposite of what it should be doing on those configurations.

The training metrics during that period did not flag the bug. Bal.Acc climbed normally. Value loss fell normally. The agent was learning to match the oracle's labels with high agreement. *The oracle's labels were wrong*, but the imprint mechanism doesn't care about the absolute correctness of the labels — it cares about agreement with them. A perfect imprint of a wrong oracle is a perfectly wrong agent.

The Causal Expert Imprinting paper warns about this in the abstract:

> "Causal confusion occurs when an imitator agent learns to associate an expert's action with incidental features of the state that are correlated with the action but do not cause it."
>
> — `docs/imprinting/Causal Expert Imprinting.txt` §1.1

The paper frames the failure mode as spurious-correlation rather than oracle-correctness, but the structure is the same: the agent imitates faithfully, the agent is wrong, and the failure is invisible from inside the training loop because the training loop's notion of *correct* is *agrees with oracle*. The eval-side benchmark is what eventually catches it. Article 14's story is the moment we learned to never, ever ship a Foxglove checkpoint without a benchmark gate after every merge.

The deeper lesson, which I have to keep relearning: the imprint is a structural transform, not an oracle of truth. It transmits whatever the oracle gives it. If you upgrade the oracle, the imprint upgrades for free. If you corrupt the oracle, the imprint corrupts for free. The technique is symmetric in its error treatment, which is exactly the property you want for a stable training method and exactly the property that makes a bad oracle catastrophic.

### Layer two: even a clean imprint is not enough

The second kickback is the larger one and it is the one Article 6 was about: even when the oracle is correct, even when the imprint converges cleanly to its 0.74 ceiling, even when every training-side metric says *this is working*, **the resulting policy does not necessarily deploy as a profitable trader.**

By 2026-05-15 the evidence was unambiguous. Three independent encoder/classifier improvements — TASK_737's auxiliary-head ablation, TASK_741's Mint per-asset injection, TASK_738's legacy-channel revert — each produced a Foxglove checkpoint with *better training-side metrics than the canonical baseline*, and each produced a *worse deployment Alpha than the baseline*. Multi-seed, paired, all-signs-negative. The mechanism, fully fleshed out in Article 6: a sharper directional classifier upstream produces a sharper value surface downstream, MCTS visit counts concentrate more aggressively, the argmax decision crosses the trade threshold on more bars, trade rate explodes, fee bleed eats any per-bet edge, Alpha collapses.

What this means *on the training side* — the side this article is about — is that the imprint phase is the **easy phase**. Its metrics are well-defined, its loss converges monotonically, its checkpoints look beautiful on the dashboard, and *the relationship between those checkpoints and downstream deployment quality is, on this stack, not the relationship the naive theory predicts*. From the canonical memory entry:

> "Better classifier (loss DOWN, Bal.Acc UP) → worse trader (Δα DOWN, all seeds negative, over-trading collapse)."
>
> — `lore/notes_2026-05-15.md` §110

The naive theory says: a better imprint → a better warm-start → a better Stage-2 policy → better deployment Alpha. The empirical finding says: a better imprint produces a policy whose *reaction surface to its own encoder* is *more aggressively decisive*, and on this fee structure and this MCTS budget *more aggressively decisive policies underperform less aggressive ones*.

The implication for the imprint phase specifically: **the imprint is a warm-start, not a fix**. It is necessary for cold-start avoidance, it is sufficient for training-side metric production, and it is *not* sufficient — and may not even be load-bearingly upstream of — deployment Alpha. The classifier-Alpha anti-correlation lives downstream of the imprint. The imprint can be perfect and the trader can still trade badly. The trader will, in fact, *more reliably* trade badly *when the imprint is sharper*, because the sharpened upstream is what the downstream stack overreacts to.

The performance-bound math earlier in this article, the one that says J(π\*) − J(π̂) ≤ C/(1−γ)² · ε_shift, was not wrong. It was simply not the binding constraint. The binding constraint on this stack, on this fee model, on this MCTS depth, is the *policy's response function to encoder sharpness*, which lives in a completely different part of the loss landscape than the imprint loss touches. The imprint phase optimizes the encoder. The deployment failure is happening in the gap between the encoder and the trade.

Article 6 covers the eval-side discovery of this. This article is the training-side acknowledgment: *the technique works, it just doesn't carry the load people assume it carries.*

---

## What's left

It is 2026-05-22 and Foxglove still runs imprinting. The pipeline I described above is the canonical Stage-1 in active use on `workstation` right now — the 3090 is grinding through a smoke validation of an old checkpoint as I write this, oracle labels precomputed, multi-discrete softmax, dual-branch synthetic + historical, the whole shape. Nothing about the May 2026 kickback has caused us to abandon the imprint phase. The cold-start problem is real, RL on financial data from scratch is still a sample-efficiency catastrophe, and Causal Expert Imprinting is still the cleanest cold-start fix on offer. The technique is load-bearing for the warm-start. It will keep running.

What changed is the *frame*. We stopped grading Stage-1 checkpoints by their training-side metrics alone. Every checkpoint that emerges from Foxglove now goes through the eval-MCTS recalibrated benchmark (TASK_759, Article 10's spine) before it is allowed to feed into anything downstream. Bal.Acc and value loss are diagnostic; paired Δα over three seeds against the canonical baseline under the deploy-truth operator is the only success metric we trust. From the canonical promotion rule:

> "Trust **paired Δα over 3 seeds** as the success metric going forward. Don't promote on Bal.Acc or loss alone."
>
> — `project_task741_result.md`

The other thing that changed, and this is the direction the structural pivot is pushing on as I write, is **the expert itself**. TASK_761's first lane (`tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`) is **Phase-1 retrain with richer oracle (k=8 or k=12)**. The thesis: the imprinting math is fine, the imprinting implementation is fine, the bottleneck is the *quality of the demonstrations the imprint is consuming*. A k=4 oracle solves a four-bar lookahead, which is short enough to be dominated by microstructure noise in some regimes. A k=8 or k=12 oracle has more horizon to work with, prefers different trades, produces a different label distribution. If the published bound J(π\*) − J(π̂) ≤ C · ε_shift is the binding one, *upgrading the expert is the move*. If it isn't binding — if the downstream policy-response-to-encoder-sharpness is what kills Alpha regardless of expert quality — then richer-oracle won't help and we'll know that, too. Either result is informative. Either result is a Tier-3 direction that doesn't repeat the encoder-side mistakes of January through April.

There is a third move on the shelf that the published technique recommends and we never implemented: the **Variational Information Bottleneck** (β · I(Z, X) penalty on the encoder representation) and the **DQfD margin loss** (force expert action's Q-value to dominate every other action by δ). Both are explicit regularizers that would, in principle, restrict the encoder's capacity to overfit to non-causal features and prevent the policy from skewing too sharply on small classifier improvements. Neither is in the active queue right now — they are noted in the TASK_761 backlog as small-perturbation candidates for after the richer-oracle result lands.

The seedling held the shape. The shape was correct in the propagation tray. The wind, when it reached the bed, did things the propagation tray could not have prepared for. The gardener changes neither the seedling nor the shape, but the propagation tray itself. *A richer oracle is a different propagation tray.* That is the structural pivot, in the apothecary's frame.

---

## Closing

The imprint is real. It works. It does what it advertises: it solves the cold-start problem, it converges stably, it produces checkpoints whose Bal.Acc is reproducible across seeds and useful as a warm-start for Stage-2 RL. The published technique is rigorous, the implementation in this stack is faithful to it, and the training-side artifact is exactly what the math predicts.

The artifact is also, on its own, not a trader. It is a network that has learned the shape of an expert. Whether that shape carries forward into deployment Alpha is a separate question — a question the imprint loss does not see, cannot answer, and must not be asked to answer. The eval-side benchmark is what answers it. The training-side dashboard reports what it can report.

I keep Foxglove in the canonical pipeline because the alternative — tabula rasa RL on financial data — is worse for reasons unrelated to whether Foxglove is sufficient. The imprint phase is necessary. It is not sufficient. The whole rest of the project, from Article 6's classifier-Alpha anti-correlation to Article 10's structural pivot, is the slow accumulation of evidence about exactly *what* the imprint is necessary *for*, and exactly *what* it is not.

> *The gardener does not water the leaf. The gardener waters the root. The leaf is what the gardener watches. The root is what the gardener tends.*

---

## Sources

- `docs/imprinting/Causal Expert Imprinting.txt` — Gemini Deep Research synthesis commissioned early 2026; theoretical spine of the technique. Quoted §1, §1.1, §10.
- `docs/imprinting/compass_artifact_wf-7a81fbfc-...md` — companion artifact on BC+RL mixing, demonstration selection, HER-for-returns.
- `tasks/TASK_125_CAUSAL_EXPERT_IMPRINTING.md` — the retrospective-expert-buffer design that fixed TASK_119's future-leakage AMAX. The temporal-separation quote is from here.
- `tasks/TASK_670B_EXPERT_IMPRINTING_STABILITY.md` — latent-drift, Turtle local optimum, support-resolution stability fixes. The Turtle mode framing in this article comes from this task.
- `tasks/TASK_698_FOXGLOVE_HISTORICAL_TRAINING.md` — dual-branch e-dojo + historical-market integration, data leakage analysis table, benchmark-after-merge gate.
- `train_easy.py` (~3525 LOC) — Foxglove training entrypoint, OnTheFlyGenerator, historical branch.
- `easy_trainer.py` (~573 LOC) — Stage-1 loss function. The per-asset multi-discrete cross-entropy quoted in the implementation section is around lines 165–205.
- `mcts_trader/jax_agent/path_solver.py` — the oracle. `compute_optimal_path()` with `n_actions_per_asset=5` is the load-bearing call.
- `tools/orchestration/CLAUDE.md` and `tools/orchestration/e-dojo_v1.py` — orchestrator that drives Foxglove's dual-branch + merge + benchmark loop.
- `lore/SYNTHETIC_CURRICULUM_DESIGN.md` — 17-regime synthetic curriculum that SegmentPool cycles over.
- `lore/notes_2026-05-15.md` §110 — canonical "better classifier, worse trader" memory entry, quoted in the kickback.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four-lane structural pivot; this article's "what's left" section cites the richer-oracle lane.
- Article 6 (`06_classifier_alpha_anticorrelation.md`) — the eval-side companion to this piece.
- Article 14 (the path-solver-sign-bug saga, working title) — the demonstration that the imprint inherits the oracle's bugs in full.
