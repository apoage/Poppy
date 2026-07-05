# GRPO and Why I Didn't Switch to It

*A design-register entry on Group Relative Policy Optimization, the loss function DeepSeek published in January 2025 that briefly looked like it ought to replace mine, and the three reasons — economic, methodological, and lineal — I left it on the shelf.*

> *Observation: an apothecary in the next valley distills a tincture from the same root I use, but with a different solvent. The dose-response curve they describe is cleaner than mine. I prepare a small batch using their method to compare. The colour is correct. The smell is correct. The yield, on my soil, is not their yield. I record this without judgement and continue with the preparation I have.*

---

## A January paper, a February task file

`workstation` is on. The single 3090 is idling at thirty-eight degrees while a checkpoint validation grinds in the next pane; the desk fan is quiet. In a tab I haven't closed since the morning, `tasks/TASK_139_GRPO_DEEPSEEK_R1.md` is open at the equations section. The task was filed on the fourth of February, two and a half months after the DeepSeek R1 paper landed and roughly two weeks after I had finished telling everyone within earshot — Gemini sessions, ClickUp comments, my own diary — that this was the algorithm I was about to switch to.

I did not switch to it. Today is May 22, 2026, and TASK_139 is still in the queue with the status field reading "Research Complete / Implementation Pending." The status field has been honest about that for three and a half months. The plan was always to evaluate first, decide second; the evaluation said "not yet, and possibly not at all, and here is why." This article is the long-form version of that decision, the one I want on the record before I forget which arguments were load-bearing and which were just rationalisation.

The short version goes like this. **GRPO — Group Relative Policy Optimization — is the right algorithm for the problem DeepSeek was solving. The problem DeepSeek was solving is not, in three specific ways, the problem I am solving.** The first way is economic: GRPO's main saving is the value-function critic, and on this stack the critic is small to begin with, so the saving is rounding error. The second is methodological: the campaign that was running when TASK_139 was filed had already entered the regime where switching the loss architecture would add a confound to every subsequent measurement, and the cost of that confound exceeded the expected gain. The third is lineal: GRPO was designed for LLM-as-policy, where the policy is enormous and the rollout cost dominates; the lineage transfers partially, not cleanly, to RL-with-deep-net-policy on a 200-step trading episode. None of those three is a fatal objection. Stacked, they are the reason the task file's status hasn't moved.

I want to walk all three carefully. The walk is also an excuse to write down what GRPO actually is, in formulas, because I think the algorithm is genuinely interesting and the version of it that floated through my Twitter feed in January 2025 was about half right and half wishful.

## What's in this stack, before we change anything

It is worth pinning down what I would have been replacing. The Trader project trains a MuZero-style architecture, which is the lineage from Schrittwieser et al. 2020 with a TRM head bolted on top, and the loss is not vanilla PPO. There is no advantage computation, no clipping ratio, no on-policy importance sampling in the textbook actor-critic sense. There are three heads, three targets, and three losses summed at the end of the step.

The **policy head** is trained against the visit distribution produced by an MCTS rollout. For each state in the replay batch, we run a tree search rooted at that state, count visits to each action at the root, normalise to a distribution `π_MCTS(a|s)`, and minimise cross-entropy or KL divergence between the network's prior and that target. The MCTS visits are the supervision; the network is being told "this is what a search budget concluded about this state," and is asked to predict the conclusion before doing the search. The math is

```
L_policy = - Σ_a  π_MCTS(a|s)  ·  log π_θ(a|s)
```

and that is all. No advantage. No baseline. No ratio.

The **value head** is trained against bootstrapped n-step returns. From state s we record the actually-realised reward sequence for n steps (n = 5 in most of my configs) and bootstrap the tail with the network's own value prediction at step s+n, then take a Huber loss against the target. The math is

```
G_n = Σ_{k=0..n-1} γ^k r_{t+k}  +  γ^n V_θ(s_{t+n})
L_value = HuberLoss( V_θ(s_t),  G_n )
```

and again that is all. No critic-vs-actor architecture, just a value head sitting on top of the same encoder the policy head reads from.

The **dynamics function** — this is the part that makes it MuZero rather than AlphaZero — is trained to predict the next latent state and the immediate reward, given the current latent and an action. The supervision is the actually-observed reward and the encoder's encoding of the next observation. The dynamics loss is a small term, but it is the difference between *knowing what state you are in* and *imagining what state you would be in if you tried each action*. The MCTS at training time and at inference time both rely on the dynamics function to roll forward without consulting the real environment.

Everything is end-to-end. The encoder, dynamics function, value head, and policy head share a representation and are optimised against the sum of the three losses, plus an L2 regulariser, plus whatever auxiliary terms the current experiment has added (rank loss, oracle distill, the Bella shape-match term back when it was alive). There is no separate critic network. There is no separate actor network. There is one encoder and three heads and a learned transition model, trained jointly.

**GRPO does not drop in here as a like-for-like swap.** It is a different loss-architecture entirely. The honest framing of TASK_139 is not "swap our PPO for GRPO," because we are not running PPO. It is "replace the value-and-policy training pair with a sampled-group-advantage objective, and figure out what happens to the dynamics function and the MCTS visit supervision." That is a much bigger surgery than the casual reading of the R1 paper suggests, and the question I had to answer in February was whether that surgery was justified.

## What GRPO actually is

DeepSeek's R1 paper presents GRPO in Section 2.1, equations one through three. The paper is *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, DeepSeek-AI 2025, arXiv 2501.12948. I will reproduce the equations because they matter, and because the version of GRPO that circulates in tweet form is usually missing one of the three.

**Equation 1 — the objective.** For a prompt `q` and a group of `G` sampled trajectories `{o_1, …, o_G}` drawn from the *old* policy `π_θ_old`, with current policy `π_θ` and reference policy `π_ref`:

```
J_GRPO(θ) = E_{q, {o_i}}  [ 1/G Σ_i  ( min( ratio_i · A_i ,  clip(ratio_i, 1−ε, 1+ε) · A_i )
                                       −  β · D_KL( π_θ ∥ π_ref ) ) ]
```

where `ratio_i = π_θ(o_i | q) / π_θ_old(o_i | q)`.

This is recognisably PPO's clipped surrogate, with one critical substitution: the advantage `A_i` is not computed from a value function. It is computed from the group statistics, which is the second equation.

**Equation 3 — the group advantage.** For each trajectory in the group, with realised return `r_i`:

```
A_i = ( r_i  −  mean(r_1, …, r_G) )  /  std(r_1, …, r_G)
```

That is it. The advantage is a z-score within the group. If `r_i` is one standard deviation above the group mean, `A_i = +1`. If it's one below, `A_i = -1`. The policy gradient pushes probability mass towards above-average trajectories and away from below-average ones, all within the *same group of samples drawn from the same prompt*. There is no learned baseline. The group is the baseline.

**Equation 2 — the KL penalty.** The reference policy `π_ref` is the frozen policy from before this round of training (in R1, it gets re-set every 400 steps). The KL term is the standard "don't drift too far from where you came in," but DeepSeek uses an unbiased estimator that I find aesthetically pleasing:

```
D_KL( π_θ ∥ π_ref ) = π_ref(o|q)/π_θ(o|q)  −  log( π_ref(o|q)/π_θ(o|q) )  −  1
```

That estimator is always non-negative and unbiased, and it doesn't require summing over the full vocabulary; you just evaluate it on the sampled trajectories. Small detail. Nice detail.

That is the whole algorithm. The mechanical advantages over PPO are real:

- **No critic network to maintain.** PPO trains a value function alongside the policy; GRPO does not. The value function is replaced by `mean(r)` over the group, computed from forward rollouts only.
- **Self-normalising advantage.** The z-score is bounded in practice and doesn't drift with the reward scale, which means the learning rate is less sensitive to reward-magnitude changes than PPO's GAE-based advantages.
- **One forward pass per sample, no value bootstrapping.** Compared to PPO's training step, the per-sample compute is leaner — you don't have a value head doing a separate backward.

DeepSeek also reports one empirical finding worth flagging because it is counter to PPO folklore. In Section 3.2.1 they note that the clip ratio `ε` plays a "crucial role" and that *a lower value can lead to the truncation of gradients for a significant number of tokens, thereby degrading the model's performance*. They use `ε = 10` (yes, ten), where standard PPO uses 0.2. The R1 result is partly a result of letting the policy move much faster per step than PPO conventionally allows. I am not sure how much of that result is GRPO-specific and how much is "PPO with ε=10 would also have worked, nobody tried." The literature is starting to ask this question; in February 2025 it had not yet.

Compared to **AlphaZero-style** training — which is what my MCTS visit target most resembles — GRPO is doing something fundamentally different. AlphaZero distils search into the policy: the visit distribution from MCTS is the supervision, and the supervision encodes the search's reasoning about *all* actions, not just sampled ones. GRPO is closer to **vanilla policy gradient with a smarter baseline**: sample a group of complete trajectories, compute their returns, push probability mass towards the better ones. There is no tree, no expansion of unsampled actions, no propagation of value through a model. The supervision is whatever the group of forward rollouts happens to express.

That distinction is the seed of the whole evaluation. AlphaZero uses information that is *implicit in the search*; GRPO uses information that is *implicit in a sample of rollouts*. The information content is not the same. On a domain where rollouts are cheap and the policy is the expensive part (LLM tokens, where a 671B model's forward pass dwarfs everything), the GRPO trade is wonderful. On a domain where the rollouts are expensive and the policy is small (a 200-step trading episode where each step involves a learned dynamics model, an encoder pass, and a search), the trade is, at minimum, less obviously good.

## The evaluation, in three questions

TASK_139's analysis was structured around three questions, which I'll walk in order.

**Question 1: does GRPO fit the trader's loss structure?**

Partially. The policy head's update could be reframed as a GRPO-style group-relative gradient, and the math even works out — sample G trajectories from the current policy, compute their cumulative trading rewards, z-score within the group, take the clipped surrogate. That's a clean substitution for the *one* of the three loss terms.

But the value head still needs a target, and the dynamics function still needs a target. GRPO is mute about both. In a pure GRPO-LLM setup these don't exist — there is no learned value head (it's been removed by design), and there is no dynamics function (the LLM is its own model of the world). On the Trader, removing the value head means giving up bootstrapping, which means giving up the n-step return that lets me train on partial trajectories instead of full 200-step rollouts. And removing the dynamics function means giving up the model-based planner, which is the whole MuZero spine of this project. I do not want to remove either.

So the realistic shape is **GRPO replaces the MCTS-visit policy loss; the value head and dynamics function keep their current targets.** That is a partial replacement, not a full swap. The complexity reduction the R1 paper sells you — fewer hyperparameters, simpler training loop — does not apply, because the loss is no longer simple. I'd be running three losses still, with the GRPO term *added* to a stack that already has cross-entropy against MCTS visits to deal with. The reasonable form would be to *replace* the MCTS-visit policy loss with GRPO, but now I have to ask whether the GRPO gradient is informationally richer than the MCTS-visit gradient, and the answer there is not obvious. The MCTS gradient distils a tree's worth of look-ahead reasoning; the GRPO gradient distils a group's worth of forward rollouts. These are not the same information.

**Question 2: does the group-advantage statistic work on trading rollouts?**

The group-relative z-score wants a group of `G` trajectories that are *meaningfully different from each other*. If the policy is near-deterministic — which it tends to be after Foxglove imprinting, when the oracle has already collapsed the policy onto a near-greedy strategy — then `std(r_1, …, r_G)` is small and the advantages are noisy. The signal-to-noise of the GRPO update depends on the policy's *exploration variance*, which on this stack is something I have to engineer rather than something I get for free.

The compute side is the other half. For LLM reasoning, `G = 4` to `G = 16` is the reported sweet spot. R1 itself uses `G = 16`. On a 200-step trading episode, `G = 16` means sixteen rollouts per gradient step. Each rollout is a forward pass through the encoder plus the dynamics function plus the policy and value heads, repeated 200 times, with the current Bella-generated price tape (or historical replay) as the environment. That is a sixteen-fold multiplier on the rollout compute per gradient step compared to my current training loop, which uses replay-buffer transitions and does not roll out at training time at all. The training step right now is "sample a batch from replay, do one search per state, take a gradient step." A GRPO step would be "for each state in the batch, perform G full rollouts, score them, compute the z-score, take a gradient step." That is, charitably, an order of magnitude more compute per training-step.

On a single 3090, that order-of-magnitude is not absorbable. The current training loop is already filling the GPU. Sixteen-times-more rollouts means sixteen-times-fewer gradient steps per wall-clock hour. Even if GRPO's per-step learning is better (and the R1 paper claims it is, at scale, on LLM benchmarks), the campaign rate goes down, and the campaign rate is what produces the paired-Δα measurements that are the only honest signal I have about whether things are working.

**Question 3: is the win worth the integration cost?**

TASK_139 itself estimates Phase A (GRPO core) at one to two days, Phase B (trading reward function) on top, Phase C (integration with `train_easy.py`) on top of that, and Phase D (ablation studies — group size, clip ratio, comparison to MuZero value-based) as the bulk of the work. Realistically, two to three weeks of focused work to get a GRPO variant trained end-to-end on this stack and produce a paired-seed comparison against the current loss. That is roughly the same envelope as TASK_734 (flow matching) or TASK_740 (Bella distillation) — both of which got run, both of which produced clear signals (Article 6 has the verdicts).

The question is whether the *expected Alpha return* of GRPO is comparable to the expected Alpha return of those other tasks. My prior in February was: probably not. GRPO is a *known-good* algorithm in its home domain; in mine it would be a transplant, and transplants on this stack have a poor record. Article 6 ran the numbers on five mechanisms drawn from the same neighbourhood (encoder sharpening, classifier rebalancing, Bella distillation, oracle perturbation, dataset replacement), and the single-mechanism class was decisively closed at the end of that campaign. GRPO would have been mechanism six. The prior calibration on mechanism six, after five negative results on mechanisms one through five, is not "this one will be the breakthrough."

The decision in February was: defer. Not kill. The math is correct. The algorithm is solid. The integration is moderate. The expected return, on this stack, on the prior calibration available at the time, was not enough to displace the experiments already in the queue.

---

> *The Apothecary records: the tincture from the neighbouring valley was prepared correctly. The colour matched the description. The recipe was not at fault. The soil here yields a different oil, and the same solvent does not draw the same compounds from it. The recipe is filed, not discarded, and the journal is updated.*

---

## The kickback, in three layers

This is where the article earns its place in the design-register. The way I framed the decision in February was a deferral; the way I understand it in May, after five more months of campaign and the closure of the single-mechanism class, is sharper. Three layers, in increasing order of generality.

**Layer one: GRPO is correct in its setting, not necessarily in ours.** DeepSeek's economic argument for GRPO is that the value-function critic is the genuinely-expensive component to maintain when your policy is a 671B-parameter language model. The critic, in their setup, has to be roughly the same size as the policy to be useful — you can't pair a tiny critic with a giant policy and expect the critic to track the policy's reasoning. Removing the critic, then, removes a model the same size as the policy. That is a 2x parameter reduction at training time, which on a model of that scale is hundreds of GPUs of saving. The economic argument is enormous.

On this project, the value head is small. It is a few-hundred-parameter linear head on top of a shared encoder. Removing it does not remove anything close to half the model — it removes a fraction of a percent. The economic argument for GRPO, the one that motivated DeepSeek's design choice, does not transfer. *I am paying the same compute either way; I am not saving a critic-sized model by removing the critic, because the critic was never critic-sized.*

This is the most concrete of the three layers and the one I should have written down in February. Algorithmic choices in published RL work are usually *jointly optimal with their cost model*. The cost model for LLM-RL has the policy as the dominant cost; the cost model for small-network RL has the rollouts and the environment as the dominant cost. The same algorithm, in the second regime, optimises a different bottleneck and can produce a different verdict.

**Layer two: TASK_139 was the right call to defer, but not because GRPO is bad.** This is the methodological layer. By early February the campaign was deep into single-mechanism exploration. Article 6 and Article 8 between them tell the story: encoder-side mechanisms anti-correlated with Alpha; multi-seed validation was overturning single-seed claims; the paired-Δα protocol had been formalised as the only honest signal. To switch the policy loss to GRPO in the middle of that would have added a confound to every subsequent measurement.

TASK_139's status note hints at this: *"No changes needed to model architecture (works with existing TRM)."* That is structurally true but methodologically misleading. The architecture would have been the same; the *training dynamics* would not have. A GRPO-trained policy would have different exploration variance, different early-training behaviour, different sensitivity to the reward landscape, and — most importantly for paired comparison — would have produced a different baseline. Every subsequent mechanism test would have had to be re-baselined against the GRPO version of the trader. The campaign's measurement infrastructure was tuned to the existing loss; replacing the loss meant rebuilding the infrastructure.

The cost-benefit was: small theoretical gain (GRPO might be better, on a prior that has subsequently weakened), large measurement-disruption cost (every subsequent mechanism test needs a new baseline). In a campaign that was about to discover, over the next three months, that *none* of the mechanisms it tested were producing real lift (Article 10's eval-MCTS recalibration was the final word on this), the measurement-disruption cost was the load-bearing one. The campaign's value came from measurement, not from any individual mechanism. Disrupting the measurement to chase a marginal mechanism would have cost the campaign.

I want to be precise about what this means. **TASK_139 being the right deferral is not the same as TASK_139 being a wrong adoption.** If the campaign had been in a different phase — if I had been at the start of an architecture-shopping era rather than the middle of a mechanism-shopping one — the calculus might have gone the other way. The deferral was phase-specific. The math on the shelf will be evaluated again when the prior calibration changes.

**Layer three: LLM-trained-with-RL is a different problem than RL-with-deep-net-policy.** This is the lineal layer, and it pairs directly with Article 23's observation about hindsight-optimal. The literature transfers partially. Algorithms that work on one regime don't automatically work on the other, and the partiality is structural, not incidental.

Concretely. LLM-RL has:
- Massive policy, modest rollout cost per token, sparse-but-verifiable reward at the end of a long generation, no learned world model (the LLM *is* the model), and an action space that is the vocabulary.

Trading-RL on this stack has:
- Small policy, expensive rollout (200 timesteps with a learned dynamics function), dense-but-noisy reward at every timestep, an explicit learned dynamics model the policy plans through, and an action space that is a portfolio allocation.

GRPO is designed against the first cost model. It is optimal-ish for the first cost model. On the second cost model, two of its main savings (no critic, no model) are either inapplicable (the model is the whole point) or unimportant (the critic was small). The methodological transfer is partial. The algorithm is not wrong; the assumptions it was tuned against are wrong, for this stack.

This is the same observation I made about hindsight-optimal demonstrations in Article 23. The lineage is correct. The papers are correct in their published benchmarks. What I have been learning, mechanism by mechanism, is that *the gap between the lineage's success in its home domain and its productivity in mine is structural*, not incidental, and not something I can close by being more careful with the integration. Cross-pollination from LLM-RL to small-network RL is real but partial. Cross-pollination from hindsight-optimal robotics-RL to financial-RL is real but partial. The Trader is a specific environment with a specific cost model and a specific reward landscape, and the algorithms tuned against other cost models and other landscapes need translation, not transplantation. TASK_139 was an early case of recognising the gap before paying the integration cost.

## What's left

The task file is still in the queue. *Status: Research Complete / Implementation Pending.* The status field has not lied; it has just gotten old. If I were rewriting it today I would update it to "Research Complete / Implementation Deferred Pending Phase Change," because the deferral has a specific condition attached. The condition is: the prior calibration changes, or the campaign enters a phase where loss-architecture changes are no longer measurement-disruptive.

The first condition fires if one of TASK_761's directions lands. TASK_761 is the structural-pivot roadmap that came out of the eval-MCTS recalibration in May: four directions, none of which are single-mechanism tweaks, all of which involve changing something structural about the training regime (reward shaping, entropy regularisation, oracle-k, incompatible-dataset). If any of those produce a genuine paired-Δα gain, the campaign exits the single-mechanism era and enters one where I am exploring a new design space. In that new design space, GRPO is back on the table — not because it has changed, but because the calibration around it has.

The second condition fires if the campaign reaches a *baselining moment* — a point where I am willing to rebuild the measurement infrastructure around a new training regime. That moment will come if a structural change forces a re-baseline anyway, in which case bundling GRPO into the same re-baseline is approximately free.

Either way, the math is on the shelf. The DeepSeek paper is still in `docs/research/arXiv-2501.12948v2/`. The task file's equations section is still correct. The clip ratio finding (ε = 10) is still the most empirically-interesting thing in the paper and the part I would test first if I revisited. The group-size schedule is still an open question; sixteen for trading rollouts is not obviously the right number.

What is no longer on the shelf is the *naive* adoption case — "DeepSeek did it, it must be better than PPO, switching is free." That case did not survive February. The two minutes of euphoria when the R1 paper landed gave way, by the time I had typed out the equations into a task file, to a sober "this is a different cost model than mine." By the time I finished walking the three questions, the deferral was the obvious call. By the time the single-mechanism class closed in May, the deferral had retroactively become the *right* call, in a way that I could not have articulated in February but which the structure of the deferral had implicitly bet on.

This is what the design-register is for. Article 26 — sorry, the article on a design that didn't ship — argued that the value of recording a non-adoption is in the *reasons*, not the *outcome*. TASK_139 is a clean example. The reasons were three: economic mismatch, methodological cost, lineal partiality. None of them required experimental falsification. All three were available at the design table, if you were willing to slow down and ask the questions in the right order.

I want to be careful not to over-claim. There is a version of this story where I was just lucky — where GRPO would have produced a clean Alpha bump and the deferral cost me three months of progress. I cannot prove that version is false. What I can say is that the three reasons above are reasons-from-structure rather than reasons-from-hope, and the structural argument has only strengthened with time. The single-mechanism closure in May is the strongest version of "you would not have gotten there from here" that the campaign has produced. GRPO would have been mechanism six in a regime where mechanisms one through five all failed and the post-hoc reason was that the bottleneck was upstream of the loss architecture entirely.

The task file's last edit date is February 4. The status is still "Implementation Pending." I will leave it that way until the calibration changes.

---

## Sources

- DeepSeek-AI 2025. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv 2501.12948v2. Local copy: `docs/research/arXiv-2501.12948v2/sn-article.tex`. Equations 1–3 in Section 2.1, lines 217–232.
- `~/Trader/tasks/TASK_139_GRPO_DEEPSEEK_R1.md` — primary task file, status "Research Complete / Implementation Pending," last edit 2026-02-04.
- Schrittwieser et al. 2020. *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. arXiv 1911.08265. The MuZero lineage this stack descends from.
- `~/Trader/mcts_trader/jax_agent/muzero_model.py`, `path_solver.py` — the three-head, MCTS-visit, n-step-bootstrapped value, dynamics-prediction loss architecture described in section 2 above.
- `~/Trader/lore/notes_2026-05-20.md` — eval-MCTS recalibration verdict; single-mechanism class closed.
- `~/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four-direction structural pivot, the condition under which GRPO returns to the table.
- Article 6 — *Classifier-Alpha Anti-correlation*. The three-way confirmation of the mechanism-class pattern.
- Article 8 — *Five Seeds and a Funeral*. Multi-seed validation overturning single-seed claims; why paired Δα is the only honest signal.
- Article 10 — *Eval-MCTS Asymmetric Lift*. The recalibration that closed the single-mechanism class.
- Article 23 — *Hindsight Optimal: The Research Lineage Behind the Oracle*. The companion piece on lineage transfer being partial, not clean.

*The neighbour's tincture was prepared correctly. The soil here yields a different oil.*
