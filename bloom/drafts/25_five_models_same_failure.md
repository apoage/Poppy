# Five Models Diagnose the Same Failure

> *The Apothecary takes the failing specimen, wraps its roots in linen, and walks it across the city to five different houses. Each apothecary inspects the leaves, palpates the stem, and writes a diagnosis on a separate sheet of paper. The Apothecary returns home with five papers and one specimen. She does not yet know which paper is right. She knows only that the specimen is still dying.*

— Lukas, 2026-05-22, `workstation`

---

## The moment I went to five doors

By the middle of May 2026, the campaign had landed on the noise floor. Article 10 documented it from the inside: every single-mechanism variant we had thrown at the Stage-B stack — Mint, shape-distill, dense-sign, flow-matching at sims=16, Titans-MAG, Bella encoder distillation — came back null-or-worse under the recalibrated eval-MCTS protocol. Paired Δα across three seeds, same data, same operator, mechanism flipped: the distribution of results was visibly *centered on zero*. The 738–758 wave was, in the most expensive sense of the word, an answer. The answer was: not here.

`workstation` is on as I type. The desk fan is on. `nvidia-smi` shows the single EVGA 3090 at 47°C, idle — the last `train_phase2.py` finished an hour ago and I have not started another one because I do not yet know what to start. The git symlink to `/dev/sdb1` is mounted; the project's history is intact. The state of the project, however, is not. We had a pathology and no fix. The roadmap file `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` was, until that week, empty.

The pathology had a name by then. The first time I'd written it down was December, when the dual collapse had first declared itself: a MuZero-style agent on synthetic markets whose policy converged to deterministic HOLD inside fifteen epochs, with a Neural-ODE prediction head whose MSE plunged from 0.4 to 0.01 by *learning the identity function*. Both numbers looked like learning. Neither was. Once I understood the coupling — collapsed policy starves the world model of action-conditioned trajectories, collapsed world model returns no value differential to the policy, both components lock into a stable useless equilibrium — I called it **Coupled Policy-Prediction Collapse**, and the name stuck. The full Gemini Deep Research report I'd commissioned back then is the canonical citation for that framing; Articles 6 and 10 both lean on it.

But by May the question wasn't *what is the pathology*. The question was *what is structurally wrong with our stack such that no single mechanism dents it*. Article 10 ended on a structural pivot: reward shaping, entropy regularization, oracle k reformulation, an incompatible dataset experiment. Four directions, drawn from somewhere. This article is about where they came from. Specifically: about the week I commissioned the same diagnostic question from five different model perspectives and read all five reports back-to-back, looking for the consensus underneath the dialects.

The five doors I knocked on were Gemini Deep Research, DeepSeek (DS3), Claude CLI, Gemini CLI, and Kimi. A sixth document — a compass-style synthesis — sat at the end as a meta-reading of the first five. I wanted to know what each model saw, where they agreed, and — this turned out to matter more than I expected — where they disagreed. The disagreements were the part I almost stopped reading early. They turned out to be the most useful part.

## What I asked them

I will not pretend I composed the prompt freshly each time. I composed it once, carefully, in mid-November, and reused it across all five sessions with only cosmetic changes. The prompt is preserved verbatim inside `docs/AI_bleed research/kimi.txt`, where Kimi quotes it back at me before answering. The core of it:

> *We are training a MuZero-style model-based RL agent for trading with an auxiliary prediction head (neural ODE for price forecasting). We observe a "dual collapse" phenomenon where two seemingly independent components collapse together. POLICY COLLAPSE: The policy entropy decreases over training, converging to a single dominant action (HOLD) despite entropy regularization. Starts ~4.5 (near-uniform over 320 actions), collapses to ~3.0 within 10–15 epochs. Adaptive entropy boosting temporarily reverses but doesn't prevent recurrence. PREDICTION COLLAPSE: Prediction MSE drops from 0.4 to 0.01 rapidly. The predictor learns to output "no change" (trivial solution). COUPLING: These collapses are linked — when policy converges to HOLD, the prediction head only sees flat trajectories, so "predict no change" becomes optimal.*

> *— prompt to five models, November 2025, verbatim from `docs/AI_bleed research/kimi.txt`*

Then I listed seven specific questions — how do production AlphaZero implementations prevent this, what causes auxiliary heads to trivialize, how should entropy regularization be weighted, are there techniques to decouple policy and world-model training, what is the relationship between MCTS simulation count and policy entropy, how do exploration bonuses interact with learned value functions, and what are known failure modes when training world models on policy-generated data — and I asked each model for academic papers, blog posts, GitHub issues, theoretical analysis, and production-system techniques.

The prompt was *deliberately neutral* about which side I thought was causal. I described both symptoms. I described the coupling. I did not say "the policy is the upstream collapse" or "the world model is the upstream collapse." I wanted to see which model picked which.

What I got back, in the five replies, was something I had not expected. The reports were not the same.

## The five diagnoses

### Gemini Deep Research — the canonical framing

The Gemini Deep Research report (`docs/AI_bleed research/big gemini.txt`) is the one I have already used in Articles 6 and 10 as the canonical reference. It is the most structurally complete of the five, and it is the one that first put the words *Coupled Policy-Prediction Collapse* on the page. Its opening sentence is the one that named the problem for the rest of the project:

> *"The most pernicious of these is the 'Coupled Policy-Prediction Collapse,' a phenomenon where the agent's decision-making policy and its internal world model degrade simultaneously, entering a self-reinforcing feedback loop that halts learning."*
>
> *— `docs/AI_bleed research/big gemini.txt`, §1*

Gemini Deep Research's core move is to decompose the dual collapse into *three* deep-learning pathologies running concurrently: **entropy collapse in actor-critic optimization**, the **dormant neuron phenomenon** in non-stationary learning, and the **trivial solution trap** in Neural ODE optimization. Each of these has its own literature, and Gemini Deep Research walks all three. The dormant-neuron piece was the one I had not internalized: even if entropy regularization eventually pulls the policy distribution back, the neurons responsible for action-relevant features may already be dead. Plasticity loss is irreversible without intervention. ReDo (Recycling Dormant Neurons) is the surgical fix. The same report's section on the Neural ODE's collapse to identity — *"the network learns that the safest path to low loss is to predict nothing happens"* — is what made me give up on training the ODE head against MSE as the only signal. Flow matching, as a velocity-target objective, came directly from this report's §5.2.

What Gemini Deep Research got right that the others missed: the *neuron-level* mechanism. It is the only report of the five that gives more than a paragraph to dormant neurons as a structural cause of the irreversibility. Several of the others mention plasticity in passing. Gemini Deep Research treats it as a third axis of the failure.

### DeepSeek (DS3) — the engineering-handle report

DeepSeek's report (`docs/AI_bleed research/DS3.txt`) reads like a sharper, more compact version of Gemini Deep Research, with one major shift in emphasis: it foregrounds **optimizer pathology** as a direct cause of policy collapse. The Adam-hyperparameter discussion, which appears as a footnote in Gemini Deep Research, is one of DeepSeek's top recommendations:

> *"Default Adam hyperparameters can exacerbate collapse; adjusting moment buffers and using L2 regularization can mitigate it. […] Set β₁ = β₂ (e.g., both 0.9) and use aggressive L2 regularization to prevent sudden weight changes that cause collapse."*
>
> *— `docs/AI_bleed research/DS3.txt`*

This is the kind of recommendation I would have ignored if I had only read Gemini Deep Research, because Gemini's framing puts it inside the *mitigation* section as one bullet among many. DeepSeek elevates it. It also pulls in the **Clip-Cov / KL-Cov** family of methods from Cui et al. 2025 — the idea that the entropy collapse is driven by a *positive covariance between action probability and advantage*, and that you can restrict updates of high-covariance tokens to slow the collapse without disabling the policy gradient. That's a structural fix at the loss level, not a hyperparameter knob.

DeepSeek's diagnostic angle is *the optimizer is part of the pathology*. Adam's momentum behavior under non-stationary advantages is itself a contributing cause. This is not wrong; it is, however, a thing none of the other four models foregrounded.

What DeepSeek got right that the others missed: the **RobustZero** angle — the asymmetric projection-predictor architecture with stop-gradient on one branch. DeepSeek is the report that cites RobustZero most centrally and treats it as the canonical fix for auxiliary-head collapse. Stop-gradient as a structural intervention to break the symmetry that allows trivial solutions: this is the move that several follow-on experiments in TASK_741 / TASK_757 ended up touching, whether or not they got it right.

### Claude CLI — the causal-diagram report

The Claude CLI report (`docs/AI_bleed research/cl_cli.txt`) does something the others don't: it draws the causal diagram by hand, in ASCII, top-to-bottom, with feedback amplification arrows looping back. The first page of the file is the diagram; the rest is the ranked list of contributing factors with line-level evidence.

Its core diagnostic move is to identify **MCTS as the collapse amplifier**, not as a neutral search operator:

> *"MCTS takes the current value function V(s) and produces a refined policy π_mcts by lookahead search. This acts as a 'sharpening' operator: if V slightly prefers HOLD (lower variance, no fee), MCTS amplifies this. […] Higher sims = faster convergence to HOLD."*
>
> *— `docs/AI_bleed research/cl_cli.txt`, §1*

This is a claim none of the other four reports made with this specificity. It says: MCTS is not a passive consumer of the value function. It is an *amplifier*. The cross-entropy policy loss then locks the amplified target into the network, because the network is being trained to match `π_mcts` and `π_mcts` is already collapsed. The deeper the search budget, the faster the collapse. Claude CLI's data-distribution-shift analysis is also the most concrete: it walks through exactly *what the prediction head sees* when the replay buffer is HOLD-heavy, and exactly *why* the trivial solution becomes optimal under MSE — `target = price_change ≈ market_noise; optimal prediction: output 0`.

What Claude CLI got right that the others missed: the **oracle k** angle, by implication. If MCTS-as-amplifier is the primary driver, then the right intervention is not adding entropy regularization on the *network's* policy distribution (which is what Kimi and DS3 emphasize), but adding entropy regularization on the *MCTS policy itself* — temperature, Gumbel scale, prior-strength adjustments. The "k" in our oracle reformulation work — reducing the supervised oracle's commitment to a single answer — comes downstream of this insight.

Claude CLI is also the most architecturally specific about the **information bottleneck in the shared encoder**. When the policy only needs the latent state to encode "current position" (binary, low-rank), the encoder stops encoding subtle market features. The prediction head, downstream of the same encoder, then has nothing rich enough to predict against. The collapse is not in three places; it is in one place — the encoder — and the three symptoms are downstream of the one collapse. This is a structural claim. If it's right, the fix has to be applied at the encoder level. If it's wrong, the fix has to be applied at the head level. The reports disagree about which.

### Gemini CLI — the practical-remedy report

The Gemini CLI report (`docs/AI_bleed research/gm_cli.txt`) is the most engineering-flavored of the five, and the most experimentally prescriptive. It opens with a one-sentence diagnosis and then immediately proposes an intervention table:

> *"The system is caught in a Martingale Trap where the 'Identity Function' (Predicting no change) is a strong local optimum for the prediction head, which in turn causes the Value head to see no profit opportunity, driving the policy to a cost-minimizing 'HOLD' equilibrium."*
>
> *— `docs/AI_bleed research/gm_cli.txt`, §1*

Where Gemini Deep Research gives the literature, Gemini CLI gives the *experiment plan*. Its four-experiment diagnostic — the Signal Injection Test (run training on synthetic data with clear trends to verify the architecture isn't the bottleneck), the Forced Dynamics Test (initialize physics parameters to be active and freeze them, see whether the value head propagates), the Reconstruction Loss intervention, and the Contrastive Prediction intervention — is the most actionable single page in any of the five reports. Read it sideways and it is a four-step debug protocol with predicted outcomes per step.

Gemini CLI also names the **Martingale Trap** explicitly. Several of the other reports describe this implicitly — the identity function is a strong local minimum because of the martingale-like property of efficient markets — but Gemini CLI is the one that puts the financial-economics name on the mathematical pathology. That naming was useful because it told me the trap is not specific to our architecture. It is specific to the *domain*.

What Gemini CLI got right that the others missed: the **incompatible dataset** insight. Its Signal Injection Test was the proximal source for what eventually became one of TASK_761's four structural directions — *try training on a dataset where the dual-collapse equilibrium is provably not optimal*. If the agent learns to trade on synthetic-level-3 (clear trends) but fails on real data, the pathology is data-conditioned, not architectural. That distinction matters for every downstream decision.

### Kimi — the production-system report

The Kimi report (`docs/AI_bleed research/kimi.txt`) is the longest of the five, and it has the most production-system context. It is the report that most clearly emphasizes that **original MuZero and AlphaZero do not actually include entropy regularization**, and that production systems like AlphaStar and OpenAI Five rely instead on *population-based training* and *forced diversity injection*:

> *"If entropy < 3.0, inject random actions for 100 steps. Separate behavior policy: collect data with more exploratory policy than learning policy."*
>
> *— `docs/AI_bleed research/kimi.txt`, §8 (OpenAI Five exploration strategy)*

Kimi's framing is the theoretical/production one. It is the report that walks the SAC-style automatic entropy tuning math (`H_target = 0.5 × log(num_actions)` ≈ 2.9 nats for our 320-action space), it is the report that explains why Dirichlet α=0.03 was tuned for Go's 361 actions and why we should be using α=0.1–0.3 instead, and it is the report that ties our specific failure mode to the **representation-collapse literature in self-supervised learning** (SimSiam, BYOL, MuDreamer's batch normalization fix).

Its position on entropy regularization is the strongest of the five: fixed coefficients fail, adaptive boosts provide only temporary relief, and the only thing that works reliably is target-entropy scheduling with the SAC-style automatic α. Kimi is also the report that argues most strongly for **alternating optimization** — freeze the policy, train the auxiliary head; then freeze the auxiliary head, train the policy — as a structural decoupling mechanism. This is the same family of ideas as DS3's stop-gradient, but framed as a training schedule rather than an architectural choice.

What Kimi got right that the others missed: the **production-systems angle**. The reminder that AlphaStar maintained a *league* of agents with explicit exploiter agents specifically trained to find weaknesses in the main agents. The reminder that PBT with 8–16 agents optimizing entropy coefficient jointly with learning rate is the only reliable way to get the entropy schedule right *in domains where the optimal schedule changes during training*. Single-agent training is structurally fragile in non-stationary domains. This was a thing I had been avoiding hearing, because population-based training on a single 3090 is not a thing I can do.

### The compass synthesis — the meta-document

The compass-style synthesis document (`docs/AI_bleed research/compass_artifact_wf-2955a86b-...md`) sits at the end as a long-form integration. It does not introduce a new perspective so much as it *consolidates*. Its single most useful sentence is the one I keep coming back to:

> *"Since the collapse is a stable equilibrium, breaking it at only one point allows the other component to pull it back. Effective solutions target multiple points simultaneously."*
>
> *— `docs/AI_bleed research/compass_artifact_wf-2955a86b-...md`*

That sentence is the *meta-finding*. It is what none of the five individual reports said in quite that compressed form. The compass synthesis is also the document that explicitly names **multi-point intervention** as the strategy — Point 1 (policy entropy), Point 2 (world model training), Point 3 (coupling between components) — and walks each. It is the document that proposes detection triggers (`if policy_entropy < 0.5 * target_entropy: entropy_coef *= 2.0; if prediction_variance < threshold: shrink_and_perturb(world_model)`). It is the document that names the trading-specific exacerbating factors most cleanly: HOLD's lower-variance reward distribution, sparse and delayed rewards, the inadequacy of short planning horizons for multi-step position-building strategies.

What the compass got right that the individual reports did not, *in isolation*: the recognition that no single fix would work. Multi-point intervention is what the synthesis sees that the individual diagnostics, however correct, cannot.

## The agreements and the disagreements

Reading all five back-to-back, the consensus is striking. Five reports, five model perspectives, five rhetorical styles. They all converge on the same core diagnosis:

- **The collapse is a stable equilibrium**, not an unfortunate intermediate state. All five.
- **The coupling between policy and prediction is causal, bidirectional, and self-reinforcing**. All five.
- **HOLD is structurally privileged** in any reward landscape with transaction costs and noisy initial value estimates. All five.
- **Fixed entropy regularization is insufficient**. All five. The phrase "the entropy bonus is added to the loss but gets overwhelmed by the advantage gradient" appears, in some form, in every single report.
- **Distribution shift in the replay buffer is the proximal cause of prediction collapse**. All five.

That is the consensus. It is solid. It is the part I trust the most because the five model perspectives are reasonably independent — they were trained on overlapping but distinct corpora, they have different RLHF heritage, and they were asked the same question on different days. Convergence across five independent priors on the same diagnosis is much stronger evidence than any one report alone.

The disagreements are also informative:

| Question | Gemini DR | DS3 | Claude CLI | Gemini CLI | Kimi |
|----------|-----------|-----|------------|------------|------|
| Primary cause | Three-way pathology (entropy/dormant/identity) | Optimizer + architecture | MCTS amplification | Martingale local min | Distribution shift + plasticity |
| Where to intervene first | Policy entropy + flow matching | Adam + stop-gradient | MCTS temperature | Reconstruction loss | PBT + SAC-α |
| Recommended fix for ODE collapse | Flow matching | Asymmetric architecture | Reduce shared-encoder load | Reconstruction or contrastive | Temporal consistency SSL |
| Stance on intrinsic rewards | Mentioned | Recommended | Skeptical | Not central | Strongly recommended (RND) |
| Stance on population-based training | Not central | Mentioned | Not central | Not central | Strongly recommended |

The disagreements cluster on the *where to intervene first* axis. DS3 says optimizer. Claude CLI says MCTS temperature. Kimi says PBT. Gemini Deep Research says flow matching plus ReDo. Gemini CLI says reconstruction loss. These are not contradictions — they are different bets about which intervention point is the *upstream-most* one such that fixing it propagates downstream. The literature itself does not have a settled answer here. The disagreement *between the models* is reflecting the unsettled-ness of the literature.

That unsettled-ness is itself data. If all five models agreed on the intervention point, I would have one cheap experiment to run. They don't, so I have four expensive experiments to run, and the choice of which to run first is mine.

---

> *Five apothecaries write five papers. Three of the diagnoses share a root word; two more share a different one. The Apothecary lays the five papers side by side on the workbench. The specimen is still on the linen. She does not yet know which paper is right. She picks up the pen.*

---

## The kickback

There are three layers of kickback here, and I want to take them in order.

**First — five reports, one stack. The synthesis is multiplicative.** If I had only read Gemini Deep Research, I would have the framing (Coupled Policy-Prediction Collapse) but not the MCTS-amplifier sharpening that Claude CLI surfaces. If I had only read Claude CLI, I would have the amplifier diagnosis but not the dormant-neuron / plasticity-loss axis that Gemini Deep Research foregrounds. If I had only read Kimi, I would have the production-systems context but not the engineering-handle Adam-tuning insight from DS3. If I had only read Gemini CLI, I would have the experiment plan but not the literature behind why the experiments were the right experiments. The compass synthesis would not have been writable from any single report. It is, structurally, a multi-source document — and the reason the synthesis is more useful than any individual report is that the perspectives are *partially independent*. They are not redundant. They are *partially redundant*, and the non-redundant slices are the parts I would have missed otherwise.

This is the strongest argument for multi-model research that I have personally experienced. I had read it argued before — "ensembling LLM responses outperforms any single model" — and I had filed it under "true but expensive." Reading these five reports back-to-back made it tactile. There were *specific concrete claims* in each report that did not appear in the other four, and several of those specific claims ended up in the TASK_761 roadmap.

**Second — multi-model is not free.** Five Deep Research reports cost real money in tokens. The total bill was not catastrophic — under a hundred dollars all-in across the five sessions — but it was not zero. The deeper cost was time. Each report is twenty to forty pages. Reading five reports carefully, cross-referencing claims, building the comparison table I just gave you, took two evenings. Then writing the synthesis took another. Five days of wall-clock from "I should ask several models" to "I have a roadmap." For a routine question — *which optimizer should I use for this fine-tune* — this would be absurd overkill. For a question at a strategic decision point — *what do I do now that the entire single-mechanism class has been exhausted* — the cost-yield ratio was good. The decision is when to do it. The cost-yield trade is real; do not invoke the five-model protocol for routine work.

**Third — model disagreement is data.** This is the layer I had not anticipated and the one I now think is the most important. When DS3 says *the optimizer is part of the pathology* and Kimi says *no, it's the lack of population diversity*, and Claude CLI says *no, the MCTS is the amplifier* — they are not just disagreeing. They are mapping the disagreements *in the literature*. The literature on policy collapse is unsettled. The literature on dormant neurons is unsettled. The literature on world-model training is unsettled. The model disagreements *encode* the unsettled regions. Where the five reports converge, the literature is converged. Where they disagree, the literature itself is contested, and any single-model answer would be smuggling in a confidence the field doesn't have.

That last layer is what changed my reading practice. I now treat single-model deep-research reports the way I treat single-seed experiments. They are *one draw from a distribution*. To learn the shape of the distribution — and especially to learn its variance — you need multiple draws.

## Where the four directions came from

Article 10 ended on the structural pivot: four directions sketched in `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`. I owe a citation for each.

**Reward shaping.** Strongest in Gemini Deep Research, which devotes a section to it under §5.4 (Adaptive Entropy Regularization plus advantage-aware reward shaping) and §5.6 (asymmetric reward shaping that penalizes excessive HOLD). The compass synthesis echoes this with its "asymmetric reward shaping that penalizes excessive HOLD sequences" recommendation, attributed to transaction-aware RL literature. Kimi mentions it but does not foreground it. The other two are quieter on this axis. The Gemini Deep Research framing — that the optimal policy under transaction costs and noise is *structurally* HOLD-biased and that you cannot fix this with entropy alone — is what made reward shaping a top-tier direction.

**Entropy regularization** (the more aggressive forms: SAC-α target tuning, Clip-Cov, KL-Cov, target-entropy scheduling). Strongest in Kimi, which gives the most explicit math (`H_target = 0.5 × log(320)` ≈ 2.9 nats) and the most thorough comparison of static vs. adaptive vs. target-entropy schedules. DS3 reinforces this with the Clip-Cov / KL-Cov framing from Cui et al. 2025. Without Kimi's framing this would have stayed a hyperparameter knob; with it, it became a structural direction.

**Oracle k reformulation.** Comes most directly from Claude CLI's MCTS-amplifier diagnosis. If the search operator is amplifying small value biases, the right intervention is at the *search prior* level — softening the oracle's commitment to a single trajectory, increasing search temperature, allowing top-k actions rather than top-1. The "k" in oracle k is a relaxation of the supervised target the agent is trained against. Claude CLI's section 6 (Entropy Regularization Structural Failure) is the proximal source: *"Entropy regularization on the MCTS policy itself (e.g., temperature, Gumbel scale)."*

**Incompatible dataset.** Comes most directly from Gemini CLI's Signal Injection Test. If the dual collapse is data-conditioned — if it only occurs when the signal-to-noise ratio is below some threshold — then the cleanest diagnostic experiment is to train on a dataset where the pathology *should not* occur. If we still collapse on synthetic-level-3, the pathology is architectural. If we don't collapse, the pathology is data-conditioned, and the fix lives in the reward or in the data, not in the network.

The roadmap is a consolidation. No single report contains all four directions with the right weight. The roadmap is the synthesis.

## What's left

The five-models pattern survives as a methodology. I do not run it weekly. I do not run it for every research question. I run it when the cost of being wrong is large enough that the cost of five reports is small in comparison — which is to say, at strategic decision points where the next experimental wave will consume weeks of GPU time. For everything else, one model is fine.

The deeper lesson is about *what model disagreement encodes*. When I started this project I treated agent responses as truth-claims to be filtered. A single confident answer was a single confident answer. Five confident answers that disagreed were five contradictions to be resolved. I now treat the disagreements as information about the *shape of the unsettledness in the literature*. When all five converge on a claim, I trust it more than I would trust any single source. When they diverge, I take the divergence as a signal that the underlying field is contested and that my next experiment has to actually adjudicate, not just commit.

The TASK_761 roadmap, sitting on the workstation now, is not a plan to fix the project. It is a plan to *learn which of the five diagnoses is the right one for our stack*. Reward shaping will tell us if the HOLD-attractor is the upstream collapse. Entropy regularization will tell us if the policy is the upstream collapse. Oracle k will tell us if MCTS amplification is the upstream collapse. Incompatible dataset will tell us if the data is the upstream collapse. Four experiments, each one designed to falsify a different one of the five diagnoses. By the time the roadmap is finished, the literature will be slightly less unsettled — at least for our stack — and the next set of decisions will be cheaper.

I am writing this in the same Python environment the next training run will launch from. The single 3090 in `workstation` is still idle. The roadmap is still on disk. The five reports are still in `docs/AI_bleed research/`. I will start the first experiment tomorrow.

> *Five papers on the workbench. The Apothecary reads them once more, in order, and then again, sideways. She circles the words that appear in three of the five. She underlines the words that appear in exactly one. The diagnosis she finally writes down is shorter than any of theirs, and is in her own hand.*

---

## Sources

- `docs/AI_bleed research/big gemini.txt` — Gemini Deep Research, *Coupled Policy-Prediction Collapse in Model-Based Reinforcement Learning: A Comprehensive Analysis of Dual Failure Modes in Stochastic Trading Environments* (canonical reference; primary source for Articles 6 and 10).
- `docs/AI_bleed research/DS3.txt` — DeepSeek (DS3), *Synthesis report on policy/prediction collapse, optimizer pathology, and stop-gradient architectures*.
- `docs/AI_bleed research/cl_cli.txt` — Claude CLI, *Causal-diagram analysis of MCTS as collapse amplifier; ranked contributing factors*.
- `docs/AI_bleed research/gm_cli.txt` — Gemini CLI, *Martingale-trap framing; four-experiment diagnostic plan*.
- `docs/AI_bleed research/kimi.txt` — Kimi, *Deep research report on production-system collapse-prevention; SAC-α target-entropy framing; population-based training*.
- `docs/AI_bleed research/compass_artifact_wf-2955a86b-...md` — Long-form synthesis document, multi-point intervention framing.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — Internal roadmap consolidating the five reports into four experimental directions (reward shaping, entropy regularization, oracle k, incompatible dataset).
- `bloom/drafts/06_classifier_alpha_anticorrelation.md` — Article 6 (cites Gemini Deep Research).
- `bloom/drafts/10_eval_mcts_asymmetric_lift.md` — Article 10 (cites Gemini Deep Research; ends on the structural pivot this article documents the sources for).
