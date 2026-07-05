# Hindsight Optimal: The Research Lineage Behind the Oracle

*An architecture essay on the decade of reinforcement-learning literature that the Trader project's supervised target descends from, what those papers actually proved, and what the same idea has so far failed to do on this particular stack.*

> *Observation: the gardener keeps a journal that catalogues, by season, every plant they could not have known would survive at the time it was sown. Each entry has two dates: the day the seed went into the soil, and the day, in the next year or the year after, when the question of survival became answerable. Between the two dates the journal is silent. The Apothecary reads this journal more often than the planting log.*

---

## What "hindsight-optimal" means, and why I have a directory full of it

`workstation` is on. The single 3090 is sitting at thirty-nine degrees with the desk fan ticking. In one terminal an eval-MCTS recalibration is grinding through a benchmark gate; in another I have an editor open to a directory called `docs/Hindsight-optimal/`. It holds eight arXiv papers and six reference codebases. I assembled it over the winter, slowly, the way you collect field notes — one paper here from a Gemini Deep Research session, one there from a Reddit thread, one because a paragraph in an EfficientZero readme pointed at it. By spring it had become a small library. By May 22, 2026 — today — it is the most heavily revisited directory in my docs tree.

The directory has a name and the name is the thesis. **Hindsight-optimal**: a training regime in which the teacher signal is computed *with information the agent at inference time does not have*. The student sees only the past. The teacher saw the future when it computed the label. The training task is to imitate what the teacher chose without acquiring a structural dependence on the privileged information the teacher used to choose it.

This is exactly what the Trader project's oracle does. `path_solver.py` looks at a window of prices, computes the trade sequence that — knowing the future, exactly — would have captured the most return, and emits per-asset action labels. The agent sees only the present and the past. The agent learns to imitate the teacher. The Foxglove supervised phase, Articles 15 and 17 of this notebook, is hindsight-optimal imprinting in its most literal form. The TASK_697 oracle-as-search-distillation idea (Article 17, Oracle 6) is hindsight-optimal *planning*, taking the teacher's tree as the target. The whole architectural spine of Phase 1 is hindsight-optimal in the LUPI sense — Learning Using Privileged Information — and the directory in question is the lineage that named the trick.

I am going to walk that lineage in this article. There are eight papers, in five thematic clusters, spanning 2017 to 2024. They include some of the most influential demonstration-augmented RL methods of the last decade. Each one, in some way, shaped a design decision in this stack. None of them, individually, has been sufficient to produce a deployment-Alpha gain on the Trader project's data. That is not a failure of the literature. It is a finding about the *transfer* of the literature into one particular regime — a regime where the demonstrations come from a closed-form solver rather than a human, where the action space is multi-asset portfolio trading, where the reward landscape is contaminated by transaction costs and a planning-side fee structure that the literature's benchmarks never had to handle.

I want to be careful about the tone. This is not a takedown of any of these papers. The papers are correct in their published benchmarks. The methods work where they were tested. What I'm trying to do is record, honestly, the lineage that produced my design and the gap between the lineage's success in its home domain and its productivity in mine. The lineage is also still alive — TASK_761's richer-oracle direction is, at the deepest level, a wager that the hindsight-optimal ceiling can be lifted by rebuilding the teacher. The literature's open questions, framed in 2017 by DQfD and DDPGfD, are still my open questions in 2026. The decade has not closed them.

## The lineage, in chronological order

### 2017 — DQfD: demonstrations as a priority-replay seed (Hester et al.)

The first paper in the directory is *Deep Q-Learning from Demonstrations* by Hester et al., AAAI 2018, arXiv 1704.03732. It is the founding move of the modern demonstration-augmented RL literature, and its claim is direct:

> "We present an algorithm, Deep Q-learning from Demonstrations (DQfD), that leverages small sets of demonstration data to massively accelerate the learning process even from relatively small amounts of demonstration data and is able to automatically assess the necessary ratio of demonstration data while learning thanks to a prioritized replay mechanism. DQfD works by combining temporal difference updates with supervised classification of the demonstrator's actions."
> — DQfD abstract, Hester et al. 2017

The mechanism is mechanically simple: keep demonstration transitions in the replay buffer with permanently elevated priority, mix them with on-policy transitions, train Q with a sum of TD loss, a one-step margin classification loss against the demonstrator's action, an n-step return loss, and an L2 regularizer. The margin loss — `max_a [Q(s,a) + l(a*,a)] − Q(s,a*)` — is the part that became canonical: it forces the Q-value of the expert's action to dominate every other action by a fixed margin δ. The paper shows DQfD reaches scores on the first million Atari steps that PDD DQN needs 83 million steps to match.

The hindsight content here is straightforward: in Atari, "demonstrations" are human play, which is privileged in the trivial sense that the demonstrator sees the screen rendered and reacts. The agent eventually sees the same screen, so the privilege is more about *strategy memory* than literal information access. But the architectural template — *imitation loss as a fixed term alongside TD updates, demonstration data living permanently in the replay buffer* — is the template my Foxglove imprint inherits.

The margin loss specifically is the one I did not adopt, and Article 15 records the reasoning: I want Stage 1 to leave probability mass on non-oracle actions so Stage 2 RL can explore. The margin would close that off. The choice cost me a small amount of training-side sharpness in exchange for a downstream search budget. On the May 2026 evidence, where the failure mode is *over-confidence rather than under-confidence*, that was the right tradeoff. DQfD's recommendation is correct for its setting; my setting bent its recommendation the other way.

### 2017 — DDPGfD: the same idea in continuous control (Vecerik et al.)

*Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards*, Vecerik et al., arXiv 1707.08817. Same year as DQfD, continuous-control variant. The architectural translation is one-to-one: take DDPG, add a permanently-resident demonstration buffer with high priority, mix demonstration transitions into actor and critic updates.

The DDPGfD codebase in my directory (`docs/Hindsight-optimal/DDPGfD-main/README.md`) spells out the priority formula explicitly:

> "Always keep the demonstration data in the replay buffer and assign them with high priority. Specifically, priority for a transition tuple is `p_i = δ_i² + λ · |∇_a Q(s_i, a_i)|² + ε + ε_D`, where δ_i is the last TD error, the gradient term is the actor loss, ε is a small constant (0.001), and ε_D is a large demonstration priority constant (0.99)."
> — `docs/Hindsight-optimal/DDPGfD-main/README.md`

The paper's contribution is mostly empirical: it shows that demonstrations can replace reward shaping on real robotics manipulation tasks. For my purposes the load-bearing observation is the *prior on demonstrations*. ε_D = 0.99 is large compared to the per-transition TD error term; in effect, demonstrations are *always near the top of the sampling distribution*, not just early in training. The agent does not graduate away from them. They are a permanent anchor.

In my stack the analogue is the **anchor branch** of the e-dojo orchestrator — the synthetic-Bella branch that stays in the curriculum throughout training, alongside the historical branch, to prevent the policy from drifting off the oracle's distribution. The branch-merge cadence is the priority mechanism: synthetic gets weighted equally with historical even after the policy has long since outgrown the simple synthetic episodes, because if I let synthetic decay, the policy loses the directional anchor and benchmark Alpha collapses. The mechanism is different in implementation; the *prior* — "demonstration-shaped data stays high-priority forever" — is from DDPGfD.

### 2017 — Overcoming Exploration with Demonstrations (Nair et al.)

*Overcoming Exploration in Reinforcement Learning with Demonstrations*, arXiv 1709.10089. Same fall, third paper in the demonstration-augmented family. Where DQfD and DDPGfD treat demonstrations as a *seed*, Nair et al. add two distinct mechanisms: a behavior-cloning auxiliary loss on actor parameters, with a Q-filter that suppresses the BC term whenever the critic believes the policy's action is better than the demonstration's; and *resetting from demonstration states* — initializing rollouts from random points along the demonstration trajectory, not the environment's nominal start state.

> "Our method, which builds on top of Deep Deterministic Policy Gradients and Hindsight Experience Replay, provides an order of magnitude of speedup over RL on simulated robotics tasks. It is simple to implement and makes only the additional assumption that we can collect a small set of demonstrations. Furthermore, our method is able to solve tasks not solvable by either RL or behavior cloning alone, and often ends up outperforming the demonstrator policy."
> — Nair et al. 2017, abstract

The "outperforming the demonstrator" claim is the one I find most useful to revisit. Demonstrations bound the warm start; they do not bound the final policy, provided downstream RL has a path to improve. The Q-filter is the gating mechanism: BC only when BC is better than the critic's current opinion, otherwise let TD updates lead. The architectural lesson — that demonstration loss should *decay or gate* as the policy matures — shows up in my stack as the curriculum boundary between Foxglove (pure supervised) and Phase 2 MCTS self-play. I do not use a Q-filter inside Foxglove because Foxglove has no Q to filter against. The function is taken over by the phase boundary itself. Cross the boundary, the BC loss is gone.

The reset-from-demonstration-states trick I never adopted. In retrospect — and TASK_761 has put this on the "consider later" list — it is the most interesting unused idea in the paper. The hindsight oracle in my project produces full trajectories, every state of which is on a near-optimal path. Initializing some training rollouts from mid-trajectory oracle states would broaden the coverage of the policy's support without changing the demonstration source. I have not run this. It is on the shelf.

### 2021 — Proto-RL: prototypical representations (Yarats et al.)

The lineage steps sideways here. *Reinforcement Learning with Prototypical Representations*, Yarats, Fergus, Lazaric, Pinto, arXiv 2102.11271, ICML 2021.

Proto-RL is not strictly a demonstrations method. It is a *representation* method that uses self-supervision over learned prototypes — discrete cluster centers in latent space — to pre-train an encoder before downstream RL begins. The hindsight content is more subtle: at pre-training time the framework has access to the full trajectory and can compute prototype assignments over the entire dataset; the resulting representation is "task-agnostic" in the paper's language but is shaped by *whatever the prototype assignment is allowed to see*. The abstract is explicit about the unbinding:

> "We pre-train these task-agnostic representations and prototypes on environments without downstream task information. This enables state-of-the-art downstream policy learning on a set of difficult continuous control tasks."
> — Proto-RL abstract, Yarats et al. 2021

I put Proto-RL in this directory because of the *two-phase* structure, not because of prototypes specifically. The pattern — *pre-train a representation under a richer objective than the downstream task affords, then plug the encoder into a smaller policy head and fine-tune* — is exactly the Foxglove → MuZero handoff in my stack. The Proto-RL paper is the cleanest statement I know of the *encoder-as-substrate* hypothesis: representation quality is what limits downstream sample efficiency, and pre-training the encoder under privileged objectives is the right place to spend compute. It is a hypothesis that, by Article 6 of this notebook, my stack has empirically tested and found wanting in its strong form — encoder improvements that improve representations do not, on this fee structure, improve deployment Alpha. The Proto-RL framing remains correct; the binding constraint in my stack just happens to live downstream of where Proto-RL puts it.

### 2021 — EfficientZero: MuZero with auxiliary self-supervision (Ye et al.)

*Mastering Atari Games with Limited Data*, Ye et al., arXiv 2111.00210. This is the paper that, more than any other in the directory, shaped my actual loss function. EfficientZero adds three components to MuZero: a self-supervised consistency loss in latent space (forcing the dynamics model's prediction to match an encoded next-state target, SimSiam-style), a value-prefix prediction loss (predicting the *sum* of upcoming rewards rather than each one individually), and end-to-end model-based off-policy correction. The headline:

> "We propose a sample efficient model-based visual RL algorithm built on MuZero, which we name EfficientZero. Our method achieves 194.3% mean human performance and 109.0% median performance on the Atari 100k benchmark with only two hours of real-time game experience. […] This is the first time an algorithm achieves super-human performance on Atari games with such little data."
> — EfficientZero abstract, Ye et al. 2021

The hindsight content: EfficientZero's self-supervised consistency loss is a hindsight-optimal regularizer. At training time you have the actual next observation; you encode it; you require the dynamics model's prediction to match. This is the trick that makes MuZero's learned dynamics model stable at low sample counts. In my stack the equivalent is the *oracle-mask auxiliary head* (TASK_695, Article 17's Oracle 5): the model is trained to predict not just the oracle's chosen action but the oracle's *perception* — the binary selection mask of which assets are on the optimal path. This is not the same mechanism as SimSiam consistency, but it is the same architectural pattern: *use hindsight to construct auxiliary targets that constrain the encoder's latent geometry*. Same family of move.

The value-prefix idea I have not adopted directly, but my value-head categorical support over scaled returns is in the same neighborhood — discretized value targets to handle the long-tailed reward distribution. The directional intent (encode the *shape* of returns, not just their pointwise sum) is shared.

### 2022 — TF-C: Time-Frequency Consistency pre-training (Zhang et al.)

*Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency*, Zhang et al., NeurIPS 2022, arXiv 2206.08496. This one is sideways from the RL line, but it sits in my directory because the encoder side of the Trader stack is, fundamentally, a time-series representation problem.

> "We posit that time-frequency consistency (TF-C) — embedding a time-based neighborhood of an example close to its frequency-based neighborhood — is desirable for pre-training. […] Experiments against eight state-of-the-art methods show that TF-C outperforms baselines by 15.4% (F1 score) on average in one-to-one settings."
> — TF-C abstract, Zhang et al. 2022

TF-C's claim is that for time-series pre-training, the right invariance is *agreement between time-domain and frequency-domain views of the same window*. The paper is mostly about transfer across heterogeneous sensor datasets — EEG to EMG, that sort of thing — but the construction is generic.

My stack does not use TF-C directly. Bella, the market generator, sits in roughly the same architectural slot (Article 16), and its training objective is latent diffusion against historical market windows, not contrastive consistency between time- and frequency-views. But TF-C influenced one specific decision: the encoder family used in TASK_732 and forward (the cross-sectional channels work) includes a frequency-domain channel — running mean/variance over rolling FFT windows — as an auxiliary input to the TCN. That channel is in there because the TF-C paper made it impossible to ignore that you were *probably* leaving frequency-domain structure on the table by training only on time-domain views.

The richer question — whether I should run a *full* TF-C pre-training as a third pre-training pass before Foxglove — is on the same shelf as the Nair-style reset-from-demonstration-states trick. Considered. Not run. The TASK_761 backlog has a note.

### 2022 — ROSMO: regularized one-step look-ahead (Liu et al., MuZero Unplugged variant)

*Regularized One-Step Model-based Learning for Offline Reinforcement Learning*, arXiv 2210.05980. This is the most pointed paper in the directory for my current dilemma. Its premise is that MuZero Unplugged — the offline-RL variant of MuZero, trained from a fixed dataset with the Reanalyze algorithm — requires *expensive MCTS at training time* to refresh targets, and that this expense is structurally avoidable.

> "MuZero Unplugged presents a promising approach for offline policy learning from logged data. […] This paper investigates a few hypotheses where MuZero Unplugged may not work well under the offline RL settings, including 1) learning with limited data coverage; 2) learning from offline data of stochastic environments; 3) improperly parameterized models given the offline data; 4) with a low compute budget. We propose to use a regularized one-step look-ahead approach to tackle the above issues."
> — ROSMO abstract, Liu et al. 2022

What ROSMO replaces MCTS with is, structurally, a hindsight-shaped target: a one-step advantage estimate derived from the learned model, regularized toward the data distribution. The headline number is that ROSMO matches or beats MuZero Unplugged with 5.6% of the wall-clock time — under one hour versus seventeen-point-eight hours for the same IQM normalized Atari score.

This paper is directly relevant to two of my recurring questions. First, *do you need MCTS during pre-training, or can a cheaper hindsight signal substitute?* — which is exactly the TASK_697 Oracle 6 question, framed in reverse: my Oracle 6 wants to import MCTS-shaped targets into supervised pre-training; ROSMO wants to *remove* MCTS from offline RL by substituting a one-step advantage. Both papers are arguing that MCTS-at-training-time is not load-bearing, given the right hindsight construction. The conclusions point in opposite tactical directions but rest on the same observation. Second, the *low-data-coverage* hypothesis on the failure side of MuZero Unplugged is uncomfortably aligned with the data-coverage regime of my training set — financial windows with limited regime diversity, especially in bull regimes, which `project_bull_blindness.md` has flagged as the binding constraint on the project for months.

I have not implemented ROSMO. I have re-read its hypothesis list in TASK_761 planning sessions more times than is probably healthy.

### 2024 — EfficientZero V2: a unified framework (Wang et al.)

*EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*, Wang et al., ICML 2024 Spotlight, arXiv 2403.00564. The successor to EfficientZero. Extends to continuous action spaces, adds search-with-Gaussian-policies (an MCTS variant that handles continuous actions cleanly), and reports:

> "EfficientZero V2 outperforms the current state-of-the-art (SOTA) by a significant margin in diverse tasks under the limited data setting. EfficientZero V2 exhibits a notable advancement over the prevailing general algorithm, DreamerV3, achieving superior outcomes in 50 of 66 evaluated tasks across diverse benchmarks, such as Atari 100k, Proprio Control, and Vision Control."
> — EfficientZero V2 abstract, Wang et al. 2024

The reason this paper sits in my directory is the *generality* of its claim — same framework across discrete and continuous, visual and low-dimensional. The directional bet is that the MuZero family is the right substrate for sample-efficient model-based RL, full stop. My stack is in the MuZero family. EZV2 is the strongest existence proof that the family generalizes. It also raises the version question for me, periodically and uncomfortably: am I working with the right MuZero variant? Should I be importing EZV2's continuous-action search? Discrete is sufficient for portfolio trading at the bar-level granularity I use, but the search-with-Gaussian-policies has properties — implicit exploration noise calibrated by the policy's own variance — that I have wanted at various points in the project.

I have not migrated. The reason is opportunity cost: a search-substitution would invalidate every prior paired-Δα measurement in the project, the same way the richer-oracle re-imprint would. The campaign verdict (Article 10) is sitting on the structural-pivot decision tree right now and I am not willing to spend the disruption budget on a search upgrade until the oracle question is resolved one way or the other.

### 2024 — RDES: demonstrations selected by RL (Wang et al.)

The latest paper in the directory is *Demonstration Selection for In-Context Learning via Reinforcement Learning*, Wang et al., ICML 2025, arXiv 2412.03966. This is genre-bending — it is an LLM in-context-learning paper, not an RL paper for control. Its claim is that *which demonstrations you select for an in-context prompt* should itself be optimized by reinforcement learning, balancing diversity (label distribution coverage) against relevance (similarity to the task input).

> "This paper introduces the Relevance-Diversity Enhanced Selection (RDES), an innovative approach that leverages reinforcement learning (RL) frameworks to optimize the selection of diverse reference demonstrations for tasks amenable to in-context learning (ICL), particularly text classification and reasoning, in few-shot prompting scenarios."
> — RDES abstract, Wang et al. 2024

The paper sits in my directory because it inverts the usual question. The 2017 papers ask: *given demonstrations, how do you train against them?* RDES asks: *given a population of possible demonstrations, which ones should you train against?* In the Foxglove setting, where the oracle can produce arbitrarily many demonstration trajectories (just run the path solver on more episodes), the *selection* problem is real and underexplored. The SegmentPool curriculum (TASK_710/712) is a manual answer — cycle all 17 regimes uniformly — but it is not optimized. RDES suggests an RL outer loop over the selection policy. I have not built that. It is on the shelf, in a different drawer from the others, in the *"things to think about once the inner-loop question is settled"* part of TASK_761.

## The codebases, and what survived into the Trader codebase

A reference codebase is a strong claim about how a paper actually works. Reading the codebase tells you what the paper omitted, what was load-bearing in the implementation but not in the algorithm description, where the hyperparameters came from. I have six of them in the directory.

The **DQfD-master** codebase (Felix Kerkhoff's Bachelor's thesis implementation, faithful to Hester 2017) is the one I read closest in February 2026 when I was deciding the loss-balance for Foxglove. Its `loss.py` makes explicit the four-way sum — TD + one-step margin + n-step return + L2 — and the relative weight on the margin term. I borrowed the *structure* (a sum of weighted heterogeneous losses, each addressable by a config flag) and discarded the *margin* specifically. The four-way decomposition pattern, where each loss can be ablated independently, became the template for how `easy_trainer.py`'s loss function is laid out: policy + value + auxiliary mask + (optional) rank, each gated by a weight that can be set to zero.

The **DDPGfD-main** codebase (the open-source implementation in my directory) is small and pedagogical; it is the one I cite when explaining the *priority formula* to people because the formula is one line of Python in its replay buffer. The closest analogue in my stack is not a buffer-level priority but the dual-branch orchestrator's *branch weighting* during Fisher-merge, which is conceptually the same operation at a coarser temporal granularity.

The **EfficientZeroV2-main** codebase is large and well-organized; I have read its `model.py` and `loss.py` more times than any other reference. The consistency-loss block in particular — the SimSiam-style projection plus stop-gradient on the target encoder — is a clean piece of code. I did not import the consistency loss directly, but the *target-network pattern* — having a frozen-copy encoder produce targets that the learning encoder is regularized toward — is a pattern I use in the Bella-distillation experiment (TASK_740, Article 17) and in the value-target smoothing on the Foxglove side. The pattern survived; the specific SimSiam formulation did not.

The **Muzero-unplugged-main** codebase is the one closest to my actual MuZero core. Its README spells out the demonstration-buffer/reanalyze-buffer pair:

> "The Demonstration buffer, which is a collection of expert demonstrations that guide the agent's learning. […] The Reanalyze buffer, which is a collection of agents' demonstrations. The introduction of these two new buffers allows the algorithm to work with environments that do not have a simulator, or to reduce the amount of simulation needed."
> — `docs/Hindsight-optimal/Muzero-unplugged-main/README.md`

My stack does not separate these. Foxglove has a single buffer that holds (observation, oracle-target) pairs. There is no reanalyze buffer because there is no on-policy data in Foxglove — the on-policy data appears only in Phase 2, where it is again held in a single buffer alongside oracle warm-start examples. The two-buffer pattern from MuZero Unplugged would be the natural extension if I implemented Oracle 6. I have not.

The **TFC-pretraining-main** codebase I have read once and not adopted from. The frequency-domain channel mentioned earlier in this article is the only piece that touched my actual code path, and even that came in through TASK_732's cross-sectional augmentation, not through a direct port.

The **proto-main** codebase is the cleanest of the six, in my opinion as a reader — small, well-commented, idiomatic PyTorch. I did not adopt its prototype-clustering machinery. Its *experimental harness*, though, the way it factors pre-training and downstream tasks into separate entry-point scripts with a shared encoder checkpoint, was directly influential on my e-dojo orchestrator design. The orchestrator's pattern of *one checkpoint, multiple downstream evaluations* is in part proto-main's pattern.

The summary: of six codebases, two contributed *architectural patterns* (DQfD's four-way loss decomposition, proto-main's harness factoring), three contributed *individual mechanisms or formulae* that landed in different parts of my stack (DDPGfD's priority concept, EZV2's target-network pattern, TF-C's frequency channel), and one (Muzero-unplugged-main) defined the boundary for what I have *not* yet implemented but probably will if Oracle 6 ever gets unblocked. None of them was imported wholesale. The lineage works at the level of *idioms*, not at the level of code reuse.

---

> *The Apothecary records the seedlings that did not take. The journal lists, in the same script as the survivors: the variety, the soil, the season, and the practitioner who recommended it. The Apothecary does not strike these entries through. They are the inheritance against which the surviving plants are measured.*

---

## What the lineage contributed to the Trader project

The contribution is real and it is structural rather than mechanical. Let me lay it out cleanly.

The **two-phase architecture** — supervised pre-training under privileged information, then RL self-play — is from DQfD/DDPGfD/Nair and codified by Proto-RL and the EfficientZero family. My Foxglove → Phase 2 split is one instantiation of this template. Without the literature there would still be a temptation to put all the compute into RL self-play from scratch, and the cold-start math (Article 15) would still be unforgiving. The literature gave me permission to spend two thirds of the project's compute budget on a supervised phase against a constructed teacher, and the permission was load-bearing for the project's overall shape.

The **demonstration-as-permanent-prior** idea — keep the teacher in the loop forever, do not let the agent graduate away — is from DDPGfD. My dual-branch e-dojo, with synthetic-Bella alongside historical at all times, is the implementation. The cost is curriculum complexity. The benefit is that the policy does not drift off the manifold of episodes where the oracle is meaningful.

The **margin-vs-soft-target decision** is from DQfD, and I went the other way deliberately. The literature recommends margin; my deployment context (downstream MCTS that overreacts to encoder sharpness) made margin the wrong choice. This is the cleanest case I have of *the lineage prescribing one thing and the data prescribing the opposite*, and the lineage was correct in its setting and wrong in mine.

The **auxiliary-head pattern** is from EfficientZero (consistency), elaborated by the Klissarov et al. 2025 RL²F paper cited in TASK_695, and lands in my stack as the Oracle 5 mask-prediction head. The pattern is *use hindsight to construct auxiliary objectives that shape the encoder's latent geometry without directly shaping policy outputs*. It is the cleanest architectural inheritance the lineage gave me, and it is the one that — Article 17's verdict — also did not move deployment Alpha.

The **MCTS-as-hindsight-target** idea, Oracle 6 in Article 17, is the most ambitious unrealized inheritance. It descends from both EfficientZero (search-as-substrate) and ROSMO (search-substitution by regularized one-step look-ahead, the inverse direction). TASK_697 is the spec. It is on hold because it is a trainer reformulation, not a target reformulation, and the cost is two weeks of engineering. The lineage *frames* this as the canonical next move; my project has not yet been able to afford it.

The **richer-oracle direction** of TASK_761 / TASK_764 is the deepest inheritance and the one currently most active. The premise — that the hindsight teacher itself is the binding constraint, and that lifting the teacher's quality is the move that breaks the campaign verdict's null result — is what the entire LUPI literature has been pointing at since Vapnik first proposed Privileged Information in the early 2010s. The Trader project's specific instantiation (raise lookahead k from 4 to 8 or 12) is a parameter change, but the *theory* under it is the whole lineage's central thesis: the hindsight teacher determines the bound, and the bound determines the deployment ceiling.

## The kickback

This is the part I want to land honestly and without sourness.

The Hindsight-optimal lineage is good. The papers are correct. The benchmarks they report are not, as far as I can tell from reading the codebases, fabricated or inflated. DQfD does outperform PDD DQN on Atari. DDPGfD does work on real-robotics insertion tasks. EfficientZero does hit super-human Atari scores at 100k frames. EfficientZero V2 does generalize to 50 of 66 tasks. ROSMO does cut wall-clock time by an order of magnitude. The literature does what it says.

The Trader project, taking this lineage as a foundation, has built — over six months — a Foxglove → Phase 2 pipeline that:

- imprints a 64-asset multi-discrete policy against a hindsight-optimal expert,
- converges training-side metrics (Bal.Acc → 0.74, value loss falling cleanly) as the theory predicts,
- supports dual-branch synthetic + historical curricula with periodic merging,
- exposes auxiliary heads (mask, rank) under the EfficientZero pattern,
- has tested fourteen distinct mechanism variants under paired three-seed eval-MCTS recalibrated benchmarks,
- and has produced **zero promotions to deployment Alpha** across the entire campaign (Articles 8, 10).

That last bullet is the kickback. The lineage produced a beautiful pre-training pipeline whose training-side metrics behave exactly as the theory predicts and whose deployment behavior does not track the training-side improvement. Six months of mechanism-level experimentation has confirmed this with a sample size that no longer leaves much room for "maybe the next variant works." Article 6 named the proximate failure mode (classifier-Alpha anti-correlation: better encoder, sharper policy, more aggressive trades, fee bleed, Alpha collapse). Article 10 named the campaign-level evidence (fourteen-zero). Article 17 named the deeper structural diagnosis (the oracle is the binding constraint, single-mechanism encoder/auxiliary work cannot break it).

The lineage is correct in its theory and limited in its transfer to this specific stack. The reasons, as best I can articulate them in May 2026:

- *The reward landscape is different*. The Hindsight-optimal lineage trained on Atari and robotics. Atari rewards are sparse but unambiguous; robotics rewards are sparse and shapeable. Financial rewards are *contaminated by fees* — a sharp policy that trades more crosses more fee surface, and per-bet edge has to clear the fee bar before any cumulative Alpha shows up. The literature's benchmarks did not stress this regime.
- *The action space is wider*. Atari has 18 actions; manipulation has a low-dimensional continuous joint space. Portfolio trading at 64 assets has a 5⁶⁴ joint action space, factored into 64 independent 5-way heads. The class imbalance — under k=4 lookahead, 93.75% of per-asset labels are HOLD — is structurally different from anything in the literature, and the "Turtle local optimum" (Article 15) is the consequence. The lineage's loss-balancing methods do not natively address it.
- *The demonstrator is closed-form, not human*. The lineage's demonstrations are human or scripted-policy trajectories; they have human imperfection baked in, which the literature treats as a feature (the Q-filter exists because the demonstrator is suboptimal in places). My oracle is *exactly optimal under its own objective* — k=4, lookahead=20, transaction cost threshold — and the suboptimality is in the *objective*, not in the trajectories. The literature has less to say about this regime. The literature mostly assumes the demonstrator's *trajectories* are the limiting noise; in my case the demonstrator's *definition of optimal* is the limiting structure.
- *The campaign-noise floor is real*. Article 10 measured the per-experiment noise floor on this stack at roughly ±3pp paired Δα. Many of the lineage-inspired mechanisms I have tested produce Δα in the −3 to +3 range, which is statistically null. The literature's effect sizes (DQfD's 83×, EfficientZero's 500×) are an order of magnitude larger than my noise floor. The lineage's methods may be working in my stack at effect sizes I cannot resolve. The honest version of this statement: I cannot rule out that the lineage is helping by 1pp here, 2pp there, and I am unable to distinguish that from noise.

I want to be careful not to over-claim a negative result. The lineage might still produce deployment Alpha gains in this stack — under a different oracle (TASK_764), under a different planner (EZV2's continuous-action search), under a different reward shaping (TASK_762), under a different entropy regularizer (TASK_763). What the evidence rules out is the simplest version: "import the lineage's methods as-is, expect the lineage's effect sizes." That hypothesis is closed. The richer version — "the lineage's *frame* is correct and the *binding constraint* in this stack is downstream of where the lineage's published experiments stressed" — is consistent with everything I have measured.

## What's left

TASK_761's four pivot directions are all, in their own way, lineage-shaped. Direction 1 is reward shaping — a hindsight-shaped reward signal layered over the per-step PnL, which is the financial analogue of the auxiliary-loss pattern from EfficientZero. Direction 2 is entropy regularization — a planner-side intervention that the lineage does not directly address but that follows from the diagnosis (the policy is over-confident; the planner over-exploits; entropy reg should soften both). Direction 3 is the richer-oracle re-imprint — the most directly lineage-derived move, the explicit lifting of the hindsight teacher's quality bar. Direction 4 is the incompatible-dataset direction — train on a fundamentally different distribution and see if the lineage's methods transfer differently, which is closest in spirit to the TF-C cross-domain pre-training experiments.

None of the four is in production yet. TASK_762 and TASK_763 are the cheap ones and they run first. TASK_764 is the disruptive one — it breaks the paired-Δα measurement contract with all prior experiments, which is why it sits last in the queue even though it is the most directly indicated by the campaign verdict. The lineage prescribes its own next move and the prescription is expensive.

The lineage's open questions are, after six months of close engagement with eight papers and six codebases, still the right questions. *How rich does the hindsight teacher need to be?* — open. *How do you select demonstrations from a population of equally-correct teachers?* — open, and RDES (the 2024 paper) is the only entry I have on this thread. *Can search be the supervised target rather than the run-time mechanism?* — open, and Oracle 6 is the spec. *What is the right invariance for representation pre-training on time series in a financial regime?* — open, and TF-C is the closest published answer.

I keep the directory. I add to it when new papers come out. The journal of seedlings I could not have known would survive is the one I read most often. Whether any of the entries in it will, in the next year or the year after, produce the deployment-Alpha gain that the Trader project is built around, is a question whose answer is on the same shelf as the rest. The seed is in the soil. The dates of survival are not yet written.

> *The Apothecary closes the journal. The most-read entries are not the ones that flowered. They are the ones that explained, in the season after they did not flower, why the seedling next to them did.*

---

## Sources

- `docs/Hindsight-optimal/arXiv-1704.03732v4` — Hester et al., *Deep Q-Learning from Demonstrations* (DQfD), AAAI 2018.
- `docs/Hindsight-optimal/arXiv-1707.08817v2` — Vecerik et al., *Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards* (DDPGfD), CoRL 2017.
- `docs/Hindsight-optimal/arXiv-1709.10089v2` — Nair et al., *Overcoming Exploration in Reinforcement Learning with Demonstrations*, 2017.
- `docs/Hindsight-optimal/arXiv-2102.11271v2` — Yarats, Fergus, Lazaric, Pinto, *Reinforcement Learning with Prototypical Representations* (Proto-RL), ICML 2021.
- `docs/Hindsight-optimal/arXiv-2111.00210v2` — Ye et al., *Mastering Atari Games with Limited Data* (EfficientZero), NeurIPS 2021.
- `docs/Hindsight-optimal/arXiv-2206.08496v3` — Zhang, Zhao, Tsiligkaridis, Zitnik, *Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency* (TF-C), NeurIPS 2022.
- `docs/Hindsight-optimal/arXiv-2210.05980v2` — Liu et al., *Efficient Offline Policy Optimization with a Learned Model* (ROSMO), 2022. Title verified from the codebase reference; arXiv abstract used in the quoted block.
- `docs/Hindsight-optimal/arXiv-2403.00564v2` — Wang, Liu, Ye, You, Gao, *EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data*, ICML 2024 Spotlight.
- `docs/Hindsight-optimal/2412.03966v2.pdf` — Wang, Wu, Yuan, Cai, Li, Jia, *Demonstration Selection for In-Context Learning via Reinforcement Learning* (RDES), ICML 2025.
- `docs/Hindsight-optimal/DQfD-master/` — Felix Kerkhoff, faithful PyTorch implementation of Hester et al. 2018.
- `docs/Hindsight-optimal/DDPGfD-main/` — community implementation of Vecerik et al. 2017; priority-formula reference.
- `docs/Hindsight-optimal/EfficientZeroV2-main/` — Wang et al. official code release; consistency-loss and target-network pattern reference.
- `docs/Hindsight-optimal/Muzero-unplugged-main/` — DHDev0's PyTorch MuZero Unplugged port; demonstration-buffer / reanalyze-buffer pattern reference.
- `docs/Hindsight-optimal/TFC-pretraining-main/` — Zhang et al. official code release; frequency-domain channel reference.
- `docs/Hindsight-optimal/proto-main/` — Yarats et al. official PyTorch implementation; harness-factoring pattern reference.
- Article 6 (`06_classifier_alpha_anticorrelation.md`) — eval-side classifier-Alpha anti-correlation finding.
- Article 10 (`10_eval_mcts_asymmetric_lift.md`) — eval-MCTS recalibration and the fourteen-zero campaign verdict.
- Article 15 (`15_causal_expert_imprinting.md`) — training-side companion on Causal Expert Imprinting.
- Article 17 (`17_five_oracles.md`) — the five-oracle lineage in `path_solver.py`.
- `tasks/TASK_695_ORACLE_MASK_PREDICTION_HEAD.md` — auxiliary-head pattern, RL²F-inspired.
- `tasks/TASK_697_ORACLE_AS_MCTS_DISTILLATION.md` — Oracle 6, the unrealized MCTS-as-supervised-target plan.
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the four-direction structural pivot triggered by Article 10's campaign verdict.
- `tasks/TASK_764_ORACLE_K_REFORMULATION.md` — the richer-oracle re-imprint specification, lineage-derived.
- `mcts_trader/jax_agent/path_solver.py` — the hindsight-optimal teacher.
- `easy_trainer.py` — the loss function in which the lineage's auxiliary-head and four-way-decomposition patterns landed.
