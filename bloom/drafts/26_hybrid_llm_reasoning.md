# Article 26 — Hybrid LLM Reasoning: A Design That Stayed in the Drawer

> *Observation: there is a plant in the catalogue that the apothecary studies but does not plant. It draws too much sun, and the rest of the bed yellows in its shadow. She keeps the seed in a labeled envelope in the third drawer of the bench. She re-reads its monograph each winter, when the beds are empty and the light is fair. She has never put it in the ground. She is not, she tells herself, refusing it. She is waiting for a bed large enough that its shadow falls on nothing she is trying to grow.*

---

**by Lukas — May 22, 2026**

## The temptation, plainly

`workstation` is on. The single 3090 is at 41 °C; the desk fan is on its quiet setting; `nvidia-smi -l 2` is ticking in the next window through the last validation pass of the single-mechanism campaign I declared dead on Wednesday. To my left, a second pane has Claude Code open in this same working tree, halfway through a refactor of `tools/orchestration/e-dojo_v1.py` that has nothing to do with this article. The agent is *already on the desk*. It has been on the desk for nine months. It writes the boilerplate, finds the shape bugs, drafts the dashboards, queries the database, reads its way around `mcts_trader/` faster than I can.

The temptation, somewhere around the middle of 2025, was as obvious as a tool sitting next to a workpiece: *if the agent is already on the desk, why not let it into the trade?* Not as a coder. As a *reasoner*. Read the news. Parse a 10-K. Structure a thesis. Pass a hint to MuZero. Let the planner consult an oracle that has, by definition, *read more financial text than any single human ever could*. The planner is good at the inside of the chart. The LLM is good at the outside of the chart. The hybrid would, on first inspection, beat either alone.

I wrote a design document. I gave it a name. I gave it two names, in fact — the first one in January 2026, the second the same document had grown a different opinion about itself by February. The document is `docs/HYBRID_LLM_REASONING_DESIGN.md`. It is 3376 words, and the title at the top reads:

> *"Codename: Compact Intelligence → **Dense Latent Mind**"*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, header

The arrow is the whole story of the document. The first half is the practical hybrid I sat down to specify. The second half is the philosophical inversion I talked myself into a month later. Both halves are interesting. Neither half is in the codebase. The seed is in the third drawer.

This article walks the design, walks the math of the integration I would have built, walks the three layers of kickback that kept the seed in the drawer, and ends with the deeper kickback that *closed the case retroactively*: by May 2026 the bottleneck in the system was demonstrably not in the reasoning side. An LLM augment, however elegantly wired, would have been a beautiful solution to a problem the project did not have.

The lineage is honest. The 2014 origin prompt for the trader — the one Article 0 quotes from `_ORIGIN_PROMPT.md` — already contains the seed. The MoE framing of the original sketch includes, almost as an aside, the line:

> *"…a potential LLM for sentiment analysis if incorporated…"*
> — `bloom/drafts/_ORIGIN_PROMPT.md`

The hybrid LLM design is what that aside grew up into across eleven years. From "LLM for sentiment" in 2014, to a structured reasoning-orchestrator design in January 2026, to a roadmap for a mobile-deployable Dense Latent Mind in February, to the drawer, in May, after the bottleneck had been measured and found elsewhere. Article 0 is the seed. Article 2 is the prettiest physics module I built and never wired in. Article 26 is the prettiest *system architecture* I designed and never built at all. Two of those went into the drawer. One of them did not survive contact with the bottleneck. The article order matters.

---

## What the design proposed

The document opens with a vision statement that is, for once, honest about the trade-off it is making:

> *"A reasoning system that combines: Frozen small LLM (2-4B params) — knowledge and generation; Trained MCTS controller (~10M params) — decision making; Trained value model (~10M params) — path evaluation; Vector index — external knowledge retrieval. Key insight: Don't train the LLM. Train a tiny 'reasoning orchestrator' that learns WHEN and HOW to use the LLM optimally."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Vision

I want to land what is *interesting* about this framing before I land what is *wrong* with it.

The interesting part: the design refuses, on day one, to fine-tune the LLM. Fine-tuning a 7B model takes 28 GB of gradient and optimizer state — a number the document quotes correctly in its efficiency section — and the workstation has 24 GB. Fine-tuning was not on the menu. So the design pivoted to a much sharper question: *given a frozen LLM you cannot afford to retrain, what is the smallest learned object you can put around it to make it useful for a specific task?* The answer the document lands on is a 25M-parameter orchestrator: a state encoder, a policy head (the "controller"), and a value head. Three small things that learn *when to ask the LLM*, *what to ask it*, and *how much to trust the answer*. The LLM stays a tool. The brain is small, separable, and trainable on the box under my desk.

The orchestrator's action space is the cleanest part of the document. Twelve discrete reasoning moves:

> *"0: retrieve_from_index — RAG lookup. 1: retrieve_with_rewrite — Rewrite query, then RAG. 2: llm_generate — Direct generation. 3: llm_with_context — Generate with retrieved context. 4: llm_decompose — Break problem into parts. 5: llm_verify — Ask LLM to verify current answer. 6: calculator — Arithmetic. 7: code_execute — Run Python snippet. 8: backtrack — Abandon current path. 9: commit_answer — Finalize and return. 10: rephrase_problem — Restate problem differently. 11: request_clarification — Ask for more info (if interactive)."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Action Space

That is the API of the design. Twelve verbs. MCTS searches over sequences of those verbs. The value head says *how close are we to a correct answer given the current trace*. The controller says *given where we are, which verb next*. The LLM gets called only when verbs 2–5 fire, which means the orchestrator can learn to *not* call the LLM when the calculator or the index would do.

For the trading domain the same structure mapped over with almost too-clean symmetry:

> *"This architecture directly mirrors the trading system: Market state → Problem + context. Trade actions (320) → Reasoning actions (12). PnL reward → Correctness reward. Price dynamics → Reasoning dynamics. MCTS search → MCTS search. Value function → Value function. Same algorithms, different domain."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Connection to Trading System

I read that table now and what I see is a sleight of hand. The mapping is *correct in its rows and wrong in its column*. The rows are real — MCTS is MCTS, value heads are value heads — but the column on the right is a *reasoning* domain (QA, GSM8K, HotpotQA) where the LLM has, by virtue of its training, deep priors that are useful. The column on the left is a *trading* domain where the LLM has, by virtue of its training, nearly no useful priors at all. The next paragraph in the doc papers this over with one of those gentle lies a designer tells himself at the worst moment:

> *"The trading system is a proof that this architecture works. If it can learn market patterns, it can learn reasoning patterns."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Connection to Trading System

Read that backward. *The trading system proves the architecture, therefore the architecture will work for reasoning.* That is the order in which the doc would have me believe the inference goes. But what I was actually planning to *deploy* was the reverse direction: the architecture from the reasoning domain, ported back into trading. *Generalize to QA → return to trading with the LLM hook now battle-tested.* If the reasoning case had been built and validated, the trading case would have gotten the LLM hook for free. The doc reads in one direction; the deployment plan ran in the other. I did not see that asymmetry when I wrote it. I see it now.

A month later, on February 11 2026, the document grew an inversion. The header changed: *Compact Intelligence → Dense Latent Mind*. The body acquired a new section titled *The Architectural Inversion* whose opening line was:

> *"OLD THINKING (LLM-centric): LLM is the intelligence. MCTS orchestrates the LLM. Language is the representation. NEW THINKING (World Model-centric): Dense latent world model IS the intelligence. Language is just one interface/expression layer. MCTS enables deliberate reasoning in latent space."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Crystallized Vision

This was the moment the design talked itself out of the practical hybrid and into a more ambitious target. The doc then made a quantitative claim I will quote in full because it is the load-bearing assumption of the entire February rewrite:

> *"LLMs store ~95% 'ballast' from an intelligence perspective: Memorized facts ('Paris is capital of France'); Redundant phrasings (10,000 ways to say hello); Training data noise (Reddit, SEO spam); Only ~5% is actual reasoning capability."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §95% Ballast Hypothesis

The 95% Ballast Hypothesis is the inversion's load-bearing argument. If most of an LLM's parameters are dead weight from an intelligence perspective, then *building the 5% directly* is not absurd. The doc then describes a 96 GB hard-limit dense Mamba backbone, GARCH-grade compression schedules, an AMD AI Max 395 development path, and a *Pixel 10 deployment target*. Twenty-eight hundred words of how to build the future of intelligence on a phone.

The two halves of the document — January's practical hybrid, February's dense-latent-mind inversion — are not actually contradictory. The doc itself closes with the line:

> *"Original: Practical near-term (use existing LLMs efficiently). Crystallized: Ultimate goal (build intelligence from scratch). Path: Original → validate architecture → Crystallized."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Evolution of This Document

Practical hybrid as the stepping stone. Dense latent mind as the destination. Bella and Foxglove as the *bootstrap* — earn money, fund compute, build the real thing. A perfectly coherent multi-year plan, written on a Sunday afternoon on a workstation with one GPU. The plan never executed. Below I want to walk through *why*, with the same care.

But first the math.

## The math of the integration

The hybrid's clean part is the action space. The hard part — the part the doc waves at and does not actually solve — is the *interface*. If you take the design at its word and try to wire it into the existing MuZero trader, the question becomes: *how does a structured LLM output feed into the planner's state representation?* The doc does not contain that diagram. I sketched it three times across January and February and never committed any of them to disk. Here is the version that, on reflection, I would have built — if I had built anything.

The MuZero trader's planning loop, in pseudocode, is:

```
for sim in range(n_simulations):
    leaf = tree.select(root)
    obs = leaf.state                      # (27, 64, 64) market obs
    z = encoder(obs)                      # (D,) latent
    pi, v = policy_head(z), value_head(z)
    z' = dynamics(z, action)              # next-state latent for expansion
    tree.expand(leaf, pi, v)
```

`z` is the latent the value and policy heads consume. The integration question is: *where in this loop does the LLM output enter?*

There are three architecturally honest answers. I'll walk all three because the design landed on the third but the first two are not silly.

**Option A — concatenate.** Run the LLM once per market day on whatever text corpus is relevant (a curated news feed, the most recent 10-K of the asset, a sentiment summary). Project its hidden-state output through a linear layer into a fixed-size vector `u ∈ R^k`, and concatenate `[z; u]` before the heads:

```
pi, v = policy_head([z ; u]), value_head([z ; u])
```

Cheap. Trivially wired. The gradient flows back into the projection but not into the LLM. The disadvantage is that `u` is the *same vector* across every simulation in the search tree — it cannot adapt as the planner descends into a counterfactual branch. The LLM is, at this point, a static feature.

**Option B — cross-attention.** Run the LLM with output_hidden_states, get back a sequence of token embeddings `T ∈ R^{L × d_LLM}`, project once into the planner's dimension, and let the planner's latent attend over the token sequence at every tree level:

```
z' = z + CrossAttn(query=z, keys=T, values=T)
```

Now the *same* token stream gets re-queried at every search depth, with a different query each time. The LLM behaves like an external memory the planner reads from. This is closer to how a human trader would use a research note — re-read it through the lens of the current decision.

**Option C — FiLM injection at a specific encoder layer.** This was the option the design (implicitly) landed on, because it has the cleanest gradient properties and the cheapest inference cost. Run the LLM once per day; produce two vectors γ, β ∈ R^c via a small projection head; and use them as a feature-wise linear modulation of the encoder's intermediate activations at a chosen layer:

```
h_l = encoder_layer_l(h_{l-1})
h_l' = γ ⊙ h_l + β             # FiLM, where γ, β = MLP(LLM_summary)
```

Gradient flows: from the loss, back through the planner heads, into the encoder, into the FiLM projection MLP, *and stops at the boundary of the frozen LLM*. The LLM's parameters never update; only the small projection learns. This matches the document's "don't train the LLM" insistence at the architectural level: the FiLM head is the entire *trained surface* between the LLM and the rest of the system, and it has, by my back-of-envelope, perhaps 500K parameters at most.

The reason FiLM wins on paper: it injects conditioning *deep* in the encoder, where the planner's representation is still spatial enough that a per-channel scaling has real effect, but the gradient path back to the LLM projection is short. Cross-attention (Option B) is more *expressive* but has gradient paths that are an order of magnitude longer and inference cost roughly proportional to `L × n_simulations`. Concatenation (Option A) is too shallow; the heads can ignore `u` if the gradient finds it noisy, and there is no way to ensure the conditioning *propagates*.

FiLM also has the property that, if the LLM-derived γ, β are close to (1, 0), the system reduces to the un-conditioned baseline. That is the soft fallback I wanted: if the LLM signal is noise, the network learns γ ≈ 1, β ≈ 0, and the FiLM layer becomes a no-op. The architecture has an explicit *off switch* that gradient descent is allowed to throw.

So the math, written down once for the record, is:

```
LLM_summary = LLM(news_text_of_day)                    # frozen, cached per day
γ, β        = MLP_FiLM(pool(LLM_summary))              # 500K params, trained
h_l'        = γ ⊙ h_l + β                              # conditioning
z           = remaining_encoder(h_l')                   # standard path
pi, v       = heads(z)                                  # standard heads
```

That is the integration. Three lines of math, one cached forward pass through the LLM per market day, ~500K trainable parameters in the FiLM head. On the training side, the loss is exactly the loss MuZero already optimizes — no auxiliary objective, no separate reasoning-trace reward, no need to align the LLM's notion of correctness with the trader's notion of PnL. The LLM gets to *flavor* the encoder. Nothing more.

This is the design I would have built. It is, I think, the right design. The kickback is not that the design is wrong. The kickback is that *the design is wrong about its own importance*.

---

> *The Apothecary opens the third drawer. The envelope is dry. The label has not faded. She lifts it to the light, holds it for a moment, returns it to its place. The drawer closes without ceremony.*

---

## The kickback

The kickback has three layers, and a fourth that closes the case from underneath.

**Layer one — latency.** The trader's decision loop is on the order of milliseconds. Forward pass through the encoder plus the heads, on `workstation`, is well under one millisecond. A 32-simulation MCTS expansion lives in the tens of milliseconds. Add even a small LLM and the numbers stop being friends. Phi-3 Mini at int4 quantization, on a 3090, generates at perhaps 50 tokens per second; a 200-token summary takes four seconds. The design's "Per problem: LLM inference: 32 sims × 3 calls × 50ms = 4.8s" math is, even at its most optimistic, four orders of magnitude slower than the planner.

The architectural fix the design implicitly proposed — and the FiLM integration I sketched above doubled down on — is to *precompute the LLM features once per market day and cache them*. Run the LLM offline. Store the γ, β as a per-day feature vector. The training loop touches a hash table, not the LLM. Inference latency goes back under a millisecond.

The fix works. But notice what the fix has done: *the LLM is no longer a reasoner*. The LLM is a feature extractor. It is doing exactly what a sentence-transformer would do, with vastly more parameters and vastly more compute. The whole *point* of the hybrid was that the LLM would *reason in the loop* — verify hypotheses, decompose problems, query the index. The moment you precompute its output, those twelve reasoning verbs from the design collapse into a single static *summary token*. The orchestrator is gone. The MCTS-over-reasoning-actions is gone. What remains is, at the architectural level, a 3B-parameter feature extractor wired into a FiLM head.

A `all-MiniLM-L6-v2` sentence encoder is 22M parameters and produces a fixed-size summary vector for free. It would feed the *same* FiLM head with the *same* gradient surface. If I want LLM-derived day-summary features, the small model gives them at one one-hundredth the inference cost. The Phi-3 is overkill for the role the latency constraint forces it into.

That is the first kickback: *the architectural choice that makes the latency budget work strips the LLM of the property that made it worth using*.

**Layer two — cost.** Even granting that the LLM runs offline, the token economics did not work for a single-3090 hobby project. The design speculates a Phi-3 Mini int4 footprint of 2 GB and a per-problem latency budget of seconds; what it does not price in is the *training-time* cost, which is the cost that actually matters. A multi-year MuZero training run on the trader touches on the order of 10^7 daily-state observations across all assets and rotations. Even at one LLM forward pass per day-per-asset, that is on the order of 10^6 inference calls. At 4 seconds each on a 3090, that is roughly 4 × 10^6 GPU-seconds, or 46 GPU-days, for one pass through the dataset. The trader's *entire* Phase 1 imprint, the canonical Foxglove campaign, costs 93 GPU-days end to end (Article 9). The LLM augment would, at the most conservative back-of-envelope, *halve the throughput of the whole project*. For an integration whose expected lift is, by every honest reading of the literature, somewhere in the single-digit-Alpha-percentage-points range.

The 93 GPU-days are not an idle benchmark. They are the cost of the canonical Bella precompute the project paid in March, and Article 9 is the record of how much that hurt. Paying it again — to give the encoder a feature it could probably learn from price action alone — is the kind of cost that requires a proof of necessity, not a proof of plausibility. The design did not have one.

**Layer three — the harder version of the cost.** This is the one I want to be specific about, because it is the kind of cost-against-budget reasoning the single-GPU constraint forces. The 3090 has 24 GB of VRAM. The trader's training-time memory budget is already tight — JAX preallocates ~70% of VRAM, the MCTS replay buffer eats a few gigabytes, the encoder + dynamics + heads are a few gigabytes more in mixed precision, and the optimizer state for the trainable parts is non-negligible. Add a 2 GB int4 LLM resident in VRAM during training and the memory math gets ugly. Add an fp16 LLM and the training simply OOMs. The "cache to disk" workaround means a disk read per minibatch, which the project's efficiency-philosophy CLAUDE.md explicitly forbids:

> *"Anti-patterns (NEVER): Pre-generate training data to npz/pkl then load. Repeated disk reads in training loop."*
> — `/home/lukas/Trader/CLAUDE.md`, Efficiency Philosophy

I would have been writing a feature extractor whose output had to be streamed from disk during training, on a workstation whose published operating rule forbids exactly that. There are workarounds — RAM-resident cache, careful batching — but each of them is a refactor of the data pipeline, and each of them eats into the same engineering budget the project's actual experiments are competing for.

**Layer four — the deeper kickback.** This is the one Article 6 named, Articles 7 and 8 confirmed, and Articles 12 and 15 collectively buried. *The bottleneck is not in the reasoning side.* By May 2026 the campaign's strongest finding was that encoder improvements — better classification, lower loss, higher balanced accuracy on the imprint task — *anti-correlate* with deployed Alpha. The line is in my notes:

> *"The bottleneck is downstream of the encoder."*
> — `~/Trader/lore/notes_2026-05-15.md` §110

The hybrid LLM design's *entire mechanism of effect* on the trader is upstream of the encoder: the FiLM head conditions an intermediate encoder layer; the encoder produces a better latent; the heads consume a better latent; the planner produces better trades. The causal chain is exactly the one the May findings showed *does not produce Alpha lift*. Even a perfectly wired, perfectly cached, perfectly trained hybrid would have improved the latent representation by some amount, made the classifier sharper, made the value head more confident, and — by the three-way confirmed pattern of TASK_737, TASK_741 and TASK_738 — likely *cost* paired-Δα through trade-rate inflation. The same pattern I dragged through fourteen failed mechanisms in May, dressed up in a fancier feature extractor.

The 2014 prompt's "LLM for sentiment" line, twelve years later, has the same fate as the rest of the 2014 mixture-of-experts framing in Article 0: the *kind* of failure mode the design tried to address was real, but the *location* in the stack was wrong. Sentiment, news, structured-thesis priors are upstream-of-encoder interventions. The bottleneck is downstream. The cure was aimed at the wrong organ.

This is the same shape as Article 2's NeuralOGCM physics core. Article 2 quotes a January note explicitly, and the same line applies here:

> *"I built a beautiful solution to a problem that turned out, when measured, not to be the bottleneck."*
> — `bloom/drafts/02_physics_core.md`, §IV

Two articles, two designs, one diagnosis. The physics core was an upstream world-model intervention. The hybrid LLM is an upstream feature intervention. Both targeted the same wrong location. The fact that I wrote them in different months, with different reading lists, and arrived at the same architectural mistake from two different directions tells me something I want to name: *upstream interventions are the natural reach for a person whose engineering reflexes were formed on perception problems*. CNNs, encoders, feature engineering, world-models — those are the muscles. The bottleneck living in the policy head and trade-rate calibration was an unfamiliar shape, and unfamiliar shapes do not invite designs. Familiar shapes do. So I designed twice into the familiar shape, and twice the familiar shape was not where the patient was sick.

## What's left, what I learned

The design is in the drawer. It is not dead in the strict sense — `docs/HYBRID_LLM_REASONING_DESIGN.md` is still in the repo, the FiLM math sketched above is reproducible from the document plus a weekend of work, and the LLMs that would run inside it have only gotten cheaper and faster in the year since I wrote the doc. If the diagnosis ever changes — if the project ever finds itself with an upstream bottleneck that an external knowledge source could relieve — the design is on the shelf, labeled, dated, ready.

The lineage is the part I want to keep on the record, because it is honest in a way the design itself was not.

In 2014, in the original prompt Article 0 quotes, the seed was three words: *LLM for sentiment*. A reflex, a fashionable inclusion, a hand-wave at how a system might be augmented if a thing-that-did-not-yet-exist were ever to exist. In January 2026, the seed had grown into a 12-action reasoning orchestrator with code skeletons, a training procedure, and a mobile deployment plan. In February 2026, it had inverted itself into a 96 GB dense-latent-mind blueprint with a hardware roadmap, a ballast hypothesis, and a deployment target on a phone that hadn't been announced yet. In May 2026, it sat in the drawer next to the physics core, both of them well-stoppered, both of them aimed at the wrong organ.

What I keep from the design is not the design. What I keep is the *separation-of-concerns argument* that opens it:

> *"The LLM already 'knows' things. We're training a tiny brain to USE it well."*
> — `docs/HYBRID_LLM_REASONING_DESIGN.md`, §Why This Works

That sentence applies — *applies right now, applies in the room next to this draft* — to the way I work with Claude Code in 2026. The agent already knows things. I am, in a very real sense, the tiny brain that learned when and how to ask it. The hybrid LLM trading system would have been an attempt to teach a *neural net* the same skill at machine speed. The fact that I learned it as a human at human speed instead is not a substitute for the original design, but it is the *form* of intelligence the design was pointing at, and it is the form that is currently load-bearing for the whole project. The existence of this article series — twenty-six pieces written in five weeks against a daily training campaign — is the existence proof. The agent did not earn its compute *inside the trader*. It earned a thousand other things outside the trader, in the codebase around it, in the documents on the desk, in the chat panes that produced and audited every claim in the lore.

The version of the design that survived is the version that did not need to be wired into the planner. *LLM as collaborator on the engineering, not the trading*. Article 0 names the four bricks that made the project actionable in 2025; the fourth brick was the LLM-as-collaborator era. That brick is in place. It is doing the work the hybrid design imagined a tiny orchestrator would learn to do. The orchestrator stayed in the drawer because *I am the orchestrator*.

The standing question I have not answered, and may never answer: *if the diagnosis changes — if the post-May structural pivot finds an upstream-bottleneck branch worth exploring — does the hybrid design come back out of the drawer?* The honest answer is *probably not in that form*. The 2026 LLM landscape moves fast enough that the practical-hybrid design of January is, in May, already partly obsolete: Mamba-architecture models have shipped, the int4 quantization story has tightened, the inference-on-3090 economics have improved, and the right design for *the same problem* in May would look meaningfully different from the design I wrote in January. The seed is in the drawer. The seed will not be planted as-is. If the bed is ever ready for it, the seed will have evolved, and what I will plant will be a descendant.

## Closing

The prettiest *system architecture* I ever wrote sits in a 3376-word file in `docs/`, fully sketched, untested, and unimplemented. The prettiest *module* I ever wrote sits in three files in `mcts_trader/jax_agent/prediction/`, fully tested, benchmarked at sub-millisecond, and unused. The two designs are siblings. They were aimed at the same patient, at the same organ, with the same diagnostic instinct, and the diagnostic instinct turned out to be wrong about where the patient was sick.

That is not, in either case, a failure of design. It is a failure of *triage*. The project's discipline for the second half of 2026 — the structural pivot the `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` document is now driving — is, finally, to design for the diagnosed location instead of the suspected one. Reward shaping. Entropy regularization. Oracle-k sweeps. *Downstream-of-encoder* interventions. The hybrid LLM design and the physics core are the two specimens in the drawer that taught me, by their joint absence from the active codebase, what the next year's specimens cannot afford to be.

The drawer is not empty in the sense that matters. It contains two complete designs, each pointing exactly where I will not aim next.

> *The seed will keep. The drawer is dry. The catalogue is open at a different page.*

---

## Sources

- `/home/lukas/Trader/docs/HYBRID_LLM_REASONING_DESIGN.md` — primary source. 3376 words. The January practical-hybrid design, the February dense-latent-mind inversion, the 95% ballast hypothesis, the 96 GB hard limit, the action space, the mobile deployment target, the trading-system mapping, and the "LLM is a tool, not the brain" framing all quoted from this file.
- `/home/lukas/Trader/bloom/drafts/_ORIGIN_PROMPT.md` — the 2014 sketch and its *"potential LLM for sentiment analysis if incorporated"* line. The lineage starts here.
- `/home/lukas/Trader/bloom/drafts/00_before_alphago_had_a_name.md` — Article 0, the prequel naming the LLM-as-collaborator era as the fourth load-bearing brick.
- `/home/lukas/Trader/bloom/drafts/02_physics_core.md` — Article 2, the design-document trilogy entry this article is paired with. Same shape of kickback, different module.
- `/home/lukas/Trader/bloom/drafts/06_classifier_alpha_anticorrelation.md` — Article 6, the canonical statement of the downstream-of-encoder finding.
- `/home/lukas/Trader/lore/notes_2026-05-15.md` §110 — *"The bottleneck is downstream of the encoder."*
- `/home/lukas/Trader/CLAUDE.md` — the Efficiency Philosophy section, including the "NEVER pre-generate training data to npz/pkl" rule cited above.
- `/home/lukas/Trader/tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the post-May pivot away from upstream interventions.
