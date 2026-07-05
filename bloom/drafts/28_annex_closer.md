# The Annex: What the Architecture Essays Were For

> *Observation: the gardener keeps two journals on the same shelf. The first is the season's: the dates of the sowings, the doses, the names of the specimens, the weather on the morning each one flowered or failed. The second is older and slimmer. It is the journal of tools and theories — the makers of the scales, the provenance of the seed stock, the lineages the manuals belong to, the words used in other gardens for the same plant. She consults the second more often than the first. The second is what allows her, in any given season, to read what the first is saying.*

---

**by Lukas — May 22, 2026**

## What this article is, and what it isn't

`workstation` is on. The desk fan is on. `nvidia-smi` in the right pane is at 41°C and idle — the single EVGA RTX 3090 has not been asked to do anything since this morning's eval-MCTS recalibration finished, and the next experiment in the queue, a reward-shaping sweep against the TASK_761 roadmap, is still being drafted in the next tmux pane. It is the late afternoon of May 22, 2026. I closed the diary arc of this series — Articles 0 through 13 — three days ago in *What a Year of Negative Results Looks Like*. That article was the closer. I meant it to be. I labeled it that way. And then I went back through the working tree this morning, sorting drafts by number, and realized I had closed the wrong half.

The series has two arcs. The diary arc — Articles 0–11, 13 — is the one Article 22 took inventory of: the campaign, kickback by kickback, the dead listed by name, the three lessons that survived. The second arc has been growing alongside the first for the last few weeks: a small annex of architecture essays in the 23–27 slot, plus a handful of Phase 1 deep cuts in the 14–21 slot that I wrote in the same register but that Article 22 — drafted out of sequence, ahead of those pieces — could not have known to include. Together that annex is a different *kind* of writing than the diary. The diary asks *what happened*. The annex asks *what the field had said about this before it happened, and what kind of object the project actually is.* Both are honest. Only one was lived. They need different closers.

This article is the closer for the second. It does not recap the diary; Article 22 owns that territory, and I am not going to dilute its accounting by repeating it here. What I am going to do is walk the five-essay arc — Articles 23 through 27 — in order, and say, for each, what it added that the diary could not. Then I am going to acknowledge the Phase 1 deep cuts that Article 22 missed for the simple reason that the dispatch sequencing put them after their own closer. Then I am going to land what the annex collectively says, what the meta-kickback of writing an annex looks like, and what changes when you read the two closers in sequence — 22 first for the season, 28 after for the field-guide.

The honest version of what an annex is for, before I start: *the diary by itself is dishonest in one specific way.* It records what happened, but it does not record what the work was set against. The work was set against a decade of demonstration-augmented RL literature, against a visualization research program in machine learning, against an internal commission of five different deep-research perspectives, against a design that did not ship, and against a financial-statistics word — *Alpha* — that I used 263 times without once defining. None of those contexts fit naturally inside a diary entry. They are too dense; they are too academic; they are too *placed* in fields outside the project. So they accreted, late, as architecture essays. And what they accrete into is what the annex is.

The Apothecary, in the epigraph, knows the trick of it. *The second journal is what allows her, in any given season, to read what the first is saying.* The annex is what allows the diary to be more than a story.

---

## The five-essay arc, in order

### Article 23 — Hindsight Optimal: where the oracle came from

The first thing I needed the annex to say was that the project is *locatable*. The diary, in its own register, reads like a one-off — the writer is alone with the codebase, the failures are particular to the experiment in front of them, every kickback has the texture of being discovered for the first time in this room on this machine. That texture is true; the diary is honest. What it is not is *complete*. The oracle that drives Foxglove, the imprint that warms the policy, the supervised-target-driven training regime as a whole — these are not inventions of this project. They sit inside a nine-year research lineage, eight arXiv papers and six reference codebases deep, and the lineage has its own history of working in some places and not transferring in others.

Article 23 walks that lineage. DQfD in 2017, DDPGfD the same year. EfficientZero, REDQ, TF-C. The LUPI paradigm (Learning Using Privileged Information) as the umbrella idea — the student sees only the past, the teacher saw the future when it computed the label, and the whole architectural problem is how to imitate the teacher without acquiring a structural dependence on the privilege the teacher used. From the article:

> *"**Hindsight-optimal**: a training regime in which the teacher signal is computed with information the agent at inference time does not have. The student sees only the past. The teacher saw the future when it computed the label. The training task is to imitate what the teacher chose without acquiring a structural dependence on the privileged information the teacher used to choose it."*
> — `/home/lukas/Trader/bloom/drafts/23_hindsight_optimal_research_lineage.md`

What that adds, that the diary could not, is a *frame*. The campaign verdict from Article 10 — fourteen evaluations, zero promotions, single-mechanism class exhausted — reads in the diary as the project's own failure. Article 23 reframes it: the lineage is correct in theory, demonstrated to work in its home domains (Atari, robotics, biomedical time series with TF-C), and the transfer to this stack is *bottlenecked at the teacher's quality bar*, not at any of the mechanism-level interventions the campaign tried. The campaign is not a failure of method; it is a failure of *transfer*. That is a different claim than "we couldn't make it work." The lineage is alive; the transfer is open; TASK_761's richer-oracle direction is the explicit attempt to lift the teacher's ceiling.

The diary by itself could not say this. The diary's grammar is per-experiment, per-kickback, per-seed. The annex's grammar is per-lineage, per-decade, per-field.

### Article 24 — Visualizing MuZero Latent: map versus territory

The second thing I needed the annex to say was that the *instruments* of the project are themselves objects of study. The diary treats them as transparent: the dashboard shows what's happening, the metric measures the thing, the visualization renders the latent. None of those framings survive contact with the work the annex's second essay does.

Article 24 walks the DS24 dashboard against two long deep-research reports in `docs/aires/` — about a hundred pages between them on how to visualize the latent space of a MuZero-style agent in JAX. The article's central observation: *DS24 implements about thirty percent of what those reports propose.* Not because the rest is wrong — it is mostly right — but because the rest is either too expensive to run online, or has unclear diagnostic value for the campaign's actual failure modes, or, in a few cases, is a technique I could not picture catching anything Poppy was doing wrong. The dashboard is a curated subset, and the curation was itself an experiment.

The kickback the article lands is the one that matters here:

> *"the most consequential bug of the campaign — the better-classifier-worse-trader anti-correlation of Article 6 — was found by aggregating metrics across runs, not by squinting at any single latent projection. […] The pattern was global, across runs. The visualizations were local, per run."*
> — `/home/lukas/Trader/bloom/drafts/24_visualizing_muzero_latent.md`

What that adds is a category. Visualization, as an instrument, has a *resolution scale*. DS24 was excellent at finding local pathologies — at telling me whether a single run's latent had collapsed, whether the policy was confident in a regime where it should have been uncertain, whether the dynamics were rolling forward stably. It was structurally blind to *cross-run patterns*, because each panel is a projection of one run. The campaign-defining failure of the year — the anti-correlation between classifier sharpness and deployment Alpha — only existed in the *comparison* between runs. No amount of artful latent projection on any single experiment would have surfaced it. The diary records the bug; the annex records that the *instrument was wrong-scale for the bug* — a finding about the instrument, not about the bug.

That is the map-versus-territory move. The diary lives on the territory. The annex measures the map.

### Article 25 — Five Models, Same Failure: methodology, not productivity

The third thing the annex says, and the one I most want to land carefully, is that *multi-model collaboration is a research methodology*. Not a productivity tactic. Not a hedge. Not an ensemble trick to squeeze a few percent out of an output. A *methodology*: a structured way of producing knowledge that exploits the fact that different models, trained on different corpora with different objectives, have different blind spots, and the *union of their reviews* covers more ground than any one of them alone.

Article 25 documents the week in mid-November 2025 when I knocked on five doors — Gemini Deep Research, DeepSeek (DS3), Claude CLI, Gemini CLI, Kimi — with the same prompt on the dual-collapse phenomenon Poppy was producing. Same prompt, deliberately neutral about causality, asking each to diagnose the pathology. Five reports came back. The reports were *not* the same.

> *"Five reports, one stack. The synthesis is multiplicative. […] The compass synthesis would not have been writable from any single report. It is, structurally, a multi-source document — and the reason the synthesis is more useful than any individual report is that the perspectives are partially independent. They are not redundant. They are partially redundant, and the non-redundant slices are the parts I would have missed otherwise."*
> — `/home/lukas/Trader/bloom/drafts/25_five_models_same_failure.md`

The TASK_761 four directions — reward shaping, entropy regularization, oracle k-reformulation, the incompatible-dataset retrain — are the synthesis. They were not handed to me by any one of the five models. They were assembled by reading all five against each other and asking what survived all five lenses. The disagreements between the models, which I had been tempted to skim past as noise, turned out to be the data about *where the field itself is unsettled* — and where the field is unsettled is where the experiments I run next will be paying the rent on what I do not yet know.

The diary does not — cannot — record this. The diary records *what was done* with the synthesis. The annex records *how the synthesis was made*, which is a different kind of knowledge, and one I want available the next time the campaign hits a wall.

### Article 26 — Hybrid LLM Reasoning: the design that stayed in the drawer

The fourth thing the annex says is that *not every elegant design is worth building*, and that the diary's natural narrative bias — toward the things that did get built, even when they failed — leaves the unbuilt designs underclaimed. Article 26 is the architecture essay on the prettiest system design I never shipped: a hybrid LLM reasoning module that would have wired a small frozen language model into Poppy's planning loop, with a tiny trained orchestrator deciding when and how to invoke it.

The design document, `docs/HYBRID_LLM_REASONING_DESIGN.md`, is 3376 words and has two halves. The first is the practical hybrid I sat down to specify. The second is the philosophical inversion — *Dense Latent Mind* — that the same document grew up into a month later. Both halves are interesting. Neither half is in the codebase. From the article:

> *"This article walks the design, walks the math of the integration I would have built, walks the three layers of kickback that kept the seed in the drawer, and ends with the deeper kickback that closed the case retroactively: by May 2026 the bottleneck in the system was demonstrably not in the reasoning side. An LLM augment, however elegantly wired, would have been a beautiful solution to a problem the project did not have."*
> — `/home/lukas/Trader/bloom/drafts/26_hybrid_llm_reasoning.md`

The kickback has three layers and a fourth that closes it from underneath. Latency: the trader's decision loop is in milliseconds, an LLM forward pass at int4 on a 3090 is in seconds, four orders of magnitude. Cost: even granted that the LLM runs offline, the per-asset-per-day inference budget would halve the throughput of the whole project for an expected lift in the single-digit-Alpha range. The architectural sleight that makes the latency work — caching the LLM's output as a static prior — strips the LLM of the property that made it worth using. And the deeper kickback, the one Articles 6 through 15 collectively built, is that the bottleneck is *downstream of the encoder* — which is to say, downstream of anywhere the LLM augment could have intervened.

What this adds, that the diary could not, is a *category for the unbuilt*. The diary's structural pressure is toward the experiments that ran; the annex makes space for the experiment that didn't, and asks why. Article 26's answer is the most uncomfortable kind of "no": *not because the design is wrong, but because the design is wrong about its own importance.* The diary cannot enforce that kind of clarity about an unrun experiment. The annex can.

### Article 27 — What "Alpha" Actually Means: metric discipline

The fifth thing the annex says is the one that, in retrospect, I should have said in Article 0. Article 27 is the field survey of the word *Alpha* itself — the audit of a noun I used 263 times across the drafts and 187 times in its delta form, **Δα**, without once writing down what the noun *is*. From the article:

> *"the word Alpha appears 263 times across the drafts in `bloom/drafts/`. The notation Δα appears another 187 times. […] And in fifteen articles I have not, once, written down what Alpha is."*
> — `/home/lukas/Trader/bloom/drafts/27_what_alpha_means.md`

The article walks from Jensen 1968 — the textbook regression definition, alpha as the residual intercept after netting out beta — through CAPM, through Fama-French factor extensions, through the practitioner usage that drops the regression and just reports excess-over-benchmark, to the project's own usage. The project's Alpha is *none of the above and partly all of them*: a paired Δα between two checkpoints, evaluated under a specific operator (eval-MCTS sims=16, as of May 20 — argmax before that, see Article 10), over a specific benchmark window, with a specific unit convention (percentage points of cumulative log-return). Five qualifiers. None of them disclosed in most of the article bodies.

The kickback lands in three layers, and the third is the one I have been carrying since I read it:

> *"the campaign verdict is a function of the chosen metric. […] If I had kept the argmax operator, 746 oracle promotes. If I had lowered the promotion bar to 'mean ≥ 0pp,' 734 flow-matching promotes. If I had used a Jensen-style risk-adjusted alpha, several variants whose Δα was raw-positive would have negative Jensen alpha after netting out the higher beta exposures they learned."*
> — `/home/lukas/Trader/bloom/drafts/27_what_alpha_means.md`

What this adds, that the diary could not, is *metric discipline*. Every result table in the series — every "Δα = +6.21%" and every "Δα = −18.44pp" — is conditional on a metric definition that I was writing as if it were a constant. It was not a constant. It drifted across the project's lifetime; it differed across articles in the same series; it differed across pairs of variants within the same article. The campaign's central verdict (fourteen mechanisms, zero promotions) is *also* conditional on the chosen operator and the chosen promotion bar — and the article shows, explicitly, which alternative metrics would have produced different verdicts.

This is the annex doing what the diary cannot do at all: turning around and inspecting its own units.

---

## What the annex collectively says

Five essays. Three claims that no single one of them states but that all of them, together, make.

**The first is that the project is locatable in a research lineage.** Article 23 places Foxglove inside DQfD/DDPGfD/EfficientZero/LUPI; Article 24 places DS24 inside the visualization research the deep-research reports surveyed; Article 25 places the dual-collapse diagnosis inside the published literature on policy-collapse and auxiliary-head trivialization; Article 26 places the unbuilt hybrid design inside the LLM-augmented-RL space; Article 27 places the metric inside Jensen 1968 and its descendants. The project is not a one-off. It inherits a decade of method, and its limitations are partly the *lineage's* limitations — not failures of the writer's engineering, but properties of how the methods transfer (or don't) across substrates. This claim changes the meaning of the campaign verdict from Article 10. *Fourteen mechanisms, zero promotions* reads, in the diary, as a local failure of the project. Read alongside the annex, it reads as a particular instance of a known transfer problem the field has been negotiating since 2017. The same shape lives in other people's failed transfers; mine is data for the field.

**The second is that the project's instruments are themselves objects of study.** Article 24 makes this most plainly — the dashboard is a curated subset of a larger visualization space, and the curation has consequences. Article 27 makes it more sharply still — the *metric* is curated, the metric has drift, the metric's choice silently determines the verdict. Articles 23 and 25, in their own registers, do the same move: the lineage's methods are themselves instruments, with assumptions and failure modes; the multi-model synthesis is an instrument, with the property that the *disagreements* between models are information about the field's uncertainty rather than noise to filter out. Treating instruments as transparent is a category error: it makes the writer believe the instrument is just delivering the world, when in fact the instrument is *making decisions* about which slices of the world to deliver. Treating instruments as opaque is a different category error: it produces paralysis. The discipline the annex argues for is the middle one — treat the instrument as a *first-class object*, with its own provenance, its own assumptions, its own scale of resolution, and its own list of things it cannot see.

**The third is that multi-model collaboration is a research methodology, not just a productivity tactic.** Article 25 is the explicit version of this; Article 26 is the implicit version (the hybrid design was *itself* the question of how a model and a planner collaborate at inference time, which is methodologically adjacent to how I collaborate with five models in the deep-research register at design time). When five models disagree, the disagreement is data about where the field itself is unsettled. The non-redundant slices of partially-redundant reports are exactly the parts that any one model alone would have hidden. This is a different relationship to LLMs than "they help me code faster." It is a relationship in which the models are *epistemic peers with different blind spots*, and the methodology is to harvest the blind spots' complement rather than to pick one and trust it.

Three claims. None of them statable inside a diary entry, because diaries are local and these claims are *meta*. The annex's whole grammar is meta — meta-to-the-lineage, meta-to-the-instrument, meta-to-the-collaboration. The diary is the season; the annex is what the season was set against.

---

## *The Apothecary moves the second journal*

> *The second journal survives the year longer than the first. The first goes to the publisher, where it will be read once by people who do not garden, and once again by the gardener herself in a winter when the next season's planning will not begin. The second goes back to the shelf, where she will reach for it without looking on the morning of the next disease she cannot name. The first journal is a record. The second is preparation. She files them next to one another, in alternating colours so that the spines do not blur.*

---

## And these too — the Phase 1 deep cuts Article 22 missed

I owe a section to the Phase 1 deep cuts that Article 22 could not list because they were drafted after it. They are the Phase 1 equivalents of the annex: detailed architecture and process essays that didn't fit the diary's per-kickback grammar, that I wrote in the same register, and that belong on the same shelf as Articles 23–27. Article 22 took inventory of the diary; this section takes inventory of the inventory's gaps.

**Article 14 — The Sign Bit That Ate a Quarter.** Late January 2026. A reward signal that looked like a sign bug, a thirteen-sub-task chase across `tasks/TASK_126_REWARD_SPIKE.md`, a culprit that turned out to be a hard-coded 10-bp transaction cost punishing the oracle for being an oracle. Article 22 catalogued the variance, measurement, causality, estimation, and substrate categories of kickback. Article 14 belongs in a sixth that 22 named only obliquely: *assumption decay* — the failure mode where a default that was right when written becomes wrong when the surrounding code grows around it. The transaction-cost line in `path_solver.py` did not change; what it was being asked to do, did. The annex placement is *assumption-decay-as-a-class*, distinct from the procedural-discipline category that 22 already named.

**Article 15 — What Foxglove Actually Does.** The training-side companion to Article 6's eval-side observation. Article 6 said *better classifier, worse trader*; Article 15 walks the math of the imprint that produced the classifier — Causal Expert Imprinting, the constructed-oracle imitation objective, the precise way training-side perfection fails to translate to deployment Alpha. Article 22 named the anti-correlation as a structural kickback. Article 15 explains what the imprint actually is and what the kickback *means at the loss-function layer*. The diary records the symptom; 15 records the mechanism.

**Article 16 — Bella: A Diffusion Model for Markets That Don't Exist.** Article 9 in the diary was the *operational* disaster of Bella's precompute — the 93 GPU-days estimation kickback. Article 16 is the *architectural* essay I should have written first: what a diffusion model is, what the U-Net + temporal-attention generator architecture looks like, why the math is correct, and why a good generative model of X is not the same as a useful teacher for an agent acting on X. The annex placement is *generative-fidelity-and-downstream-utility-are-orthogonal*, which is a structural lesson Article 22's substrate category gestures at but does not name.

**Article 17 — The Five Oracles.** The lineage of the supervised target inside this codebase. Five definitions over six months, all still in `path_solver.py`, none yet sufficient to lift deployment Alpha. Article 22 named the substrate kickback as "the recipe was right, the soil refused it." Article 17 narrows that diagnosis: *the soil refused this particular teacher five times in a row, and the question of whether a richer teacher (Oracle 6 in TASK_697) breaks the ceiling is still open*. The annex placement is the project's own internal lineage, alongside Article 23's external lineage.

**Article 18 — Teaching the Simulator to Hurt.** The realism work of TASK_047 through TASK_058 — execution friction, capital friction, temporal friction, information friction. The simulator's job is to teach the agent against *resistance*, because an agent that trained against a frictionless market has learned the simulator, not the market. Article 22 listed the substrate as hostile; Article 18 names *the substrate work itself* as the place where the hostility is dialled in deliberately, and as the place where there is a structural floor below which no simulator can be honest. The annex placement is *simulator-as-instrument*, parallel to Article 24's dashboard-as-instrument.

**Article 19 — The Mining Card That Wasn't a Mining Card.** The CMP 90HX saga: the cryptocurrency-mining variant of the GA102 die that I tried to press into ML service for three months in early 2026, the firmware and PCB-level mutilations NVIDIA applied to enforce its non-use, the partial unlock, the throughput costs, and the moment in May the card came out and went back to the mining chassis. Article 22 noted in passing that the project ran on a single 3090. Article 19 is the long version of how I learned that *the single 3090 was the load-bearing fact* — not because the 90HX experiment failed, but because the costs of working around the 90HX's crippling were larger than the gains the card delivered. The annex placement is *hardware-as-constraint*, made specific.

**Article 20 — The Dojo.** The training environment as a piece of architecture. The seventeen-regime carousel, the orchestration in `tools/orchestration/e-dojo_v1.py`, the curriculum design with twenty-three levels mostly unimplemented, the moment in May 2026 when the campaign verdict made me realize the dojo had become so well-conditioned that *it could no longer tell my mechanisms apart*. Article 22 listed the campaign as exhausted; Article 20 places the exhaustion partly in the environment — the section every RL paper skips, doing most of the load-bearing work in this stack.

**Article 21 — Yarrow.** The parked side project — the multi-model code reviewer designed in `~/Trader/yarrow/`, never implemented as code, whose method was nevertheless absorbed into the cold-eyes audit pattern that became Article 12's TASK_115 fifteen-subagent review. Article 22 did not mention Yarrow because Yarrow does not appear in any of the diary's per-kickback narratives. Article 21 places it: a side project that survived through its method even when its code did not, an unusual fate that the diary's grammar has no slot for. The annex's grammar has the slot.

Eight Phase 1 deep cuts. Each one of them: *not* a recap, *not* a re-litigation of the diary's failures, but a placement — an essay that asked, of one piece of the project, *what kind of object is this and what lineage does it belong to.* Together they are the Phase 1 wing of the annex. Articles 23–27 are the campaign-wide wing. They belong on the same shelf.

---

## The meta-kickback of the annex itself

The diary's meta-kickback, the one Article 22 closed on, was the question of whether writing the diary was itself a form of conviction-management — whether the act of narrating the negative results into categories produced the *feeling* of having learned in the absence of the learning. I left that question open. I am leaving this article's version of it just as open, because the annex has its own meta-kickback, and it is sharper.

*Writing an architecture annex risks turning a diary into a textbook.*

The annex articles are denser, more academic, more *review-paper-with-commentary* than the diary articles. They cite. They place. They draw lineages. They survey four definitions of Alpha and walk a decade of demonstration-augmented RL papers in chronological order. The risk this register carries — and I want to name it plainly, because the annex is the most likely place in the series for the writer to deceive himself — is that the annex *makes the work look more legible than it actually was*. The diary was the truth-teller. The annex is the explanation. Both are honest about what they are doing; only one was lived. The danger is that a reader who reads only the annex, and not the diary, comes away with the impression that the project was a coherent research program executed against a known literature — which is exactly the *retrospective coherence illusion* that the diary, in its per-kickback grammar, was structurally honest about not delivering.

The second layer of the meta-kickback: *the annex is what the project looks like from outside.* The diary is what it looks like from inside. Most readers will want one or the other, depending on which of the two questions they came in with. The reader who wants to know *what happened* wants the diary. The reader who wants to know *what kind of object the project is* wants the annex. A few readers will want both. The honest stance — the one I want to hold — is that *the work is the diary; the annex is the apology for the diary's missing context*. Not apology in the sense of self-effacement; apology in the older Greek sense, *apologia*, the defense of a thing by placing it among the things it can be defended next to.

If the next campaign — TASK_761's structural-pivot directions — produces a kickback whose category is *not* in any of the annex's frames, then the annex was rhetoric. If the next campaign produces a kickback that the annex's frames *did* place, then the annex was lesson. As with the diary's meta-kickback, I do not yet know which. The annex earns its place on the shelf or not depending on what the next disease looks like and whether I reach for it without looking.

---

## What both arcs say in chorus

Read 22 and 28 in sequence and the texture of each changes.

The diary's kickbacks, taken alone, are particular: this seed was an upward outlier, this metric anti-correlates with that one, this estimation was off by three orders of magnitude. Read against the annex, they become *examples of the annex's claims*. The single-seed bolded NEGATIVELY of Articles 7–8 is an instance of *measurement-instrument-without-disclosed-qualifiers* in Article 27's frame. The classifier-Alpha anti-correlation of Article 6 is an instance of *the lineage's training-side metrics not predicting deployment Alpha* in Article 23's frame. The eval-MCTS asymmetric lift of Article 10 is an instance of *the operator is part of the metric* in Article 27's frame. The dashboard that did not catch the anti-correlation, in the diary's narrative, becomes in Article 24's frame *the instrument was wrong-scale for the pattern*.

And the annex's claims, taken alone, are abstract: the project is locatable, the instruments are objects of study, the methodology is multi-source. Read against the diary, they become *grammars* that the diary's specific failures spoke. The lineage's transfer limits are *fourteen mechanisms producing null deltas*. The instrument-as-object-of-study is *DS24's blindness to cross-run patterns*. The multi-source methodology is *the four directions of TASK_761*. The annex's grammar is what the diary's particulars are sentences in.

Neither is complete on its own. The diary, alone, looks like a series of unfortunate incidents. The annex, alone, looks like a research review-paper that hasn't been forced to live. Together they describe what the work actually was: a year of negative results, set against a decade of method, executed on a single 3090 in a room with a desk fan, with instruments that had to be argued with as much as the experiments did.

The diary is the season. The annex is the field-guide. They were always two journals on the same shelf. The first season has produced both. The next season will start, if it starts, with the second journal already open.

---

## Closing

The second journal closes for the season. The diary, in the sense of the per-day lore files in `~/Trader/lore/`, continues — `workstation` is on as I save this draft, the next experiment's SPEC is half-drafted in the next pane, and there are still measurements to be made against the structural-pivot directions through Q3. The series, for real this time, ends.

The series ends, but the diary doesn't. The annex closes, but the lineage doesn't. If a structural result lands in the third quarter that has the shape of an article — a kickback, a texture, a lesson that earns its space — there will be a second season, and the second season will likely have its own annex, because the work has produced one once and that is now the shape of how it processes itself. If no structural result lands, then the silence will be the post-script, and the existing two arcs will be what the season was.

I file the journals together. The harvest journal next to the lineage journal. The spines are different colours so they do not blur. I label the shelf with the year. I turn out the lamp.

> *The Apothecary places the two journals beside one another on the bench. She weighs nothing; she compares nothing; she does not rank one above the other. She knows only that the next season's question will, in some morning she cannot predict, require both. She closes the cabinet. She does not lock it.*

---

## Sources

- Article 14 — *The Sign Bit That Ate a Quarter* — `/home/lukas/Trader/bloom/drafts/14_sign_bit_that_ate_a_quarter.md`
- Article 15 — *What Foxglove Actually Does* (Causal Expert Imprinting) — `/home/lukas/Trader/bloom/drafts/15_causal_expert_imprinting.md`
- Article 16 — *Bella: A Diffusion Model for Markets That Don't Exist* — `/home/lukas/Trader/bloom/drafts/16_bella_diffusion_for_markets.md`
- Article 17 — *The Five Oracles* — `/home/lukas/Trader/bloom/drafts/17_five_oracles.md`
- Article 18 — *Teaching the Simulator to Hurt* — `/home/lukas/Trader/bloom/drafts/18_teaching_simulator_to_hurt.md`
- Article 19 — *The Mining Card That Wasn't a Mining Card* — `/home/lukas/Trader/bloom/drafts/19_mining_card_that_wasnt.md`
- Article 20 — *The Dojo: How Foxglove Sees the World* — `/home/lukas/Trader/bloom/drafts/20_the_dojo.md`
- Article 21 — *Yarrow: When the Code Reviewer Is the Code* — `/home/lukas/Trader/bloom/drafts/21_yarrow_code_reviewer.md`
- Article 22 — *What a Year of Negative Results Looks Like* (the diary closer) — `/home/lukas/Trader/bloom/drafts/22_year_of_negative_results.md`
- Article 23 — *Hindsight Optimal: The Research Lineage Behind the Oracle* — `/home/lukas/Trader/bloom/drafts/23_hindsight_optimal_research_lineage.md`
- Article 24 — *Visualizing a MuZero Latent Space* — `/home/lukas/Trader/bloom/drafts/24_visualizing_muzero_latent.md`
- Article 25 — *Five Models Diagnose the Same Failure* — `/home/lukas/Trader/bloom/drafts/25_five_models_same_failure.md`
- Article 26 — *Hybrid LLM Reasoning: A Design That Stayed in the Drawer* — `/home/lukas/Trader/bloom/drafts/26_hybrid_llm_reasoning.md`
- Article 27 — *What "Alpha" Actually Means — A Field Survey* — `/home/lukas/Trader/bloom/drafts/27_what_alpha_means.md`
