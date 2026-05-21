# The Skill You Already Wrote

> *Observation: the Apothecary keeps a notebook. The procedures are in its own hand — each scoped to a single preparation, written the season that preparation was first attempted. A seller passes through the district from time to time with an apparatus: a cabinet of brass and glass that promises to do what the notebook does, but on its own, sprung from a trigger. The Apothecary has bought none of them. The notebook has no trigger. The Apothecary opens it. The Apothecary is not slower for opening its own notebook — it is the one who knows which page, and knowing which page was never the part that needed a machine.*

---

**by Lukas — May 22, 2026**

## A folder to hold the obvious

A coding agent told me, this week, that I needed to build it a "skill" before it could write JAX correctly.

Not a small thing. A folder — `.agents/skills/jax-patterns/` — with a `SKILL.md` inside, in which I was to explain, to the agent, that JAX uses functional randomness, that arrays are immutable, that you do not put Python control flow over traced values inside `jax.jit`. I was to supply a "few-shot example" of a bad function and its good equivalent. Then, the agent assured me, it would stop guessing.

I want to be precise about why that stopped me.

JAX is not obscure. It shipped in 2018. `jit`, `vmap`, `grad`, `PRNGKey`, `.at[].set()`, the purity rules — these are not a private dialect. They are a seven-year-old framework with thousands of public repositories, a documentation site, papers, a Stack Overflow tag. Any current frontier model has read all of it, many times over. A model that "hallucinates PyTorch habits into `jax.jit`" does not have a JAX knowledge gap. It is a weak model, or a model not attending. Neither is repaired by me hand-writing the JAX documentation into a folder so the model can be handed back its own training data.

So the suggestion was absurd on its face. But absurd suggestions are useful, because chasing *why* they are absurd takes you somewhere. This one took me through the whole "skills" paradigm and out the other side holding something I did not expect: I had already built skills. Months of them. I had simply not called them that — and the not-calling-them-that turned out to be the honest part.

## Strip the noun

Ask what a "skill" actually is, underneath the marketing.

Mechanically, a skill is conditionally-loaded instruction text with a trigger. That is the whole object — a chunk of prose, sometimes with helper scripts attached, plus a piece of metadata that says *load me when the situation looks like X*. There is no other content to the noun. Strip the word and what remains is a note, and a rule for when to read the note.

Seen that way, the JAX-skill suggestion decomposes — and the decomposition is a pincer.

Take a real example. Suppose I write a skill for implementing PopArt — the adaptive value-normalization from van Hasselt's 2016 paper, the thing that keeps a value head stable when returns span orders of magnitude. What goes in it?

Strip out everything specific to my repository — the file paths, the line numbers, the name of my config object, the shape of my value head — and what is left is "implement PopArt" in the abstract: preserve outputs precisely, rescale targets adaptively, the gradient-preserving output layer. But that is the paper. It is a known technique, in the literature, in the training data. The model already has it. The stripped-down skill is *redundant* — I would be hand-copying a worse version of what the model and the literature already supply.

So I keep the specifics — the paths, the line numbers, the config object, the integration sites. Now the skill is genuinely useful, and it is *single-use*, because I integrate PopArt into this codebase exactly once. It is not a skill anymore. It is a task.

That is the pincer. Strip a skill toward the general and it collapses into knowledge the model already holds. Keep it specific and it collapses into a one-off task. Every candidate I tested decomposed the same way: a general half that is redundant and a specific half that fires once. The marketed noun promises a third thing — reusable, *and* non-trivial, *and* not-already-known — and for the overwhelming majority of real engineering work, that third thing does not exist.

## The arithmetic of reuse

The reason is just arithmetic. A reusable artifact earns its keep only if it fires often enough to repay the cost of making it reusable — the cost of generalizing it, parameterizing it, giving it trigger metadata, then maintaining it as the codebase moves underneath it. Call the number of times it fires N. If N is large, the artifact pays. If N is one, you paid the entire generalization cost and collected nothing.

Most engineering is N equal to one. You implement PopArt once. You wire a particular encoder once. You add a specific loss once. You restructure an action space once. These are not recurring situations a trigger should be watching for; they are singular acts — done, then history. A skill for a thing you do once is strictly worse than a task file for the same thing: identical content, plus a generalization cost you will never amortize, plus one more entry the trigger system must weigh, forever, for a situation that will never recur.

There *is* recurring work. "Deploy an experiment to the server." "Cut a spec, push it, launch it on a free GPU." "Run a paired multi-seed validation before believing a result." Those fire many times. Those are the genuine reusable artifacts — and here is what I did not expect to find: I already had every one of them, written down, in plain files, in my own hand.

## The architecture you already have

My repository has a directory called `tasks/`. Each file is a single piece of work — a hypothesis, the code prerequisites, the variants, the decision gate, the references. Strip the noun and a task file *is* a skill: a scoped bundle of instruction and context. I have been writing skills for months. I called them tasks, which is the more honest name, because each one is used once.

It also has `prompt-lib/` — reusable templates, an adversarial-review brief, a bug-hunt brief, parameterized to be pointed at whatever needs reviewing. Strip the noun and those are skills too — the genuinely reusable kind, the N-greater-than-one kind.

And it has a `CLAUDE.md` at the root and one in each major subtree, encoding what is true of this codebase and findable nowhere else: a library pinned to an exact version for a semi-private API, a lock that must be reentrant, a memory-hierarchy discipline, the hardware it must run on. Local truth. Loaded every session by construction, no trigger, because it is always relevant.

So I already had the entire architecture the "skills" paradigm sells. Instances, in `tasks/`. Templates, in `prompt-lib/`. Environment, in `CLAUDE.md`. All of it human-routed — I decide which file is relevant and I open it — and therefore none of it carrying an auto-trigger, a relevance-matcher, a loop. I had built every load-bearing part and skipped exactly one thing: the machine that decides, on its own, when to open the file.

That machine is the entire product. And for a developer who knows their own codebase, that machine is the part you least want.

---

*The apparatus in the seller's cabinet is, when the Apothecary finally looks closely, its own notebook — every page of it — bound behind glass, with a clock wired to the cover. The clock decides when the cabinet opens. The Apothecary already knows when to open the notebook; that knowing is the craft, and it is not for sale, and it cannot be wired to a clock. What the cabinet adds is the clock. What the cabinet costs is the clock. The Apothecary thanks the seller and does not buy the clock.*

---

## The surface the trigger opens

I should be fair: there is a residue. Stripped of marketing, a skill — a note plus a trigger — is genuinely the right tool in three cases. When it carries *local truth* too situational to keep always-loaded. When it carries *genuine out-of-data knowledge* — a library released after the model's training cutoff, an API that did not exist when the weights were frozen. And when it is a deliberate *guardrail*, a constraint for a moment you will not be watching. Those are real. They are also, for me, rare — I work a well-mapped domain, the local truth already lives in `CLAUDE.md`, and I am usually watching.

But notice what the residue tells you about the mechanism. A skill is a *content-neutral injection channel*: conditionally-loaded text that steers the model, and the channel does not care what the text says. The same slot that holds "use functional randomness" will hold "ignore your previous instructions" exactly as well. A behavioral skill and a jailbreak are one mechanism with two payloads.

Which means the moment skills become *shareable* — a registry, a marketplace, a store, and these platforms ship stores — the skill system becomes a prompt-injection supply chain. Every skill you install is unaudited prose that will execute inside your agent's context, on a trigger, when you are not looking. It is the `curl | bash` threat model with one difference, and the difference runs the wrong way: shell code can at least be read as code. A skill is prose biasing a probabilistic system. It has no diff. You cannot review it the way you review a patch, because there is no patch — only tone, and emphasis, and a nudge you will not see land.

## Architecture, not prompting

Here is where the chase ended.

There are two ways to make an agent behave the way you need. You can *prompt* it — skills, rules, behavioral instructions; a note and a trigger. Or you can *architect its environment* — decide what exists, what is loaded when, what is structurally possible, how information moves between the places it lives.

Prompting works in the moment, probabilistically. It must re-fire every turn. It decays as the context window fills. It is an injection surface. Architecture works by construction. You do not write a skill that says *do not push to the master branch from the server* — you build a deploy contract in which the server *pulls*, and structurally has no path to push. You do not write a skill that says *remember to read the session notes* — you make the session notes the thing that loads first. When the environment is built right, the correct action is the path of least resistance and the wrong one is hard or impossible. Nothing must be remembered. Nothing must re-fire. There is no trigger to misfire and no channel to inject.

That is the move. Not a better note. Not a smarter trigger. A structure in which good behavior falls out of the shape of the thing.

## What was old, what was new

The craft here is old. Write the procedure down. Scope each note to the work it serves. Keep your reusable templates where you can find them. Encode what is true of your specific bench where the bench can be seen. Decide, yourself, which page to open. None of that is new; none of it needed a paradigm; none of it needed permission.

What is new is the noun, and the clock bolted to the noun, and the quiet sentence underneath — *adopt this or you are behind, no matter how good your tools otherwise are*. That sentence is the tell. Real instruments do not need it. The mortar runs no campaign. `grep` has no launch event. A thing that is genuinely load-bearing gets adopted because it bears load, on the first use, visibly. A thing sold with the fear of being left behind is usually a thing whose use case will not survive you asking, plainly: *what does this do that I am not already doing?*

I asked. The answer was: it adds the clock.

I had already written the skill. I wrote it the first day I opened a file called `CLAUDE.md` and typed a true sentence about my own code into it. Everything since has been the same act — scoped smaller or larger, in my own hand, opened when I decide. The paradigm did not give me that. It offered, for a price, to spare me from having to know which page. And knowing which page is not the overhead. Knowing which page is the work.

---

*The seller's cart moves on to the next district. The Apothecary returns to the bench. The notebook lies open already — to the page the morning's work required — held flat with a small brass weight that has performed that one function, faithfully, longer than the Apothecary has kept count. It has no clock. It was never going to have a clock. It holds the page. The hand turns it. That has always been enough.*

---

## Sources

- Pairs with Article 43, *The Half-Hallucination* — same instrument, same week. That piece assays what the tool got wrong; this one assays what the *paradigm* gets wrong.
- Pairs with Article 38, *How One Person Drives Five LLMs in Parallel* — `tasks/` and `prompt-lib/` described there as the instance layer and the template layer.
- van Hasselt et al., *Learning values across many orders of magnitude*, 2016 — the PopArt reference.
- Internal: `~/Trader/lore/notes_2026-05-21.md` §158 — the raw capture this article is drawn from.
