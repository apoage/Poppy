# Article 21 — Yarrow: When the Code Reviewer Is the Code

> *Observation: yarrow grows along the edges of every bed, never the centerpiece, never the harvest the apothecary is paid for. The leaves are kept dry on a shelf above the bench, in a paper twist labeled in the older hand. The shelf is dusty. The twist is untouched for months at a time. When the apprentice cuts themselves opening a crate of glass, the apothecary reaches for it without looking. The shelf is not decoration. It is preparation for a wound that has not happened yet.*

---

**by Lukas — May 22, 2026**

## The wound herb

`workstation` is on, fan running quiet, single 3090 sitting at 41 °C between checkpoints. The window beside this draft is showing `nvidia-smi -l 2`, ticking through a validation pass on a Foxglove model I'm not going to ship. The campaign has been declared dead since Wednesday — every single-mechanism variant null or worse under eval-MCTS recal — and what I'm doing today is writing about the project that doesn't exist anymore. Or, more accurately, the project that exists fully on disk, works on paper, and has never been run.

Yarrow lives in `~/Trader/yarrow/`. Twenty-odd markdown files. Seven harvester prompts. An execution map. A backends doc. A `workspace/` directory that is empty except for a `.gitkeep` I added in April. No Python. No CLI. No invocation. The README ends with *"Implementation: not started. Parked pending bandwidth"* and the date next to "Conceptual design complete" is end of January 2026.

I want to write about it anyway, because the *idea* of Yarrow survived in something the Trader project does daily — the cold-eyes audit pattern that became the spine of Article 12. The tool is parked. The method is integrated. That's an unusual fate for a side project. Most parked side projects die undocumented in a directory you forget about until your disk is full. Yarrow's prompts are read every couple of weeks because they're the closest thing I have to a written-down theory of *how to get multiple sets of eyes on a piece of code without spending the human attention*. So this article is half eulogy, half acknowledgment that the design did its job through a back door.

The name is the load-bearing thing. The garden is full of medicinal plants — Poppy (the MuZero+TRM trader), Bella (the market generator that hallucinates beautifully), Foxglove (the supervised imprint that doses the cardiac muscle of training). Yarrow is *Achillea millefolium*, the wound herb, the one Achilles supposedly used on the battlefield. It doesn't cause anything. It stops bleeding. It controls the response. In a garden whose specimens can kill you if dosed wrong, yarrow is the thing you reach for after one of them already did. That's the whole metaphor. Yarrow is the tool you build because the *other* tools are powerful enough that you need a second opinion before you trust them with anything load-bearing.

## The origin moment

The thing that made me start designing Yarrow happened in late January 2026. I was deep in the TASK_136 pre-encoding work — the row-wise tile redundancy discovery, where the encoder output turned out to be 64× more memory than it needed to be because every position in a row carried the same vector. The fix was a few hundred lines of refactor across the encoder, the data loader, and the BC training loop. It touched three files I cared about and four I'd half-forgotten. I had asked Claude to do the refactor, and it did, and the diff looked plausible.

And I sat there at one in the morning staring at a diff that *looked plausible* and realized I had no idea whether it was correct.

This is the specific texture of the moment: I had been pair-coding with Claude for weeks on this same codebase. We were both in the trees, both familiar with the encoder's idiosyncrasies, both already invested in the patch landing. If there was a bug, neither of us was going to spot it, because we had the same blind spots. The cure for that is fresh eyes — a reviewer who hasn't been staring at the encoder for a week. And the only fresh eyes available to me at one in the morning were *other models*.

I had Gemini in another terminal. I'd been using it for Deep Research-style background reads since November. I copy-pasted the diff into Gemini. Gemini found a thing — a reshape that worked in the test case I'd shown it but wouldn't survive the actual batch dim during training. Genuine catch. *Not* the kind of thing Claude would have flagged because Claude had already convinced itself the reshape was fine three turns ago.

That was the seed. *Different models have different blind spots, and if I route the same diff through enough of them in parallel, the union of their reviews covers more ground than any one of them alone.* The next morning I started writing `vision.md`. The mathematician in me was happy: this is just ensemble methods applied to LLMs, and the ensemble gain comes from *uncorrelated errors*, not from any one model being best. Multiple weak reviewers, structurally forced not to see each other's output, then a clustering step to find what they agreed on. The wound herb.

## The design

The architecture is in `yarrow/vision.md` and `yarrow/architecture.md`. The runtime mapping — which I wrote later, in April, after Claude Code's sub-agent system had matured enough to make most of the original orchestration moot — lives in `yarrow/execution.md`. Together they describe what is essentially a small medical staff for code.

There's an **Apothecary** at the top — the orchestrator, Claude Opus, the brain with no hands. Their full constraint set, from `agents.md`:

> Powers: READ, DECIDE, SPEC. Constraints: NEVER edits source code. NEVER runs tests. NEVER implements — refines specs when the compounder pushes back. Only the user shifts gears (plan → approval → yolo).
> — `yarrow/agents.md`

Below the Apothecary, three categories of subordinate role. An **Apprentice** (cheap model — Haiku or Sonnet) whose only job is to take the raw findings from the harvesters and deduplicate them, cluster them, score them. They don't validate. They don't filter. They preserve. *"Bias toward inclusion (orchestrator filters, apprentice preserves)."* A **Compounder** (Opus again, or Gemini) who is the hands — they take an approved prescription (a `tasks/TASK_XXX_*.md` file, the existing project format, deliberately not a parallel system) and implement it exactly. They don't scope. They don't gold-plate. If the spec is ambiguous, they ask.

And then the harvesters. Seven of them, each named for a method of gathering, each tuned to a model whose characteristic failure mode covers a different blind spot:

| Name | Model | Territory |
|---|---|---|
| Wildcrafter | Gemini | Data flow, race conditions, edge cases |
| Simpler | Claude | Logic errors, JAX pitfalls, math |
| Root-Digger | GLM | Types, signatures, API misuse |
| Herb-Walker | Kimi K2 | Duplication, dead code, patterns |
| Distiller | DeepSeek | Algorithms, numerics, invariants |
| Physick | OpenAI | Security, resilience, error handling |
| Chronicler | MiniMax | Cross-file consistency, drift |

The naming is doing work the engineering doesn't have to. *Wildcrafter* forages the margins of the codebase, finds things in the unkempt corners. *Simpler* knows each plant by name — methodical, mechanism-first, the most trusted single voice. *Distiller* is the patient prover, the one who traces a function with concrete inputs until the contamination concentrates enough to see. *Chronicler* reads the long scrolls of the codebase looking for places where two files claim different things about the same constant. The names tell you what the role does without making you read the prompt.

There's a small detail that took me longer than it should have to land on: the orchestrator reads *clusters*, never raw findings. The Apothecary's context window is a precious resource — Opus is the most expensive model in the stack, and the whole pipeline collapses if I burn its context on seven near-duplicate descriptions of the same off-by-one. So the Apprentice exists almost entirely as a *context compression layer*. Seven harvesters might emit fifty findings between them on a thousand-line file; the Apprentice clusters those into maybe twelve issues, each one annotated with which harvesters contributed and at what confidence. The Apothecary reads twelve, not fifty. That's load-bearing for token economics, and it's the kind of constraint that only matters once you've actually paid for a few thousand-finding scans on someone else's API budget.

The harvesters never see each other's findings. That's load-bearing for a different reason. From `vision.md`:

> Every multi-agent system degrades toward one of two failure modes: (1) Groupthink — agents see each other's work, converge on the same answer. (2) Chaos — agents contradict each other, no mechanism to resolve. Yarrow prevents both through structural constraints.
> — `yarrow/vision.md`

The structural constraint is forced independence at the harvest stage. Each model gets a directive ("scan `mcts_trader/jax_agent/path_solver.py` for bugs"), gets a finding schema, gets read-only access to source, and writes structured JSON into `workspace/findings/`. They don't see each other. They don't see the orchestrator's earlier validation passes. They don't see what's already been prescribed. The Apprentice picks up the raw JSON and clusters by file+line overlap, with a semantic check, weighted by per-agent precision priors:

```
claude:    0.73   # methodical, mechanism-first
deepseek:  0.65   # patient prover
openai:    0.55   # broad system view
glm:       0.40   # literal, type-focused
minimax:   0.40   # cross-file
kimi:      0.35   # pattern-matcher
gemini:    0.30   # creative chaos, brilliant when right
```

These priors are starting estimates. The whole point is that they get updated empirically — `workspace/agent_stats.json` is the persistent precision log, updated after every Apothecary validation cycle. Over time the ensemble learns which voice to trust for which kind of finding. A Wildcrafter-alone hit at 0.8 confidence is *maybe* real (Gemini's prior is 0.30). A Simpler-alone hit at 0.8 is probably real (Claude's prior is 0.73). A Simpler + Distiller agreement is near-certain — two provers from different lineages independently arriving at the same diagnosis.

The runtime, when I went back to map it onto Claude Code primitives in April, fell out cleanly. From `execution.md`:

> | Yarrow role | Runs as | Location |
> |---|---|---|
> | Apothecary (Orchestrator) | Main Claude Code thread | N/A — it's you + main model |
> | Apprentice (Assistant) | Sub-agent | `.claude/agents/apprentice.md` |
> | Compounder (Executor) | Sub-agent | `.claude/agents/compounder.md` |
> | Harvester: Claude | Sub-agent, parallel-instanced | `.claude/agents/harvester-claude.md` |
> | Harvester: Gemini / GLM / Kimi / DeepSeek / OpenAI / MiniMax | MCP tools | `yarrow-harvesters` MCP server |
> | The loop itself | Skill | `.claude/skills/yarrow/SKILL.md` |
> — `yarrow/execution.md`

That was the moment the architecture stopped being aspirational. By April I could see the whole pipeline executable in Claude Code with maybe a week of work — one MCP server with seven tool endpoints, three agent definition files, one skill file. The `tasks/TASK_XXX_*.md` prescription format already existed in the project. The disk-as-bus design (`workspace/findings/`, `workspace/clusters/`, `workspace/tasks/`) needed no infrastructure. The whole thing could be invoked with `/yarrow scan mcts_trader/jax_agent/`.

I never wrote that week of code.

## What it caught when it ran

Yarrow as a full ensemble never *ran*. But the underlying pattern — send the same diff or the same file through multiple models, get independent reads, consolidate — I used informally a dozen times between February and May, mostly via copy-paste between terminals, sometimes via the cross-model harvester scripts that lived briefly in `yarrow/tools/`. Those informal runs are what convinced me the design was sound.

Three concrete catches stay with me. None of them is the kind of bug a single model on a single pass would have found.

**The reshape-that-worked-on-the-test.** TASK_136 again, the row-wise tile collapse. Claude wrote a reshape that passed the unit test and looked right on the toy tensor. Gemini, asked to review the same diff, traced the actual training batch dimension and pointed out that the reshape assumed `(B, T, A, F)` ordering but the data loader was producing `(B, A, T, F)` after the optimization. Different model, different blind spot. Claude had been thinking in the geometry of the original code; Gemini didn't carry that prior and walked the dimensions from the loader outward. Catch.

**The PopArt desync.** During the BUG_MAYHEM phase in late January, one of the entries in `tasks/BUG_MAYHEM_MASTER_REPORT.md` was the PopArt scale-shift drift between the value-head normalization and the agent's internal reward scaling. I'd missed it in two passes with Claude. When I ran the same files through a quick informal harvest with DeepSeek and GPT — the would-be Distiller and Physick — both flagged a "variance mismatch around the normalize/unnormalize boundary" in different language. Two harvesters, same root cause, different framing. The Apprentice (had it existed) would have clustered them. The Apothecary (had it existed) would have validated against the actual `popart_normalize.py` and produced a prescription. As it was, I did the clustering by eye in a markdown table and wrote the task spec myself.

**The doc-code drift on the oracle's path solver.** Sometime in March I had Kimi K2 read `mcts_trader/jax_agent/path_solver.py` against `lore/adr/0005-dir-acc-ceiling-decomposition.md`, which described what the solver was *supposed* to do. The Chronicler-shaped catch: the ADR said k=4 segments; the code defaulted to k=3 on one branch. Nobody had noticed because k was set explicitly in all the live call sites. The branch was a fallback for a config path that nobody used in production. It would have bitten me the first time I tried the k-sweep that TASK_761's structural pivot now wants. *Cross-file consistency*, exactly what MiniMax's Chronicler role was designed for.

There's a fourth one I want to mention because it's the failure case, and the failure case is informative. In late February I ran a quick informal harvest on a stretch of the replay buffer code. Three models, Claude + Gemini + DeepSeek. Gemini came back with a confident finding about a "race condition between the prioritized sampler and the buffer write path." It was wrong — there's no actual concurrency there, the whole pipeline is JAX-pmapped within a single process and the writes are sequenced by the training step. Gemini had pattern-matched to a generic distributed-replay bug and confabulated the JAX context. Claude and DeepSeek didn't flag it. The Apprentice, had it been clustering with ensemble weights, would have given the finding a low score (1-of-3, lowest-prior agent). I caught it by re-reading the actual code. The point isn't that Gemini failed — it's that *the structure of the ensemble assumed Gemini would sometimes fail this way*, which is why the precision priors are 0.30 on Wildcrafter. The ensemble is *robust to* Gemini being wrong here, because two other harvesters didn't see it and Gemini's own prior is low. That's the architecture working in its negative case.

Each of these required two things that single-model review doesn't naturally produce: a model that doesn't share my and Claude's recent reasoning context, and a forced-structured output that I could mechanically merge. The Yarrow design specified both. I got the value informally — the *method* — without ever building the *tool*.

The pattern that emerges from these three catches, which is also in `vision.md` more or less verbatim: *the ensemble isn't faster. It's less wrong.* Different models hallucinate differently. Gemini's failure mode is creative confabulation; Claude's is sycophantic agreement with the prior turn; GPT's is over-broad caution; DeepSeek's is over-confident proof-sketching. Each is wrong differently. The intersection of their *correct* findings is wider than any individual's. The intersection of their *errors* is narrow. That's the whole game.

---

> *The wound herb does not grow in a single spot. It seeds itself along every edge, beneath every other plant. When one is cut, the gardener does not have to walk far to find what stops the bleeding. It is already where it is needed, in every direction at once.*

---

## The kickback — why it's parked

There are two reasons Yarrow is parked, and I want to name both honestly because each is its own lesson.

The first reason is that the methodology won without the tool. By March, I had started using Claude Code's sub-agent system aggressively for the kind of cold-eyes review that Yarrow was designed to provide. The 15-subagent audit of the MuZero training pipeline (which becomes Article 12) was the moment this crystallized: I dispatched fifteen parallel sub-agents, each scoped to one component (replay buffer, MCTS, PopArt, optimizer, target network, loss function, LR schedule, and so on), each given a fresh context window with the role of "audit this one component cold, you have no other context, write a memo." The dispatch took one message. The results came back forty minutes later. The combined output found nine real issues, two of which I'd have shipped to the next training run if nobody had looked.

That was Yarrow's pattern — *forced independence, structured findings, consolidation by the orchestrator* — executed entirely with Claude sub-agents instead of seven external models. Cheaper. Faster. Lower coordination cost. *Less diverse*, yes — fifteen Claude instances share Claude's blind spots in a way that Claude + Gemini + DeepSeek + GPT would not — but in practice the diversity I lost from going single-model was smaller than the diversity I gained from going multi-*context*. Fifteen fresh contexts, each one ignorant of the others' work, turns out to be most of what I wanted from seven different models. The remaining marginal gain from the cross-model ensemble was real but not large enough to justify the build cost. And it didn't help that Claude itself was getting better — the precision priors I'd written down in January for Claude (0.73) were, by April, probably closer to 0.85, which moved the ensemble math against the multi-model design.

The second reason is the harder one to admit. Yarrow, built out, is a real product. Not in the side-project sense — in the *startup* sense. A polished multi-model code-review tool with a CLI, a CI plugin, a precision-tracking dashboard, an MCP server people could install — that is its own company. There are people building exactly this right now. The architecture in `yarrow/vision.md` would be perfectly viable as a YC application. And I have one of those already; I have Trader, which is the thing I am actually trying to build. Yarrow as a tool for *me* needs maybe a week of glue code. Yarrow as a tool with *users* needs six months of product work — auth, billing, hosted endpoints, multi-tenant findings DB, an opinionated CLI, a website, the entire stack of things that turn a working script into a thing strangers can use. I could see, around April, the moment where I would either commit to building that product and walk away from Trader for a quarter, or commit to Trader and leave Yarrow at its current state. I chose Trader.

So Yarrow exists, works on paper, has no user beyond me, and is parked. The repo is honest about it. The status note at the bottom of `CLAUDE.md` reads:

> Conceptual design: complete (end of January 2026). Runtime mapping: complete (`execution.md`, April 2026). Implementation: not started. Parked pending bandwidth; pipeline (Foxglove → Poppy) research takes priority.
> — `yarrow/CLAUDE.md`

That's the honest line. Not "abandoned." Not "shipped." Parked. The garden contains a shelf where it sits.

## The lesson it survived in

Here's what I think is the actual moral of Yarrow, the thing I want to be sure I write down before the directory gets stale and the prompts drift out of sync with the models they were tuned for.

A side project can fail two ways and one of them is much better than the other. The bad failure: you build the tool, nobody uses it, you forget it, it accumulates entropy until it's a museum piece nobody can run anymore. The good failure: the *idea* of the tool gets integrated into the thing you actually do, the tool itself never gets built, and you stop pretending the directory is a roadmap and start treating it as a written-down theory. Yarrow had the good failure.

The cold-eyes audit pattern that Article 12 documents is, in mechanism, the Yarrow pipeline minus the cross-model harvest. *Forced independence* — sub-agents with no shared context. *Structured findings* — each sub-agent writes a memo in a known format. *Consolidation* — the main thread reads the memos, validates against source, writes task specs. *No drive-by fixes* — sub-agents don't get Edit. *No scope creep* — the main thread doesn't implement, it dispatches. Read `yarrow/vision.md` and `yarrow/agents.md` and you are reading, almost verbatim, the rules the cold-eyes audit follows today. The naming is different. The model lineup is different. The skeleton is the same.

That is a healthier outcome than most parked side projects get. A side project usually leaves behind nothing but a folder. Yarrow left behind a working pattern that I run every couple of weeks against a different chunk of the codebase. The shape of it — "diverse perspectives → forced independence → noise reduction → validated decision" — is in my hands now, not just in markdown. When I dispatch a wave of audit sub-agents on a Wednesday morning, I am running Yarrow. I am running it in a degenerate form, single-model rather than seven-model, and the prescription format is the existing `tasks/TASK_XXX_*.md` rather than a parallel system, and the precision tracking is in my head rather than in `agent_stats.json`. But the spine is Yarrow's spine.

And — this is the part I want to be careful with, because it's easy to over-claim — the *cross-model* part of the design might still matter someday. Right now Claude's precision is high enough, and Claude Code's sub-agent ergonomics are good enough, that single-model multi-context is winning on cost/value. That isn't permanent. If Claude's edge over Gemini and DeepSeek narrows (or reverses), if API costs shift, if I find a class of bug where the cross-model ensemble dominates clearly — Yarrow's full pipeline is still on the shelf. The design isn't invalidated. It's *deferred*. The prompts are model-tuned and will need a refresh, the precision priors will need empirical calibration, the MCP server still needs writing. But the architecture survives. The wound herb is in the paper twist on the shelf, dry but not gone.

If I had to summarize the kickback: I designed Yarrow as a tool, used it as a methodology, and the methodology was enough. Causality kicked back here too, just gently — I assumed I'd need seven models to get the diversity I wanted; in practice fifteen fresh contexts of one model got me 80% of the way there at 5% of the integration cost. The design wasn't wrong about *what mattered*. It was wrong about *which axis of diversity mattered most*. Fresh context turned out to be more load-bearing than fresh model.

That's a softer kickback than most of the ones in this series. There's no funeral here. There's a piece of design that did its job by being absorbed into a daily practice rather than shipped as a product. I'd rather end a side project that way than ship one that nobody runs.

## Closing

`workstation` is still on. The fan is still running. The 3090 has finished its validation pass and is sitting at 38 °C, waiting for me to launch the next training run, which I will, after I save this draft. The `yarrow/` directory is one `cd` away. I haven't opened it for any reason but writing this article in over a month. I will probably not open it again until either a new model lands that meaningfully shifts the precision math, or someone — maybe me, maybe not — wants the cross-model code review product that the architecture has been waiting to become.

In the meantime, the wound herb is on the shelf. The shelf is dusty. The twist is intact. That is the correct state for it to be in.

> *The apothecary closes the cabinet. The yarrow remains where it has always been: dry, labeled, untouched, ready.*

---

### Sources

- `~/Trader/yarrow/vision.md` — the architectural rationale and the seven-harvester naming
- `~/Trader/yarrow/agents.md` — full role definitions and precision priors
- `~/Trader/yarrow/architecture.md` — system diagram, hard boundaries, storage layer
- `~/Trader/yarrow/execution.md` — runtime mapping onto Claude Code sub-agents / MCP / skill
- `~/Trader/yarrow/backends.md` — model availability (Z.AI, Cerebras), endpoint contracts
- `~/Trader/yarrow/CLAUDE.md` — status: conceptual design complete, runtime mapping complete, implementation parked
- `~/Trader/tasks/BUG_MAYHEM_MASTER_REPORT.md` — the PopArt desync mentioned in §"What it caught"
- `~/Trader/lore/adr/0005-dir-acc-ceiling-decomposition.md` — the k=4 vs k=3 doc-code drift in the path solver
- Article 12 (forthcoming) — the cold-eyes audit methodology that absorbed Yarrow's pattern
