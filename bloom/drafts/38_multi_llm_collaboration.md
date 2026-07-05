# How One Person Drives Five LLMs in Parallel

> *Observation: the gardener keeps five apprentices, and none of them can do the whole work alone. One knows the soils. One knows the weights of the seed-pans. One reads weather. One copies labels in a neat hand. One walks across the city to fetch a second opinion when a leaf turns. The gardener herself does nothing the apprentices could not, in principle, do better — except this: she knows which apprentice to send to which bed at which hour, and she remembers, between the beds, what each of them last said.*

---

**by Lukas — May 22, 2026**

## The article you're not getting from me

The article you would find on a generic blog post — *Five Tips for Using AI to Code Faster* — is not this article. I am not going to tell you to use Cursor's tab completion or to write better prompts or to enable chain-of-thought. Those pieces are everywhere. They are also, mostly, written by people who have not shipped a project of nontrivial complexity end-to-end using these tools, and so they have the texture of advice in the abstract.

This is the article about *how a project of nontrivial complexity was actually delivered by one person using a stack of language models, and what the daily reality of doing that looks like — including the parts that look impressive in screenshots and the parts that don't*.

`workstation` is on as I write. The desk fan is on. The single EVGA RTX 3090 FTW3 is idle at 41 °C; the last `train_phase2.py` finished an hour ago and I have not started another one because I am writing instead of training. In one tmux pane to my left is the Claude Code session this document is being drafted in — the prompt is loaded with this repository's `CLAUDE.md`, with `bloom/CLAUDE.md`, with `AGENTS.md`, and with the auto-loaded project memory. In the pane below it is a bash with `trader_env` activated, ready to take a `python -c` if I need to sanity-check something. To my right, in a different tmux window, is a Gemini CLI session that I opened yesterday to audit a docstring inconsistency and have not closed. In the browser, behind this terminal, is a Claude.ai tab with a long, casual conversation about a paper I read last week. On the desktop is the icon for the internal SSD mounted at `/mnt/data…` — the snapshot of January's Claude Code sessions, 121 jsonl files, 32,094 timestamped messages. That mount point is the project's memory before resume was on.

Six contexts. One developer. None of those contexts can do the work of any of the others, and the work is not the sum of what they can do — the work is the part *I* do, which is to know which one to point at which problem, and to carry information between them when none of them can.

This article is the methodology piece the series has avoided so far. It pairs with Article 12 (the 15-subagent audit, which is the canonical use of the subagent batch pattern), Article 25 (the five-model deep-research diagnosis, which is the canonical use of cross-model triangulation), and Article 28 (the annex closer, which is the canonical demonstration of how an annex of research artifacts gets folded back into a working diary). Read those three for what each context *did*; read this one for how all six fit together.

I want to be careful not to oversell. The setup looks like a productivity stack. The lived experience is closer to *spending most of the day deciding what to put where, and the rest of the day moving facts between places that should already know them*. The kickback at the end of this article is that the dependency on this stack is structural now, in a way that is uncomfortable to admit, and the right response to that discomfort is not to deny it but to say so.

---

## The six contexts

### 1. Claude Code, main session

The primary coding driver. The session that has the full project loaded — `CLAUDE.md` at the repo root, the per-subtree `CLAUDE.md` for whichever component I am editing (`mcts_trader/`, `diff_teacher/`, `tools/orchestration/`, `yarrow/`), the auto-loaded `MEMORY.md`, and the AGENTS.md pointer file that exists so non-Claude tools can find the same conventions. This session knows the file structure. It knows the recent diffs. It knows the active task. It knows, by the time it has read its config, that the project's hardware constraint is *a single 3090*, that JAX preallocates VRAM and we don't change that, that `optax` is pinned to 0.2.6 because we use a semi-private API on it, that `numpy<2` for TF/numba compatibility, that we use `threading.RLock` not `Lock`, and that we don't silently delete failed runs.

The main session is where the *codebase changes*. When a line of `trainer.py` needs to move, the main session moves it. When a new ADR goes into `lore/adr/`, the main session writes it. When a `tasks/TASK_NNN_*.md` is being authored against a binary decision gate, the main session drafts it and I edit. The main session is the one that runs `git diff`, that runs the linter, that watches the test output, that talks to the orchestration dashboards in `tools/DS24/`. It is the heaviest context I run, the most expensive in tokens, the one I am most careful to keep on-task.

Two disciplines apply to this session. First: I read what it does before I accept it. The session is fluent; fluency is not correctness. Article 12's kickback is on the table every time — *named infrastructure existing is not evidence that the infrastructure is used*. Second: when the conversation has accumulated more than a few thousand turns, I take a hand-rolled checkpoint into `lore/notes_YYYY-MM-DD.md` and start fresh. The reboot-document hygiene from Article 13's territory is what makes this survivable. A reboot is not a regression. A reboot is *a deliberate purge of stale priors*.

### 2. Subagent batches, dispatched from inside the main session

The main session has a tool — the Agent tool — that spawns short-lived subagents. Each subagent gets cold context: it does not see the parent session's prior turns. It gets a brief, a set of file paths, and a return contract. It does its work and returns a structured report. Then it goes away.

Article 12's audit was the canonical use of this pattern in this project. The training pipeline was broken into eight components; each component was audited by two or three subagents independently; the discipline was *no synthesis until every subagent had returned*. The result was 40 candidate findings, 29 confirmed bugs after validation, in a pipeline that I and a daily-collaborator agent had declared closed less than a week earlier. Article 28's mega-batch is the second canonical use — a dozen subagents dispatched in a single sweep to validate the annex's article-23-through-27 cross-references against the actual source files.

The subagent batch's value is not that the subagent is smarter than the main session. The subagent is the same model. The subagent's value is the *cold context*. The main session has been looking at `trainer.py` for two months; the parts of that file the main session and I have re-read into furniture are exactly the parts neither of us can audit anymore. A subagent has not been looking at `trainer.py` for two months. It has been looking at `trainer.py` for thirty seconds. The thirty-second eye catches what the two-month eye long stopped seeing.

The pattern, in practice: I draft the brief in the main session, hand it off as an Agent call, the subagent comes back with a report, the report goes into `tasks/` as a `TASK_NNN_*.md` artifact, and the main session then *consumes the artifact* — not the subagent's transcript, the artifact — when synthesizing. The cold-eye/warm-synthesis separation from Article 12 is not optional. Mixing them re-introduces the daily-collaborator blindness through the back door.

### 3. Gemini CLI, parallel coding/audit driver

In a separate tmux pane runs Gemini CLI. It looks at the same repository, holds its own state, and is occasionally pointed at the same problem the main Claude Code session is on. Article 4 and Article 5 quote from the era — *"by gemini task 120"* in a January 2026 prompt is the literal transcript of a relay handoff, where I had been working a problem in Gemini, declared a milestone, and switched to Claude to continue.

I use Gemini when I want a *second pair of eyes on the same problem from a different model family*. Sometimes that is because Claude has been circling and a fresh substrate breaks the loop. Sometimes it is because the problem is one I know Gemini is structurally better at — long-context survey of a deeply nested file tree, for example. Sometimes it is because I want the disagreement: if two different model families, given the same brief, produce the same recommendation, that recommendation is more trustworthy than either one alone.

There is also a *failure-mode benefit*. Different models hallucinate in different shapes. Claude's hallucinations tend to be confident, structurally plausible, and wrong about the names of functions in a file. Gemini's tend to be more verbose, less structurally tight, and wrong about a different set of things. When I run the same audit through both and look at the *overlap*, the overlap is mostly real. The non-overlap is interesting; sometimes one model caught a real thing the other missed, and sometimes one model fabricated a thing the other correctly did not see. Either way the diff is informative.

The Gemini pane is not as continuous as the Claude Code pane. It is more like a tool I open when I need it and close when I am done. The relay model — *Gemini does task 120, hands it to Claude to continue* — was the early-2026 pattern; by spring I had settled on *Gemini for second-opinion audits, Claude for the main thread*. Roles are not strict. They reflect a slowly converged preference.

### 4. Deep research reports

These are long-form synthesis runs commissioned from external research-mode tools — Gemini Deep Research, Claude in long-form mode, DeepSeek, Kimi. Each one accepts a prompt of a few hundred words and returns a report of tens of pages, with citations, after running for several minutes (sometimes longer). I treat them as *commissioned literature surveys*. They are not coders; they are not auditors; they are readers who have read more than I have.

Article 25 walks the canonical use. One question — *what is the literature on coupled policy-prediction collapse in MuZero-style agents* — sent to five different models, the same prompt verbatim, with deliberate neutrality about which side I thought was causal. Five reports came back. They were not the same. Gemini Deep Research named the pathology and decomposed it into three concurrent mechanisms including the dormant-neuron piece I had not internalized. DeepSeek foregrounded optimizer pathology that Gemini had treated as a footnote. The two were not contradicting each other; they were *emphasizing different axes of the same problem*. The consensus across the five was where my pivot roadmap came from. The disagreements were where the surprises lived.

I do not commission a deep research report every week. They are expensive in time-to-result (several minutes of wallclock per query, sometimes hours if I batch them), and they are expensive in attention (a thirty-page report needs an hour of careful reading, with annotations, before it is useful). I commission them when I have hit a question where the literature is plausibly the answer and I do not yet know which literature. Article 33's dormant-neurons material came directly from a deep-research commission. Article 25's whole structure is the commissioning + reading workflow made explicit.

The discipline that makes deep research reports work is *triangulation* — never one report, always three to five, prompt held constant, disagreement read with as much care as agreement.

### 5. Claude.ai, browser tab

The casual layer. The original Feb 11 2025 conversation that Article 0 documents — *"How feasible would be alpha go aproach to trading let's say in gpu optimized space"* — happened in this layer. So did, three months later, the May 10 formalization. So does, in May 2026, the conversation I had last week about a paper that may or may not be relevant to flow matching's failure to promote.

Claude.ai is for *thinking out loud*. It does not have the project loaded. It does not know `CLAUDE.md`. It does not see `trainer.py`. It is, in the cleanest sense, *talking to a smart friend who has read the field but does not yet know the codebase*. That property is structurally useful. The main Claude Code session, having read `CLAUDE.md`, is biased by `CLAUDE.md` — it knows what the project considers true and proceeds from there. Claude.ai, knowing none of that, will sometimes ask the question the main session has stopped asking. *Why are you doing it this way? Have you considered the simpler thing?* The naive question is occasionally the load-bearing one.

The casual layer is also where the *idea check* happens. *Is this idea even worth trying?* — typed into Claude.ai, with one paragraph of setup and zero project context. The answer is sometimes "yes, here are three things to check first," sometimes "this is a known dead end in the literature, see X," sometimes "this is interesting but is downstream of a question you haven't asked yet." None of those answers are decision-grade. All of them are *triage-grade*. The cost of asking is low; the upside of not running a multi-day experiment on a dead-end idea is high.

I keep the Claude.ai tab open as a matter of habit. When something pops into my head that does not yet belong inside the project's working memory, it goes there. If it survives the casual layer, it gets *written into a `tasks/TASK_NNN_*.md` draft* and proposed to the main session. If it dies in the casual layer, no harm done. The conversion rate from casual idea to actual task is — I have not measured it, but eyeballing it — maybe one in eight.

### 6. SSD archive of earlier sessions

The 2 TB internal SSD mounted at `/mnt/data…` holds the project's prehistory. 121 jsonl files from the Phase 1 era, January 11 to February 13, 2026. 32,094 timestamped messages. Each file is a Claude Code session that auto-cleaned itself out of the live workspace because resume was not on at the time. They are recoverable only because the snapshot was taken before the cleanup.

This is the project's *memory*. Not active memory — the live sessions don't read from it. *Lookup memory*. When I ask myself *what did I think six months ago about why TASK_136 was choking the encoder*, the answer is not in any of the live contexts. It is in a 4,857-line jsonl file from January 31, opener *"ok i had crash .. pos[itions?]"*, where the conversation that walked the diagnosis lives in full. The way I get to that answer is a Python one-liner grepping the jsonl for the substring I half-remember, returning the matching lines with their timestamps, and from there I can pull the surrounding turns by file offset.

The SSD archive is *read-only* by discipline. I do not write back to it. I do not extend the conversations stored there. If I want to act on something I found in an archived session, I copy the relevant passage into the live session as a quoted block — *"in `9774f034-…jsonl` on 2026-01-11 I wrote: 'X'"* — and proceed from there. The archive is a record. The record stays a record.

The archive also taught me something the live sessions cannot, which is *how much I had already learned and then forgotten*. The 32,094 messages contain ideas I am rediscovering in May 2026 as if for the first time. Sometimes the rediscovery is genuine — context has changed, the idea means something different now. Sometimes the rediscovery is just *I forgot*. The archive's existence is what makes the second case visible.

---

## The orchestration

The six contexts are the labor. The human is the *strategy*.

The choice of which context to use is itself the work. It is not automated. It does not get easier with experience in the way that, say, vim becomes muscle memory; the choice has to be made fresh each time because the question is fresh each time. A small triage protocol has nevertheless emerged over six months, and writing it down here is the closest this article gets to *advice*.

- **Code change in an active file, within a known component**: main Claude Code session. Costs: a few hundred to a few thousand tokens. Risk: low, because the session has the file in view and the edit is local. This is 70% of all touches.
- **Cross-cutting audit, multiple files, multiple components**: subagent batch from inside the main session. Costs: a few hundred thousand tokens for the batch, plus the synthesis pass. Risk: high if the synthesis is done badly; the value is *only* unlocked by the cold-eye/warm-synthesis discipline.
- **Second opinion on a contested edit, or a problem the main session has been circling**: Gemini CLI. Costs: a separate session's worth of tokens. Risk: low; even disagreement is informative.
- **Literature question, algorithm-choice question, "what does the field think about X"**: deep research, three to five models, prompt held constant. Costs: time-to-result (minutes to hours of wallclock), plus the reading time. Risk: the reports can mislead; triangulation is the mitigation.
- **"Is this even worth trying?" question, or a free-form idea I do not yet know how to scope**: Claude.ai browser tab. Costs: a single conversation. Risk: low; the casual layer is *meant* to be lossy.
- **"What did I think six months ago about X?" question**: SSD archive, Python grep into the jsonls, surrounding turns by file offset. Costs: a few minutes of search. Risk: none; it's a lookup.

The protocol is not strict. Categories bleed. A code change sometimes turns into an audit; an idea check sometimes turns into a deep research commission. What the protocol gives me is a *default*, which is what I use when I am not paying attention. The defaults matter because most of the day I am not paying attention to the meta-choice; I am paying attention to the work, and the meta-choice has to be implicit enough not to consume the foreground.

There is a deeper claim under the protocol, and Article 3 already made it in a different register. The berserker contradiction generalizes. The agent is structurally good at the *tactical* — the local code change, the cold-context audit, the literature pull, the docstring rewrite — and structurally bad at the *strategic* — which problem is worth solving, which result is real, which direction the project should pivot toward. The strategy is the thing the LLMs are *not*. The strategy is the thing the human is *for*.

This is uncomfortable for the LLM-skeptic reading because it sounds like I am claiming the agent is doing the hard work. It is uncomfortable for the LLM-enthusiast reading because it sounds like I am claiming the human is doing the hard work. Both readings are wrong. The hard work is the *integration*. The agent labor is real labor and the human strategy is real strategy and the project ships only because the integration of the two — which context for which job, which output to trust, which to discard, which to reconcile, which to escalate — actually happens. The integration is what one person does. The integration is, in a real sense, the *job*.

---

*The Apothecary's apprentices each do what they have been trained to do. None of them looks up from the bed they are tending. The gardener walks among them, carrying a small notebook, and writes down what each has said. By the end of the day the notebook is longer than any single apprentice's report. The garden is not in the notebook. The garden is in the beds. The notebook is in the gardener's pocket, and it is what makes tomorrow possible.*

---

## The kickback

There are three layers to this, and I am going to walk them in order from the most visible to the most uncomfortable.

### Layer one: it looks impressive and is mostly tedious

The six-context stack, described as above, sounds like a force multiplier. In the screenshot it sounds like *ten times more productive*. In practice it is closer to *context-switching all day*.

The reason is that the six contexts do not share state. The main Claude Code session does not know what the casual Claude.ai conversation just decided. The Gemini CLI audit does not know what the subagent batch concluded. The SSD archive does not know what either of the live sessions is doing. *I* am the bridge between all of them. Every fact that needs to move from one context to another moves through me — through my reading of the source context, my retyping or quoting into the destination context, and my judgment about which facts are worth moving in the first place.

That bridge is work. It is not glamorous work. It is *copy-paste with provenance*. *"Gemini said X on Tuesday; here is the quoted passage; here is the file it lives in"* — typed into the main Claude Code session before I ask the session to act on X. The provenance is what keeps the bridge honest; without it, facts arrive in the destination context without their source, and the destination context — having no way to weight them — either over-trusts them or under-trusts them. Neither failure mode is recoverable cheaply.

The aggregate experience is, on a good day, the feeling of running a small newsroom out of one head — checking sources, weighing them, copying between desks, synthesizing for the editor (me) who has to make a call by end of day. On a bad day it is the feeling of *losing track of what was said where*. The casual Claude.ai conversation had a fact the main session needs but does not have. The Gemini audit said something I half-remember. The SSD archive holds a passage I have not yet looked up. The contexts proliferate; the bridge stretches; the cognitive load is real.

The thing I have not yet automated, and I am not sure can be automated, is the *judgment about which facts deserve to cross*. Most facts do not. Most exchanges in the casual layer should die in the casual layer. Most subagent findings are false positives or low-priority. Most deep-research bullet points are background. The judgment about which 5% to lift across is the judgment that is most expensive, and it is the judgment that is least visible from outside.

### Layer two: token budget is a real constraint

The 28 articles in this season alone consumed somewhere in the low single-digit dollars of API spend, *roughly* — I have not audited the exact number, and the figure depends heavily on which mix of contexts each article drew on. The 200+ subagent jsonls in the project's history total context-window-equivalents in the thousands. None of this is free.

The economics, as of mid-2026, work for me. They work because the alternative to *me + this stack* is *me + a researcher I would have to hire*, and that alternative is more expensive by several orders of magnitude. They also work because the project is a personal one — I am the only consumer of its outputs, the labor cost is mine to allocate, and the marginal cost of an additional subagent batch is small compared to the marginal value of the audit it produces.

The economics do not work indefinitely. I have watched my monthly spend climb as the project has matured and the contexts have multiplied; the trajectory, extrapolated, is not stable. The deep-research commissions are the single biggest line item per query, because each one consumes a long-form generation budget. The subagent batches are the second, because parallelism means I am paying for n simultaneous contexts where I used to pay for one. The main session itself is the third, because it has grown — the loaded context now includes `CLAUDE.md`, four subtree `CLAUDE.md`s, `MEMORY.md`, the active task spec, and whatever files the current turn has opened.

There are mitigations. The reboot-document hygiene from Article 13 is one — it lets me start a fresh main session with the relevant priors compressed into a single document, which keeps per-turn token usage flat instead of monotonically rising. The subagent batches are themselves a mitigation, because cold-context work is cheaper to commission than warm-context continuation. The deep-research budget I gate explicitly; I commission a multi-model survey only when I have hit a wall I am not getting past otherwise.

The honest version is *this stack is affordable now and the affordability is a feature of where I am, not a property of the stack*. If my circumstances changed — if the day job stopped funding the research, if the rate of model improvement plateaued, if the per-token pricing moved the wrong way — the stack would have to shrink. I have a triage of which contexts I would cut first. The casual Claude.ai layer is the cheapest in money and the hardest to give up emotionally; the deep-research layer is the most expensive in money and the easiest to gate. The subagent batches sit in between. The order in which I would shed contexts under pressure is something I have already thought about, which itself tells you something.

### Layer three: the dependency is structural

This is the layer that is hardest to write down and I am going to write it anyway because the alternative is to pretend.

If language models disappeared tomorrow, this project would freeze.

Not because the models do the thinking. Article 26's kickback closed that question. The thinking — the strategy, the framing, the diagnosis, the choice of what experiment to run next — is mine. The models are not, in any structural sense, the source of the project's intellectual content. They have been wrong too often, in too many directions, for me to delude myself otherwise. Article 7 (TASK_737 — the single-seed champion that wasn't), Article 8 (Five Seeds and a Funeral), Article 10 (the eval-MCTS asymmetric-lift discovery that nullified a campaign's worth of results) — every one of those kickbacks was *the human noticing the model had told them something untrue*. The intellectual content is mine because the auditing of the intellectual content is mine, and the auditing is what creates the content in the first place.

But.

The project would freeze, all the same, because the *tax* the models absorb is structural. The boilerplate I do not have to write. The library APIs I do not have to memorize. The doc-reading I do not have to do. The formatting I do not have to fuss over. The repetitive code refactor I do not have to type. The shell script I do not have to author from scratch. The first-draft prose I do not have to compose. The literature search I do not have to crawl. Each of these is small. Each of them, summed across a six-month project, is *the difference between shipping and not shipping*.

Without the stack, the strategy would still be possible. The strategy is mine. But the strategy alone does not produce a working repository. The strategy plus the tax-absorption produces a working repository. The tax-absorption is what the stack does. The stack is what allows the strategy to convert into artifacts at all.

This is uncomfortable to admit because it sounds like a weakness. It is also, I think, the right diagnosis. The right diagnosis is rarely the comfortable one. The project depends on a specific availability of specific tools, and the availability is not under my control. If the tools change — if pricing moves, if access tightens, if a model family I rely on shifts — the project's velocity drops sharply. Not to zero. To something between half and a quarter.

I am writing this down because I would rather have it on the page than discover it later. The dependency is what it is. Naming it does not change it. Naming it does, however, change my obligations: the disciplines I built on top of the stack — the multi-seed rule, the operator-vs-metric distinction, the cold-eyes audit pattern, the reboot-document hygiene — those have to be *tool-independent*. They are the parts of the methodology that survive the tools.

Which brings me to the closing.

---

## What survives if the tools change

The tools will change. They have already changed multiple times in the six months of this project — Claude Code has updated, Gemini's deep research mode has rolled out new versions, DeepSeek and Kimi have each had model bumps, the Claude.ai interface has reorganized at least twice. The specific configuration I described in section one of this article is a snapshot of May 22, 2026. By August it will be different in ways I cannot predict.

The methodology, however, is not the configuration. The methodology is what I do with whichever configuration is available.

The disciplines that survive any tooling shift, that this project has earned the hard way and that I now treat as load-bearing in the way that file-system invariants are load-bearing:

- **Multi-seed before claim.** Article 8's funeral was the final lesson on this one. Single-seed Alpha gaps ≤ ~5pp on this stack are within noise. Three seeds minimum before any claim of improvement; five if the gap is small. This is tool-independent. It applies whether the run is dispatched by Claude Code, by a shell script, or by hand.
- **Operator versus metric.** Article 10's discovery — the same checkpoint scores +0.72 Alpha under argmax and −1.20 Alpha under eval-MCTS at sims=16 — generalized into a rule I keep on the wall: *the metric you report and the operator you deploy under must be the same operator*. The tool stack can dispatch the eval; the rule about which eval to dispatch is the human's, and it survives any change in the dispatching.
- **Cold-eyes audit.** Article 12's audit methodology — cold-context subagents in parallel, no cross-talk, warm synthesis afterward separately — does not depend on the agents being Claude. It depends on the *cold-context boundary*. If the agents were human contractors, the methodology would be the same. If they were a different model family, same. The boundary is the load-bearing object.
- **Reboot-document hygiene.** When a session has accumulated more context than it can usefully integrate, a hand-rolled checkpoint into `lore/notes_YYYY-MM-DD.md` is not a regression. It is *a deliberate purge*. This applies to any model that has a context window, which is to say every model.
- **Multiply the arithmetic onto the page.** Article 28's annex closer made this one explicit — when the math gets dense, the answer is not to wave at the math, it is to write the math out, line by line, into the document. The Apothecary's discipline. The tool that holds the page can change. The discipline of putting the arithmetic on the page does not.
- **Triangulate across models for hard questions.** Article 25's pattern. Same prompt, three to five models, read for consensus, read for disagreement, read for *what each one foregrounded that the others did not*. The disagreements are often more informative than the agreements.
- **Provenance on every fact that crosses a context boundary.** *"Gemini said X on Tuesday; here is the file"* — this is the only thing that keeps the six-context stack honest. The provenance discipline is mine, and it is not specific to any tool.

What I would say to someone setting up a similar stack today is this: the tools are the cheap part. The disciplines are the expensive part. The disciplines were learned by losing the same kind of bet enough times that the loss became the lesson. None of them came from a productivity blog. They came from kickbacks. The series is, in that sense, a long inventory of what the disciplines cost to acquire.

The Apothecary, in the closing image of the epigraph, *remembers between the beds*. That is the line. The apprentices each do one thing. The gardener carries the notebook. The notebook is what makes tomorrow possible. The notebook is not in any one bed; it is in the gardener's pocket, and the gardener walks.

I write this from the same workstation the training loop will run on tomorrow. `workstation` is on. The desk fan is on. The single 3090 is still at 41 °C. There is a `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` open in another pane that I have not yet committed. The next experiment in the queue is a reward-shaping sweep. The next article in this series — number 39 — has a title but not yet a draft. The six contexts are all still open, each on its own pane or tab or mount point, and none of them has done anything in the last twenty minutes because I have been typing this.

That is the methodology. Not the stack. The standing-in-front-of-the-stack, deciding.

---

## Sources

- `/home/lukas/Trader/CLAUDE.md` — project identity, components, hardware constraints, hard rules.
- `/home/lukas/Trader/bloom/CLAUDE.md` — publishing pathway, voice spec, source-material discipline.
- `/home/lukas/Trader/AGENTS.md` — non-Claude-tool compatibility layer pointing back at `CLAUDE.md`.
- `/home/lukas/Trader/bloom/drafts/_CHRONOLOGY.md` — timeline of the project's LLM tooling, including the SSD-archived January 11 → February 13, 2026 sessions (121 jsonls, 32,094 messages).
- `/home/lukas/Trader/bloom/drafts/_GENESIS_CHAT.md` — verbatim Claude.ai conversation extracts from the Feb 11, May 10, and July 14 2025 sessions that preceded the active build.
- Article 12, *The 15-Subagent Audit and What It Caught* — the canonical cold-eyes batch.
- Article 25, *Five Models Diagnose the Same Failure* — the canonical cross-model triangulation.
- Article 26, *Hybrid LLM Reasoning* — the prior kickback on what the models do and do not do for the thinking.
- Article 28, *The Annex: What the Architecture Essays Were For* — the canonical demonstration of an annex folded back into the diary.
