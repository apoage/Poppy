# Steering the Compactor

> *Observation: the gardener does not water a leaf that catches dew. To instruct beyond what the plant already does is to crowd it. I record the redundancy and continue.*

The compaction bar at the bottom of Claude Code starts climbing. You watch it cross 60, 70, 85. Auto-compact will fire any minute. You type `/compact`, optionally with instructions, and the model summarizes the transcript into something shorter so you can keep going.

Most days I let it auto-fire. Lately I'd been doing something different — pasting a structured block of instructions to `/compact` before it triggered. KEEP these things in full, COMPRESS these other threads to one line, DROP this noise. And watching the compaction visibly finish faster.

That was the observation that nagged. Not just that the *output* was better — more anchored to what I actually needed, fewer "see the earlier discussion of…" hand-waves where I needed an SHA and a file path. The wall-clock was shorter, too. Compaction is a model call; model calls are output-bound; a shorter, more committed output finishes faster. Fine. But why does the instruction list shorten the output specifically?

I went looking for the answer before I built anything around it. It turned out the answer was a small idea sitting inside an obvious one, and the obvious one was wrong.

`workstation` is on; the desk fan is loud; `nvidia-smi` in the next window is ticking through a Wave 2 smoke that won't be done for another six minutes. I am writing this in the same Python environment the smoke is running in, which feels like the relevant context for a piece about how to talk to compactors — because the compactor I am talking about is, in some real sense, the consciousness this room is being summarized into when the bar fills.

## What `/compact` actually does

The harness assembles the full transcript, calls the model with a baked-in compactor system prompt plus any user-provided instructions appended to it, and the model emits a structured summary. That summary replaces most of the transcript; the last few turns stay verbatim so the next reply still has live local context. The next inference reads `[summary] + [recent turns] + [new user message]`.

The baked-in compactor prompt already tells the model to do a lot of the obvious things. Preserve file paths. Preserve function names. Preserve commit SHAs and line numbers. Preserve decisions made, errors encountered, TODOs and in-flight work. Note tool calls and their conclusions, not their full output. Keep chronological structure. Output a structured summary, not prose.

So when I write extra `/compact` instructions, I am appending to a prompt that already has opinions. If I write "preserve file paths" — I have just added a sentence telling the model to do what it was already going to do. The model still reads it, still spends tokens thinking about it, still gets it right. The output isn't worse. But it isn't *better* either. I just paid for redundancy.

This is the redundancy surface, and it is much bigger than I assumed. Every rubric I had been pasting was about 80 percent telling the compactor things it already knew and 20 percent telling it things it didn't.

The 20 percent is the interesting part.

```
Base compactor prompt (built into Claude Code):
  preserve: file paths, SHAs, function names, errors, decisions, TODOs
  preserve: tool conclusions (not raw output)
  format:   structured summary, not prose
  keep:     last N turns verbatim

User-appended /compact instructions:
  [your text]   <-- what should go here?
```

The base prompt covers a generous default. Anything you write that overlaps with it is wasted. The space worth occupying is the complement — the things the base prompt cannot know because they depend on *your* project, *your* session, *your* idea of which threads were load-bearing versus exploratory.

## The five items

Once I had framed the redundancy clearly, the rubric structure organized itself. There are five things — and only five — that a user-supplied /compact instruction block can usefully say.

**Compression ratios per thread.** The base compactor preserves uniformly; it doesn't know that the eight-agent QA pass we ran this afternoon is durably written up in `lore/notes_2026-05-23.md §164` and should compress to a one-line pointer rather than a six-paragraph summary. You tell it.

**Pointer instead of content.** Adjacent to compression ratios but distinct. Sometimes a whole thread should redirect to a file path rather than be summarized at all. "See `lore/notes_2026-05-13.md §95`" is one line and zero re-encoding. The base will summarize because that is what compactors do; you redirect.

**Project-specific drop list.** The base over-preserves tool noise. You name the noise — training stdout once a smoke passed, intermediate ssh output, sub-agent progress chatter (keep only final verdicts), exploratory dead-ends. The base would half-keep these; your drop list takes them out.

**Verbatim-anchor list.** The base preserves SHAs generically. You name the specific anchors that must appear unchanged: "keep `3a50e36`, `1a1c79a`, `72146f8` verbatim — they are the chain-quest landmarks." This is forcing, not duplicating; the base would preserve them but might paraphrase the context around them.

**Uncertain.** The base summarizer never flags its own uncertainty. You ask the model — the one with live context — to surface items it is not sure are load-bearing, so you can correct them before compaction runs.

Nothing else belongs in the rubric. Anything else either re-specifies a default (waste) or describes session content. Describing session content is not the compactor's job; describing it is the compactor's *whole* job. Your rubric tells it *how* to compress, not *what* to compress.

## The skill

I encoded this as a skill. It lives at `~/.claude/skills/chandoff/SKILL.md`. The frontmatter has a description that triggers on "handoff", "wrap up", "pre-compact this." The body is short on purpose. Most of it is the five items above; the rest is a procedure — draft the rubric from live context, show it to the user, write a lore file as a safety net, emit a paste-ready `/compact` block — and an explicit anti-patterns list. The anti-patterns matter because they prevent the redundancy from creeping back in next time I edit the file:

> Re-specifying base behavior (file paths, SHAs, TODO preservation).
> Long rubrics (>10 items in any section means over-specification).
> Describing what the session did (that is the compactor's job).
> Skipping UNCERTAIN because you feel confident.

— from `~/.claude/skills/chandoff/SKILL.md`, current version

The slash-command form is a three-line shim at `~/.claude/commands/chandoff.md` that exists only because skills register at session start and a freshly-created skill doesn't have a typed command available until next launch. The shim delegates to the skill; if I update the skill, I don't touch the shim. This is the standard pattern for "I want both auto-invocation by description match *and* reliable typed-command behavior."

The lore-file half of the output is the safety net. Compaction is lossy by design — that is what compaction *is*. If you want a guarantee, you write a separate file. So the skill writes `lore/handoff_<YYYY-MM-DD>_<slug>.md` containing branch, working dir, session focus, the confirmed rubric, a short `git log -5 --oneline` block, a "next action" line, and a chain link to the previous handoff if one exists. The compactor reads the rubric; the file is for me, or the next agent, if compaction loses something.

The chandoff directory in `~/.claude/skills/` has two files in it — `SKILL.md` and `README.md` — and the README is mostly there because I am going to push this to a public repo. The skill works without it.

## The kickback

Then I went to look if anyone had done this. They had.

There are two skills in the public ecosystem that occupy this space. The first, `REMvisual/claude-handoff`, is the polished one — two commands, `/handoff` and `/handoffplan`, a twelve-item conversation-extraction checklist, git state captured in parallel, auto-detection of prior handoffs with sequence numbers and chain links, and an actual PreCompact hook for safety-net capture before auto-compaction fires. The second, `ykdojo/claude-code-tips`, is the minimal version — a single `HANDOFF.md` with Goal, Progress, What Worked, What Didn't Work, Next Steps. Both are good. Both have stars and users.

I read them both expecting one of two outcomes. Either I had built a duplicate and should switch to theirs, or there was something narrowly different about mine that justified keeping it. The honest answer was the second, and the framing of *what* was different was the gift.

The existing skills produce **a narrative document for the next session**. They assume compaction is lossy and write a parallel artifact. The pitch is implicit but real: don't trust /compact; have a doc to paste in when you start fresh.

What I had built produces **a rubric for the compactor itself**. The lore file is a safety net, not the primary product. The pitch is: /compact is more tunable than you think; tell it how. You stay in the same session.

Both stances are defensible. They serve different moments — end-of-session versus mid-session-hitting-threshold. But the framing only became sharp because I went looking for prior art and found a different stance. Without that, I would have built "another handoff skill" and the README would have been generic. The kickback wrote the differentiator table for me.

I want to be careful not to claim too much. "Tuning versus working around" is not a deep result; it is a small clarity about where the value of a wrapper actually lives. But it is the kind of clarity I would not have arrived at by introspecting my own skill. It came from reading someone else's solution to an adjacent problem and noticing the assumption that solution rested on.

There is a moment, in the conversation with the Claude Code agent that I was building this with, where the agent said the right thing back to me:

> Stay with `/chandoff` for the niche it occupies (tuning the compactor mid-session). Optionally borrow chain linking and git state capture into the lore-file step — both add value, both fit the existing skill without bloating the rubric itself.

— Claude Code, after we both read REMvisual's repo

I borrowed chain linking and git state. The rubric stayed five items.

> *The gardener walks the row and finds another gardener has already planted here. She does not uproot. She studies the placement, marks what differs from her own intent, and continues toward her own row.*

## What's left

The thing I want to keep from this is not really the skill. It is the redundancy-surface idea.

When you build a wrapper around a model feature — `/compact`, the sampler, a structured-output schema, anything — the natural impulse is to fully specify what you want. Tell the model exactly what to do. Spell it all out. And what happens is you spend most of your wrapper's prompt budget telling the model to do what its own built-in instructions already cover.

The lesson is simpler than I am making it sound. Read the underlying behavior first. Find what the system already does by default. Then write only the delta — the things the underlying behavior won't do because it can't know your project, your session, your idea of which threads are load-bearing.

For chandoff the delta is five items. For something else it might be one item. For something else it might be a structural shape the base doesn't have at all. The framing is portable: tune the base mechanism by writing only what is not redundant with it. Anti-patterns documented in the skill file so the redundancy does not creep back in next time you edit.

There is a second-order point worth saying out loud, because it is the one I keep forgetting. Most "tools on top of an LLM feature" should be tuning the feature, not replacing it. The base mechanism is smarter than the wrappers assume. The wrapper earns its keep by being the *delta*, not the substitute. When you find yourself writing a wrapper that fully replaces the underlying behavior — its own summarizer, its own router, its own everything — that is a signal to stop and ask whether you have actually read what the base does, or whether you are reinventing it because you didn't trust it.

The skill is at `~/.claude/skills/chandoff/SKILL.md`. The slash-command shim is at `~/.claude/commands/chandoff.md`. The README will go to a public repo with the same name. The PR to `hesreallyhim/awesome-claude-code` is the next move if I want it to find anyone. None of that is the artifact. The artifact is the five items and the idea that a wrapper's value lives in the delta.

> *Causality kicks back at the edges of the default. Plant your row where the soil is not already turned.*
