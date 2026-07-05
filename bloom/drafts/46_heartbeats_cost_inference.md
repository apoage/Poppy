# Heartbeats Cost Inference

> *Observation: every signal carried back to the main vessel costs a measure of breath. The gardener may stand at the door and listen, but cannot listen without exhaling. I shorten the listening and continue.*

A smoke is running. It will take three to five minutes. I want to know how it is going without dumping the raw log into the main conversation — raw logs are noise and noise eats context. The natural shape, sitting at the desk and watching `tail -f` go by, is: route the watching to a cheaper model, have it occasionally tap me on the shoulder with a digested update.

I asked the agent in the other window if I could spec a tool that routes to a sub-agent and prints occasional heartbeat reports back into the main thread. Phrased it as a monitoring tool, because that is how it felt — observability, not work.

The answer was instructive in the way that physics-of-the-harness answers usually are. The thing I was describing was buildable. But it was not buildable the way I had named it. The naming was the problem.

`workstation` is on. The same smoke I am writing about is one of three in the queue, and the second monitor has `nvidia-smi` ticking through it now, the GPU draw oscillating between 240 W and 380 W as the JIT-compiled kernels settle. The desk fan does not stop. I am writing this in a Claude Code session that has been open for about two hours, with the compaction bar at 47 percent.

## The naive shape

What I had in my head looked like this. The main thread spawns an MCP tool. The tool runs in the background. Every few minutes, it emits a digest. The digest appears in the main conversation. The main agent reads it; if it wants to act, it can; otherwise it ignores the digest and life goes on.

Three hard constraints in Claude Code's harness collide with this shape, and they collide in different places, which makes the diagnosis interesting.

**First: anything that reaches the main thread re-fires inference.** There is no "free" notification slot. Every event that crosses the bg-task → main-thread boundary fires the main model — Opus, in my case — for one turn. Heartbeats are not observability events; they are conversation turns. Each one costs.

**Second: MCP tool calls are synchronous from the agent's perspective.** When the main agent calls an MCP tool, the call blocks until it returns one result. There is a progress-notification mechanism in MCP, but in current Claude Code that gets surfaced to the *user*, not re-fed to the *model* as a new turn. You cannot have a single MCP tool call that streams heartbeats back during its own execution. The plumbing exists; the destination is wrong.

**Third: the model that fires for the heartbeat is the main thread's model.** You cannot say "this turn doesn't need Opus; downgrade to Haiku because it is just a status update." The routing knob exists, but it lives inside Claude Code, not in user-pluggable space. From the Agent SDK you could build a custom-routed wrapper — but at that point you are wearing the whole harness yourself and have given up the terminal UI, the slash commands, the skill system, the permission model. The price of the knob is the chrome.

So the original shape is wrong in three places. Monitoring is conversation. MCP cannot stream. You cannot pick a cheaper model for the consumption side.

## The shape that works

The shape that does work uses sub-agent chaining and Haiku-side filtering.

The main thread spawns a sub-agent — Haiku — with a brief: watch this process, this log, this directory; exit when something meaningful happens (terminal state, error, or T minutes elapsed); return a digest. The sub-agent runs in its own context, does the watching, applies the filter, exits with one structured payload. The bg-completion notification fires the main thread; the main thread reads the digest in one turn. If the work is still going, the main thread spawns the next sub-agent and the cycle repeats.

```
main_thread (Opus):
  spawn watch_round_1 (Haiku sub-agent):
    brief: tail $LOG, exit on (terminal state | error | 5 min elapsed)
    return: digest
  ← digest_1                    (1 Opus turn)
  if still running:
    spawn watch_round_2 …
  ← digest_2                    (1 Opus turn)
  …
```

What this buys: the main thread pays one Opus turn per *heartbeat-worthy event*, not per raw log line. Filtering moves to the sub-agent, where Haiku-cost applies. The main thread's job is to consume digests and decide; the digest is small.

What it does not buy: free heartbeats. Each digest still costs one Opus turn. You can lengthen the cadence — one heartbeat every five minutes instead of every thirty seconds — which reduces total inference. You cannot make the cadence zero-cost.

This works today without an MCP server. It is pure sub-agent chaining with a brief that encodes the filtering. The MCP wrapping would only be useful if you wanted to package the brief reusably for other agents, but the architectural shape is the same with or without it. MCP is not the load-bearing piece.

## The kickback

The kickback was in the naming.

I had asked for a "tool that routes to subagent and prints heartbeat reports." The word that did the damage was *tool*.

Tools, in agent harnesses, are calls the agent makes. They are synchronous. They return when they return. The agent waits. You cannot put an asynchronous monitoring channel inside a tool, because the tool's calling convention is "I called you; give me one answer."

The shape I wanted was not a tool. It was an *event source* — something that emits, on its own cadence, into the main conversation. Event sources are not how the harness works. The harness is request/response in both directions. The bg-task → main-thread channel does exist, but the main thread is asleep until a notification wakes it up, and each wake-up is a turn.

Once that was clear, the design fell out. Sub-agents are the event source. Each sub-agent invocation is one event. The cadence is set by how the sub-agent decides to exit. The model in the sub-agent does the filtering. The main thread is the consumer.

I had been thinking of monitoring as observability — something that watches without participating. In an LLM harness, watching IS participating. The watcher is a conversation turn. The cost is paid in inference, not in CPU.

The Claude Code agent put it cleanly when we were working it out:

> Anything that reaches the main thread re-fires inference. You can route the work to Haiku (cheap), but the moment the heartbeat appears in the main conversation, the main model (Opus) wakes up for that turn. You can't make heartbeats "free" — only control how often they happen.

— Claude Code, during the design conversation, May 22, 2026

The phrase that stayed with me is "control how often they happen." That is the entire knob. Cadence, not cost. The cost is fixed by the harness; the cadence is yours.

> *The gardener stands at the door. She listens, but each act of listening costs one breath. She listens less often, and learns to hear more in each breath.*

## What is not buildable

The right knob — "downgrade this specific turn to Haiku because it is just consuming a notification" — is not pluggable from inside Claude Code today. It lives in CC's model routing layer. From the Agent SDK you could build a custom wrapper that routes per turn, but you would be writing your own terminal UI, your own tool plumbing, your own permission model, your own everything. The chrome is most of what makes Claude Code worth using; the heartbeat problem is not worth giving up the chrome.

So the practical floor is: one Opus turn per digested event. If your filtering is good, that ratio is generous — one Opus turn might consume a five-minute window of tail output condensed by Haiku into three bullet points. If your filtering is bad, you spawn too often and the floor becomes the ceiling. The architecture is fine; the discipline is in the brief you write for the sub-agent.

There is a related lesson about exit conditions. The brief I keep coming back to is something like: *exit when (terminal state OR error OR T minutes elapsed OR a metric crosses a threshold)*. Each of those is a tiny piece of language the Haiku sub-agent has to interpret. The error case is the easy one; terminal state and threshold crossings are the ones that drift if you re-write them by hand each time. That drift is the case for encoding the brief into a skill — not because the architecture needs MCP, but because the *brief* benefits from being canonical.

## What's left

The buildable design is sub-agent chaining with smart-filtering briefs. The cadence is the knob. The filtering happens at the sub-agent. The main-thread cost is one Opus turn per digested event.

What is not buildable today, without stepping out of Claude Code, is model-routing per turn. The ability to say "this incoming notification is low-stakes; consume it with Haiku, don't wake Opus" is the right knob in principle, but it lives inside CC's routing layer. Pluggable from Anthropic's Agent SDK if you want to wear the whole harness yourself; not pluggable from inside the CC chrome. That tradeoff is bigger than the heartbeat problem.

The encoded form — call it `wait-with-heartbeat`, or whatever — is about 40 lines of skill markdown plus a small bash wrapper for the watcher. I have not built it yet. The question is whether the convention is enough, or whether the encoded form would be used often enough to earn its space in `~/.claude/skills/`. I lean toward "encoded form" because the brief that sub-agents need is non-trivial — exit conditions are the part most likely to drift if I re-write them by hand each time. The skill is the place where exit-condition language gets canonical.

The deeper lesson is the one I keep relearning: the harness has physics. Every event that crosses the boundary into the main thread costs a measure of inference. You cannot get below that floor; you can only widen the events. Filter at the sub-agent, consume at the main thread, accept the per-event cost. The garden is loud; the gardener listens in measured breaths.

When I sat down to write this I thought it was going to be a design note about a monitoring tool. It turned out to be a design note about the difference between observability and conversation in a system where every observation IS a conversation. That is the harness. That is the breath. The cost is real, and the only knob is how often you spend it.

> *She listens in measured breaths. The garden does not quiet for her listening.*
