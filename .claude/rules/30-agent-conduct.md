---
description: How agents in this repo reason and report — source of truth, refutation discipline, transient API failures, lore tone. Deliberately not path-scoped.
---

# Agent conduct

## The repo is the source of truth, not a summary

Read `lore/notes_*.md` (latest) + `tasks/QUEUE.md` + `tasks/DECISIONS.md` before acting. Neither
the root `CLAUDE.md` nor any memory entry is authoritative for **code state** — component facts
(encoder version, PopArt range, defaults) drift within days, and an agent reading a stale summary
will confidently decide on wrong information. Order of authority: `lore/` (why) → `tasks/`
(what) → subtree `CLAUDE.md` (component facts) → `experiments/LEDGER.md` (verdicts).

Governing decision: `lore/adr/0008-per-component-claude-md-and-lore-source-of-truth.md`.

## A refutation's replacement needs its own adversarial pass

When a diagnosis **refutes** a task's premise, the replacement explanation gets the same
adversarial verification the refuted claim got. Being right about the negative half buys the
positive half nothing.

*Factory run 48 (TASK_913, 2026-08-31):* the row correctly killed the "a process is hogging the
ext4 journal" premise, then confidently named a new culprit — "failing SATA link / bad cable" —
which a judge demolished on four counts: **causal inversion** (the `SATA link down` recovery
happened ~90 s *after* the device went silent; the artifact even wrote "the link-up is the instant
our sampled stall ended" without noticing that refutes its own chain), **a dependent variable read
as evidence** (1.5 Gbps was libata's escalation *response* to timeouts), **confidence bounded on
data never requested** ("the dmesg ring doesn't reach further back" while `journalctl` held 4 GB of
persistent journal containing the disconfirming evidence), and **n=1 dressed as corroboration**.

Practical rules that fall out of it:
- Before claiming a physical/hardware cause, check the disconfirming log source first —
  `journalctl`'s persistent journal, not just the `dmesg` ring.
- Ask of every piece of evidence whether it is a **cause or an effect** of the phenomenon.
- **A metronomic period is a timer until proven otherwise.**
- The cost is asymmetric: the replacement half is what produces an operator recommendation.

## A transient subagent failure can masquerade as an auth error

Long-running implementer subagents here hit transient API failures, and one surfaces as
`API Error: 401 OAuth access token has expired. Re-authenticate to continue.` **even when the
session is perfectly authenticated.** Run 37 (2026-08-20) saw three failures on one agent; none
was an auth problem.

On any subagent API failure, 401-shaped included:
1. Check whether your **own** tool calls still work. If they do, the session is authenticated —
   do not send the operator to `/login`.
2. Assess the tree before assuming loss: `git status`, `git diff --numstat`, compile the touched
   modules, verify fixture/DB invariants. In every run-37 case the tree was intact.
3. **Resume the agent with SendMessage rather than re-spawning it** — resumption keeps its
   context and loses no work. Re-spawning restarts a long build from zero.
4. Tell it in the resume message what you already verified, so it doesn't re-derive it.

Corollary: long implementer runs need resume-tolerance by design. Preserve work in-tree plus a
patch copy so any interruption is a resume point, not a loss.

## `lore/` is a neutral technical record

Lore preserves state across session boundaries: facts, exact numbers, `file:line`, commands,
decisions, reversals. Capture reversals and failures with the same care as successes.

Write **neutrally** — no dramatisation, no emotional editorialising, no emoji, no narrative
flourish. Bullet-point precision beats storytelling; emotional tone is noise to be pruned, and it
biases future sessions. The earlier dual-role framing (lore as raw material for Bloom publication)
was **retired 2026-07-07** when `bloom/` was pruned. Write for exactly one audience:
future-session-me needing the technical record. If a sentence carries no fact, cut it.

## The closing-pass exception to the cycle limit (D76, ruled 2026-09-02)

`halt_after_failures` binds every new-solution attempt. It has exactly ONE named exception:
a **closing pass** may exceed the limit when it (a) replaces an artifact the SAME RUN has
proven defective, (b) only transcribes or reassembles evidence a judge has already collected
and confirmed — no new solution attempt, no new evidence-gathering of its own — and (c)
discloses the deviation in the verdict itself, never in a footnote. Anything else past the
limit halts to a DECISIONS row. The exception exists to UN-ship a wrong artifact (run 48:
cycle 4 replaced the run's own wrong SATA-cable culprit); it is never a fourth try. Run 49
(exactly 3 cycles, the last pre-declared) is the compliant precedent.

## Briefs state denominations in the gate's own quantity (D81(c), ruled 2026-09-05)

Every orchestrator brief must state, **for each check it specifies, the DENOMINATOR and the
direction argument in the SAME quantity the gate consumes** — population vs rate, events vs
time, per-pair vs pooled — and every verification panel must include a lens pointed at the
**brief's own direction claims**, not only at the implementation. Rationale: three consecutive
runs (50, 51, 52) had the orchestrator's brief mis-state or under-specify a denomination, and
in each case the implementer faithfully built what the brief asserted (run 52: a fail-closed
argument stated in population size for a gate that divides by a rate — fail-open in fact).
The recurrence is the brief-writer's, not the implementer's, and no cycle limit binds it.


## Verification instruments get their own adversarial pass (D83(i), ruled 2026-09-06)

Every new guard, harness, or verification instrument is itself a claim, and gets the same
adversarial treatment a diagnosis's replacement gets: **ask where the defect it cannot see
sits, and prove the guard fires when the defect is planted exactly there.** A guard whose
reach does not cover what its prose certifies is a MAJOR even when every test is green —
"the cap never binds" and "the cap was never applied" produce identical tables (run 53's
TASK_820 finding), and a rebuilt `builtins.open` name intercepts zero `io.open` calls while
affirmatively certifying "traversals=1.000" for a double-traversing reader (run 53's 914).
Measured recurrence: seven such guards/claims in run 55 alone across two rows; five were
caught only by an adversarial lens, none by their author. Authors plant poisons where the
mechanism is STRONG; the adversarial pass plants them where it is BLIND.

## Every asserted number carries its artifact and revision (D83(ii), ruled 2026-09-06)

A brief, spec, or task file that states a number must state, for that number, **the artifact
and revision it was measured on** (file+sha, run, or instrument named). Unattributed numbers
are provenance failures waiting to happen: all five factory errors in run 54 were numbers
asserted without artifact+revision, and the planner's own task files inherited dead-panel
estimates that measurement then cut down (a "+0.3 Sharpe" log-bias measured at −0.043; a
"twice the cost stack" borrow line that was a 2× denomination error). Digest files and
consultation summaries are LEADS, not numbers — recompute before quoting, or quote with the
source named and the revision stated.

**Staging rider (run-55 process note, adopted under D83): `git add -A` is forbidden while
any deliberate fence stands in the tree** (a halted candidate, a preserved base). Stage by
explicit path. Run 55's mid-ritual `git add -A` staged the fenced 914 files and was caught
only by the staging listing.
