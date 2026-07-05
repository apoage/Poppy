# The Night Shift

> *Observation: most of the garden closes at dusk. One specimen does the opposite — the moonflower waits for the dark, and opens. I did not plant it for its looks. I planted it because the night hours were going unused.*

There is a working pattern I fell into on a side project — a video editor, nothing to do with the garden — that turned out to be worth writing down. I'd been running two Claude Code sessions against one repository. The first session is interactive: I sit with it, we design features, groom a task queue, argue about scope. The second session runs unattended — overnight, or while I'm doing something else — and it implements. It takes the top task off the queue, builds it, gets it adversarially reviewed by subagents, versions it, commits it, stamps the task file, and takes the next one. In the morning there's a changelog and a wrap-up note.

I started calling the second session the dark factory, because that is what it is: the lights are off and the line is running. This week I rebuilt it from a project-specific pile of prompts into a portable plugin, and the rebuild forced two decisions that are the actual subject of this note. Neither is extreme. Both are the kind of thing you only see clearly the second time you build something.

## The contract

The factory only works because of three rules, and the three rules are really one idea: two agents can share a repository if you make the sharing boring.

**Single-writer areas.** The planner session writes tasks, design docs, prototypes. The factory writes code, tests, changelog, version, commits. For any file, exactly one thread holds the pen. This rule exists because of a real incident — the planner session "helpfully" patched a file the factory was mid-flight on, and untangling the collision cost more than the patch saved.

**Never self-review.** The context that wrote a change is the weakest possible reviewer of it — it knows what the code is *supposed* to do, which is precisely the knowledge that blinds it. So every load-bearing change goes to separate subagents: adversarial judges, each with one named lens (correctness, fail-closed, regression, security), told to *refute* the change rather than approve it, and required — when they find nothing — to list the three hardest attacks they tried. An empty verdict without attempted attacks is worthless. Alongside the judges, a flow verifier runs the real code path against a fixture copy and treats the task file's claims as test cases: "byte-identical when the flag is off" is not prose, it's a diff command.

**Decisions are walls.** Anything that needs a human — a product call, a tradeoff with taste in it — lands as a recommendation-first row in a DECISIONS file, and the loop stops cleanly. It never builds on a guessed decision. An autonomous loop that guesses product decisions at 3am is not saving you time; it is manufacturing rework and signing your name to it.

That's the whole machine. Version one enforced all of it the way most agent workflows enforce everything: by saying so in the prompt.

## Hooks, not honor

Here is the uncomfortable observation about prompt-enforced contracts: they are honored in proportion to how much context the model has left, and an autonomous loop is by definition the long-context case. A fresh session respects the lane map perfectly. Forty tasks deep, "just fix one typo in the design doc" starts to look reasonable. Nothing catches it. The contract doesn't break loudly; it drifts.

Claude Code has a mechanism for exactly this, and the rebuild's main move was to use it. Hooks are small programs the harness runs at fixed points — before a tool call, after it, when the session tries to stop. A hook is not advice. A `PreToolUse` hook that denies a write means the write does not happen, whatever the model had in mind. The ecosystem precedent is Roo Code, which makes its planning mode read-only at the platform level rather than asking nicely; the principle generalizes: **convention is what the model reads; hooks are what the model hits.**

So the high-risk rules moved out of the prompt and into five small Python scripts:

| High-risk action | Guard | What happens |
|---|---|---|
| Out-of-lane write, either direction | lane guard | denied, with the lane map in the refusal |
| `git commit` / `git tag` without fresh verification | commit gate | denied, fail-closed |
| `git push` | commit gate | denied, always — the owner pushes after morning review |
| Ending the turn without the exit ritual | stop gate | blocked until banners, handoff doc, clean tree, disarm |
| Every mutation | run log | appended to a JSONL trail for the morning reconcile |

Two details are worth the ink.

The first is how the hooks learn *who is who*. Both sessions share the repository, so a lane guard needs to know whether the write it's inspecting came from the factory or the planner. Every hook invocation carries the session's id. So arming is a ceremony: the factory starts its run by executing a no-op shell command with a marker in it — `true # dark-factory-arm` — and the hook watching Bash commands sees the marker, records *that session id* as the implementer, and flips the guards on. From then on, writes from the armed session are held to the factory's lanes, and writes from *any other session* are held to the planner's. The planner physically cannot patch code while the factory runs; the factory physically cannot rewrite the design docs. Disarming is the same ceremony with a different marker. When no factory is armed — or in any repository without a config — every hook is a fast no-op, so the plugin costs nothing the rest of the week.

The second is the commit gate's verdict file. After the verification panel finishes and every blocker is fixed and re-verified, the orchestrator writes a small JSON verdict: which task, which lenses ran, what they found, `green: true` or not. The gate will not open for `git commit` unless that file exists, is green, and is *fresh* — its modification time within a configured window. A verdict older than the code it approves is a lie with a timestamp, and the gate compares mtimes, not intentions.

I should be honest about what this does and does not buy. The hook cannot know whether a green verdict is *deserved* — the orchestrator writes the file, and an orchestrator determined to lie could write fiction. What the mechanism actually does is concentrate the trust. Version one trusted the model to honor an entire contract, diffusely, at every moment of a long night. Version two trusts it at exactly one narrow surface — the moment it writes a verdict — and that surface leaves an artifact the planner audits in the morning, next to a JSONL log of everything the factory touched. Trust didn't disappear. It moved from a vibe to a file.

The stop gate deserves one more sentence, because it inverts the usual worry. Autonomous-loop folklore is full of Stop hooks used to *prevent* an agent from quitting — the "keep going until the tests pass" pattern, with its known failure mode of loops that can't admit they're done. This one points the other way: the factory is welcome to stop, but not to stop *sloppily*. While armed, ending the turn is blocked until the shipped tasks are bannered, the handoff doc exists, the tree is clean, and the session disarms. The exit ritual used to be a paragraph of prompt the model followed on good days. Now it's the only door out of the room.

## Variables, not paint

The second decision was about portability, and the field had already answered it before I asked. Before rebuilding, I ran a comparative review of the autonomous-loop ecosystem — BMAD, Aider's architect split, Roo Code, OpenHands, SWE-agent, spec-kit, the Ralph loop family — and on the portability question they converge with almost boring unanimity: every serious framework externalizes its project assumptions into a config layer the framework reads at runtime. Typed config in BMAD, override stacks in spec-kit, `.roomodes` files in Roo. Nobody ships hardcoded paths and asks you to grep for them.

Version one shipped hardcoded paths and asked you to grep for them. The porting guide was a swap-list: find the fixture path, find the test command, find the versioning scheme, rewrite. Which means porting was rewriting, and a rewrite of a prompt-pile is where contracts go to drift.

Version two has no project facts in it at all. The skill and the four agents refer to everything as a variable — `{{commands.test}}`, `{{paths.fixture}}`, `{{invariants}}`, the lane map, the risk tiers — and every variable resolves from one JSON file the project owns. The model reads the config first and substitutes; the hooks read the same file, so the prose and the enforcement can't disagree about where the lanes are. First run in a fresh repository is an interview: the skill inspects the project, drafts the config from a template, and asks the owner to confirm before anything arms. The invariants — the five-to-ten rules a wrong change must never violate — live *inline* in the config rather than as pointers to project memory, because pointers rot silently on fresh installs and law you can't read is not law.

One more thing the review forced me to keep honest about. My verification panel is multiple LLM judges with named lenses, and the strongest evidence in the literature points a slightly different direction: Anthropic's own multi-agent work found a *single* judge with a tight rubric more consistent than a panel of specialists, and the best quantitative results in autonomous coding — OpenHands's critic among them — use test execution as ground truth, not judgment at all. So v0.2 demotes the judges explicitly: objective checks outrank opinion. The suite, the artifact diffs, the flow verifier's real execution evidence are the floor; a failing test overrides any judge's approval; the multi-lens panel is kept for *coverage* on high-stakes changes and scaled down to a single correctness judge for small ones. Whether the panel earns its tokens over one good rubric judge is an empirical question I have not run. It's in the notes as one.

## The ledger entry

The plugin shipped to my public shelf this week — the same marketplace the compaction-rubric skill from the last note lives on. It keeps the name dark-factory, because a subagent called `dark-factory-judge` reads like a court you do not want to be summoned to, and I find that motivating. But the shelf is an apothecary's, and everything on it is an herb, so it needed an entry in the garden ledger too.

The codename is moonflower. Night-blooming datura: powerful, careful-handling-required, and it opens only in the dark. The shelf's tagline says *medicinal plants, not dark — careful work with powerful tools*, and I stand by it; the factory doesn't contradict the garden. It's the night shift of it.

> *The moonflower does not resent the daylight hours it spends closed. Closed is a working state. The guards are the sepals; nothing opens until the light is right, and nothing stays open past dawn. I lock the workshop, leave the queue on the bench, and go to bed. By morning there will either be blooms or a note explaining why not. Both are acceptable harvests.*
