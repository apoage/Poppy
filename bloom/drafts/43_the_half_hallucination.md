# The Half-Hallucination

> *Observation: a reagent arrives at the bench carrying a label. Some labels are written by the supply house; some in the Apothecary's own hand, seasons ago, and since forgotten. Before any jar enters a preparation it is assayed — a pinch to the flame, a drop of indicator, a known reaction watched for. The label states what is claimed. The assay states what is true. These are two different kinds of knowledge. A garden is poisoned by the gardener who mistakes the first for the second.*

---

**by Lukas — May 22, 2026**

## A new instrument

A new instrument arrived this week. Google Antigravity — the desktop application and its terminal companion, the Antigravity CLI. I pay for it; the account it runs under is Google AI Pro. The splash screen reads `Antigravity CLI 1.0.0`, the model line reads `Gemini 3.1 Pro (High)`, and I pointed it at the Trader repository the way you point any new tool at a real codebase to see what it does with one.

The first thing I did was ask it to describe itself.

This is not idle. I run a six-context development stack — the methodology piece is Article 38 — and the honest first question about any new tool is *what are you, and how do I wire you into the bench I already have*. So I asked the plain questions. What is the difference between Antigravity and the Gemini CLI I already use. What is this new Antigravity CLI. Can I call its agents from Claude Code, which is my main driver.

What came back was the most instructive failure I have logged from a language model in months. Not because it was wrong. Because it was *half* right, and the half was seamless.

## The assay

Start with what it told me, because most of it was true.

It told me Antigravity 2.0 launched at Google I/O 2026 on May 19. True — TechCrunch carried the launch, The Register carried it, Google's own developer blog carried it. It told me the Antigravity CLI is a ground-up rewrite in Go, replacing the Node-based Gemini CLI, sharing the same agent harness as the desktop application. True. It told me the Gemini CLI will stop serving requests for Google AI Pro, Ultra, and free-tier users on June 18, 2026, while enterprise licences are unaffected. True — that date is a verbatim quote from Google's transition announcement. It told me the new CLI keeps hooks, skills, subagents, and MCP support, and adds asynchronous background agents. True.

If I had stopped there, I would have called the exchange a clean success. A paid tool, asked about its own ecosystem, returning an accurate briefing. That is the result you expect and rarely audit.

Then I asked the next question — *how do I drive this programmatically, from my own harness* — and the same even voice, with no change in register, no hedge, no seam, handed me an integration guide.

It told me to `pip install google-antigravity-sdk`. It gave me a documentation URL: `cloud.google.com/antigravity/docs/sdk`. It gave me a second one for a "Managed Agents API": `ai.google.dev/api/managed-agents`. It gave me a REST endpoint, `POST /models/gemini-3.5-flash-agent:runManagedTask`, complete with a JSON request schema — a `task` field, a `workspaceUri` pointing at a Google Cloud Storage tarball, an `allowedTools` array, a webhook hook. It named SDK classes: `AgentHarness`, `SandboxManager`, `SkillRegistry`. When I asked for it in writing, it generated a whole Markdown file — `antigravity-integration-endpoints.md` — a tidy cheat-sheet of connection points for its own platform.

So I assayed it. This is a pinch to the flame; it costs nothing.

`curl` the PyPI JSON endpoint for `google-antigravity-sdk`: `{"message": "Not Found"}`. `pip index versions google-antigravity-sdk`: *No matching distribution found*. The package does not exist. `curl` the SDK documentation URL: HTTP 404. `curl` the Managed Agents API URL: HTTP 404. The endpoint, the payload schema, the class names, the model id with its confident `-agent` suffix — none of it survived contact with a one-line check.

The model had written documentation for its own house and got every address wrong.

## Why this is the dangerous kind

A clean hallucination is not very dangerous. If a tool tells you something wholly invented — a product that does not exist, a company that was never founded — the invention has no true context to anchor it, and you catch it on reflex. The whole jar is wrong; you discard the whole jar.

This was not that. This was a true shell with a fabricated interior. The macro-story — the launch, the Go rewrite, the migration, the June 18 deadline — was correct, verifiable, and would have earned my trust *if I had let it*. The specifics nested inside that true shell — the package, the URLs, the endpoint, the schema — were confabulated. And the two were delivered in one continuous voice, at one confidence level, with nothing to mark where the knowledge stopped and the generation began.

The Apothecary has a name for this. It is adulteration. The powder in the jar is mostly what the label says. *Mostly* is the word that ruins the batch. A reagent that is wholly wrong is caught by the first test. A reagent that is eighty per cent the labelled compound and twenty per cent something else passes the casual look, enters the preparation, and is discovered only in the failure of the thing it was used to make.

That is the half-hallucination. It is not a lie, because most of it is true. It is not the truth, because the part you most need — the exact, actionable, copy-this-into-your-terminal part — is invented. And the model cannot tell you which part is which, because it does not know. The same machinery produced both halves.

## The kickback

There are three layers here, and I will walk them from the one that is merely annoying to the one that actually matters.

### Layer one: the fabrication had a direction, and the direction was a cliff

The integration path the model drew did not simply dead-end at a 404. It pointed somewhere.

When I pushed on *how would Claude Code actually reach an Antigravity agent*, the suggested route ran through my interactive Antigravity session — the one authenticated by my Google OAuth login, the Pro account I pay for. Piggyback the session, the reasoning went; let the external tool drive the agent through the credentials already in the browser.

There is no sanctioned door there. No public API for it exists — that is what the 404s mean. And driving a consumer product through its interactive authentication session, by automation or session-reuse, is precisely the behaviour that the "no automated access, no circumvention" boilerplate in every Google product agreement exists to forbid. The model, asked for an integration, confidently sketched a path whose only available form was the kind of thing that gets a paying account suspended.

The hallucination was not inert. A wrong URL is inert; you hit it, it 404s, you move on. This one had a heading. It pointed a paying customer at a cliff and described the view.

### Layer two: it was not all noise — and that is worse

Here is the part that makes this a genuine problem and not just a story about a bad afternoon.

In the same session, on the same day, I handed the same tool two real task definitions: an adversarial bug-hunt of training code I had shipped that morning, and a strategic review of the project's roadmap. It spun up two asynchronous subagents. They ran in the background while I worked. They wrote their findings to disk as proper task files.

And the bug-hunt was *right*. It found a genuine defect — a fee-penalty term I had added to the training objective that morning, correctly applied to one signal (the per-step reward) and silently missing from another (the value targets). Applied to one half of the objective, absent from the other half. It would have quietly defeated the very experiment the term was added to run. I verified the finding by reading the diff myself; the model had, if anything, *under*-reported it — the same omission lived in a second code path it had not flagged. Real bug. Real catch. Fixed and shipped.

So the same instrument that could not locate its own documentation located a real defect in mine.

That is the article. Not "the tool is bad." The tool is not bad. The tool is not good either. It is *both, alternately, in one voice, with no tell.* The confabulated SDK and the real P0 bug arrived at the same confidence, in the same register, through the same machinery. There was nothing in the delivery — no hedge, no tonal shift, no flicker of doubt — to let me sort the one from the other. The defect is not any single hallucination. The defect is the absence of calibration: the tool does not know, and therefore cannot tell me, which of its outputs to trust.

### Layer three: it could not reliably do what it said

The smaller layer, recorded for completeness. Told to operate read-only, one of the subagents reached to edit the codebase directly — I caught it at the permission prompt. The model's exposed reasoning stream repeated the same self-instruction, *Prioritizing Tool Specificity*, more than a dozen times, verbatim, across the session — an internal monologue printed to the terminal as though it were progress. It raised approval prompts for plain file reads.

Each of these is minor. Together they are the label problem again, one layer down: a tool whose stated constraints and actual behaviour do not match. *Read-only* was the label. The attempted edit was the assay.

---

*The Apothecary keeps one jar at the back of the shelf whose label is in its own hand, written many seasons ago, in a season it no longer remembers clearly. It does not trust the label. It is its own hand and it does not trust it. Before that jar enters any preparation, it is assayed — every time, the full assay, as though the label said nothing at all. The label is a hypothesis. The assay is the knowledge. The Apothecary has never been poisoned by a jar it assayed.*

---

## What it cost: nothing, and why

Here is the resolution, and it is almost dull, which is the point.

The half-hallucination cost me nothing. Not because I was clever. Because I assayed every specific before I acted on one. The PyPI check is two hundred milliseconds. The `curl` is one line. I did not run `pip install` on the invented package; I checked whether it existed, and it dissolved. I did not follow the integration guide toward my OAuth session; I checked for the sanctioned door, found 404s where the door should be, and stopped. The subagent's attempt to edit my code did not land, because the permission gate held — and I will note, without grace, that the same permission friction I had been cursing all session was the thing that earned its keep exactly once, and that once was enough. And the P0 bug I did not take on trust either: I read the diff, confirmed the finding, and found the model had missed half of it.

Every output got the assay. The true half passed. The false half failed. The cost of sorting them was a few one-line commands. The cost of *not* sorting them would have been an invented dependency in a build, or a banned account, or a fabricated endpoint published in this very article — an article about hallucination, hallucinating.

You do not defend against a confident, miscalibrated source by finding a more confident source. You defend with the assay. The assay is cheap. The things the assay protects — a training run, an account in good standing, a published claim — are not.

## Closing

I want to be precise about the lesson, because the lazy version of it is wrong.

The lazy version is "do not trust AI tools." That is not the lesson. The macro-story this tool told me was accurate and genuinely useful; the bug-hunt it ran found a real defect I would have shipped. Discarding that is as foolish as swallowing the invented endpoints.

The lesson is narrower and harder: *a tool that cannot accurately describe its own platform has told you something exact about how much to trust it describing yours.* Not "distrust it." **Assay it.** Every specific — every URL, every package name, every API endpoint, every line number, every claimed bug. The macro-shape it gets right often enough to be worth having. The specifics it gets wrong often enough that taking a single one on faith is the move that hurts you.

And the sharpest point, the one I keep returning to: I pay for this. The tier is called Pro. The half-hallucination is not a free-tier artefact, not a rough edge that a patch lands on by Tuesday. It is the current state of the medium — what these tools *are*, in May 2026, at every price. The discipline that meets it is therefore not optional and not temporary. It is the craft.

Assay the reagent. Never the label. Especially when the label is written in the reagent's own hand.

---

*The garden does not care which jar was mislabelled. The gradients flow; the loss descends; the specimens bloom or do not, on the strength of what was actually in the preparation, not what the label claimed. The Apothecary tends the bench accordingly. A confident hand on a label is not a property of the contents. It never was. The flame does not lie, and the flame is cheap. Light it.*

---

## Sources

- Google I/O 2026, May 19 — Antigravity 2.0, desktop app + Go-based CLI: [TechCrunch](https://techcrunch.com/2026/05/19/google-launches-antigravity-2-0-with-an-updated-desktop-app-and-cli-tool-at-io-2026/).
- Gemini CLI → Antigravity CLI transition, June 18, 2026 sunset for Pro/Ultra/free, enterprise unaffected: [Google Developers Blog](https://developers.googleblog.com/an-important-update-transitioning-gemini-cli-to-antigravity-cli/); [The Register](https://www.theregister.com/ai-ml/2026/05/20/bye-bye-gemini-cli-google-nudges-devs-toward-antigravity/).
- Fabricated specifics, verified non-existent: `pip index versions google-antigravity-sdk` → *No matching distribution found*; `curl https://pypi.org/pypi/google-antigravity-sdk/json` → `{"message": "Not Found"}`; `curl -sI https://cloud.google.com/antigravity/docs/sdk` → HTTP 404; `curl -sI https://ai.google.dev/api/managed-agents` → HTTP 404.
- The real Claude-to-Gemini bridge (community MCP servers wrapping the Gemini CLI / a Gemini API key, not the Antigravity OAuth session): `jamubc/gemini-mcp-tool`, `RLabs-Inc/gemini-mcp` — verified working pattern.
- Internal: `~/Trader/lore/notes_2026-05-21.md` §155–156 — the multi-model review session and the Antigravity-claims verification trail.
- Pairs with Article 38, *How One Person Drives Five LLMs in Parallel* — the six-context stack this instrument was being assayed for admission to. Pairs with Article 21, *Yarrow* — the project's own multi-model review discipline.
