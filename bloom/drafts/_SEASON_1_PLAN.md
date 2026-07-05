# Season 1 — territory classification & article plan

Source-material classification for the bloom series after Articles 0–10 landed. Goal: identify what's article-shaped, what would be repetitive, and what should wait. User instruction: "up to 60 additional articles … no repetitive slop … scope of current bloom is this season."

Realistic season-1 estimate: **11 done + 12–18 more** = 23–29 articles total. Beyond that, repetition risk climbs sharply because the project has a limited number of *distinct* kickbacks and structural lessons; further articles would mostly be detail-amplifications of existing arcs.

---

## Inventory snapshot

- **`tasks/`** — 294 files. Many are sub-task fragments of larger arcs (e.g. TASK_115a–r is one bug-hunt epic in 14 pieces; TASK_127a–r is the master training audit in 15+ pieces).
- **`lore/`** — 144 files. ~40 design docs in caps, ~100 daily session notes.
- **`docs/`** — research papers + Gemini Deep Research reports.
- **SSD snapshot** — 121 Claude Code sessions, Jan–Feb 2026.
- **`~/.gemini/tmp/trader/chats/`** — Gemini CLI sessions, Nov 2025 onward.
- **`Claude.ai export`** — 823 conversations, of which ~6 are trading-related (the rest are mining business + life). The Feb 11 / May 10 / Jul 14 Janus traces.

---

## Twelve thematic arcs that have NOT yet been articled

### A. **TASK_115 — the shape-mismatch bug-hunt epic** (14 sub-tasks)
A multi-week mosaic of OOM, shape mismatches, JIT failures, numerical stability, silent failures, integration seams. Article 5 covered BUG_MAYHEM at Phase 1; this is a *different*, later, multi-front campaign. Each sub-task has a distinct lesson — but one omnibus article is probably the right shape: *"Fourteen Days of Shape Mismatches"*. Risk: overlap with Article 5's tone. Mitigation: lean into the *integration-seam* lessons that Article 5 didn't touch.

### B. **TASK_127 — the master training audit** (15+ sub-audits)
MuZero dynamics, replay buffer, PopArt, entropy/policy collapse, MCTS, optimizer, target network, loss function, LR. This was a *systematic* audit, not a bug hunt — different posture. Article-worthy: *"The 15-Subagent Audit and What It Caught"*. Each subagent audited one component cold. Strong article on the *epistemic methodology* (cold-eyes audit vs daily-collaborator blindness) — Article 4's "Gemini caught it" observation expanded into a process.

### C. **TASK_128 — the epoch-90 training hang** (6 sub-tasks)
Training would deadlock at epoch 90, every run, no clear cause. Sub-tasks walk root-cause analysis, debug plan, fix. Article: *"The Hang at Epoch 90: An RCA"*. Postmortem-of-a-deadlock. Pairs technically with Article 5 but has a different shape — debug vs survey.

### D. **TASK_126 — the reward-spike / path-solver-sign-bug saga** (~13 sub-tasks: a–m)
A reward function went catastrophic; the chase found a sign bug in the oracle's path solver. Article: *"The Sign Bit That Ate a Quarter"*. Concrete, dramatic, has a clear villain (one minus sign).

### E. **Causal Expert Imprinting (TASK_125, plus docs/imprinting/)**
The supervised-imprint phase of Foxglove has its own theoretical lineage — the "Causal Expert Imprinting.txt" paper, the imprinting-as-distillation framing. Article: *"What Foxglove Actually Does: Causal Expert Imprinting"*. Pairs with Article 6 but on the *training-side* instead of the *eval-side*.

### F. **The Bella generator deep dive (TASK_108–114, TASK_753–755)**
Bella's architecture — latent diffusion, skeleton-conditioned, Mamba, conv diffusion, regime conditional, trajectory shape distill. Article 9 covered the *operational disaster* but never explained *what Bella is* in detail. Article: *"Bella: A Diffusion Model for Markets That Don't Exist"*. Math-heavy. Pairs with Article 2 (physics core) — both about generators of synthetic markets, different routes.

### G. **The Oracle saga (TASK_119, 125, 133, 141, 157, 673, 681, 695, 697)**
The oracle is the supervised target. It has been re-formulated five+ times. Each reformulation is a kickback in miniature. Article: *"The Five Oracles: How a Supervised Target Keeps Outgrowing Itself"*. Pairs with Article 10's TASK_761 oracle-k direction — sets up the next campaign.

### H. **Realism & market microstructure (TASK_047–055)**
Limit-order fills, fill priority randomization, episode-length randomization, position limits, borrowing cost shorts, curriculum realism integration. Each is small; together they're a coherent "the simulator got more honest" arc. Article: *"Teaching the Simulator to Hurt"*. Single article, walks through 4–6 of the 8 sub-tasks.

### I. **The CMP 90HX saga (TASK_688, plus the removal in May 2026)**
A mining card unlocked for training use, eventually removed. *"The Mining Card That Wasn't a Mining Card"*. Has a clear arc, has hardware-specific texture, is publishable workstation-only (no server detail). Bonus: connects to the mining-business backdrop from Article 0.

### J. **The TASK_761 forward directions (TASK_762–765)**
Reward shaping, policy entropy reg, oracle k reformulation, incompatible dataset. Each could be its own article *once they run*. Right now they are pre-experiment — articles would be hypothesis-stage. Probably **wait for Season 2** to write the result articles. **Season 1 risk**: writing a hypothesis article without a result becomes the kind of "discourse" Article 0 dunked on. Defer.

### K. **Yarrow side project**
Multi-model code review (`yarrow/`). Parked per MEMORY. Article-worthy as a side note: *"Yarrow: When the Code Reviewer Is the Code"*. Light touch. Could fit as a season-1 outlier.

### L. **The dojo + curriculum system (TASK_045, lore design docs)**
DOJO_MODULAR_ARCHITECTURE, SYNTHETIC_CURRICULUM_DESIGN, SELF_PLAY_DOJO_VISION. The *training environment* itself is a piece of architecture. Could be one article: *"The Dojo: How Foxglove Sees the World"*.

---

## Proposed Season 1 batch (Articles 11–22)

These are the next 12 to dispatch. Each has clear non-overlapping source material and a distinct kickback.

| # | Title (working) | Arc | Source emphasis |
|---|---|---|---|
| 11 | Fourteen Days of Shape Mismatches | A (TASK_115) | Bug-hunt epic, integration seams |
| 12 | The 15-Subagent Audit | B (TASK_127) | Cold-eyes methodology |
| 13 | The Hang at Epoch 90 | C (TASK_128) | Deadlock RCA, postmortem shape |
| 14 | The Sign Bit That Ate a Quarter | D (TASK_126) | One minus sign, reward chaos |
| 15 | What Foxglove Actually Does (Causal Expert Imprinting) | E | Imprinting theory + training-side |
| 16 | Bella: A Diffusion Model for Markets That Don't Exist | F | Architecture of the market generator |
| 17 | The Five Oracles | G | Five reformulations, kickback in miniature each |
| 18 | Teaching the Simulator to Hurt | H | Realism/microstructure block |
| 19 | The Mining Card That Wasn't a Mining Card | I (CMP 90HX) | Hardware adventure, workstation-only |
| 20 | The Dojo: How Foxglove Sees the World | L | Training environment architecture |
| 21 | Yarrow: When the Code Reviewer Is the Code | K | Side-project note |
| 22 | What a Year of Negative Results Looks Like (season closer) | meta | A reflection article distilling the season |

That's 12 more. Total Season 1 = **23 articles ≈ 100,000 words**.

---

## Why not the full 60

After Article 22, the remaining material is:
- **Sub-task fragments** of arcs already covered (variants of Article 6's classifier-Alpha story, variants of Article 10's structural pivot).
- **Hypothesis-stage articles** (the TASK_762–765 pre-experiment specs) that would read as discourse without results.
- **Daily-note minutiae** that don't have a kickback shape.
- **Repetition risk**: a fourth bug-hunt article, a third oracle-reformulation article, would dilute the existing ones.

If Season 2 runs (after TASK_761's four directions produce results — probably Q3 2026), the natural second-season material is:
- One article per TASK_761 direction (4 articles)
- The actual results of the structural pivot (~3 articles)
- Any new kickbacks from the next campaign (~3 articles)
- One reflective season-2 closer

That'd be ~11 articles for Season 2. Estimated total over two seasons: **34 articles ≈ 150,000 words**. Well under the 60-article ceiling, with the discipline that each one has a distinct kickback or distinct technical content.

---

## Recommendation

**Dispatch Articles 11–22 in parallel** (12 writer subagents, background). This consumes the season's clearly-non-repetitive material, ends with a meta reflection article, and leaves the *post-experiment* writeups for Season 2 once results land.

If you want a shorter Season 1, the *core 7* to dispatch first are 11, 12, 14, 16, 17, 18, 19 — these have the clearest non-overlap with Articles 0–10 and the strongest standalone kickbacks. The remaining five (13, 15, 20, 21, 22) are also good but could be dropped if you'd rather conserve budget for Season 2.
