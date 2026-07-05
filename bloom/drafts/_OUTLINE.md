# Series Outline — "The Apothecary's Lab Notebook"

10 articles, ~3000 words each, ~30k words total. Each article has a kickback (a moment causality fights back). Series arc: hubris → friction → reckoning → pivot.

Style: hybrid (Lukas first-person + occasional Apothecary interludes). See `_STYLE.md`.
Source manifest per article: see `_CHRONOLOGY.md`.

---

## Article 1 — "I Thought I Was Going to Break Causality"
**Phase:** 0 (Dec 2025). **Kickback:** The very phrase "Result: superhuman performance" in my own design doc.
**Thesis:** Why I believed a small modular MuZero+TRM stack would beat quant funds. The seduction of AlphaZero analogies. The moment of writing "30–50% annual return" with a straight face.
**Spine:**
- Opening hook: my desktop wallpaper in December was a printed page of `ALPHAZERO_TRADING_VISION.md`.
- The argument for modular > monolithic.
- The math: how MuZero's learned world model + MCTS planning + adversarial market = (allegedly) emergent novel tactics.
- The "Move 37" daydream.
- Kickback: quoting my own grandiose lines back at me; the Apothecary closes with "the specimen requires attention."
**Sources:** `lore/ALPHAZERO_TRADING_VISION.md`, `lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`, `lore/MCTS_Trading_Agent_Research_2025_Update.md`, first Gemini session.

---

## Article 2 — "Physics Core: A Differentiable Market I Could Differentiate"
**Phase:** 0 (Dec 2025). **Kickback:** It was never integrated into the training pipeline. The most beautiful code I ever wrote is dead code.
**Thesis:** NeuralOGCM-style hybrid physics + neural corrector for trustworthy long-horizon prediction. Why I thought differentiable α/β/σ/γ would solve the hallucination problem. The math of the ODE solver. Why I left it on the shelf.
**Spine:**
- The dream: 200-step rollouts with no divergence.
- The architecture: physics_core.py / neural_corrector.py / market_ode.py.
- Forward pass 0.12ms benchmarks — the moment I thought I had it.
- Kickback: it never connected to the real training loop. Why.
- What I'd keep, what I'd burn.
**Sources:** `lore/NEURAL_MARKET_PHYSICS_DESIGN.md`, `lore/NEURAL_MARKET_PHYSICS_SUMMARY.md`, `lore/notes_2025-12-30.md`.

---

## Article 3 — "The Berserker That Knew It Was a Berserker"
**Phase:** 0 (Dec 2025). **Kickback:** I was the berserker. The model never had opinions; I had them for it.
**Thesis:** Designing intentional constraints — short horizon, no self-awareness, five-layer safety oath — as a design philosophy. The Buddhist trade-off of giving up capability for stability.
**Spine:**
- The metaphor (and why the metaphor is doing work).
- Five-layer oath: Self → Wrapper → Monitor → Exchange → External.
- Why short horizon is a feature.
- Kickback: the philosophy of constraint became cope when the model didn't work — "well it was *supposed* to be limited."
- What survives.
**Sources:** `lore/BERSERKER_DESIGN_PHILOSOPHY.md`, related ADRs.

---

## Article 4 — "Encoder Output, Repeated 64 Times: The Information Loss Audit"
**Phase:** 1 (Jan–Feb 2026). **Kickback:** The encoder was producing 22 GB of mostly-the-same-row tiles. Gemini found it. I had been training on water.
**Thesis:** The first real reckoning. GPU at 5–17% util. The TASK_136 back-and-forth with Gemini. The 64× compression ratio. The lesson about *trusting your own dataloader*.
**Spine:**
- Symptom: GPU starving.
- The collab: my hypothesis, Gemini's counter, my response, the fix.
- The math: why row-wise tile redundancy implies a broken pooling layer upstream.
- Kickback: this had been training for *months*.
- Direct quotes from `tasks/136a-k.md` and `tasks/136b-gemini-solution.md` (both sides).
**Sources:** `tasks/136a-k.md` + 136b/c/d-gemini-solution.md, `lore/2026-01-04_information_loss_audit.md`, `lore/notes_2026-01-20.md`.
**Snapshot Claude sessions** (mounted SSD path in `_CHRONOLOGY.md`): `9774f034*.jsonl` (Jan 11 first prompt), `55be4183*.jsonl` (Jan 13 cli check), `79d73903*.jsonl` (Jan 22 patch-verification grind), `4fe59e46*.jsonl` (Jan 28 "by gemini task 120" relay). Pull verbatim Lukas-side quotes — especially the typo-strewn one-handed prompts that capture the work texture.

---

## Article 5 — "22 Silent Killers: The BUG_MAYHEM Audit"
**Phase:** 1 (Feb 2026). **Kickback:** Each individual bug was small. Together they were the reason nothing worked.
**Thesis:** The Alpha-scale mismatch, BC masking broken, PER broken, PopArt desync, phantom rewards, "agent state doesn't reflect burn losses." A composite portrait of "code that runs but doesn't learn."
**Spine:**
- Opening: the disquieting moment of a working pipeline that learns nothing.
- Walk through 5–6 of the 22 bugs in detail (the most instructive).
- The PER bug specifically — why importance sampling weights were off by a TD step.
- Kickback: realizing the *combination* matters; you can't ablate-fix when 22 things are simultaneously broken.
**Sources:** `tasks/BUG_MAYHEM_MASTER_REPORT.md`, `tasks/BUG_HUNT_2026-01-31.md`.
**Snapshot Claude sessions:** `76fea01e*.jsonl` (Jan 29 19:53, *the* "money printing / future leakage" task prompt — primary), `09cbe342*.jsonl` (Jan 31 11:41 crash/recovery, largest session at 4857 lines), `5925cd72*.jsonl` (Jan 31 12:19 epoch-27 catastrophic-loss debug). The Claude side of the bug hunt is in these three files. Quote verbatim the task framing prompt and at least one Claude response identifying a specific bug.

---

## Article 6 — "Better Classifier, Worse Trader: The Anti-Correlation"
**Phase:** 2–3 (Apr–May 2026). **Kickback:** Three independent experimental setups, three confirmations. The encoder was not the bottleneck. It never was.
**Thesis:** TASK_737, 741, 738 each showed: loss ↓, Bal.Acc ↑, Alpha collapses. Why the policy over-trades when the classifier is sharper. The moment I stopped proposing encoder mechanisms.
**Spine:**
- TASK_738 the worst case: Bal.Acc +7.48pp, Δα −18.44pp, trade count 136 → 13,246 (50× over-trading).
- The Sharpe collapse: +8.68 → −27.20.
- The mechanism: value-head saturation under sharper logits → argmax becomes more confident → more trades → fees eat the edge.
- Kickback: my entire research direction was downstream of a misidentified bottleneck.
- *Apothecary interlude: "the specimen flowered, then withered overnight."*
**Sources:** `lore/notes_2026-05-15.md` §110, `project_classifier_alpha_anti_correlation.md`, `project_task741_result.md`.

---

## Article 7 — "TASK_737: The Champion That Wasn't"
**Phase:** 3 (May 13). **Kickback:** I drafted the victory writeup before running the second seed.
**Thesis:** A 4-cell ablation. no_aux_solo +6.21% Alpha. a1b1 baseline −1.99%. The story of writing "the two heads interact NEGATIVELY (−13pp combined effect)" as if I had measured it.
**Spine:**
- The ablation design (rank × aux 2×2).
- The result table. The Slack/diary moment of "we have something."
- The mechanism I invented to explain the gap.
- Kickback teaser: TASK_743 is loading.
**Sources:** `lore/notes_2026-05-13.md` §95.

---

## Article 8 — "Five Seeds and a Funeral"
**Phase:** 3 (May 14). **Kickback:** TASK_743. The most consequential single data point of the project.
**Thesis:** Seeds 7/13/23/31/42 → range −4.38 to +7.82, spread 12.2pp, mean +3.61%. The champion was an upward outlier. The operational lesson: anything < 5pp on a single seed is untrustworthy.
**Spine:**
- The reseed.
- Seed 13's −4.38%. The exact moment of reading that number.
- Quote the lore entry verbatim: *"Seed 42 was an upward outlier."*
- The retraction: rewriting `MEMORY.md` and `runs/737_eval/result.md`.
- Kickback: every prior single-seed claim in the project is now suspect.
- *Apothecary interlude.*
**Sources:** `lore/notes_2026-05-13.md` §99, MEMORY.md edits.

---

## Article 9 — "From 60 Minutes to 93 GPU-Days: The Bella Precompute Disaster"
**Phase:** 3 (May 14). **Kickback:** The SPEC said 30–60 minutes. The reality was three orders of magnitude.
**Thesis:** An honest postmortem of an estimation failure. 270 periods × 258k windows × k=4 at 8.6 windows/sec = 2,241 GPU-hours = 93 GPU-days. What I forgot to multiply.
**Spine:**
- The plan: precompute Bella visions for all training data.
- The path bug that masked the scale bug.
- The arithmetic I should have done on the napkin.
- Why I didn't.
- Kickback: I had spent two days waiting for a job that would have finished in 2027.
- What I did instead.
**Sources:** `lore/notes_2026-05-13.md` §100.

---

## Article 10 — "Eval-MCTS Asymmetric Lift: When the Scoreboard Was Wrong"
**Phase:** 3 (May 19–20). **Kickback:** 14 evaluations, 0 promoted. The deploy-truth operator killed every mechanism.
**Thesis:** The benchmark used argmax; the live trader used MCTS. The entire campaign was scored under the wrong operator. When I recalibrated, mechanisms that looked positive became null and mechanisms that looked negative became less so — the baseline got lifted more than the variants. Single-mechanism hypothesis space exhausted.
**Spine:**
- The discovery of the benchmark↔live operator mismatch.
- The recalibration table (746/734/749/757).
- The mechanism: value-head saturation as a ceiling.
- Why this is the most useful negative result of the project.
- The structural pivot: TASK_761 roadmap (richer oracle, reward shaping, entropy reg, incompatible dataset).
- Closing: *"Bella's vision was beautiful. It was not a market."*
**Sources:** `lore/notes_2026-05-20.md` §151, `experiments/759_mcts_eval/SPEC.md`, `runs/759_mcts_eval/result.md`, `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`, `runs/recommendation_2026-05-20/analysis.md`.

---

## Dispatch order
1, 2, 3 first (Phase 0 origin trilogy). Once style is approved, 4–5 (Phase 1 reckoning), then 6 (encoder saga as bridge), then 7–8 (the reversal twin), then 9 (the precompute disaster), then 10 (the campaign closer).
