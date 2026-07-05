# Research Index — `~/Trader/docs/` material to ventilate

The user wants the series to surface the research backbone — these are the papers, reports, and Gemini Deep Research artifacts already in the repo. Each article should cite at least 1–2 from this list where relevant. Don't summarize them in bulk; *use* them — quote a line, explain a concept, attribute the idea.

## Big-picture research reports (Gemini Deep Research / Claude / etc.)

- **`docs/AI_bleed research/big gemini.txt`** — *"Coupled Policy-Prediction Collapse in Model-Based RL: A Comprehensive Analysis of Dual Failure Modes in Stochastic Trading Environments."* Synthesis of (a) Entropy Collapse in actor-critic optimization, (b) Dormant Neuron Phenomenon in non-stationary learning, (c) Trivial Solution Trap in Neural ODE modeling. Diagnoses exactly the May 2026 symptom: policy → HOLD, prediction head → identity. **→ Cite in Articles 2, 6, 7, 8, 10. Especially 6 (classifier–Alpha anti-correlation) and 10 (eval-MCTS / structural pivot).**
- `docs/AI_bleed research/DS3.txt`, `kimi.txt`, `gm_cli.txt`, `cl_cli.txt` — other model perspectives on the same problem; multi-model triangulation.
- `docs/AI_bleed research/compass_artifact_wf-2955a86b-...md` — long-form synthesis.

## Per-article suggested citations

### Article 1 — break_causality
- `docs/research/TinyRecursiveModels-main/` — the TRM paper that justifies the recursive-reasoning architecture choice. Cite once for the "MuZero + TRM" claim.
- `docs/research/arXiv-1911.08265v2` — MuZero original (Schrittwieser et al. 2019).
- `docs/research/arXiv-2501.12948v2` — likely a recent model-based RL paper; check title and cite if relevant.
- `docs/research/mctx-main/` — DeepMind's MCTS library. Mention as the planning substrate.

### Article 2 — physics_core
- `docs/research/NEURAL_MARKET_PHYSICS_DESIGN.md` (duplicate of lore version)
- The NeuralOGCM paper (Wu et al. 2025) — mentioned in design doc; cite once.
- `docs/research/arXiv-2510.04871v1` / `arXiv-2512.11525v1` — likely Neural ODE / diffusion refs; check titles.

### Article 4 — encoder / information loss audit
- `docs/imprinting/Causal Expert Imprinting.txt` + `docs/imprinting/compass_artifact_wf-7a81fbfc-...md` — Foxglove's theoretical grounding.

### Article 5 — BUG_MAYHEM
- `docs/AI_bleed research/big gemini.txt` — Dormant Neuron Phenomenon section maps directly to several of the 22 bugs.

### Article 6 — classifier–Alpha anti-correlation
- `docs/AI_bleed research/big gemini.txt` — **primary citation.** The "Coupled Policy-Prediction Collapse" framing is *the* lens for this article.
- `docs/Algorithmic Alpha/` — check for Alpha-definition refs.

### Article 7 — TASK_737 illusion
- `docs/Hindsight-optimal/` — relevant to the question "what does the oracle even know?"

### Article 8 — Five seeds and a funeral
- Multi-seed variance is a known issue in RL; cite generally.
- `docs/research/arXiv-2501.00663v1-titans` — TASK_749 Titans-MAG ran in this window. Reference if the article touches it.

### Article 9 — Bella precompute disaster
- `docs/UNIFIED_MARKET_GENERATION_ARCHITECTURE.md`
- `docs/Market Generation Research Plan.md`
- `docs/market_diffusion_architecture.md`

### Article 10 — Eval-MCTS asymmetric lift
- `docs/AI_bleed research/big gemini.txt` — the remediation section likely proposes things TASK_761 echoes (entropy regularization, reward shaping). Cite the convergence.
- `docs/research/PREDICTION_MODEL_ANALYSIS.md`
- `docs/weight merging/` — relevant if 759/760 explored ensembling.

## Rule of thumb

Don't pad with citations. One well-placed quote from `big gemini.txt` doing real explanatory work beats four name-drops. The math comes from the paper; the *story* comes from the lore notes; the writer's job is to weave them.
