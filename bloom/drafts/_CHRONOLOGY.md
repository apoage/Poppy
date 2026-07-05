# Trader Project Chronology — "The Apothecary's Lab Notebook"

**Timeline:** Late November 2025 → May 20, 2026 (6 months)
**Scope:** MCTS+MuZero+TRM crypto trading agent, from initial vision through systematic failure of single-mechanism improvements, ending in a structural pivot.

NOTE: 2025-era Claude Code conversations were lost when sessions auto-cleaned (resume was not in use). Pre-2026-01-11 Claude reasoning must be inferred from lore docs (often co-signed "Lukas & Claude Code") or from Gemini sessions, which survive in `~/.gemini/tmp/trader/chats/`.

**SSD SNAPSHOT** mounted at `/mnt/data/home/lukas/.claude/projects/-home-lukas-Trader/` covers **2026-01-11 → 2026-02-13** — 121 sessions, 32,094 timestamped messages. This is the Phase 1 Claude side; Articles 4 and 5 draw from here. Pull quotes via `python3 -c "import json; [print(json.loads(l).get('message',{}).get('content','')[:200]) for l in open('<file>') if 'user' in l]"` or similar.

Key sessions in snapshot:
- `9774f034-d9c1-489b-a70c-94dfba6b586c.jsonl` — 2026-01-11 21:07, FIRST prompt: *"hi take look on this codebase and whole project tell me if its possible oneshoot prompt this ??"*
- `55be4183-bf68-4874-8c07-095eee979767.jsonl` — 2026-01-13 22:27, 3158 lines. Opener: *"are you up to date .. i mean cli"*
- `7ef18890-a861-43fa-a300-2f8bf927d791.jsonl` — 2026-01-19, *"please clean up your memory i dont neen fo example..."*
- `79d73903-4ef0-47a9-a778-18e83c65549d.jsonl` — 2026-01-22 22:17, 4023 lines, GPU-status / patch-verification grind. Opener: *"ok cool .. please check again your patch of im not sure if it was fully sucesfull"*
- `28d7b8a1-a297-45ad-842f-8ce3fd8e091f.jsonl` — 2026-01-24, *"hi load the context and last notes"*
- `0c105549-f36f-4e1f-8e02-b50c84825d01.jsonl` — 2026-01-25, 3533 lines.
- `4fe59e46-3e97-4731-8f87-6093498cf158.jsonl` — 2026-01-28, *"ok have rebooted implemented some fixes .. by gemini task 120 .. now need command to star env .. start ds24 start traini[ng]"* — GEMINI→CLAUDE RELAY EVIDENCE.
- `76fea01e-62b0-456f-a0b2-fb8660946da6.jsonl` — 2026-01-29 19:53, *"TASK: Find all 'money printing' and 'future leakage' bugs in this trading RL system."* — **BUG_MAYHEM SESSION for Article 5.**
- `09cbe342-43ea-4d51-9fa8-f2279c03f0c9.jsonl` — 2026-01-31 11:41, 4857 lines (largest). Opener: *"ok i had crash .. pos[itions?]"*
- `5925cd72-2ceb-4fe0-9f84-e99c41f0e8e7.jsonl` — 2026-01-31 12:19, *"PROBLEM: During training, epoch 27 showed a catastrophic..."* — loss-spike debug.
- `1f4eebe6-eb9c-4cb5-a7c2-d13d078dfd09.jsonl` — 2026-02-07, 4006 lines.
- `56d22a72-c05b-405b-bb8b-1ef793a859a3.jsonl` — 2026-02-10 20:56, *"hi there .. tell me what is my IT and AI level based on this codebase .."* — atmosphere quote for any article.

---

## Phase 0: Origin & Fever Dream (Nov–Dec 2025) — "Break Causality" Era

**Emotional beat:** Supercharged confidence. The sense of holding a tool that breaks the rules of markets. Designing beautiful physics cores, adversarial training loops, multi-market transfer learning. The future is *here*.

### Key Timeline Events
- **2025-11-30** — First Gemini sessions. Source: `~/.gemini/tmp/trader/chats/session-2025-11-30T20-27-d4edcf4a.json`.
- **2025-12-19** — `ALPHAZERO_TRADING_VISION.md` authored (Lukas & Claude Code). Core claim: Modular > Monolithic. Trading agent as "one specialized module in a multi-agent general intelligence system." Curriculum Levels 0–22, AlphaZero self-play, transfer learning across markets.
- **2025-12-19** — `NEURAL_MARKET_PHYSICS_DESIGN.md`. NeuralOGCM-inspired hybrid physics+neural ODE for prediction stability.
- **2025-12-19** — `BERSERKER_DESIGN_PHILOSOPHY.md`. Intentional constraints — short horizon, no self-awareness, five-layer safety oath.
- **2025-12-30** — Physics core + corrector + ODE integration: forward 0.12ms, 10-step rollout 1.2ms. *"All tests passed!"*
- **2025-12-31** — DS56/DS78 hyperparameter tournament dashboards; 48-config search via successive halving. Benchmark H08: bull −0.72, bear +0.12, sideways +0.01, ftx_crash +0.63.

### Notable Quotes
> "Not Building: Single-purpose narrow AI (just trading). Actually Building: One specialized module in a multi-agent general intelligence system."
> — `lore/ALPHAZERO_TRADING_VISION.md`

> "This architecture is designed for short horizons and front-line trading. Like a berserker: strong, clever, can move on the battlefield."
> — `lore/BERSERKER_DESIGN_PHILOSOPHY.md`

> "Result: superhuman performance."
> — `lore/ALPHAZERO_TRADING_VISION.md`, "Expected Performance"

### Mistakes Visible in Retrospect
- Physics core (6 learnable parameter classes) never integrated into training pipeline.
- Curriculum Levels 0–22 designed; only 0–4 implemented.
- Adversarial market agent concept abandoned silently by Feb 2026.

---

## Phase 1: First Reckoning (Jan–Feb 2026) — Gemini Audits & Bug Mayhem

**Emotional beat:** Confidence meets friction. Early training runs show mysterious failures. A 22-bug audit reveals "silent killers" in gradient flow, reward scaling, sampling.

### Key Timeline Events
- **2026-01-04** — Information loss audit begins. Gemini deployed. Source: `lore/2026-01-04_information_loss_audit.md`.
- **2026-01-20** — TASK_105 data loading hang. `dojo_v3_historical.py` hangs at Batch 5 when requested data range exceeds DB max. DB had continuous data only to ~March 2025; a sparse point at 2025-12-12 made MAX(timestamp) misleading.
- **2026-01-30 → Feb** — TASK_136 pre-encoding bottleneck (Gemini collab). Source: `tasks/136a-k.md` + `tasks/136b-gemini-solution.md` etc. GPU utilization 5–17% during BC training. Root cause: Python loop + repeated encoding. Encoder output discovered to be heavily redundant (row-wise tiles), compression 64×; memory 22 GB → 338 MB.
- **2026-02** — BUG_MAYHEM_MASTER_REPORT. 22 unique issues: Alpha-scale mismatch, BC masking broken, PER broken, PopArt desync, phantom rewards.

### Notable Quotes
> "All 27 channels follow this pattern [row-wise tile]. Compression ratio: 64×."
> — `tasks/136a-k.md`

> "Agent state doesn't reflect 'burn' losses seen by the reward signal."
> — `tasks/BUG_MAYHEM_MASTER_REPORT.md`

---

## Phase 2: Foxglove & the Encoder Saga (Mar–Apr 2026)

**Emotional beat:** Shift from "bugs to fix" to "the encoder design itself is suspect." Model learns to classify assets well (Bal.Acc ↑) but trades worse (Alpha ↓). The system is *fighting itself*.

### Key Events
- Foxglove Phase 1 training. 27-channel encoder baseline. SegmentPool (TASK_710/712) cycles all 17 regimes simultaneously.
- TASK_733 — return-weighted trade loss for asset selection precision.
- Encoder variants tested. The pattern that will define the next phase: improving classifier (loss ↓, Bal.Acc ↑) → over-confident policy → over-trading → Alpha collapse.
- TASK_737 ablation table set up (rank × aux 2×2). This becomes the spark of Phase 3.

### ADRs / Design Docs in this Phase
- `lore/adr/0003-encoder-v3-32ch-cross-sectional.md`
- `lore/adr/0005-dir-acc-ceiling-decomposition.md`
- `lore/adr/0007-film-injection-post-temporal-pool.md` (sets up TASK_741 Mint)

---

## Phase 3: The Bella/MCTS Reversal Cascade (May 2026) — Collapse & Pivot

**Emotional beat:** The most dramatic phase. A champion emerges (TASK_737 no_aux_solo +6.21%), then multi-seed destroys it (avg +3.61%, range −4.4 to +7.8). Bella precompute blocked by 3-orders-of-magnitude scale miscalculation (93 GPU-days, not 1 hour). All 14 mechanisms fail under the deploy-truth operator.

### TASK_737 — The Illusion
- **2026-05-13 morning** — TASK_737 4-cell ablation. Source: `lore/notes_2026-05-13.md` §95.
  - no_aux_solo: **+6.21% Alpha**
  - a1b1_repro: **−1.99% Alpha**
  - Conclusion (premature): "The two heads interact NEGATIVELY (−13pp combined effect)."

### TASK_743 — The Crusher
- **2026-05-14 morning** — TASK_743 5-seed no_aux_solo. Source: `lore/notes_2026-05-13.md` §99.

| Seed | Alpha | Status |
|------|-------|--------|
| 42 (ref) | +6.21% | ✅ |
| 7 | +2.84% | ✅ |
| 13 | **−4.38%** | ❌ |
| 23 | +7.58% | ✅ |
| 31 | +7.82% | ✅ |
| **AVG** | **+3.61% ±4.7pp** | **NULL** |

> "Seed 42 was an upward outlier. The TASK_737 finding is UNRELIABLE — driven by single-seed variance. Single-seed comparisons on this setup are untrustworthy ≤ ~5pp Alpha gap."
> — `lore/notes_2026-05-13.md` §99

### Bella Precompute Disaster
- **2026-05-14** — Bella precompute launched, FAILED. Source: `lore/notes_2026-05-13.md` §100.
  - Bug: relative vs absolute path
  - The scale disaster: 270 periods × 258k windows × k=4, observed rate 8.6 windows/sec → 270 × 8.3h = **2,241h ≈ 93 GPU-days**. SPEC estimate ("30–60 min") **off by 3 orders of magnitude**.

### Tier-2 Mechanism Sweep
- **2026-05-15** — Source: `lore/notes_2026-05-15.md`.
  - **TASK_738 legacy_27ch:** Δα **−18.44pp** despite Bal.Acc **+7.48pp**. Trade count 136 → 13,246 (50× over-trading). Sharpe +8.68 → −27.20.
  - **TASK_739 dense_sign:** Δα −2.63pp; dir_acc ceiling not broken.
  - **TASK_734 flow_matching:** Δα **+3.38pp**, all 6 periods positive. "First credible Tier-2 candidate."
  - **§110:** "Better classifier → worse trader" confirmed 3 independent times. *"The bottleneck is downstream of the encoder."*

### Campaign Collapse (2026-05-20)
- TASK_759 eval-MCTS recalibration. Source: `lore/notes_2026-05-20.md` §151.

| Variant | argmax Δα | eval-MCTS Δα | Shift |
|---------|-----------|--------------|-------|
| 746 oracle | +0.72pp | −1.20pp | −1.92pp |
| 734 flow | +0.20pp | −0.15pp | −0.35pp |
| 749 Titans | −3.11pp | −0.33pp | +2.78pp |
| 757 bella | −5.80pp | −0.52pp | +5.28pp |

> "Eval-MCTS is a ceiling-bounded correction. It rescues ckpts where argmax underuses the value head. High-α ckpts that already saturate don't move."
> — `lore/notes_2026-05-20.md` §151

> "14 evaluations, 0 promoted under eval-MCTS (deploy-truthful operator). Single-mechanism hypothesis space is EXHAUSTED at this stack's precision. The campaign hit the noise floor."
> — `lore/notes_2026-05-20.md`

### TASK_761 — Structural Pivot Roadmap
Four strategic alternatives (not single-mechanism tweaks):
1. Phase 1 retrain with richer oracle (k=8 or k=12)
2. Reward shaping / cost-penalty sweep
3. Policy entropy regularization sweep
4. Incompatible-dataset retrain (Iris-DEX on-chain bars)

---

## Structural Findings Summary

### The Classifier–Alpha Anti-Correlation (3× confirmed)
Pattern: improving encoder classification → policy over-confidence → over-trading → Alpha collapse.
- TASK_737: no_aux (aux OFF) better classifier BUT lower Alpha than a1b1
- TASK_741: Mint improves dir_acc BUT Δα **−6.98pp** all seeds negative
- TASK_738: legacy_27ch improves Bal.Acc +7.48pp BUT Δα **−18.44pp**

Sources: `lore/notes_2026-05-15.md` §110, `lore/notes_2026-05-13.md` §99.

### Single-Seed Unreliability (Operational Discovery)
Improvements < 5pp on single seed are untrustworthy. Evidence: TASK_737 +6.21% (one seed) → TASK_743 avg +3.61% (range −4.38 to +7.82).

### Eval-MCTS Asymmetric-Lift (Deploy-Truth Operator)
Mechanism: value-head saturation ceiling (~+6% Alpha). Baseline lifted more than variants; regressions pulled toward null. All 14 mechanisms null or negative under deploy-truth.

---

## Source Path Index

### Phase 0 (Nov–Dec 2025)
- `lore/ALPHAZERO_TRADING_VISION.md`
- `lore/NEURAL_MARKET_PHYSICS_DESIGN.md`
- `lore/NEURAL_MARKET_PHYSICS_SUMMARY.md`
- `lore/BERSERKER_DESIGN_PHILOSOPHY.md`
- `lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`
- `lore/MCTS_Trading_Agent_Research_2025_Update.md`
- `lore/notes_2025-12-30.md`, `lore/2025-12-31_session_notes.md`
- `~/.gemini/tmp/trader/chats/session-2025-11-30T*.json` and Dec 2025 sessions

### Phase 1 (Jan–Feb 2026)
- `lore/2026-01-04_information_loss_audit.md`
- `lore/notes_2026-01-20.md` (TASK_105)
- `tasks/136a-k.md`, `tasks/136b-gemini-solution.md`, `tasks/136c-k.md`, `tasks/136c-gemini-solution.md`, `tasks/136d-k.md`, `tasks/136d-gemini-solution.md`, `tasks/137-k.md`
- `tasks/BUG_MAYHEM_MASTER_REPORT.md`, `tasks/BUG_HUNT_2026-01-31.md`

### Phase 2 (Mar–Apr 2026)
- `lore/adr/0003-encoder-v3-32ch-cross-sectional.md`
- `lore/adr/0005-dir-acc-ceiling-decomposition.md`
- `lore/adr/0007-film-injection-post-temporal-pool.md`
- `~/.claude/projects/-home-lukas-Trader/*.jsonl` (from 2026-04-19)

### Phase 3 (May 2026)
- `lore/notes_2026-05-13.md` (TASK_737/743/741)
- `lore/notes_2026-05-15.md` (Tier-2)
- `lore/notes_2026-05-19.md` (Bella class closed)
- `lore/notes_2026-05-20.md` (campaign collapse + pivot)
- `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`
- `runs/recommendation_2026-05-20/analysis.md`
- `runs/759_mcts_eval/result.md`
- `experiments/759_mcts_eval/SPEC.md`
