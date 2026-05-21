# Pathfinding Review — Structural Pivot post-single-mechanism-exhaustion

> **OUTPUT**: produce a task file or task set (one or more `tasks/TASK_XXX_*.md` proposals). Do not write code. Do not run experiments. The deliverable is a re-ordered, re-scoped, or re-framed structural pivot roadmap that the workstation operator and server agent can execute.
> **Codebase root**: `/home/lukas/Trader`.
> **Operator name**: workstation runs code+SPECs; server agent on `gpu-server` runs all training; experiments cost ~12-35h server wall each.

You are a senior ML research strategist looking at a campaign that has just closed its single-mechanism hypothesis class. The operator has drafted a 4-direction structural pivot roadmap. Your job is to **challenge that roadmap, find the highest-EV direction not yet considered, and produce concrete task drafts the operator can execute.** You are not a cheerleader; if the existing roadmap is mostly right, say so in one paragraph and spend the rest of the review proposing the *one* re-ordering or the *one* missing direction that improves expected value.

## Why this review exists

The campaign has spent ~3 months running paired-Δα variants on a stack with a high noise floor (~5pp Alpha at 3-seed precision). Every single-mechanism variant (738–760, ~14 variants across encoder swap / aux head / aux distill / data augment / memory module / dynamics loss / value target / train-side search depth / eval-side decoder) returned null or negative under the deploy-truthful operator (eval-MCTS sims=16 joint, established by the 2026-05-20 recal sweep). The campaign is disciplined — zero false positives at this precision — but also unproductive.

The operator's response is `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md`, which lays out four directions:

1. **Reward shaping** (TASK_762, code shipped `b4e0158`, READY) — `reward_scale` + `fee_penalty_bps` sweep. ~15-21h server. Lowest risk. Mechanism gate: Δ-trade-rate < 0 AND Δα > 0.
2. **Policy entropy regularization** (TASK_763, code shipped, READY, sequenced after 762) — `entropy_coef` + `min_entropy` sweep. ~19-24h. Sibling of 762; attacks same failure mode via different lever. Mechanism gate: Δ-Bal.Acc < 0 AND Δα > 0.
3. **Oracle k-reformulation** (TASK_764, code not shipped, BLOCKED) — Phase-1 re-imprint with `oracle_k ∈ {8, 12}` instead of canon `k=4`. ~35h server. Breaks the Phase-1 imprint contract — every prior paired-Δα number ceases to be directly comparable. Targets the documented `[[bull-blindness]]` failure mode.
4. **Incompatible-dataset retrain** (TASK_765, blocked on Iris-DEX pipeline, STRATEGIC) — train on per-block (12s) on-chain OHLCV instead of daily CEX. ~2 weeks first attempt. Standalone alpha-positive question; not a paired Δα.

The operator's expected ordering: 762 → 763 → 764, with 765 Phase-A data-prep optionally running in parallel.

## Your job

Challenge this roadmap honestly. Specifically:

### Q1 — Is the trigger correctly calibrated?

760 closed with mean Δα **+0.35pp** (s7 +0.55, s13 +0.68, s23 −0.17) and fell in the **(−1, +1) "STRUCTURAL PIVOT"** band per TASK_761. But two of three seeds were *positive*, and the worst was only −0.17. Is the band correctly drawn? Is the operator over-reacting to a result that 5-seed extension might promote? Or under-reacting, given the gate is +3pp?

Read `experiments/760_train_mcts_sims/SPEC.md`, `runs/760_train_mcts_sims/phase3_skipped.md`, and `runs/760_train_mcts_sims/eval_*/benchmark.json` summaries. Reason about whether 760 should get a 5-seed extension before declaring structural pivot. Argue for or against.

### Q2 — Is the directions list complete?

The four directions cover: reward channel, policy regularization, oracle definition, data. What's missing? Candidates to consider:

- **Action space restructure** — current 320-dim joint action (64 assets × 5 actions). Restructure to (e.g.) 64-dim portfolio weight or 64 × 3 (Sell/Hold/Buy, dropping LimitSell/LimitBuy). Does the 5-action space carry signal the planner can't actually exploit?
- **Discount / horizon restructure** — TD discount `0.95`, mcts_discount `0.95`, oracle_value_horizon `400`, direction_horizon `20`. Are these mutually consistent? Has anyone done a discount sweep?
- **Value head ensemble / quantile regression** — value is currently a 401-bin categorical with PopArt. Quantile heads or ensemble value heads are a different inductive bias.
- **Curriculum / regime weighting** — `[[bull-blindness]]` says the model fails in bull regimes. Has anyone tried regime-conditional loss weighting or regime-specific Phase-1 fine-tuning?
- **Two-stage planner** — fast policy proposes, slow planner verifies. Current MCTS does both in one pass. Distinct from "deeper search," more like "different search."
- **Stage-B initialization** — Phase-1 imprint is fixed; what about a Phase-1.5 stage that adapts the encoder *to* the Stage-B oracle (which differs from Phase-1's oracle)?

You pick one (or two) of these — or one you came up with — and argue why it has higher EV than the lowest-ranked direction the operator already drafted.

### Q3 — Should TASK_764 (oracle-k) move up?

TASK_764 is sequenced last among the in-repo directions (765 is parallel-strategic). It's also the most disruptive (Phase-1 retrain breaks comparability) but it's the only direction that directly attacks `[[bull-blindness]]`, which has been the documented top-1 failure mode since the early campaign. Reason: should it move up to first, given:

- 762 and 763 attack the same hypothesis (over-trade collapse) from two angles. If 762 is null, what's the prior that 763 succeeds via a "different lever, same target"?
- Bull-blindness has 5+ months of evidence and was never directly tested.

But also against: 764 burns ~35h server and breaks comparability. Is that EV-positive vs running 762/763 first (~21h each) to either rule out the cheap hypothesis or land a win that doesn't require a Phase-1 retrain?

### Q4 — Is the 765 (Iris-DEX) framing right?

`tasks/TASK_765_INCOMPATIBLE_DATASET.md` frames the DEX-data move as "standalone alpha-positive question, not paired Δα." Is this concession premature? Could you frame the move so it produces a comparable signal (e.g., train both stacks on a *subset* of CEX assets that have DEX equivalents, getting paired Δα at the asset level)? Or is the framing correctly conceding that comparability is impossible, in which case the question is "is this a parallel product or a pivot," not "is this an experiment."

## Constraints on your reasoning

1. **Server is the bottleneck.** Each direction costs server wall hours. Be honest about EV-per-hour.
2. **Recal contract is fixed.** Eval-MCTS sims=16 joint is the new canonical operator (`runs/recommendation_2026-05-20/analysis.md`). Don't propose anything that re-opens that question.
3. **No false positives at 3-seed.** The campaign's noise floor is ~5pp at 3-seed. Any direction that "shows a +2pp lift in 1 seed" is not a result — it's noise. Your proposed directions must be falsifiable at the 3-seed paired-Δα ≥ +3pp gate, OR they must be standalone-alpha (like 765).
4. **Workstation does code, server runs experiments.** Any new direction has to spell out the workstation code work explicitly (line counts, files touched) so the operator can budget.
5. **`[[single-mechanism-exhausted]]` is established.** Don't propose another single-encoder/aux-distill/memory variant. That class is closed.

## Deliverable format

Produce one or more files under `tasks/TASK_XXX_GEMINI_PATHFINDING.md` (or a set: e.g., one task per new direction you propose, plus one task containing the re-ordering argument). Each direction proposal should follow the existing template style (see `tasks/TASK_762_REWARD_SHAPING.md`, `tasks/TASK_763_POLICY_ENTROPY_REG.md`):

```markdown
# TASK_XXX — <short title>

**Status**: GEMINI-PROPOSED (operator review pending)
**Created**: 2026-05-21
**Estimated wall**: <Xh / Xd>

## Hypothesis
<2-3 sentences. Mechanism, falsifiable claim.>

## Code prerequisites
<files + line ranges + estimated change size>

## Variants
<table: cell name, knob values, seeds>

## Promotion gate
<paired Δα ≥ +3pp under eval-MCTS sims=16 joint, with mechanism-specific early-exit gate>

## Risk
<what could go wrong, mitigation>

## Why this comes before/after <other tasks>
<positioning argument>

## References
<paths to lore, memory, related tasks>
```

Plus a short top-level summary file `tasks/TASK_XXX_GEMINI_PATHFINDING_SUMMARY.md` that contains:

1. Your **2-3 paragraph** verdict on the existing roadmap (defend or challenge).
2. Your **recommended re-ordering** (or "operator's ordering is correct" — say so explicitly with reasoning).
3. The **single highest-EV addition** (or "no addition needed" — say so explicitly with reasoning).
4. An **EV-per-server-hour ranking** of all directions, old + new.

Total deliverable: ≤ 1500 lines across all files. Concision beats thoroughness.

## Out of scope

- Do NOT review the just-shipped code (`b4e0158`). That's the bughunt prompt's job.
- Do NOT propose changes to the eval operator (eval-MCTS sims=16 joint is fixed by recal contract).
- Do NOT propose another single-mechanism encoder/aux/memory variant. That class is closed.
- Do NOT propose changes to Bella, Yarrow, Aconite, or other side projects — focus on the Poppy training loop and its data.
- Do NOT propose buy-side LLM-feature additions ("add sentiment", "add news") — out of scope for this stack.

## Context files to read (in this order)

1. `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — the roadmap you're challenging.
2. `tasks/TASK_762_REWARD_SHAPING.md`, `tasks/TASK_763_POLICY_ENTROPY_REG.md`, `tasks/TASK_764_ORACLE_K_REFORMULATION.md`, `tasks/TASK_765_INCOMPATIBLE_DATASET.md` — the four directions.
3. `experiments/762_reward_shaping/SPEC.md`, `experiments/763_entropy_reg/SPEC.md` — the SPECs that have shipped.
4. `runs/recommendation_2026-05-20/analysis.md` — the recal sweep that closed the single-mechanism class.
5. `runs/760_train_mcts_sims/phase3_skipped.md` + `eval_*/benchmark.json` — the 760 result that triggered the pivot.
6. `lore/notes_2026-05-20.md` §151, `lore/notes_2026-05-21.md` §153 — operator's reasoning trail.
7. `~/.claude/projects/-home-lukas-Trader/memory/MEMORY.md` plus the linked memory files for `single-mechanism-exhausted`, `bull-blindness`, `classifier-alpha-anticorrelation`, `flow-matching-promising`, `bella-column-mismatch`. These are the operator's persistent findings — they're authoritative summaries of what's been tried and what failed.
8. `CLAUDE.md` + subtree CLAUDE.mds (`mcts_trader/CLAUDE.md`, `diff_teacher/CLAUDE.md`, `tools/orchestration/CLAUDE.md`) — component facts.
9. `docs/easy_mode.md` — original ExIt design doc; useful for understanding what the Phase-1 imprint contract guarantees.

## Constraints

- Read-only mode for codebase mutation: do not edit existing files (the deliverable task files are the only NEW files you create), do not run training scripts, do not execute smoke tests, do not push or write to git, do not run anything that could corrupt files, kill processes, or affect system stability.
- **Read-only terminal commands are fine and encouraged**: `git log`, `git show`, `git diff`, `git blame`, `ls`, `cat`, `grep`, `rg`, `find`, `wc`, etc. Use them to ground arguments in the actual codebase state.
- **Write the deliverables directly to disk** under `tasks/` as markdown files (one per direction proposal, plus the SUMMARY file) using your file-writing tools. Do not return them only as chat artifacts — the operator processes task files from the on-disk repo.
- Use the actual codebase to verify claims about code paths, configs, defaults.
- Cite memory files (`[[name]]` syntax) and lore sections (`§N`) where they're load-bearing for an argument.
- Don't restate what's already in TASK_761 — assume the reader has read it. Add value, don't echo.
- If your verdict is "operator's roadmap is correct as-is," that is a valid deliverable — produce a SUMMARY file with that verdict and a one-paragraph defense per direction. Don't pad.
