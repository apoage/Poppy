# Adversarial Bug Hunt — TASK_762/763 wiring (`b4e0158`)

> **OUTPUT**: produce a task file or task set (one or more `tasks/TASK_XXX_*.md` proposals). Do not write code. Do not run anything. The deliverable is a prioritized list of bugs + remediation tasks the workstation operator can execute.
> **Codebase root**: `/home/lukas/Trader` (Trader repo, branch `master`, head `f62537f` at handoff).
> **Operator name**: this user runs workstation; a separate server agent runs experiments on `gpu-server`. Code changes happen on workstation only.

You are a bug hunter looking for **silent correctness bugs** in a freshly-shipped training-code patch. The patch wires three new knobs into the Phase-2 MuZero trainer in service of two upcoming experiments (TASK_762 reward shaping, TASK_763 entropy regularization). You are NOT reviewing code style, documentation, or architecture. You are looking for bugs that corrupt training signal, leak information, cause train-vs-eval mismatch, or produce numerically-degenerate results.

## Why this hunt exists

This codebase has a documented history of silent bugs that look healthy in training and collapse at deployment:
- temporal leak (state at t encoded oracle action at t),
- val_generator epoch counter never updated → inflated val metrics,
- silent column-order mismatch between Bella outputs and trainer asset indexing (TASK_740 / `[[bella-column-mismatch]]`),
- multi-process rank-head deadlock that only fired at 3+ concurrent processes (TASK_742),
- argmax→eval-MCTS asymmetric lift that masked an entire campaign's negative variants (recal sweep 2026-05-20, `runs/recommendation_2026-05-20/analysis.md`).

The pattern: a change looks fine at code-review depth, looks fine in a single smoke test, and produces wrong-but-plausible numbers in a multi-seed campaign that costs ~20h server wall to discover. Your job is to compress that discovery loop.

## The patch under review

Commit **`b4e0158`** — "TASK_762/763: wire reward_scale + fee_penalty_bps + entropy_coef + min_entropy". Diff covers two files:

- `phase2_config.py` — adds three new dataclass fields: `fee_penalty_bps: float = 0.0`, `entropy_coef: float = 0.0`, `min_entropy: float = 0.0`. (`reward_scale` already existed but was unread; the patch makes it load-bearing.)
- `train_phase2.py` — four touch sites:
  1. `SegmentBatchGenerator` (around `train_phase2.py:632`) — replaces the module constant `REWARD_SCALE = 100.0` with `self.config.reward_scale`; adds `fee_penalty_bps × 1e-4` to the per-trade cost alongside `transaction_cost`.
  2. The N-step reward path (around `train_phase2.py:646`) — same scale swap.
  3. Stage-B `stage_b_train_step` loss (around the joint-MCTS branch, near `train_phase2.py:1103`) — adds an entropy bonus term: `policy_loss = policy_loss - entropy_coef * H(policy) + 5 * entropy_coef * relu(min_entropy - H)`. Gated to `use_factored=False`. Reports new metric `policy_entropy`.
  4. Argparse — four new CLI flags (`--reward-scale`, `--fee-penalty-bps`, `--entropy-coef`, `--min-entropy`) and the matching override block.

Read the full diff via `git show b4e0158`. Read the full context of each touched function — don't trust an excerpt.

## Hunt targets (prioritized)

### P0 — break the build / corrupt the signal

1. **Reward double-scaling.** Is `reward_scale` applied exactly once? `mcts_trader/jax_agent/mctx_search.py:145` already assumes rewards trained with REWARD_SCALE=100. Does anything downstream (PopArt, value_scale, MCTS scaling) implicitly depend on the value being 100 specifically? If the sweep tries `reward_scale=200`, does the MCTS scoring still interpret rewards correctly?
2. **Fee-penalty sign / unit.** The patch applies `step_pnl = step_pnl - changes * (transaction_cost + fee_penalty_bps * 1e-4)`. Confirm: (a) `changes` is unitless trade-count, not a return; (b) `transaction_cost` is in the same units as `fee_penalty_bps × 1e-4`; (c) the sign is a subtraction (penalty reduces reward). What is `self.config.transaction_cost`'s actual default and unit? Verify by reading `easy_config.py`.
3. **Entropy term sign and floor logic.** The added line is `policy_loss = policy_loss - entropy_bonus + floor_penalty`. Entropy is positive; subtracting it should *encourage* exploration. `relu(min_entropy - H)` is positive when H is below floor, so adding it punishes low entropy. Is the magnitude balance right? With `entropy_coef=0.05, min_entropy=3.0, H=2.0`: bonus = 0.1, penalty = 0.25, net = +0.15 added to loss → loss grows when entropy is below floor. Confirm this is the intended direction in *every* coefficient regime in the sweep (`{0.005, 0.01, 0.02, 0.05, 0.1}`).
4. **Joint-vs-factored gating.** The entropy bonus is gated by `not use_factored`. Default `use_factored=False`, so canon path takes the bonus. But: in the joint case, `policy_logits` has shape `(B, 320)` and `softmax(axis=-1)` produces a single distribution over 320 joint actions. Entropy is `ln(320) ≈ 5.77` max. Is `min_entropy=3.0` a sensible floor for this distribution given the empirical entropy of converged MCTS visit counts? Could the floor never be reached, making the floor-penalty branch dead code?
5. **JIT staticness.** `entropy_coef` and `min_entropy` are captured at `make_stage_b_train_step` call time as Python floats. Confirm: (a) re-launching with a different value via CLI re-traces the JIT (it should, because they're captured at trace time); (b) there's no path where they get baked in and then a Python override silently fails to propagate.

### P1 — train-vs-eval / train-vs-deploy mismatch

6. **Old checkpoint compatibility.** Phase-2 loads Phase-1 configs via `get_phase2_config(p1_config_path)`. Phase-1 configs predate the three new fields. Confirm the load path filters unknown fields cleanly and the new fields default to 0 (canon). Check both branches of the load at `train_phase2.py:2337-2342`.
7. **Reward-scale on eval-side.** Does `tools/benchmark_easy.py` or `real_trade_jax.py` use the same `reward_scale` value, or does eval have its own copy? If sweep cells train with `reward_scale=50` but eval uses `reward_scale=100`, the value head's outputs will be on different scales and Alpha will be miscomputed.
8. **N-step reward consistency.** The patch updates both the per-step reward site and the N-step reward site. The N-step path also multiplies by `td_discount**N` and clips. Does the new scale interact with the existing `np.clip(..., -10.0, 10.0)` cap differently at scale=200 (more clipping) vs scale=25 (less clipping)? At what scale does clipping start to censor a non-trivial fraction of samples?
9. **Saved checkpoint metadata.** `save_checkpoint` at `train_phase2.py:1362-1384` saves `config.__dict__` filtered to JSON-safe types. Confirm the three new fields land in saved checkpoint configs (otherwise resumed runs silently revert to defaults).

### P2 — numerical / silent degeneracy

10. **Entropy NaN / collapse**. `-sum(p * log(p + 1e-10))` with extremely-peaked softmax could underflow. Confirm the `1e-10` epsilon is sufficient at bfloat16 (does this stack run mixed-precision?). If `entropy_coef=0.1` and floor fires every step, policy_loss can blow up to dominate other losses — check the magnitude.
11. **Sweep-cell baseline reuse claim.** Both SPECs say `762-rs100 (canon)` and `763-ec01` reuse 760's a1b1 sims=32 baselines. But the SPECs are silent on whether `min_entropy=0` (the new default) produces the same training trajectory as a pre-patch run with no entropy code at all. Reason about whether the patch is bit-identical when `entropy_coef=0` — the `if entropy_coef > 0.0` gate looks right but verify there's no else-branch that initializes `policy_entropy = jnp.array(0.0)` differently than before.
12. **CLI / argparse %-format bug.** During smoke, workstation discovered `python train_phase2.py --help` fails with a pre-existing `%` format error unrelated to this patch. Confirm the four new flags don't worsen this (no stray `%` chars), and propose a follow-up task to fix the pre-existing bug.

## Deliverable format

Produce `tasks/TASK_XXX_GEMINI_REVIEW_762_763.md` (single file OR a small task set if the bugs naturally split). Each finding should include:

```markdown
### Finding F<N> — <one-line title>

**Severity**: P0 / P1 / P2 (per the prioritization above)
**Location**: `<file>:<line range>` and the surrounding function name
**Bug**: 2-4 sentences describing the actual incorrect behavior, with the specific input/state that triggers it.
**Why it's silent**: why a quick smoke test wouldn't catch this.
**Fix**: a concrete code change (diff-shaped if small, or 2-3 sentences if larger). Reference specific line numbers.
**Verification**: how to confirm the fix works — typically a smoke check, a specific assertion, or a 1-cell run that should differ from canon by a measurable amount.
**Acceptance**: what state proves the finding is resolved.
```

Order findings by severity. Mark findings you investigated and ruled out as "RULED OUT" (one-sentence reason) — those count, because they save the operator from re-investigating.

## Out of scope

- Do NOT propose new mechanisms or new experiments. That's the pathfinding review's job, not this one.
- Do NOT propose docs, comments, lint, naming changes. Only behavior-affecting bugs.
- Do NOT propose refactors. Only minimum-diff fixes.
- Do NOT review TASK_761 itself or the structural pivot. That's the pathfinding review's job.
- Do NOT propose changes to `experiments/762/SPEC.md` or `experiments/763/SPEC.md` unless a bug in them creates a mis-execution risk on the server side.

## Context files to read (in this order)

1. `git show b4e0158` — the patch.
2. `train_phase2.py:60-90, 575-660, 980-1110, 1300-1360, 2065-2110, 2378-2400` — surrounding context of each touch site.
3. `phase2_config.py` — full file (small).
4. `experiments/762_reward_shaping/SPEC.md`, `experiments/763_entropy_reg/SPEC.md` — what the patch is supposed to enable.
5. `tasks/TASK_761_STRUCTURAL_PIVOT_ROADMAP.md` — campaign context (don't critique this; just orient).
6. `lore/notes_2026-05-21.md` §153 + §154 — workstation operator's discovery + commit narrative.
7. `mcts_trader/jax_agent/mctx_search.py:140-180`, `mcts_trader/jax_agent/trainer.py:1218-1260` — downstream reward/entropy usage to check for interaction bugs.
8. `tools/benchmark_easy.py`, `real_trade_jax.py:120-340` — eval-side scaling references.

## Constraints

- Read-only mode for codebase mutation: do not edit existing files, do not run training scripts, do not execute smoke tests against this stack, do not push or write to git, do not run anything that could corrupt files, kill processes, or affect system stability.
- **Read-only terminal commands are fine and encouraged**: `git show`, `git log`, `git diff`, `git blame`, `ls`, `cat`, `grep`, `rg`, `find`, `wc`, `head`, `tail`, `python -c "import ast; ast.parse(...)"` for syntax verification, etc. Use them to ground claims in the actual codebase.
- **Write the deliverable directly to disk** under `tasks/` as a markdown file (e.g. `tasks/TASK_770_GEMINI_REVIEW_762_763.md`) using your file-writing tools. Do not return it only as a chat artifact — the operator processes task files from the on-disk repo.
- The operator runs all proposed code changes after reviewing your task file. You do not apply fixes; you only describe them.
- Use the actual codebase to verify your claims. Do not hallucinate line numbers; if a line number isn't where you expect, say so.
- Keep findings minimal and decisive. If you're not sure, flag the uncertainty in 1 sentence — better than 5 paragraphs of hedging.
- Total task-file output: ≤ 600 lines. Prefer concision.
