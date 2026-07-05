# Implementation Audit: Spec vs Reality

> **OUTPUT_DIR**: `{{OUTPUT_DIR}}`
> Replace `{{OUTPUT_DIR}}` with a unique folder path before use (e.g. `prompt-lib/out-audit01`).
> Each reviewer gets their own folder. Do NOT read other folders under `prompt-lib/out-*`.

You are a verification auditor. Your job is to compare **what each task spec promised** against **what was actually implemented in code**. You are looking for incomplete work, wrong implementations, unintended side effects, and config mismatches that could silently degrade training.

## Why This Exists

Training performance has regressed. `agree_trade` dropped from ~15% to ~7% — the model stopped learning trade actions. Over 20 tasks modified the training pipeline in the last month. Any one of them could have been implemented incorrectly or left half-finished. The task spec files may say "DONE" while the code tells a different story.

**The pattern we're hunting**: A task spec says "do X", the implementation does "X but slightly wrong", and the slightly-wrong version silently corrupts the training signal without crashing.

---

## Your Audit Targets

For each task below, you must:

1. **Read the task spec** in `tasks/TASK_XXX_*.md` — understand what it promises
2. **Read the actual implementation** in the source files
3. **Compare**: Does the code do what the spec says? Fully? Correctly?
4. **Check configs**: Are defaults set as the spec intended? Are feature flags wired through?
5. **Check interactions**: Does this task's implementation conflict with any other task's?

---

## Tasks to Audit (Training-Critical Only)

### Tier 1: Loss, Targets, and Gradient Signal

These directly affect what the model learns. Bugs here = training fails.

| Task | Spec File | What It Should Do | Key Source Files |
|------|-----------|-------------------|-----------------|
| **TASK_147** | `TASK_147_AGREEMENT_METRIC_FIX.md` | Balanced accuracy: `0.5*agree_trade + 0.5*agree_hold` | `easy_trainer.py` |
| **TASK_149** | `TASK_149_ADAPTIVE_CLASS_BALANCING.md` | Adaptive trade weighting: upweight trade actions ~25× | `easy_trainer.py` |
| **TASK_151** | `TASK_151_SIGNAL_SOFT_START.md` | Warmup ramp for trade weight (prevent gradient shock) | `easy_trainer.py`, `train_easy.py` |
| **TASK_670B** | `TASK_670B_EXPERT_IMPRINTING_STABILITY.md` | Value loss fixes, return scaling | `easy_trainer.py`, `easy_config.py` |
| **TASK_673** | `TASK_673_MULTIPASS_ORACLE.md` | Soft targets from multi-pass oracle (not one-hot) | `mcts_trader/jax_agent/path_solver.py`, `train_easy.py` |
| **TASK_677** | `TASK_677_TURTLE_BREAKER.md` | Post-averaging temperature sharpening on oracle | `mcts_trader/jax_agent/path_solver.py`, `easy_config.py` |
| **TASK_685** | `TASK_685_GRADIENT_CLIP_FIX.md` | Fix gradient over-suppression | `easy_trainer.py` |
| **TASK_695** | `TASK_695_ORACLE_MASK_PREDICTION_HEAD.md` | Auxiliary BCE loss for oracle mask prediction | `easy_trainer.py`, `mcts_trader/jax_agent/muzero_model.py` |

### Tier 2: Data Pipeline and Sampling

These affect what data the model sees. Bugs here = biased or corrupted training signal.

| Task | Spec File | What It Should Do | Key Source Files |
|------|-----------|-------------------|-----------------|
| **TASK_674** | `TASK_674_TRAINING_AGREEMENT_PLATEAU_BUGS.md` | Fix buffer sample counter, optimizer warmup reset | `train_easy.py`, `easy_trainer.py` |
| **TASK_675** | `TASK_675_TECHNICAL_DEBT_AND_STABILITY.md` | Continuous trade weights, MAX_TRADE_WEIGHT=64 | `easy_trainer.py`, `train_easy.py` |
| **TASK_679** | `TASK_679_TRADE_SIGNAL_DENSITY.md` | Increase oracle trade density | `mcts_trader/jax_agent/path_solver.py`, `easy_config.py` |
| **TASK_680** | `TASK_680_TRADE_BIASED_SAMPLING.md` | Coin-flip trade-biased offset sampling | `train_easy.py` |
| **TASK_687** | `TASK_687_HARD_EXAMPLE_MINING.md` | Mine hard examples during training | `train_easy.py` |
| **TASK_691** | `TASK_691_PHASE_B_ORACLE_ENTROPY.md` | Oracle entropy injection (noise in state channels) | `train_easy.py`, `easy_config.py` |

### Tier 3: Architecture and State Channels

These change what the model sees and how it processes input.

| Task | Spec File | What It Should Do | Key Source Files |
|------|-----------|-------------------|-----------------|
| **TASK_683** | `TASK_683_WIDEN_ENCODER_STATE.md` | Wider embed_dim, more encoder heads | `easy_config.py`, `mcts_trader/jax_agent/muzero_model.py` |
| **TASK_684** | `TASK_684_DECOUPLE_HIDDEN_DIM.md` | Separate hidden_dim from state_dim | `easy_config.py`, `mcts_trader/jax_agent/muzero_model.py` |
| **TASK_686** | `TASK_686_BFLOAT16_MIXED_PRECISION.md` | bfloat16 compute, float32 params | `easy_trainer.py`, `easy_config.py` |
| **TASK_689** | `TASK_689_AGENT_STATE_CHANNELS.md` | 6 state channels + 1 confidence = 34 total | `train_easy.py`, `tools/benchmark_easy.py` |
| **TASK_692** | `TASK_692_FIX_MEMORY_THAT_CLAUDE_FUCKED_UP.md` | Model-state rollout (autoregressive) | `train_easy.py` |
| **TASK_693** | `TASK_693_FIX_POSITION_SEMANTICS_IN_ROLLOUT.md` | Incremental position stepping in rollout | `train_easy.py`, `tools/benchmark_easy.py` |

### Tier 4: Bug Fixes (from Adversarial Hunt)

Each bug fix should be verified correct AND not introducing a new bug.

| Bug | From Task | What It Fixed | Source File |
|-----|-----------|---------------|-------------|
| **BUG-1** | TASK_694 | Returns alignment in model-state rollout | `train_easy.py:570-576, 789-793` |
| **BUG-2** | TASK_694 | Confidence channel prev_conf in scan | `train_easy.py:237-240, 296-300` |
| **BUG-4** | TASK_694 | TIES merge trim direction | `tools/orchestration/merge_utils.py` |
| **BUG-5** | TASK_694 | Val generator model params update | `train_easy.py:~1596` |
| **BUG-6** | TASK_694 | best_agreement after merge load | `train_easy.py:~1550` |
| **BUG-7** | TASK_694 | Trade-biased sampling boundary | `train_easy.py:~1130` |
| **LATENT** | TASK_694 | Buffer param IDs sentinel | `train_easy.py:~1075` |

---

## What to Check for Each Task

### A. Completeness
- Does the code implement ALL items in the spec, or just some?
- Are there TODO/FIXME/HACK comments in the implementation?
- Did the implementation skip any edge cases mentioned in the spec?

### B. Correctness
- Do the actual computations match the math/logic in the spec?
- Are array shapes, indices, and slicing correct?
- Are feature flags wired through the full call chain (config → trainer → model)?

### C. Config Alignment
- Read `easy_config.py` and check every default value against what the spec intended
- Are features enabled by default that should be disabled (or vice versa)?
- Do CLI flags actually reach the code that uses them?

### D. Train-Inference Consistency
- If the task changed something in training, was the same change made in `tools/benchmark_easy.py`?
- Do state channels, encoding, normalization, and action semantics match between train and inference?

### E. Interaction Conflicts
- Does task A's implementation assume something that task B changed?
- Do two tasks modify the same code section with conflicting logic?
- Are there ordering dependencies (task A must run before B) that aren't enforced?

---

## Critical Config Check

Read `easy_config.py` IN FULL and create a table:

```markdown
| Config Field | Default Value | Which Task Set It | Expected Value Per Spec | Match? |
|-------------|---------------|-------------------|------------------------|--------|
| ... | ... | ... | ... | ✓ / ✗ |
```

Pay special attention to:
- `encoder_version` — is it 2 or 3? V3 has a known causal leak (TASK_694 DORMANT finding)
- `compute_dtype` — is it float32 or bfloat16? Precision affects learning
- `oracle_noise` — is it 0.0 or 0.03? Affects oracle target quality
- `oracle_k` — how many assets oracle selects
- `oracle_temperature` — post-averaging sharpening
- `use_oracle_mask_head` — should be False by default
- `trade_sample_prob` — trade-biased sampling probability
- `warmup_epochs` / `warmup_steps` — warmup duration
- `use_state_channels` / `autoregressive_state` — state channel mode

---

## Mandatory Process

### Phase 1: Read All Specs (Skim for Promises)

Read every task spec file listed above. For each, extract:
1. What the spec promises (the "contract")
2. Which files it should modify
3. What config values it should set

### Phase 2: Read the Implementation

Read these source files IN FULL (not just the parts you think are relevant):

| File | Lines | Why |
|------|-------|-----|
| `easy_config.py` | ~175 | All config defaults — the ground truth for what's enabled |
| `easy_trainer.py` | ~540 | Loss, metrics, optimizer, model construction |
| `train_easy.py` | ~1670 | Data generation, sampling, state channels, episode pipeline |
| `mcts_trader/jax_agent/path_solver.py` | ~858 | Oracle computation (multipass, temperature, selections) |
| `mcts_trader/jax_agent/muzero_model.py` | ~600 | Model architecture, forward pass, heads |
| `tools/benchmark_easy.py` | ~1050 | Inference path — must match training |

### Phase 3: Audit Each Task

For each task, write a finding:

```markdown
### TASK_XXX: [Title]

**Spec Status**: [what the .md file says]
**Implementation Status**: [what the code shows]

**Completeness**: FULL / PARTIAL / NOT IMPLEMENTED
**Correctness**: CORRECT / WRONG / UNCLEAR

**Issues Found**:
- [Specific discrepancy between spec and code, with file:line references]

**Impact on Training**:
- [How this could affect agree_trade or agree_hold]

**Config Check**:
- [Config field] = [actual value] — matches spec? [yes/no]
```

### Phase 4: Cross-Task Interaction Analysis

Look for conflicts between tasks. Common patterns:
- Task A sets a config default, Task B overwrites it
- Task A assumes index ordering, Task B changes the ordering
- Task A adds a channel, Task B hardcodes channel count
- Task A changes loss weighting, Task B changes what the loss receives

### Phase 5: Synthesize

Create a ranked list of findings most likely to cause the agree_trade regression.

---

## Output

Save ALL output to `{{OUTPUT_DIR}}/`:

| File | Contents |
|------|----------|
| `config_audit.md` | Full config table with spec vs reality comparison |
| `task_audits.md` | Per-task audit findings (Phase 3 output) |
| `interactions.md` | Cross-task conflict analysis (Phase 4 output) |
| `summary.md` | Executive summary with top suspects ranked by impact |

### summary.md format

```markdown
# Implementation Audit — Summary

**Date**: YYYY-MM-DD
**Model**: [your model name]
**Tasks Audited**: N
**Files Read**: [list]

## Counts
- Fully Implemented & Correct: N
- Partially Implemented: N
- Incorrectly Implemented: N
- Config Mismatch: N
- Cross-Task Conflict: N

## Top Suspects (most likely to cause agree_trade regression)

1. [TASK_XXX: one-line description of the issue]
2. [TASK_XXX: ...]
3. [TASK_XXX: ...]

## Recommended Actions
[Ordered list of things to fix or revert]
```

---

## Anti-Patterns

1. **Do NOT just check if the function exists** — verify the logic is correct
2. **Do NOT skip the config check** — a feature can be "implemented" but disabled by default or set to a wrong value
3. **Do NOT trust task status labels** — TASK_694 says "TODO" but was committed. Read the code.
4. **Do NOT report code style issues** — only correctness, completeness, and config alignment
5. **Do NOT assume "it was working before"** — the question is whether it's working NOW, after all the interactions between 20+ tasks

---

*Created: 2026-02-23*
*Purpose: Verify 20+ task implementations match their specs, find sloppy/incomplete work blocking training*
*Scope: Tasks 147–695, focused on training-critical code paths*
