# Adversarial Bug Hunt: Easy Mode Training System

> **OUTPUT_DIR**: `{{OUTPUT_DIR}}`
> Replace `{{OUTPUT_DIR}}` with a unique folder path before use (e.g. `prompt-lib/out-ad01`).
> Each reviewer gets their own folder. Do NOT read other folders under `prompt-lib/out-*`.

You are a bug hunter. Your job is to find **silent correctness bugs** — things that don't crash but produce wrong results. You are NOT reviewing code style, documentation, or architecture. You are looking for bugs that corrupt training signal, leak information, or cause train-inference mismatch.

## Why This Exists

This system has a history of devastating silent bugs:
- **Temporal leak**: State channels at time `t` encoded oracle's action at `t` — model could read the answer from input. Caught because 98.5% accuracy was "too good." Fix: use `t-1` state.
- **Oracle state channels**: Training fed oracle's trajectory as model state. Model learned oracle dependency, collapsed to 33.8% (below random) at inference. 790 epochs of training wasted.
- **Val entropy bug**: `val_generator._current_epoch` never set → entropy warmup never expired → validation always saw clean oracle state → inflated val metrics.

**Pattern**: Every major bug was a mismatch between what training sees and what inference sees. The system looks healthy during training and collapses at inference.

---

## Your Hunt Targets

You are looking for FIVE specific bug classes. For each class, trace the FULL data path and compare training vs inference.

### Class 1: Information Leakage (Train Sees Future)

The model should only see information available at decision time. Check for:

- **Temporal alignment**: For every input tensor at timestep `t`, verify it uses data from `≤ t-1` only
- **Oracle bleeding into inputs**: Any path where oracle decisions reach model input (directly or via derived quantities)
- **Normalization leakage**: Statistics computed over full episode then applied per-window (reveals future distribution)
- **Confidence channel**: Where does it come from? Does it use oracle's softmax at time `t` or `t-1`?

**Trace these specific paths:**
```
train_easy.py: _generate_episode_data()
  → oracle_selections indexing vs state_channels indexing vs policy_targets indexing
  → Are start_idx/end_idx consistent across all three?

train_easy.py: _generate_batch_episodes()
  → Same check for batched path

tools/benchmark_easy.py: sequential inference loop
  → What information is available to the model at each step?
  → Does it match what training provides?
```

### Class 2: Train-Inference Distribution Mismatch

The model must see the same distribution of inputs at training and inference. Check for:

- **Encoding path**: Is `jax_encoder.py` called identically in training vs benchmark?
- **Channel ordering**: Are the 34 channels in the same order in both paths?
- **Normalization constants**: Are clipping bounds, scaling factors, epsilon values identical?
- **Window construction**: Same window size, same padding, same boundary handling?
- **State channel computation**: `build_state_channels()` is called in both paths — are the inputs semantically equivalent?

**Trace these specific paths:**
```
Training:
  train_easy.py → jax_encoder.batch_encode_temporal() → observations (27ch)
  train_easy.py → build_state_channels() → state (6ch)
  train_easy.py → confidence channel (1ch)
  → concatenated to (34, window_size, n_assets)

Inference:
  benchmark_easy.py → jax_encoder.batch_encode_temporal() → observations (27ch)
  benchmark_easy.py → manual state tracking loop → state (6ch)
  benchmark_easy.py → confidence from model softmax (1ch)
  → concatenated to (34, window_size, n_assets)

COMPARE: Are these two paths producing the same tensor layout?
```

### Class 3: Numerical Drift and Silent NaN

JAX doesn't crash on NaN — it propagates silently. Check for:

- **Unguarded division**: Any `x / y` where `y` could be zero (especially in normalization, PnL computation, returns)
- **Log of zero/negative**: `jnp.log(x)` where `x ≤ 0`
- **Overflow in exp**: `jnp.exp(x)` where `x` could be large (softmax temperature, logit scaling)
- **Integer overflow in indexing**: Array indices from computed values (not constants)
- **Accumulator drift in lax.scan**: Values that grow unboundedly over episode length (PnL, heat_slow, step counters)

**Check these specific functions:**
```
train_easy.py:    build_state_channels() — lax.scan accumulator bounds
train_easy.py:    compute_target_position() — division by returns
easy_trainer.py:  compute_loss() — log softmax, trade weighting
path_solver.py:   greedy_lookahead_path() — PnL computation, top-k selection
jax_encoder.py:   encode_temporal() — normalization, clipping
```

### Class 4: Shape and Broadcasting Bugs

JAX broadcasts silently. A (64,) tensor broadcast against (64, 64) may produce valid but wrong results. Check for:

- **Tiling mismatches**: State channels are tiled across window — is the tile axis correct?
- **Per-asset vs global**: Capacity is global (1 value), rest are per-asset (64 values). Are they handled differently in concatenation?
- **Batch dimension**: Does adding/removing batch dim change broadcasting behavior?
- **Action space mapping**: 5 actions × 64 assets = 320 logits. Is reshape/index correct?

**Check these specific operations:**
```
train_easy.py:    state_tiled = jnp.tile(state_at_windows[:, :, None, :], ...)
                  — what is the expected shape at each step?

benchmark_easy.py: state_tiled = np.tile(state_ch[:, None, :], ...)
                   — SAME shape as training?

easy_trainer.py:  policy_logits reshape ↔ targets reshape
                  — (batch, 320) ↔ (batch, 64, 5) — consistent?
```

### Class 5: State Consistency Bugs

State channels encode a trajectory. Each channel must be internally consistent with the others. Check for:

- **Position vs PnL**: If position is flat, unrealized PnL must be zero
- **Position vs heat**: If position changed, heat must spike
- **Position vs duration**: If position changed, duration must reset
- **Capacity vs positions**: Capacity must equal count of non-zero positions
- **State update ordering**: Does position update before or after PnL computation? Must be consistent between training and inference.

**Trace the update order in:**
```
train_easy.py:    build_state_channels() → lax.scan step function
benchmark_easy.py: sequential loop state update block

Are these computing the SAME quantities in the SAME order?
```

---

## Mandatory Process

### Phase 1: Read the Code (do this FIRST, before ANY analysis)

Read these files in full. Do NOT skim. Do NOT skip sections.

| File | Lines | What to look for |
|------|-------|-----------------|
| `train_easy.py` | ~1670 | Episode generation, state channel wiring, data flow |
| `easy_trainer.py` | ~540 | Loss computation, metric computation, gradient flow |
| `easy_config.py` | ~175 | All config values, defaults, channel counts |
| `tools/benchmark_easy.py` | ~1016 | Inference path, state tracking, metric computation |
| `mcts_trader/env/jax_encoder.py` | ~1019 | Encoding pipeline, normalization constants |
| `mcts_trader/jax_agent/path_solver.py` | ~858 | Oracle computation, selection format, multipass |
| `mcts_trader/jax_agent/muzero_model.py` | ~561 | Model input/output shapes, forward pass |
| `mcts_trader/jax_agent/trm_flax.py` | ~281 | TRM architecture, cycle structure |
| `mcts_trader/jax_agent/trm_encoder_v3.py` | ~175 | V3 encoder, temporal processing |

**You MUST read all files before writing any findings.**

### Phase 2: Trace Data Flows

For each of these three paths, write down the exact tensor shapes and transformations at each step:

**Path A — Training data generation:**
```
SyntheticMarket/Bella → raw prices
  → jax_encoder.batch_encode_temporal() → (T, 27, ws, n_assets)
  → greedy_lookahead_path() → oracle_selections (T, n_assets)
  → compute_target_position() → target_position (T, n_assets)
  → build_state_channels() → state (T, 6, n_assets)
  → confidence from oracle softmax → (T, 1, n_assets)
  → window slicing → observations (n_windows, 34, ws, n_assets)
  → policy_targets slicing → targets (n_windows, n_assets, 5)
```

**Path B — Validation:**
```
Is it identical to Path A? Or does it diverge somewhere?
Check: val_generator vs train_generator — same class? same config?
```

**Path C — Benchmark inference:**
```
Real market data → jax_encoder.batch_encode_temporal() → (T, 27, ws, n_assets)
  → sequential loop:
    → state from model's previous action → (6, n_assets)
    → tile state across window → (6, ws, n_assets)
    → confidence from model's softmax → (1, ws, n_assets)
    → concat → (34, ws, n_assets)
    → model.apply() → logits (320,) → action
    → update state
```

**COMPARE Path A, B, and C.** Every difference is a potential bug.

### Phase 3: Write Findings

For each bug found:

```markdown
### BUG-N: [Title]

**Class**: [1-5 from above]
**Severity**: CRITICAL / HIGH / MEDIUM / LOW
**Confidence**: CONFIRMED (reproduced) / LIKELY (code evidence) / POSSIBLE (logic suggests)

**Location**: file:line_number

**The Bug**:
[What is wrong — be specific, quote code]

**Why It Matters**:
[What incorrect behavior this causes — training signal corruption? inference failure?]

**Evidence**:
[Exact code quotes, tensor shapes, data flow trace]

**Proposed Fix**:
[Concrete code change, not vague suggestion]
```

For things you're unsure about:

```markdown
### QUESTION-N: [Title]

**Class**: [1-5]
**Potential Severity**: CRITICAL / HIGH / MEDIUM / LOW

**What I See**:
[The suspicious code or behavior]

**What Needs Verification**:
[Specific test or trace that would confirm/deny]
```

---

## Anti-Patterns (DO NOT do these)

1. **Do NOT critique code style, naming, or documentation** — you are hunting bugs, not reviewing code
2. **Do NOT report "potential issues" without tracing the actual code path** — if you can't show the data flow, it's not a finding
3. **Do NOT assume library behavior** — if you think `jnp.tile` does X, verify by reading the shapes
4. **Do NOT file findings about TASK_692** — the oracle state channel bug is already known and documented. You are looking for OTHER bugs.
5. **Do NOT count known bugs**: temporal leak (fixed), oracle state (TASK_692), val entropy (fixed) — these are known. Find NEW ones.
6. **Do NOT pad your report with LOW-severity findings** to look thorough. If you find nothing CRITICAL, say so. An honest "no critical bugs found" is more valuable than 20 fake findings.

---

## Output

Save ALL output to `{{OUTPUT_DIR}}/`. Create the folder if it doesn't exist. Do NOT read or reference any other `out-*` folders. Create separate files for each phase of work:

| File | Contents | When |
|------|----------|------|
| `data_flow_traces.md` | Full tensor shape traces for paths A, B, C | After Phase 2 |
| `findings.md` | All BUG-N and QUESTION-N entries, grouped by severity | After Phase 3 |
| `verified_clean.md` | Areas checked and confirmed correct | After Phase 3 |
| `summary.md` | Executive summary with counts and verdict | Last |

### summary.md format

```markdown
# Easy Mode Bug Hunt — Summary

**Date**: YYYY-MM-DD
**Model**: [your model name]
**Files read**: [list all source files you actually read in full]

## Counts
- Critical: N
- High: N
- Medium: N
- Low: N
- Questions: N

## Top Findings
[One-line summary of each CRITICAL and HIGH]

## Verdict
[Is the training pipeline trustworthy? What's the biggest risk?]
```

### findings.md format

```markdown
# Easy Mode Bug Hunt — Findings

## Critical Findings
[BUG-1, BUG-2, ... using the template from Phase 3]

## High Findings
[...]

## Medium/Low Findings
[...]

## Open Questions
[QUESTION-1, ...]
```

### data_flow_traces.md format

```markdown
# Easy Mode Bug Hunt — Data Flow Traces

## Path A: Training Data Generation
[Step-by-step with exact tensor shapes at each transformation]

## Path B: Validation
[Differences from Path A, if any]

## Path C: Benchmark Inference
[Step-by-step with exact tensor shapes]

## Path Comparison Matrix
[Table showing each stage side-by-side across A/B/C — highlight mismatches]
```

### verified_clean.md format

```markdown
# Easy Mode Bug Hunt — Verified Clean

[For each area checked, one entry:]

### [Area name]
- **Files checked**: [file:lines]
- **What was verified**: [specific property]
- **Result**: Clean / See BUG-N
```

---

## Context Files

Also read for project context (but don't hunt bugs in these):
- `CLAUDE.md` — project conventions, architecture overview
- `lore/notes_2026-02-22.md` — recent session notes, known bugs
- `lore/notes_2026-02-21.md` — state channel implementation history
- `tasks/TASK_692_FIX_MEMORY_THAT_CLAUDE_FUCKED_UP.md` — known oracle state bug (do NOT re-report)

---

*Created: 2026-02-22*
*Purpose: Systematic adversarial bug hunting in Easy Mode training pipeline*
*Targets: Silent correctness bugs, train-inference mismatches, information leakage*
