# TASK_667: Interleaved Dual-Mastery (Easy Mode)

## 1. Objective
Achieve **>80% Oracle Agreement** on a single model that generalizes across both **Bella (Diffusion Shapes)** and **Synthetic (Mathematical Rules, L0-L14)**.

---

## 2. The Interleaved Execution Strategy
Maximize GPU throughput and training diversity by running two distinct data sources simultaneously on one GPU without resource contention.

### Component Interleaving
*   **Instance A (Bella):** Stochastic shape generation. Stresses distributional robustness.
*   **Instance B (Synthetic):** Rule-based curriculum. Teaches specific trading mechanics.
*   **Timing Synchronization:**
    *   Target cycle: **30 seconds per epoch (+/- 5%)** for both instances.
    *   Start delay: 14 seconds (half-cycle offset) to interleave GPU-heavy backprop from A with CPU-heavy generation from B.
    *   This naturally spaces out checkpoint saving and peak VRAM spikes.

---

## 3. Implementation Phases

### Phase A: Sequential Level Mastery (Current)
*   **Goal:** Step-by-step mastery of synthetic logic.
*   **Workflow:**
    1.  Master Level $N$ (starting at L0).
    2.  Run dual instances (A: Bella, B: Synthetic L$N$) for 50-200 epochs.
    3.  **Sync:** Federated Averaging ($W_{merged} = 0.5 \times W_A + 0.5 \times W_B$) every 10 epochs.
    4.  **Advance:** Proceed to L$N+1$ when merged agreement on L$N$ is >70%.
    5.  **Persistence:** Load the merged weights into the next level cycle.

### Phase B: The "Stress Test" (Curriculum Jitter)
*   **Goal:** Prevent forgetting and force universal feature extraction.
*   **Workflow:**
    1.  Instance A runs Bella continuously.
    2.  Instance B rotates synthetic levels **every epoch**.
    3.  **Distribution:** Weighted toward harder levels (L10-L14) matching Dojo V4 Stage 2 logic.

### Phase C: Final Alignment (The 80% Threshold)
*   **Goal:** Reach production-ready performance.
*   **Workflow:**
    1.  Stable Batch Size: 1024-2048 (Large preset).
    2.  Parallel Envs: 512 (Vectorized refill).
    3.  Enable latent state caching once weights stabilize to maximize speed.
    4.  Success: >80% agreement on BOTH data sources simultaneously.

### Phase D: Historical Validation
*   **Goal:** Benchmarking.
*   **Workflow:**
    1.  Load Phase C final weights into `HistoricalMarket`.
    2.  Verify coherent trade patterns on real price action.
    3.  Optional: Final fine-tuning on mixed historical regimes.

---

## 4. Resource & Infrastructure Plan

### VRAM & Environment
*   **Virtual Env:** Must use `(trader_env)` for correct CUDA/JAX sharding.
*   **Allocator:** Use `TF_FORCE_GPU_ALLOW_GROWTH=true` and interleaved staggering.

### Performance Tuning
| Parameter | Value | Reason |
|-----------|-------|--------|
| **Steps/Epoch** | 400 | Balances cycle time to ~30s |
| **Batch Size** | 1024-2048 | Saturates Large TRM without OOM |
| **N-Envs** | 512 | Vectorizes oracle/encode into single GPU calls |
| **Model** | Large | 15M+ params for maximum capacity |

---

## 5. Weight Merging Strategy (Research-Validated)

### 5.1 Key Theoretical Foundation
> **Same-init branches are linearly mode connected** (Frankle et al., 2020)
>
> Networks sharing initialization converge to solutions connected by a LOW-LOSS linear path.
> This means simple averaging works surprisingly well - no permutation alignment (Git Re-Basin) needed.

### 5.2 The Squisher Shortcut (CRITICAL)
> **Adam's `v` accumulator approximates diagonal Fisher for FREE** (Li et al., ICML 2025)

No extra backward passes needed. Fisher information is already in the optimizer state:
```python
# In Optax, Fisher ≈ opt_state[1].nu (Adam's second moment)
fisher_a = opt_state_a[1].nu  # Already computed during training!
fisher_b = opt_state_b[1].nu
```

### 5.3 Merge Methods by Phase

| Phase | Training % | Merge Method | Interval |
|-------|------------|--------------|----------|
| **Early** | 0-10% | FedAvg (simple) | Every 50-100 steps |
| **Mid** | 10-70% | Fisher-weighted (via Adam v) | Every 200-500 steps |
| **Late** | 70-100% | Fisher + TIES (for heads) | Every 500-1000 steps |

### 5.4 Divergence Metrics & Thresholds

**Primary Metric: Loss Barrier at Midpoint**
```python
# Check if merge is safe
barrier = loss(0.5 * params_a + 0.5 * params_b) - 0.5 * (loss(params_a) + loss(params_b))
```

| Cosine Sim | Barrier | Action |
|------------|---------|--------|
| > 0.99 | < 1% | Safe. Increase interval 1.5x |
| > 0.95 | < 2% | Normal. FedAvg or Fisher merge |
| < 0.95 | > 2% | Warning. Fisher + distillation, -25% interval |
| < 0.90 | > 5% | **ALERT.** Reset to min interval, increase FedProx μ |

### 5.5 Optimizer State Handling (Post-Merge)
```python
# CORRECT approach (validated by research):
merged_v = tree_map(lambda a, b: 0.5*a + 0.5*b, v_a, v_b)  # Average second moment
merged_m = tree_map(lambda _: 0.0, m_a)                     # RESET first moment
# Then: LR warmup for 200-500 steps
```

**Why reset m?** Momentum vectors point in branch-specific directions. Averaging produces poor compromise.

### 5.6 Layer-wise Strategy
| Component | Merge Ratio (α) | Reason |
|-----------|-----------------|--------|
| Embeddings | 0.5 | Shared structure |
| Encoder (early) | 0.5 | General features |
| Encoder (late) | 0.5-0.6 | Slight bias to stronger |
| Policy Head | Fisher-weighted | Task-specific |
| Value Head | Fisher-weighted | Task-specific |

---

## 6. Anti-Forgetting Strategy

### 6.1 Experience Replay (Primary Defense)
After each merge, branches train on **mixed data**:
- Branch A: 90% Bella + **10% Synthetic**
- Branch B: 90% Synthetic + **10% Bella**

```python
# In training loop
if step % 10 == 0:  # Every 10th batch
    batch = sample_from_other_branch_buffer()
```

### 6.2 EWC via Adam's v (FREE)
```python
# Post-merge regularization loss
ewc_loss = lambda_ewc * sum(v_merged * (params - params_merged)**2)
```
Where `v_merged` is the merged Adam second moment (≈ Fisher).

### 6.3 Knowledge Distillation (For Heads)
Keep pre-merge snapshots as teachers:
```python
distill_loss = α * KL(policy_teacher_A || policy_student) + β * KL(policy_teacher_B || policy_student)
```
Temperature τ=2-4 for softer targets.

---

## 7. e-dojo_v1.py Orchestrator Specification

### 7.1 Core Loop
```python
while not converged:
    # 1. Check divergence metrics
    cosine_sim = compute_cosine_similarity(params_a, params_b)
    barrier = compute_loss_barrier(params_a, params_b, val_data)

    # 2. Decide merge action
    if should_merge(cosine_sim, barrier, steps_since_merge):
        merged = fisher_merge(params_a, params_b, v_a, v_b)
        broadcast_to_branches(merged)
        reset_momentum()
        lr_warmup(200)

    # 3. Check advancement criteria
    if agreement_on_level_n > 0.70:
        advance_to_next_level()

    # 4. Log metrics
    log_to_dashboard(cosine_sim, barrier, agreement_a, agreement_b)
```

### 7.2 Merge Decision Logic
```python
def should_merge(cosine, barrier, steps):
    if steps < MIN_STEPS_BETWEEN_MERGE:
        return False
    if cosine < 0.90 or barrier > 0.05:
        return True  # Emergency merge
    if steps > MAX_STEPS_BETWEEN_MERGE:
        return True  # Scheduled merge
    if cosine < 0.95 or barrier > 0.02:
        return True  # Normal merge
    return False
```

### 7.3 Files Structure
```
tools/orchestration/
├── e-dojo_v1.py           # Main orchestrator
├── merge_utils.py         # Merge functions (Fisher, TIES, etc.)
├── divergence_metrics.py  # Cosine, barrier, CKA, behavioral
└── replay_buffer.py       # Cross-branch experience sharing
```

---

## 8. Tech Debt Audit

| Issue | Location | Impact | Priority |
|-------|----------|--------|----------|
| **N_BUFFER hardcoded = 32** | `train_easy.py:86` | Caps parallelism at 32 | HIGH |
| **N_WORKERS hardcoded = 4** | `train_easy.py:88` | Bottleneck for 512 envs synthetic | MEDIUM |
| **Queue maxsize = N_BUFFER** | `train_easy.py:106` | Coupled to buffer size | LOW |
| **SyntheticMarket shared across threads** | `train_easy.py:137-138` | Race condition on `regenerate()` | MEDIUM |
| **`_generate_prices_bella` single-env fallback** | `train_easy.py:312` | Still n_envs=1 in one code path | LOW |
| **No historical data support** | - | Phase D blocked | MEDIUM |
| **No auto-merge** | - | Manual merge only | MEDIUM |
| **No level progression** | - | Phase A manual advancement | MEDIUM |
| **No curriculum jitter** | - | Phase B not automated | LOW |
| **SAMPLES_PER_EPISODE hardcoded = 200** | `train_easy.py:87` | Can't tune exhaustion rate | LOW |

### Critical Path for 512 Envs ✅ COMPLETE
1. [x] N_BUFFER configurable (`--buffer-size`)
2. [x] N_WORKERS scale with buffer size (`max(2, min(8, buffer_size // 4))`)
3. [x] Thread-safe market regeneration (per-thread `_get_thread_market()`)

---

## 9. Implementation Checklist

### Phase 1: Infrastructure (Week 1) ✅ COMPLETE
- [x] Add `--buffer-size` flag to `train_easy.py`
- [x] Make `N_WORKERS` scale with buffer size
- [x] Fix thread-safety for SyntheticMarket (per-thread instances)
- [x] Update `merge_utils.py` with Fisher merge (using Adam v)
- [x] Add loss barrier computation (heuristic estimate)
- [x] CSV metrics logging to `logs/`

### Phase 2: Orchestrator (Week 2) ✅ COMPLETE
- [x] Create `e-dojo_v1.py` orchestrator
- [x] Implement divergence metrics (cosine, barrier, alignment)
- [x] Implement adaptive merge scheduler
- [x] Add experience replay (`--mix-ratio`, `--secondary-levels`)
- [x] Active level progression (kill/restart branch)
- [x] VRAM control per subprocess
- [ ] Dashboard integration (DS24) - deferred

### Phase 3: Training Pipeline (Week 3) - IN PROGRESS
- [x] Optimizer state rebuild (`rebuild_optimizer_state()`)
- [x] LR warmup post-merge (count reset)
- [ ] FedProx regularization in `easy_trainer.py`
- [x] Layer-wise merge support (`layer_wise_merge()`)
- [x] TIES-Merging for policy/value heads (`ties_merge()`)
- [ ] EWC loss term (using merged v) - optional

### Phase 4: Validation (Week 4) - PENDING
- [ ] Run Phase A (L0 → L5) with auto-merge
- [ ] Run Phase B stress test
- [ ] Run Phase C final alignment
- [ ] Historical validation (Phase D)

### Minor Improvements Identified
- [ ] Uniform mixing: Interleave primary/secondary batches instead of chunk-based
- [ ] Checkpoint preservation: Add `--checkpoint` flag to `restart_branch()`
- [ ] Phase B rotation: Add level rotation logic to `_check_level_advancement()`

---

## 10. Success Criteria (Definition of Done)

### Infrastructure ✅ COMPLETE
- [x] `N_BUFFER` configurable via `--buffer-size`
- [x] `N_WORKERS` scales appropriately (formula: `max(2, min(8, buffer_size // 4))`)
- [x] Thread-safe market regeneration (per-thread `_get_thread_market()`)
- [x] `e-dojo_v1.py` orchestrator working with adaptive merge

### Merging ✅ COMPLETE
- [x] Fisher merge using Adam's v (zero extra cost via `extract_fisher_from_optimizer()`)
- [x] Loss barrier check before each merge (`_estimate_barrier()` heuristic)
- [x] Optimizer state handled correctly (`rebuild_optimizer_state()`: average v, reset m, warmup)
- [x] Layer-wise merge for encoder vs heads (`layer_wise_merge()`, `ties_merge()`)

### Performance - PENDING VALIDATION
- [ ] Merged model hits **85% agreement** on Synthetic L0-5
- [ ] Merged model hits **80% agreement** on full Synthetic L0-14
- [ ] Merged model hits **80% agreement** on Bella
- [ ] GPU utilization >80% during training

### Validation - PENDING
- [ ] Performance validated on **Historical Data** benchmark
- [ ] No catastrophic forgetting across levels (verified via experience replay)
- [ ] Loss barrier stays < 2% between merges

---

## 11. Research References

| Paper | Key Insight | Application |
|-------|-------------|-------------|
| Frankle et al., 2020 | Same-init = linear mode connectivity | Skip Git Re-Basin |
| Li et al., 2025 (Squisher) | Adam v ≈ Fisher | Free Fisher estimation |
| McMahan et al., 2017 (FedAvg) | Baseline averaging | Early phase |
| Li et al., 2020 (FedProx) | Drift prevention | Mid phase |
| Matena & Raffel, 2022 | Fisher-weighted merge | Late phase |
| Yadav et al., 2023 (TIES) | Sign conflict resolution | Policy heads |
| Karimireddy et al., 2020 (SCAFFOLD) | Control variates | Advanced drift correction |

See `docs/weight merging/` for full research reports from Gemini, GPT, and Claude.
