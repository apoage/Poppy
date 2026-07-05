# TASK_668: Dynamic Merging Research (Easy Mode)

## 1. Objective
Research and implement intelligent checkpoint merging strategies for Federated Learning between Bella (Shapes) and Synthetic (Math) branches. Go beyond simple FedAvg (0.5/0.5 averaging).

---

## 2. Research Vectors

### A. Weight Averaging Strategies
- [ ] **FedAvg**: Simple `(A + B) / 2` - current baseline
- [ ] **Agreement-Weighted Averaging**:
  $$W_{merged} = \frac{Acc_A}{Acc_A + Acc_B} W_A + \frac{Acc_B}{Acc_A + Acc_B} W_B$$
  *Hypothesis:* Prevents a "stalled" branch from degrading a "mastered" branch.
- [ ] **FedProx**: Add regularization toward global model
- [ ] **Layer-wise Merging**: Different α per layer (freeze early layers, merge late)
- [ ] **Fisher Merging (EWC Style)**: Weight by Fisher information (importance-weighted averaging)
  *Goal:* Preserve "Math" neurons while learning "Shapes"

### B. Compatibility Metrics
- [ ] **Weight Cosine Similarity**: `cos(flatten(A), flatten(B))`
- [ ] **Gradient Alignment**: Do gradients point same direction?
- [ ] **Loss Landscape Analysis**: Mode connectivity between checkpoints
- [ ] **Behavioral Similarity**: Agreement on held-out validation set
- [ ] **Branch Divergence**: If branches diverge too far, reduce merge frequency

### C. When to Merge
- [ ] Fixed schedule (every N epochs)
- [ ] Performance-triggered (when either branch improves by X%)
- [ ] Divergence-triggered (when branches drift too far apart)
- [ ] Plateau-triggered (when neither branch improves for N epochs)

---

## 3. Progressive Alignment Strategy (from TASK_667)

| Training Phase | Required Alignment | Rationale |
|----------------|-------------------|-----------|
| **Early (0-30%)** | 0% | Branches explore independently, local mastery |
| **Mid (30-70%)** | Rising (20-60%) | Gradual convergence, shared representations |
| **Late (70-95%)** | High (60-80%) | Consistent behavior across data sources |
| **Final (95-100%)** | ~85% | Production-ready unified model |

**Target**: Cross-correlation > 85% at the end of Phase C.

---

## 4. Literature to Review

- [ ] FedAvg (McMahan et al., 2017)
- [ ] FedProx (Li et al., 2020)
- [ ] Model Soups (Wortsman et al., 2022)
- [ ] Git Re-Basin (Ainsworth et al., 2022) - permutation alignment
- [ ] TIES-Merging (Yadav et al., 2023) - sign conflict resolution
- [ ] Fisher Merging (Matena & Raffel, 2022)
- [ ] EWC - Elastic Weight Consolidation (Kirkpatrick et al., 2017)

---

## 5. Implementation Plan

### Phase 1: Metrics (Research)
1. Implement weight cosine similarity
2. Implement behavioral agreement metric
3. Test on current checkpoints (jax_ckpt_easy_A vs _B)

### Phase 2: Smart Merging (Implementation)
1. Implement agreement-weighting in `merge_checkpoints.py`
2. Layer-wise merging (freeze encoder, merge heads)
3. Divergence detection and alerts

### Phase 3: Orchestrator Integration
1. Add "Cross-Validation" stats to `e-dojo_v1.py` (Test A on data B)
2. Automatic merge triggering based on metrics
3. Progressive alignment enforcement
4. A/B test against standard FedAvg (50/50 merge)

---

## 6. Success Criteria
- [ ] Documented comparison of merging strategies (mini-benchmark)
- [ ] Alignment metric implemented and validated
- [ ] Smart merging integrated into e-dojo orchestrator
- [ ] Final merged model outperforms best single branch
- [ ] Faster convergence to 80% agreement than standard merging
- [ ] Higher stability (lower variance in loss) after merges

---

## 7. Dependencies
- TASK_667: Interleaved Dual-Mastery (provides checkpoints to merge)
- e-dojo_v1.py: Orchestrator (integration target)

## 8. Priority
**LOW** - Research task, can wait until TASK_667 Phase A/B complete
