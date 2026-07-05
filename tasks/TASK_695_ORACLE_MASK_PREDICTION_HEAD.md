# TASK_695: Auxiliary Oracle Mask Prediction Head

**Status**: TODO
**Priority**: 2 (MEDIUM — Phase 3 preparation, not blocking current training)
**Depends on**: TASK_694 (bugfix batch should land first)

## Background

Paper: *"Improving Interactive In-Context Learning from Natural Language Feedback"*
(Klissarov et al., Google DeepMind, 2025) — `docs/arXiv-2602.16066v1/`

Key finding: Training a student to **predict the teacher's feedback** (not just follow
it) lets the student internalize the teacher's reasoning. At inference the student can
self-correct without a teacher — the "autodidact" result. Surprisingly, the autodidact
outperforms the teacher-student setup.

**Mapping to Foxglove**: Our oracle (path_solver with 20-step lookahead) is the
"teacher with privileged information." The model is the student. Currently we only
train the student to **imitate** the oracle (policy targets) and **predict remaining
value** (value head). We don't train it to predict **what the oracle sees** — i.e.,
which assets the oracle selected and why.

Adding an auxiliary head that predicts the oracle's top-k selection mask gives the
model an internal representation of "what a privileged observer would pick here."
During Phase 3 (MCTS self-play), this head acts as an internalized oracle — the model
can evaluate its own search candidates against what it thinks the oracle would select,
without needing the actual oracle.

## Design

### New Head: Oracle Mask Predictor

A small MLP head branching from the same shared representation as the policy and value
heads (after the TRM encoder):

```
shared_repr (from TRM encoder)
    ├── policy_head  → (batch, 320)     [existing — 5 actions × 64 assets]
    ├── value_head   → (batch, 401)     [existing — categorical value]
    └── oracle_head  → (batch, 64)      [NEW — binary selection per asset]
```

Architecture:
- Input: shared representation (same as policy/value heads)
- 2-layer MLP: `Linear(d_model, 256) → ReLU → Linear(256, n_assets)`
- Output: `(batch, n_assets)` logits — sigmoid for per-asset binary selection
- Loss: `binary_cross_entropy(predicted_mask, oracle_selection_mask)`

### Training Target

The oracle selection mask is already computed during episode generation:
```python
# path_solver.py returns: optimal_paths.selections  (T, n_assets) binary
# Already available in _generate_episode_data as oracle_selections
```

Slice to window-aligned indices (same as policy targets):
```python
oracle_mask_targets = oracle_selections[start_idx:end_idx]  # (n_windows, n_assets)
```

### Loss Integration

```python
# In easy_trainer.py compute_loss():
oracle_mask_logits = output['oracle_mask']  # (batch, n_assets)
oracle_mask_loss = optax.sigmoid_binary_cross_entropy(
    oracle_mask_logits, batch['oracle_mask']
).mean()

total_loss = policy_loss + value_weight * value_loss + oracle_mask_weight * oracle_mask_loss
```

Default weight: `oracle_mask_weight = 0.1` (auxiliary, shouldn't dominate policy learning).

### Phase 3 Usage (Future)

During MCTS self-play, the oracle head provides a "would the oracle agree?" signal:
- Model proposes candidate actions via policy head
- Oracle head predicts which assets should be active
- Agreement between policy actions and oracle predictions provides an internal
  confidence signal for tree search, replacing the external oracle

This is the autodidact mechanism from RL²F: external teacher signal (oracle) →
internalized during training → self-correction at inference.

## Implementation Steps

1. **Model change** (`muzero_model.py`):
   - Add `oracle_mask_head` MLP to `MuZeroNetwork`
   - Output `oracle_mask` key in forward pass dict
   - Only active during `initial_inference` mode (not dynamics)

2. **Data pipeline** (`train_easy.py`):
   - Add `oracle_mask` to episode data dict (sliced oracle_selections)
   - Thread through batch assembly in both single-episode and batch paths

3. **Loss** (`easy_trainer.py`):
   - Add `oracle_mask_loss` term to `compute_loss()`
   - Add `oracle_mask_weight` to config
   - Log as separate metric

4. **Config** (`easy_config.py`):
   - `use_oracle_mask_head: bool = False` (opt-in, default off)
   - `oracle_mask_weight: float = 0.1`

5. **Benchmark** (`benchmark_easy.py`):
   - Report oracle mask accuracy as metric (how well does the model predict
     which assets the oracle would select?)

## Completion Criteria

- [ ] Oracle mask head added to model (behind config flag)
- [ ] Training loss includes oracle mask BCE (weighted)
- [ ] Oracle mask accuracy logged per epoch
- [ ] Benchmark reports oracle mask agreement on real data
- [ ] No regression in policy/value loss when head is enabled
- [ ] Config flag `use_oracle_mask_head=True` activates everything

## Notes

- The head should be **small** (2-layer MLP, 256 hidden) — it's auxiliary, not the
  main learning signal. We don't want it to compete for representational capacity.
- During Phase 3, this head replaces the oracle for search guidance. The model
  effectively asks itself "what would the oracle do?" and uses that to evaluate
  candidate actions — the autodidact pattern.
- Paper reference: RL²F (arXiv:2602.16066v1), Section 4 "Towards Self-Improvement"

---

*Created: 2026-02-22*
*Source: Analysis of arXiv-2602.16066v1 applied to Foxglove architecture*
