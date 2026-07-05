GLM

# Ceiling Analysis: Breaking Through 61% Agreement

**Date**: 2026-02-20
**Context**: Stage 3 training plateaued at 60-61% oracle agreement despite extensive optimizations

---

## Executive Summary

**Root Cause Identified**: Architectural bottleneck in encoder + extreme class imbalance

The ambiguity analysis reveals the ceiling is NOT data-driven:
- **Theoretical ceiling**: 97.1% balanced accuracy (perfect model)
- **Current performance**: 61% (37pp gap)
- **Trade ceiling**: 94.5% (oracle gives high-confidence trade signals)
- **Current trade accuracy**: 43% (51pp gap)

The encoder is the critical bottleneck:
- **V3 parameters**: 0.14M (tiny! — 1% of model)
- **Reasoning parameters**: 14M (96% of model)
- **state_dim=64**: Compresses 64-asset information into 64-dim vector
- **Information loss**: 27×64×64 → 64 (compression ratio 1728:1)

**The model has sophisticated reasoning (14M params) operating on impoverished representations (64-dim).**

---

## Strategy 1: Expand Encoder Representation (HIGH PRIORITY)

### Hypothesis
The minGRU encoder is too narrow (embed_dim=32 → state_dim=64). With 64 assets and 27 channels, compressing to 64 dimensions loses critical cross-asset and temporal structure. The reasoning head (14M params) is trying to do complex multi-asset reasoning with insufficient information.

### Evidence
- Encoder params: 0.14M (1% of total)
- state_dim=64 forces information bottleneck
- Cross-asset attention operates on 64-dim per asset
- Temporal information limited to 32-dim per asset

### Implementation
**Option A: Wider encoder** (Low effort, Medium impact)
```python
# easy_config.py
embed_dim: int = 64  # Was 32
state_dim: int = 128  # Was 64
```
- Expected params: ~0.5M encoder (still <4% of model)
- VRAM: Minimal increase (encoder VRAM is tiny)
- Training: No slowdown (encoder is already fast)

**Option B: Dual-encoder architecture** (Medium effort, High impact)
```python
# Separate encoders for temporal vs cross-asset
class DualAssetEncoder(nn.Module):
    def setup(self):
        self.temporal_encoder = minGRUEncoder(dim=32)  # Time axis
        self.cross_encoder = SelfAttention(...)  # Asset axis
        self.fusion = MLP(...)  # Combine both views
```
- Provides complementary views (temporal patterns + asset correlations)
- Parameters: ~1-2M (still small)
- Proven in multi-modal learning

### Expected Impact
**High** — Addressing the root bottleneck
- Trade accuracy: 43% → 55-60% (+12-17pp)
- Balanced accuracy: 61% → 70-75% (+9-14pp)
- Better cross-asset reasoning (wider attention space)

### Risk
**Low** — Conservative width increase, well-understood architecture
- Potential overfitting if encoder too wide (monitor val gap)
- Slower convergence (more parameters to optimize)

### Implementation Effort
**Low** — Config change + checkpoint resume testing
```bash
# Modify easy_config.py
embed_dim = 64
state_dim = 128

# Resume training from current checkpoint
python train_easy.py --config jax_ckpt_easy/easy_large/config.json \
                 --encoder-version 3 \
                 --checkpoint jax_ckpt_easy/easy_large/phase1_epoch650.pkl
```

### Priority: **1 (Try first)**

---

## Strategy 2: Focal Loss for Class Imbalance (MEDIUM PRIORITY)

### Hypothesis
Current adaptive weighting (~24× for trades) is a crude global multiplier. Focal loss dynamically down-weights easy (high-confidence) examples and focuses learning on hard (low-confidence, misclassified) examples. This naturally addresses the 96% hold / 4% trade imbalance without hand-tuned weights.

### Implementation
```python
# easy_trainer.py
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal loss from Lin et al. (2017).
    gamma: Focusing parameter (2.0 is standard)
    alpha: Balance parameter (0.25 up-weights minority class)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    p_t = jnp.take_along_axis(probs, targets[..., None], axis=-1).squeeze(-1)
    focal_weight = (1 - p_t) ** gamma

    ce_loss = -jnp.log(jnp.maximum(p_t, 1e-8))
    focal_loss = alpha * focal_weight * ce_loss

    return focal_loss.mean()
```

Replace current cross-entropy with focal loss on policy head.

### Expected Impact
**Medium** — Better handling of extreme imbalance
- Trade accuracy: 43% → 48-52% (+5-9pp)
- Balanced accuracy: 61% → 66-70% (+5-9pp)
- Faster convergence on minority class

### Risk
**Medium** — Focal loss can be unstable with very small alpha
- May underweight easy holds (need to hold sometimes!)
- Requires tuning gamma/alpha hyperparameters
- Can reduce overall accuracy while improving trade accuracy

### Implementation Effort
**Low** — Loss function replacement in easy_trainer.py
```python
# Add to EasyConfig
focal_gamma: float = 2.0
focal_alpha: float = 0.25

# Modify compute_loss()
policy_loss = focal_loss(
    policy_logits,
    policy_targets,
    gamma=config.focal_gamma,
    alpha=config.focal_alpha
)
```

### Priority: **2 (Try after encoder expansion)**

---

## Strategy 3: Two-Stage Training (Hold/Trade → Action) (MEDIUM PRIORITY)

### Hypothesis
The 320-action space (5 actions × 64 assets) is overwhelming. First learn a binary hold/trade classifier (2^64 = simpler but still complex), then refine to specific actions within the trade class. This decomposes the extreme imbalance problem:
- Stage 1: Binary classification (hold vs trade) — balanced via focal/adaptive weighting
- Stage 2: Action refinement (buy/sell/limit-buy/limit-sell) — conditioned on trade decision

### Implementation
```python
class TwoStageTRM(nn.Module):
    """Hold/Trade detector + Action selector."""
    def setup(self):
        # Stage 1: Binary hold/trade detector
        self.encoder = StreamingAssetEncoder(state_dim=128, embed_dim=64)
        self.trade_detector = nn.Dense(64 * 2)  # Logits for each asset

        # Stage 2: Action selector (only for trade class)
        self.action_selector = nn.Dense(64 * 4)  # Buy/Sell/LimitBuy/LimitSell

    def __call__(self, x):
        h = self.encoder(x)  # (batch, 128)

        # Stage 1: Detect trade-worthy assets
        trade_logits = self.trade_detector(h).reshape(B, 64, 2)  # (B, 64, 2)

        # Stage 2: Select action for trade-worthy assets
        action_logits = self.action_selector(h).reshape(B, 64, 4)  # (B, 64, 4)

        return trade_logits, action_logits
```

### Expected Impact
**Medium-High** — Decomposed complexity
- Trade accuracy: 43% → 50-58% (+7-15pp)
- Balanced accuracy: 61% → 68-75% (+7-14pp)
- Better gradient signal on minority class

### Risk
**Medium** — Training complexity
- Two-stage training (train stage 1 to convergence, then stage 2)
- Joint training requires careful loss weighting
- Stage 2 depends on stage 1 quality

### Implementation Effort
**High** — New model architecture + training loop modifications
- 2-3 sessions for implementation + testing
- Requires checkpoint migration (old checkpoints incompatible)

### Priority: **3 (Try if simpler strategies fail)**

---

## Strategy 4: Curriculum Phasing (LOW PRIORITY)

### Hypothesis
Currently mixing all L0-L14 levels uniformly. L0 (pure noise) and L14 (complex arbitrage) have very different dynamics. Phased curriculum (master simpler levels before harder) may help the model build progressive representations instead of being overwhelmed by heterogeneous data.

### Implementation
```python
# train_easy.py
curriculum_schedule = {
    0:  [0, 1, 2, 3, 4],      # Epochs 0-100: Basic noise/trends
    100: [0, 1, 2, 3, 4, 5, 6, 7],  # Add mean-reversion
    200: list(range(15)),          # Epochs 200+: Full curriculum
}

def get_current_levels(epoch):
    for milestone, levels in sorted(curriculum_schedule.items(), reverse=True):
        if epoch >= milestone:
            return levels
    return [0, 1, 2, 3, 4]
```

### Expected Impact
**Low-Medium** — Progressive learning
- Trade accuracy: 43% → 46-50% (+3-7pp)
- Balanced accuracy: 61% → 65-68% (+4-7pp)
- Faster early convergence

### Risk
**Low** — Well-understood technique
- May plateau early if curriculum too conservative
- Requires tuning phase boundaries

### Implementation Effort
**Low** — Curriculum schedule in train_easy.py
```python
# Add to EasyConfig
curriculum_phase: bool = True
phase_schedule: dict = field(default_factory=lambda: {
    0: [0, 1, 2, 3, 4],
    100: [0, 1, 2, 3, 4, 5, 6, 7],
    200: list(range(15))
})
```

### Priority: **4 (Easy win, try early)**

---

## Strategy 5: Synthetic Data Diversification (LOW PRIORITY)

### Hypothesis
The model at 61% on synthetic but 48.6% on real (TASK_682) suggests a synthetic/real gap. Bella generator may be producing too narrow a distribution (insufficient regime diversity, volatility range, correlation structures). Mixing in real data or expanding synthetic diversity could improve generalization.

### Implementation

**Option A: Real data mixing** (Low effort)
```python
# train_easy.py
real_data_ratio = 0.3  # 30% real, 70% synthetic
real_dataset = load_real_data_from_db(limit=10000)
```

**Option B: Bella parameter sweep** (Medium effort)
```python
# diff_teacher/generation/generator.py
diverse_configs = [
    dict(volatility_range=[0.1, 0.5]),
    dict(volatility_range=[0.5, 1.5]),
    dict(volatility_range=[1.5, 3.0]),
    dict(correlation_range=[-0.8, -0.4]),
    dict(correlation_range=[0.4, 0.8]),
]
# Sample uniformly from configs per episode
```

### Expected Impact
**Medium** — Better generalization
- Trade accuracy: 43% → 48-55% on synthetic
- Real market performance: 48.6% → 55-60% (+6-11pp)
- Reduced synthetic/real gap

### Risk
**Low-Medium** — Data quality risk
- Real data may contain noise/bugs in pipeline
- Real data leakage (future information)
- Requires careful validation

### Implementation Effort
**Low** — Data pipeline modification
```python
# Add to EasyConfig
real_data_mix: float = 0.0  # 0.0 = synthetic only
real_data_path: str = "datasets/real_markets.npz"
```

### Priority: **5 (Invest if synthetic/real gap persists)**

---

## Rejected Strategies

### ~~Model scaling to 50-100M params~~ — REJECTED

**Why**: The encoder is the bottleneck, not reasoning capacity. Adding more reasoning parameters (H=8, L=10, hidden=512) without fixing the encoder will just make the model larger without improving representations. Current 14M reasoning params are already 96% of model.

### ~~Oracle parameter tuning (k=4→k=8)~~ — REJECTED

**Why**: k=4 already improved trade accuracy 9%→43%. k=8 would increase PnL overestimation and make oracle targets more ambiguous (worse soft targets). The ceiling is architectural, not oracle-limited.

### ~~Learning rate scheduler~~ — REJECTED

**Why**: Training is stable (grad norms 13-16, no oscillation). Loss is flat at 3.13, suggesting convergence, not optimization instability. LR schedule won't help a representational bottleneck.

---

## Recommended Action Plan

### Immediate (This Session)
1. **Implement Strategy 1A**: Wider encoder (embed_dim=64, state_dim=128)
   - Modify `easy_config.py`
   - Resume from epoch 650 checkpoint
   - Run 20 epochs, measure agreement improvement
   - **Expected**: +5-10pp if encoder is bottleneck

2. **Parallel**: Implement Strategy 4: Curriculum phasing
   - Add phase schedule to `train_easy.py`
   - Test with current encoder for baseline
   - **Expected**: +3-5pp with minimal effort

### Short-term (Next 1-2 sessions)
3. **If Strategy 1 succeeds**: Add Strategy 2 (Focal Loss)
   - Replace CE loss with focal loss
   - Fine-tune gamma/alpha
   - **Expected**: Additional +3-7pp

4. **If still plateaued**: Implement Strategy 3 (Two-Stage Training)
   - Build TwoStageTRM architecture
   - Train hold/trade detector first
   - Fine-tune action selector
   - **Expected**: +7-14pp (higher effort)

### Long-term (Post-75% agreement)
5. **Investigate Strategy 5**: Synthetic/real gap
   - If model >75% on synthetic but <60% on real
   - Mix real data into training
   - Expand Bella generator diversity

---

## Success Criteria

| Strategy | Target Agreement | Trade Accuracy | Epochs to Verify |
|----------|------------------|----------------|------------------|
| 1A (Wider encoder) | 70-75% | 55-60% | 20-30 |
| 2 (Focal loss) | 66-70% | 48-52% | 10-20 |
| 3 (Two-stage) | 68-75% | 50-58% | 30-50 |
| 4 (Curriculum) | 65-68% | 46-50% | 10-20 |
| 5 (Data mix) | 65-70% | 48-55% | 20-30 |

**Combined potential**: Sequential application of 1+2+4 could reach 75-80% agreement.

---

## Risk Mitigation

**If encoder expansion fails** (no improvement):
- Revert to state_dim=64, try Strategy 3 (two-stage)
- Investigate cross-asset attention — is 64 assets too many for current attention heads?
- Consider asset grouping (sector-based encoding)

**If focal loss destabilizes** (training diverges):
- Fall back to adaptive weighting with higher trade weight (50× vs current 24×)
- Reduce gamma to 1.0 (less aggressive focusing)

**If curriculum phasing slows learning**:
- Keep full curriculum L0-L14, add validation per-level metrics
- Identify which levels model struggles with, focus sampling there

---

## Summary

**The 61% ceiling is a representational bottleneck**, not data or optimization. The encoder compresses rich multi-asset temporal data into a tiny 64-dim vector before sophisticated reasoning. Expanding encoder representation (Strategy 1) is the highest-leverage fix.

**Recommended start**: Widen encoder (embed_dim=32→64, state_dim=64→128) + curriculum phasing. Expected impact: 61% → 68-75% in 20-30 epochs.
