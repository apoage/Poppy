# Regression Findings: Trade Action Learning Collapse

---

### REGRESSION-1: Silent Architecture Halving (hidden_dim 2048 → 1024)

**Suspect Commit**: 06c44ab (TASK_692/693) + f6669f4 (TASK_694)
**Change**: Change 8 — Model Construction parameters added to MuZeroNetworks
**Confidence**: CONFIRMED

**Why It Kills Trade Learning**:

The reasoning FFN in TRMReasoning was halved from 2048 to 1024 hidden units. This is a ~23M parameter reduction (from ~48M total to ~25.76M). The reasoning FFN is where ALL policy discrimination happens — 48 latent refiner cycles × 2-layer FFN, plus 6 output refiner cycles. Each cycle went from:

```
Dense(256 → 2048) → GeLU → Dense(2048 → 128)    # BEFORE: 786K params/cycle
Dense(256 → 1024) → GeLU → Dense(1024 → 128)    # AFTER:  393K params/cycle
```

Trades are ~3.9% of oracle decisions (k=4 out of 64 assets). Distinguishing the rare buy/sell from the dominant hold requires fine-grained nonlinear discrimination in the FFN. The relationship is super-linear: halving FFN width doesn't halve accuracy, it **collapses** it for minority classes because the model loses the capacity for fine-grained decision boundaries.

The agree_hold metric (94%) is stable because predicting hold is the "default" that doesn't require much capacity. Trade detection is the task that needs the FFN width.

**Evidence**:

1. `git show 06c44ab~1:easy_trainer.py` — Before commits, MuZeroNetworks created with 7 params only:
```python
# BEFORE (verified via git show):
self.model = MuZeroNetworks(
    state_dim=config.state_dim,
    action_space_size=config.action_space_size,
    support_size=VALUE_SUPPORT_SIZE,
    H_cycles=config.H_cycles,
    L_cycles=config.L_cycles,
    use_prediction_head=False,
    n_assets=config.n_assets,
)
# hidden_dim NOT PASSED → MuZeroNetworks default None → state_dim * 16 = 2048
```

2. `trm_flax.py:112`:
```python
hidden_dim = self.hidden_dim if self.hidden_dim is not None else self.state_dim * 16
```

3. `easy_config.py:175` — Large config sets `trm_hidden_dim=1024`:
```python
def get_large_config() -> EasyConfig:
    return EasyConfig(
        state_dim=128,
        trm_hidden_dim=1024,  # TASK_684: Constrain reasoning (was state_dim*16=2048 uncapped)
        ...
    )
```

4. After commits, hidden_dim is passed: `hidden_dim=getattr(config, 'trm_hidden_dim', None)` → 1024.

5. V2 encoder is NOT affected by embed_dim/encoder_num_heads changes — confirmed by reading `trm_flax.py:248`:
```python
# V2 encoder created WITHOUT embed_dim or encoder_num_heads:
self.encoder = AssetTransformerEncoder(state_dim=self.state_dim, dtype=self.dtype, param_dtype=self.param_dtype)
```

6. Previous run printed `Model parameters: ~48M`. Current run prints `Model parameters: 25.76M`. This is visible in the training log but may have gone unnoticed.

**Proposed Fix**:

Option A (quick): Force hidden_dim to None in easy_trainer.py for backward compat:
```python
# easy_trainer.py, model construction:
hidden_dim=None,  # Force state_dim*16 = 2048 for large models
```

Option B (proper): Update get_large_config() to use 2048:
```python
def get_large_config() -> EasyConfig:
    return EasyConfig(
        ...
        trm_hidden_dim=2048,  # Restore full reasoning capacity
    )
```

Option C (investigate): If 1024 is intentional (VRAM savings), the architecture needs
re-tuning for this capacity — e.g., more H/L cycles, different loss weighting, or
accepting slower convergence.

---

### REGRESSION-2: Oracle → Model State Channel Transition (Bootstrap Tax)

**Suspect Commit**: 06c44ab (TASK_692/693)
**Change**: Change 1 — Model-State Rollout replaces oracle state channels
**Confidence**: LIKELY

**Why It Kills Trade Learning**:

Before these commits, state channels (position, PnL, hold duration, heat, capacity, confidence) were derived from the **oracle's** actions — clean, consistent, and informative from epoch 0.

After: state channels come from the **model's own** actions. At epoch 0, the model is random → state channels are noise. The model must simultaneously learn:
1. To predict the right action (policy learning)
2. To interpret its own noisy state channels (state understanding)

This chicken-and-egg problem creates a bootstrap tax. The model can't use state channels to improve policy because the channels are garbage, and the channels can't improve because the policy is garbage.

**Evidence**:

1. `easy_config.py:91` — `autoregressive_state: bool = True` (default, always active)
2. Previous codebase had NO model-state rollout (06c44ab added it)
3. CSV shows slow upward trend (5.8% → 7.6% over epochs 60-77), consistent with gradual bootstrap
4. Previous run reached 15% at epoch 80 (with oracle channels providing immediate signal)

**Compound effect with REGRESSION-1**: The model has HALF the reasoning capacity (1024 vs 2048) while facing a HARDER learning problem (noisy state channels instead of clean ones). This is the worst-case combination.

**Proposed Fix**:

Option A: Disable autoregressive_state for first N epochs (e.g., 100), then switch:
```python
# easy_config.py
autoregressive_state_warmup: int = 100  # Use oracle channels for first 100 epochs
```

Option B: Add `--no-autoregressive` flag and compare against oracle-state baseline.

Option C: Start with oracle state channels, gradually mix in model state channels (curriculum).

---

### REGRESSION-3: Trivially-Solvable Oracle Mask Head (Noise Gradient Injection)

**Suspect Commit**: 77cb089 (TASK_695)
**Change**: Change 6 — Oracle mask prediction auxiliary head
**Confidence**: POSSIBLE (small but non-zero contribution)

**Why It Contributes**:

The oracle mask head predicts which k=4 out of 64 assets are selected. From epoch 0, the accuracy is 93.7% — exactly the "predict all zeros" baseline (60/64 = 93.75%). The head has effectively learned nothing; it just outputs large negative logits (predict no selection).

But it still generates gradients that flow through the shared latent state to the encoder. These gradients are small (0.8% of total loss) but they're **noise** — they don't carry useful learning signal because the task is trivially dominated by class imbalance.

With the halved reasoning capacity (REGRESSION-1), the model is already capacity-constrained. Adding noise gradients through the shared representation makes things marginally worse.

**Evidence**:

1. CSV: `oracle_mask_loss: 0.231`, `oracle_mask_accuracy: 0.937` (at epoch 60-77)
2. All-zeros baseline accuracy: 60/64 = 93.75% ≈ 93.7% (model hasn't learned anything useful)
3. Loss contribution: 0.1 × 0.231 = 0.023 out of ~2.8 total (0.8%)

**Proposed Fix**:

Disable oracle mask head for diagnostic:
```bash
# Remove --oracle-mask-head from launch command
# Or: set oracle_mask_weight = 0.0 in config
```

If agree_trade doesn't recover, this isn't the cause. If it improves by 1-2%, the head is adding harmful noise.

Long-term: The oracle mask head needs class-balanced loss (focal loss or per-asset weighting) to be useful — predicting 4 positives out of 64 is a severely imbalanced classification problem.
