# TASK_058: Adaptive Difficulty with Multi-Modal Noise

## Problem Statement

1. **Synthetic data too simple for TRM** - Model achieves +51 reward on synthetic but -61 on real
2. **H1/H2 cycles are cold** - Model shortcuts to H3, wasting ~4M parameters
3. **L3-L14 don't apply noise** - Bug: `_apply_noise()` only called in L0-L2
4. **No adaptive difficulty** - Static difficulty leads to memorization or frustration

## Goal

Keep model in "zone of proximal development" with reward ~10-20 by:
- Adaptive noise that responds to performance
- Multi-modal noise that forces temporal reasoning
- Gradual difficulty increase to engage all H cycles

## Noise Types

### 1. Price Noise (existing, needs extension to L3-L14)
```python
# Random walk + per-step noise + amplitude variation
noisy_prices = prices * (1 + random_walk + step_noise) * amp_variation
```

### 2. Temporal Noise (NEW)
```python
# Jitter in observation timing - model sees slightly delayed/advanced prices
temporal_jitter = rng.integers(-jitter_steps, jitter_steps+1, size=n_assets)
# Each asset's observation is shifted by different amount
```

### 3. Time Dilation (NEW)
```python
# Stretch or compress time periods - patterns appear faster/slower
dilation_factor = 1.0 + rng.uniform(-max_dilation, max_dilation)
# Resample prices to dilated time axis
dilated_prices = np.interp(dilated_time, original_time, prices)
```

### 4. Feature Dropout (NEW)
```python
# Randomly mask some channels in observation
# Forces model to reason with incomplete information
dropout_mask = rng.random(n_channels) > dropout_rate
obs = obs * dropout_mask
```

### 5. Pattern Phase Shift (NEW)
```python
# Shift pattern phase mid-episode
# Forces model to re-detect patterns, not memorize timing
if step == phase_shift_step:
    phase_offset += rng.uniform(0, 2*pi)
```

## Adaptive Controller

```python
class AdaptiveNoiseController:
    """Adjusts difficulty to maintain target reward range."""

    def __init__(
        self,
        target_reward_low: float = 10.0,
        target_reward_high: float = 20.0,
        adjustment_rate: float = 0.05,
        ema_alpha: float = 0.1,
    ):
        self.target_low = target_reward_low
        self.target_high = target_reward_high
        self.adjustment_rate = adjustment_rate
        self.ema_alpha = ema_alpha

        # Noise levels (all start at 0, grow as needed)
        self.price_noise = 0.0
        self.temporal_jitter = 0  # steps
        self.time_dilation = 0.0
        self.feature_dropout = 0.0
        self.phase_shift_prob = 0.0

        # Tracking
        self.reward_ema = 15.0  # Start at middle of range

    def update(self, epoch_reward: float) -> Dict[str, float]:
        """Update noise levels based on recent reward."""
        # Update EMA
        self.reward_ema = (1 - self.ema_alpha) * self.reward_ema + self.ema_alpha * epoch_reward

        if self.reward_ema > self.target_high:
            # Too easy - increase difficulty
            self._increase_difficulty()
        elif self.reward_ema < self.target_low:
            # Too hard - decrease difficulty
            self._decrease_difficulty()

        return self.get_noise_config()

    def _increase_difficulty(self):
        """Progressively add noise types."""
        # Priority: price noise -> temporal -> dilation -> dropout -> phase shift
        if self.price_noise < 1.0:
            self.price_noise = min(1.0, self.price_noise + self.adjustment_rate)
        elif self.temporal_jitter < 5:
            self.temporal_jitter = min(5, self.temporal_jitter + 1)
        elif self.time_dilation < 0.3:
            self.time_dilation = min(0.3, self.time_dilation + 0.02)
        elif self.feature_dropout < 0.2:
            self.feature_dropout = min(0.2, self.feature_dropout + 0.02)
        elif self.phase_shift_prob < 0.3:
            self.phase_shift_prob = min(0.3, self.phase_shift_prob + 0.02)

    def _decrease_difficulty(self):
        """Reduce difficulty in reverse order."""
        if self.phase_shift_prob > 0:
            self.phase_shift_prob = max(0, self.phase_shift_prob - self.adjustment_rate)
        elif self.feature_dropout > 0:
            self.feature_dropout = max(0, self.feature_dropout - self.adjustment_rate)
        elif self.time_dilation > 0:
            self.time_dilation = max(0, self.time_dilation - 0.02)
        elif self.temporal_jitter > 0:
            self.temporal_jitter = max(0, self.temporal_jitter - 1)
        elif self.price_noise > 0:
            self.price_noise = max(0, self.price_noise - self.adjustment_rate)

    def get_noise_config(self) -> Dict[str, float]:
        return {
            'price_noise': self.price_noise,
            'temporal_jitter': self.temporal_jitter,
            'time_dilation': self.time_dilation,
            'feature_dropout': self.feature_dropout,
            'phase_shift_prob': self.phase_shift_prob,
        }
```

## Implementation Plan

### Phase 1: Fix Existing Noise (L3-L14) - COMPLETE
- [x] Add `_apply_noise()` calls to all level generators
- [x] Verify noise is applied during data regeneration

### Phase 2: Add Temporal Noise - COMPLETE
- [x] Implement temporal jitter in synthetic market generator
- [x] Add time dilation to synthetic market generator
- [x] Add phase shift capability to pattern generators

### Phase 3: Adaptive Controller - COMPLETE
- [x] Create `AdaptiveNoiseController` class
- [x] Integrate with training loop (update after each epoch)
- [x] Pass noise config to data regeneration

### Phase 4: Feature Dropout - DEFERRED
- [ ] Add channel dropout to observation pipeline
- [ ] Make dropout rate adaptive
- Note: Feature dropout deferred - price noise + temporal noise + time dilation + phase shift provide sufficient difficulty control

## CLI Flags

```
--adaptive-noise              # Enable adaptive difficulty
--target-reward-low 10        # Lower bound of target range
--target-reward-high 20       # Upper bound of target range
--noise-adjustment-rate 0.05  # How fast noise changes
--max-temporal-jitter 5       # Max jitter in steps
--max-time-dilation 0.3       # Max time stretch/compress
--max-feature-dropout 0.2     # Max channel dropout rate
--max-phase-shift 0.3         # Max phase shift probability
```

## Expected Outcomes

1. **Reward stays in 10-20 range** - Constant challenge
2. **H1/H2 cycles heat up** - Need reasoning for temporal noise
3. **Better generalization** - Can't memorize dilated/shifted patterns
4. **Smooth difficulty curve** - No sudden jumps

## Metrics to Track (DS24)

- `adaptive_noise_level` - Current price noise
- `adaptive_temporal_jitter` - Current jitter steps
- `adaptive_time_dilation` - Current dilation factor
- `adaptive_feature_dropout` - Current dropout rate
- `adaptive_phase_shift` - Current phase shift prob
- `reward_ema` - Smoothed reward for adaptation

## Files to Modify

| File | Changes |
|------|---------|
| `mcts_trader/data/synthetic_market.py` | Add noise to L3-L14, temporal noise, dilation |
| `mcts_trader/jax_agent/adaptive_noise.py` | NEW: AdaptiveNoiseController |
| `mcts_trader/training/env_factory.py` | Pass extended noise config |
| `train_jax_fast.py` | CLI flags, controller integration |
| `tools/DS24/metrics_logger.py` | Log adaptive noise metrics |

## Success Criteria

- [x] All levels (L0-L14) apply noise when configured
- [ ] Reward stabilizes in 10-20 range with adaptive noise (needs testing)
- [ ] H1 cycle utilization increases (target: >2.0 norm, up from 1.1) (needs testing)
- [ ] Synthetic-to-real gap decreases (target: <50 points, down from 112) (needs testing)

## Implementation Status: COMPLETE (2026-01-15)

### Files Created/Modified:
- `mcts_trader/jax_agent/adaptive_noise.py` - NEW: AdaptiveNoiseController + noise helper functions
- `mcts_trader/data/synthetic_market.py` - Added multi-modal noise to all levels (L0-L14)
- `mcts_trader/training/env_factory.py` - Extended to accept noise parameters
- `train_jax_fast.py` - CLI flags + controller integration + DS24 logging

### Usage:
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --jax-env \
    --adaptive-noise \
    --target-reward-low 10 \
    --target-reward-high 20
```

## References

- TASK_049: Original synthetic noise implementation
- Dual collapse research: `docs/AI_bleed research/`
- TRM architecture: `mcts_trader/jax_agent/trm_flax.py`
