# TASK_045: Dojo v2 - Full Curriculum with Multi-Phase Training

## Status: IMPLEMENTED (2026-01-13)

## Priority: HIGH

## Overview

Complete rewrite of dojo curriculum system with:
1. **ATC integration** into trainer
2. **Three-phase training** per level (Basic → Noise → Fees)
3. **Multi-period benchmark** after each level (Bull/Bear/Sideways)
4. **Full kitchen_sink_v2 production config**
5. **Auto-graduation** from Level 0 to Level 14

---

## Part 1: ATC Integration in Trainer

### What is ATC?
Augmented Temporal Contrast - InfoNCE contrastive loss that:
- Forces temporally close states → similar embeddings
- Forces temporally distant states → different embeddings
- Prevents representation collapse during long training

### Files to Modify

**`mcts_trader/jax_agent/trainer.py`**

```python
# Add import
from mcts_trader.jax_agent.atc import compute_atc_loss

# In loss computation (around line ~400-500):
if self.config.get('atc_loss', False):
    atc_loss_val = compute_atc_loss(
        encoder_fn=self._get_encoder_fn(),
        params=params,
        observations=batch['observations'],
        k_step=self.config.get('atc_k_step', 5),
        temperature=self.config.get('atc_temperature', 0.1),
        num_negatives=self.config.get('atc_num_negatives', 32),
    )
    total_loss = total_loss + self.config.get('atc_weight', 0.1) * atc_loss_val
    metrics['atc_loss'] = atc_loss_val
```

**`train_jax_fast.py`**

```python
# Pass ATC config to trainer (around line ~629):
trainer = MuZeroTrainer(
    ...
    atc_loss=args.atc_loss,
    atc_weight=args.atc_weight,
    atc_k_step=args.atc_k_step,
    atc_temperature=args.atc_temperature,
    atc_num_negatives=args.atc_num_negatives,
    ...
)

# Log ATC stats to DS24 (in on_epoch_end call):
atc_stats = {
    'loss': train_metrics.get('atc_loss', 0.0),
    'pos_sim': train_metrics.get('atc_pos_sim', 0.0),
    'neg_sim': train_metrics.get('atc_neg_sim', 0.0),
    'temperature': args.atc_temperature,
} if args.atc_loss else None
```

### CLI Flags (already exist)
```
--atc-loss              Enable ATC contrastive learning
--atc-weight 0.1        Weight for ATC loss
--atc-k-step 5          Temporal distance for positive pairs
--atc-temperature 0.1   InfoNCE temperature
--atc-num-negatives 32  Number of negative samples
```

---

## Part 2: Three-Phase Training Structure

### New Phase Structure (Per Level)

```
Level N:
├── Phase A (Basic):     300 epochs
│   ├── No synthetic noise
│   ├── No fees/slippage
│   └── Purpose: Learn basic pattern recognition
│
├── Phase B (Noise):     250 epochs
│   ├── --synthetic-noise 0.3
│   ├── No fees/slippage
│   └── Purpose: Robustness to price perturbations
│
├── Phase C (Fees):      250 epochs
│   ├── --synthetic-noise 0.3
│   ├── --fee 16 --slippage 5
│   └── Purpose: Learn to overcome transaction costs
│
├── Benchmark:           21 epochs (3 weeks of real data)
│   ├── 7 epochs: Bull market period
│   ├── 7 epochs: Bear market period
│   ├── 7 epochs: Sideways market period
│   └── No checkpoint save, metrics only
│
└── Graduate if: avg_reward > threshold
```

### Phase Configurations

```python
PHASES = [
    {
        'name': 'Phase A (Basic)',
        'epochs': 300,
        'synthetic_noise': 0.0,
        'fee': 0,
        'slippage': 0,
    },
    {
        'name': 'Phase B (Noise)',
        'epochs': 250,
        'synthetic_noise': 0.3,
        'fee': 0,
        'slippage': 0,
    },
    {
        'name': 'Phase C (Fees)',
        'epochs': 250,
        'synthetic_noise': 0.3,
        'fee': 16,
        'slippage': 5,
    },
]
```

---

## Part 3: Multi-Period Benchmark

### Benchmark Structure

```
After each level graduation:
├── Warmup: ~7 epochs (compilation, ignored)
├── Bull period:     7 epochs (1 week)
├── Bear period:     7 epochs (1 week)
├── Sideways period: 7 epochs (1 week)
├── Total eval:      21 epochs
└── Parallel envs:   4
```

### Period Selection (Need to identify)

| Period | Date Range | Market Condition | Description |
|--------|------------|------------------|-------------|
| Bull   | TBD        | Strong uptrend   | COVID recovery or 2021 bull |
| Bear   | TBD        | Strong downtrend | 2022 crash or COVID dump |
| Sideways | TBD      | Range-bound      | Consolidation period |

**Action Required**: Query database to find representative periods.

### Benchmark Implementation

```python
def run_benchmark(checkpoint_path: str, level: int) -> Dict:
    """
    Run 3-week benchmark on real data after level graduation.

    Returns metrics for bull/bear/sideways periods.
    """
    BENCHMARK_PERIODS = [
        {'name': 'bull', 'start': '2020-07-01', 'end': '2020-07-08', 'assets': BENCHMARK_ASSETS},
        {'name': 'bear', 'start': '2022-05-01', 'end': '2022-05-08', 'assets': BENCHMARK_ASSETS},
        {'name': 'sideways', 'start': '2021-09-01', 'end': '2021-09-08', 'assets': BENCHMARK_ASSETS},
    ]

    results = {}
    for period in BENCHMARK_PERIODS:
        result = evaluate_period(
            checkpoint=checkpoint_path,
            start_date=period['start'],
            end_date=period['end'],
            assets=period['assets'],
            epochs=7,           # 1 week
            steps=1440,         # 1 day per epoch
            parallel_envs=4,
            eval_mode=True,     # No training
            save_checkpoint=False,
        )
        results[period['name']] = {
            'reward': result['avg_reward'],
            'alpha': result['alpha'],
            'sharpe': result.get('sharpe', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'action_dist': result['action_distribution'],
        }

    # Save to CSV
    save_benchmark_results(level, results)
    return results
```

### Benchmark Output

```
logs/benchmarks/
├── level_0_benchmark.csv
├── level_1_benchmark.csv
├── ...
└── level_14_benchmark.csv
```

Each CSV contains:
```
period,reward,alpha,sharpe,max_drawdown,pct_buy,pct_hold,pct_sell
bull,42.5,+318%,1.2,-5%,15,62,23
bear,28.3,+125%,0.8,-12%,8,75,17
sideways,35.1,+89%,0.6,-3%,12,68,20
```

---

## Part 4: Kitchen Sink v2 Production Config

### Base Flags (Always Applied)

```python
KITCHEN_SINK_V2_FLAGS = [
    # Core
    '--batch-size', '384',
    '--model-size', 'small',
    '--parallel-envs', '50',
    '--jax-env',
    '--bf16',
    '--synthetic-steps', '10000',

    # PopArt + GAE
    '--use-popart', '--popart-beta', '0.0001',
    '--use-gae', '--gae-lambda', '0.95',

    # MCTS
    '--sims', '24',
    '--sims-warmup', '300',
    '--sims-min', '4',

    # Prediction
    '--use-prediction-loss',
    '--prediction-loss-weight', '0.5',
    '--physics-warmup-steps', '10000',

    # Entropy control
    '--adaptive-entropy',
    '--entropy-coef', '0.2',
    '--entropy-hold-threshold', '0.55',
    '--entropy-boost-low', '0.4',
    '--entropy-boost-high', '0.6',

    # LR schedule
    '--lr-schedule', 'cosine',
    '--lr-warmup-epochs', '20',
    '--lr-min', '1e-6',

    # Gradient clipping
    '--per-head-clipping',
    '--clip-grad-value', '1.0',
    '--clip-grad-policy', '1.0',
    '--clip-grad-dynamics', '1.0',
    '--clip-grad-encoder', '1.0',

    # Exploration
    '--epsilon-greedy', '0.1',

    # Realism
    '--probabilistic-fills',
    '--borrow-rate', '0.03',
    '--episode-variance', '30',

    # Maintenance
    '--redo', '--redo-interval', '50',

    # ATC (NEW)
    '--atc-loss',
    '--atc-weight', '0.1',

    # DS24
    '--ds24-level', '3',
    '--ds24-intermediates-every', '10',
]
```

### Phase-Specific Flags

```python
def get_phase_flags(phase: str, level: int) -> List[str]:
    if phase == 'A':
        return ['--synthetic-noise', '0.0', '--fee', '0', '--slippage', '0']
    elif phase == 'B':
        return ['--synthetic-noise', '0.3', '--fee', '0', '--slippage', '0']
    elif phase == 'C':
        return ['--synthetic-noise', '0.3', '--fee', '16', '--slippage', '5']
```

---

## Part 5: Dojo v2 Main Loop

### Pseudo-code

```python
class DojoV2:
    def run_curriculum(self, start_level: int = 0):
        checkpoint = None  # Fresh start

        for level in range(start_level, 15):
            generator = CURRICULUM[level]['generator']

            # Phase A: Basic (300 epochs)
            checkpoint = self.train_phase(
                level=level,
                phase='A',
                epochs=300,
                checkpoint=checkpoint,
                reset_optimizer=(level > 0),  # Reset on new level
            )

            # Phase B: Noise (250 epochs)
            checkpoint = self.train_phase(
                level=level,
                phase='B',
                epochs=250,
                checkpoint=checkpoint,
                reset_optimizer=False,
            )

            # Phase C: Fees (250 epochs)
            checkpoint = self.train_phase(
                level=level,
                phase='C',
                epochs=250,
                checkpoint=checkpoint,
                reset_optimizer=False,
            )

            # Benchmark on real data
            benchmark_results = self.run_benchmark(checkpoint, level)
            self.log_benchmark(level, benchmark_results)

            # Check graduation
            if not self.check_graduation(level, checkpoint):
                logging.warning(f"Level {level} not passed, repeating...")
                continue

            logging.info(f"Level {level} GRADUATED!")
```

---

## Part 6: Implementation Checklist

### Phase 1: ATC Integration
- [ ] Add ATC loss computation to trainer.py
- [ ] Pass ATC config from train_jax_fast.py to trainer
- [ ] Log ATC stats to DS24
- [ ] Test ATC with 2-epoch sanity run

### Phase 2: Identify Benchmark Periods
- [ ] Query database for bull market week (2020 or 2021)
- [ ] Query database for bear market week (2022)
- [ ] Query database for sideways market week
- [ ] Validate data coverage for all periods
- [ ] Create benchmark_periods.json config

### Phase 3: Dojo v2 Core
- [ ] Create new file: `tools/orchestration/dojo_v2.py`
- [ ] Implement 3-phase training loop
- [ ] Implement benchmark runner
- [ ] Implement graduation checker
- [ ] Add all kitchen_sink_v2 flags

### Phase 4: Testing
- [ ] Test Level 0 full cycle (A→B→C→Benchmark)
- [ ] Verify checkpoint transitions work
- [ ] Verify benchmark metrics are saved
- [ ] Test Level 0→1 graduation

### Phase 5: Full Run
- [ ] Start fresh Level 0 training
- [ ] Monitor with DS24 dashboard
- [ ] Track benchmark progression

---

## Estimated Training Time

| Component | Duration |
|-----------|----------|
| Per phase | 300/250/250 epochs × ~30s = ~6-7 hours |
| Per level | ~20 hours (3 phases + benchmark) |
| Full curriculum (0-14) | 15 levels × 20h = **~300 hours (12.5 days)** |

---

## Success Criteria

1. **ATC integrated**: Loss logged, representation diversity maintained
2. **3-phase structure**: Basic→Noise→Fees working per level
3. **Benchmark**: Real data eval after each level, metrics saved
4. **Auto-graduation**: Level 0→14 without manual intervention
5. **Metrics tracking**: Benchmark shows improvement over levels

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mcts_trader/jax_agent/trainer.py` | Modify | Add ATC loss integration |
| `train_jax_fast.py` | Modify | Pass ATC config, log stats |
| `tools/orchestration/dojo_v2.py` | Create | New curriculum manager |
| `config/benchmark_periods.json` | Create | Bull/bear/sideways periods |
| `config/kitchen_sink_v2.json` | Create | Production config flags |

---

## Notes

- Starting fresh from Level 0 (no checkpoint)
- ATC enabled by default for representation health
- Benchmark periods need validation against database
- ~12.5 days for full 0→14 curriculum
