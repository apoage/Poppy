# Session Notes - 2026-01-06 to 2026-01-10

## Overview

Major focus on dual collapse prevention and stability interventions. Created modular intervention systems, ran extensive sweeps, and found best configurations.

---

## 2026-01-06: Stability Intervention Implementation

### Problem: Dual Collapse

Training exhibited coupled policy-prediction collapse:
1. **Policy collapse**: Entropy bleeds → HOLD dominates (85%+)
2. **Prediction collapse**: Loss → 0.01 (trivial "no change" solution)

Feedback loop identified:
```
MCTS produces sharp π (HOLD-heavy)
    ↓
Policy trained to match → π_network becomes HOLD-heavy
    ↓
Data distribution shifts to HOLD-only
    ↓
Prediction head sees no dynamics → learns "no change"
    ↓
Value estimates become uninformative
    ↓
MCTS produces even sharper HOLD targets
```

### Intervention Modules Created

| Module | File | Purpose |
|--------|------|---------|
| ε-greedy MCTS | `exploration.py` | Mix uniform into MCTS policy |
| Dirichlet Noise | `exploration.py` | Aggressive exploration (α=0.5, ε=0.5) |
| SAC Entropy | `sac_entropy.py` | Learnable entropy coefficient |
| Flow Matching | `flow_matching.py` | Predict velocity not state |
| Differential Reward | `differential_reward.py` | Subtract market average |

### CLI Flags Added
```bash
--epsilon-greedy 0.2        # Mix uniform into MCTS policy
--dirichlet-aggressive      # α=0.5, ε=0.5
--sac-entropy               # Learnable entropy coefficient
--sac-target-entropy-ratio  # Target % of max entropy
--flow-matching             # Predict velocity not state
--flow-velocity-weight      # Weight for velocity loss
--differential-reward       # Subtract market average
--differential-weight       # Weight for differential component
```

### Sweep Infrastructure

Created `stability_sweep.py` and `stability_tournament.py`:
- Quick mode: 50 epochs (~2h)
- Full mode: 150 epochs (~8h)
- Overnight mode: 300 epochs (~16h)

---

## 2026-01-06 to 01-07: Warmup Sweep Results

Ran comprehensive sweep testing 17 configurations.

### Top Configurations

| Rank | Config | Score | MaxRwd | HOLD>70% at |
|------|--------|-------|--------|-------------|
| 1 | kitchen_sink | 0.735 | 39.96 | epoch 27 |
| 2 | very_high_entropy_05 | 0.725 | 38.27 | epoch 27 |
| 3 | combined_v3_aggressive | 0.695 | 38.87 | epoch 15 |
| 4 | combined_v2 | 0.690 | 41.19 | epoch 8 |
| 5 | early_adaptive_060 | 0.675 | 38.71 | epoch 8 |

### Kitchen Sink Config (Best Stability)
```bash
--sims-warmup 300 --sims-min 4 --sims-max 24 \
--entropy-coef 0.2 \
--entropy-hold-threshold 0.55 \
--entropy-boost-low 0.4 --entropy-boost-high 0.6 \
--prediction-loss-weight 0.5 \
--physics-warmup-steps 10000
```

Key findings:
- Lower sims (4-24) + longer warmup (300) delays collapse
- Higher entropy coefficient (0.2) + aggressive boosts help
- Early adaptive trigger (0.55 vs 0.85) catches collapse sooner

---

## 2026-01-07 to 01-08: Stability Intervention Sweep

Tested individual interventions vs baseline.

### Individual Interventions
| Config | Result |
|--------|--------|
| epsilon_010 | Moderate improvement |
| epsilon_020 | Good exploration |
| epsilon_030 | Best single intervention |
| dirichlet_aggressive | Good diversity |
| sac_entropy_040/050 | Stable entropy control |
| flow_matching_080/100 | Reduced prediction collapse |
| differential_reward_050/100 | Mixed results |

### Combination Results
| Config | Notes |
|--------|-------|
| combo_exploration | ε-greedy + Dirichlet = good |
| combo_entropy_control | ε-greedy + SAC = very stable |
| combo_prediction_reward | Flow + differential = moderate |
| kitchen_sink_v2 | All interventions = best |

---

## 2026-01-09 to 01-10: Curriculum Training (L1)

### Level 1 Training Runs

| Run | Config | Status |
|-----|--------|--------|
| L1_eps030_dirichlet | ε=0.3, dirichlet | Completed |
| L1_kitchen_sink_v2 | All interventions | Completed |
| L1_hybrid_v1 | ε + entropy + ks + pred | Completed |
| L1_hybrid_v2 | ks base + dirichlet | Completed |

### Best L1 Result: L1_kitchen_sink_v2

500 epochs completed with:
- Stable entropy throughout
- Positive alpha generation
- No catastrophic collapse

### sweep_level1.py Created

Automated Level 1 sweep with checkpoint loading:
```python
L0_CHECKPOINT = "jax_ckpt_fast/eps030_dirichlet_longwarmup"

configs = [
    ("kitchen_sink_v2", {...}),
    ("hybrid_v1", {...}),
    ...
]
```

---

## 2026-01-10: Level 2 Training Started

Started L2 curriculum with kitchen_sink_v2 config:
```bash
python train_jax_fast.py \
    --synthetic-level 2 \
    --load-checkpoint jax_ckpt_fast/L1_kitchen_sink_v2 \
    --clear-buffer --reset-optimizer \
    ...
```

---

## Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `mcts_trader/jax_agent/exploration.py` | ε-greedy, Dirichlet |
| `mcts_trader/jax_agent/sac_entropy.py` | SAC entropy |
| `mcts_trader/jax_agent/flow_matching.py` | Flow matching |
| `mcts_trader/jax_agent/differential_reward.py` | Differential reward |
| `stability_sweep.py` | Intervention sweep |
| `stability_tournament.py` | Multi-run tournament |
| `sweep_level1.py` | L1 curriculum sweep |
| `analyze_sweep.py` | Sweep result analysis |

### Modified Files
| File | Change |
|------|--------|
| `train_jax_fast.py` | CLI args for all interventions |
| `CLAUDE.md` | Documented dual collapse, configs |

---

## Task Status Updates

| Task | Status | Notes |
|------|--------|-------|
| TASK_037 (ε-greedy) | IMPLEMENTED | exploration.py |
| TASK_038 (Differential) | IMPLEMENTED | differential_reward.py |
| TASK_039 (Dirichlet) | IMPLEMENTED | exploration.py |
| TASK_040 (SAC Entropy) | IMPLEMENTED | sac_entropy.py |
| TASK_041 (Flow Matching) | IMPLEMENTED | flow_matching.py |

---

## Key Learnings

1. **MCTS sharpening is the root cause**: High sims → sharp policy → HOLD dominance
2. **Warmup is critical**: 300 epochs of low sims (4→24) prevents early lock-in
3. **Entropy requires active management**: Adaptive triggers at 0.55 catch collapse early
4. **Combinations work better**: Kitchen sink v2 with all interventions is most stable
5. **Prediction loss helps**: 0.5 weight keeps prediction head learning
