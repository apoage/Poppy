# TASK_044: Epsilon 0.30 + Dirichlet Long Warmup Experiment

## Status: PENDING (after current sweep completes)

## Context

From stability sweep analysis (2026-01-08):
- `epsilon_030` shows excellent alpha (+438%), stable entropy (std=0.066), but early HOLD collapse (epoch 9)
- `combo_triple_entropy` delays collapse (epoch 45) but has latent magnitude collapse (51→28) and lower alpha (+94%)

**Hypothesis**: SAC entropy causes latent collapse by over-regularizing representations.

## Proposed Experiment

Combine epsilon_030's strengths with longer prediction warmup, skip SAC:

```bash
python train_jax_fast.py \
    --synthetic-level 0 --synthetic-steps 10000 \
    --epochs 500 \
    --batch-size 384 --model-size small \
    --parallel-envs 50 --jax-env --bf16 \
    --sims 24 --sims-warmup 300 --sims-min 4 \
    --epsilon-greedy 0.3 \
    --dirichlet-aggressive \
    --use-popart --popart-beta 0.0001 \
    --use-prediction-loss --prediction-loss-weight 0.2 \
    --physics-warmup-steps 20000 \
    --adaptive-entropy --entropy-hold-threshold 0.55 \
    --entropy-boost-low 0.4 --entropy-boost-high 0.6 \
    --name eps030_dirichlet_longwarmup
```

**Key changes from triple_entropy:**
| Parameter | triple_entropy | This experiment |
|-----------|---------------|-----------------|
| epsilon_greedy | 0.2 | 0.3 |
| SAC entropy | enabled | disabled |
| physics_warmup | 10000 | 20000 |
| Dirichlet | aggressive | aggressive |

## Metrics to Monitor

1. **Latent magnitude** - should stay stable (not collapse like triple_entropy)
2. **Alpha** - should stay high (~400%+) like epsilon_030
3. **Horizon accuracy** - should improve with longer warmup
4. **GARCH persistence** - watch if it stabilizes
5. **HOLD collapse epoch** - target: >30 epochs

## Expected Outcomes

- Maintain epsilon_030's high alpha and stable entropy
- Delay HOLD collapse via Dirichlet exploration
- Avoid latent collapse by skipping SAC
- Better prediction model via longer warmup

## Variants to Consider

If successful, try:
1. `eps030_dirichlet_verylongwarmup` - physics_warmup=50000
2. `eps030_dirichlet_lightSAC` - add SAC with lower target (0.3 instead of 0.5)
3. `eps025_dirichlet` - slightly lower epsilon for balance

## Dependencies

- Current stability sweep must complete first
- Results from sweep may suggest other combinations to try
