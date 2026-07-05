# TASK_026: Weight Health Monitor for DS24

## Problem

Heavy negative experience during training can cause **weight saturation**:
- Value head weights drift toward extreme negative predictions
- Policy weights saturate toward conservative actions (HOLD)
- Once saturated, model struggles to recover even when positive opportunities appear

Currently no visibility into weight health during training - only symptoms (HOLD spam, poor rewards) are visible after damage is done.

## Solution

Add weight health monitoring to DS24 Level 4+ that tracks:
1. Weight magnitude distribution per layer
2. Gradient flow health
3. Saturation risk detection with alerts

## Implementation

### 1. New Metrics (DS24 L4+)

```python
# Weight magnitude statistics per component
'weight_mean_encoder'      # CNN encoder weight mean
'weight_std_encoder'       # CNN encoder weight std
'weight_max_encoder'       # Max absolute weight (saturation indicator)

'weight_mean_trm'          # TRM reasoning weights
'weight_std_trm'
'weight_max_trm'

'weight_mean_value_head'   # Value head weights
'weight_std_value_head'
'weight_max_value_head'

'weight_mean_policy_head'  # Policy/output refiner weights
'weight_std_policy_head'
'weight_max_policy_head'

'weight_mean_dynamics'     # Dynamics model weights
'weight_std_dynamics'
'weight_max_dynamics'

# Aggregate health score
'weight_health_score'      # 0-100, lower = more saturated
'saturation_risk'          # 'low', 'medium', 'high', 'critical'
```

### 2. Health Calculation

```python
def compute_weight_health(params: Dict) -> Dict:
    """Compute weight health metrics from model parameters."""

    health_metrics = {}
    saturation_scores = []

    # Define component groupings
    components = {
        'encoder': ['conv_', 'encoder'],
        'trm': ['TRMReasoning', 'latent_refiner', 'output_refiner'],
        'value_head': ['value_head'],
        'policy_head': ['policy_', 'output_refiner'],
        'dynamics': ['dynamics', 'action_embedding'],
    }

    for comp_name, patterns in components.items():
        # Collect all weights matching patterns
        weights = []
        for key, value in flatten_params(params).items():
            if any(p in key for p in patterns) and 'kernel' in key:
                weights.append(value.flatten())

        if weights:
            all_weights = np.concatenate(weights)

            health_metrics[f'weight_mean_{comp_name}'] = float(np.mean(all_weights))
            health_metrics[f'weight_std_{comp_name}'] = float(np.std(all_weights))
            health_metrics[f'weight_max_{comp_name}'] = float(np.max(np.abs(all_weights)))

            # Saturation score: penalize extreme values
            # Healthy range: most weights in [-2, 2], max < 5
            max_abs = np.max(np.abs(all_weights))
            extreme_frac = np.mean(np.abs(all_weights) > 2.0)

            comp_score = 100 * (1 - extreme_frac) * np.exp(-max(0, max_abs - 5) / 5)
            saturation_scores.append(comp_score)

    # Aggregate health score
    health_metrics['weight_health_score'] = float(np.mean(saturation_scores))

    # Risk classification
    score = health_metrics['weight_health_score']
    if score > 80:
        health_metrics['saturation_risk'] = 'low'
    elif score > 60:
        health_metrics['saturation_risk'] = 'medium'
    elif score > 40:
        health_metrics['saturation_risk'] = 'high'
    else:
        health_metrics['saturation_risk'] = 'critical'

    return health_metrics
```

### 3. Integration with DS24 Hooks

```python
# In training_hooks.py

def _log_weight_health(self, epoch: int, trainer_state: Any):
    """Level 4+: Log weight health metrics."""
    if trainer_state is None:
        return

    params = trainer_state.params
    health = compute_weight_health(params)

    # Log to CSV
    for key, value in health.items():
        self.logger.current_row[key] = value

    # Print warning if risk is high
    risk = health.get('saturation_risk', 'low')
    if risk in ('high', 'critical'):
        score = health.get('weight_health_score', 0)
        print(f"  ⚠️ WEIGHT SATURATION {risk.upper()}: health={score:.0f}/100")

        # Identify worst component
        worst_comp = None
        worst_max = 0
        for key, value in health.items():
            if key.startswith('weight_max_') and value > worst_max:
                worst_max = value
                worst_comp = key.replace('weight_max_', '')

        if worst_comp:
            print(f"     Worst: {worst_comp} (max={worst_max:.2f})")
```

### 4. Dashboard Visualization

Add to DS24 web dashboard (Analysis tab):

```javascript
// Weight Health Gauge
<div class="health-gauge">
    <span class="health-label">Weight Health</span>
    <div class="health-bar">
        <div class="health-fill" style="width: ${healthScore}%"></div>
    </div>
    <span class="health-value">${healthScore}/100</span>
    <span class="risk-badge risk-${riskLevel}">${riskLevel}</span>
</div>

// Weight Distribution Chart (per component)
// Stacked bar showing mean/std/max for each component
```

### 5. Automatic Snapshot on Critical Risk

```python
# In on_epoch_end(), after computing weight health:
if health['saturation_risk'] == 'critical':
    print(f"  🚨 CRITICAL SATURATION - Forcing snapshot!")
    self._save_tensor_snapshot(epoch, trajectory, trainer_state)

    # Optionally: suggest rollback
    print(f"     Consider rolling back to earlier checkpoint")
```

## Files to Modify

| File | Changes |
|------|---------|
| `tools/DS24/training_hooks.py` | Add `_log_weight_health()`, call from L4+ |
| `tools/DS24/metrics_logger.py` | Add weight health columns |
| `tools/DS24/server.py` | Add weight health to analysis |
| `tools/DS24/dashboard.js` | Add weight health visualization |

## CSV Columns (Level 4+)

```
weight_mean_encoder, weight_std_encoder, weight_max_encoder,
weight_mean_trm, weight_std_trm, weight_max_trm,
weight_mean_value_head, weight_std_value_head, weight_max_value_head,
weight_mean_policy_head, weight_std_policy_head, weight_max_policy_head,
weight_mean_dynamics, weight_std_dynamics, weight_max_dynamics,
weight_health_score, saturation_risk
```

## Testing

```bash
# Run training with L4 logging
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --model-size small \
    --ds24-level 4 \
    --name weight_health_test

# Check CSV for weight health columns
head -1 logs/training_weight_health_test.csv | tr ',' '\n' | grep -i weight

# Verify health score in output
grep -i "health\|saturation" logs/training_fast_weight_health_test.log
```

## Success Criteria

1. Weight health metrics logged at DS24 L4+
2. Saturation risk classification working (low/medium/high/critical)
3. Console warning when risk is high/critical
4. Dashboard shows weight health gauge
5. Automatic snapshot on critical saturation

## Priority

Medium - Useful diagnostic for curriculum training, especially on harder levels where negative experience is common.

## Dependencies

- DS24 Level 4 logging infrastructure (done)
- Tensor snapshot saving (done)

## Estimated Effort

~150 lines of code across 4 files
