# TASK_044: DS24 Dashboard Enhancement

## Status: COMPLETED (2026-01-11)

## Priority: MEDIUM

## Problem Statement

### 1. Missing Feature Logging
The DS24 dashboard doesn't track several newly implemented features:
- **TASK_022**: GAE returns (use_gae, gae_lambda)
- **TASK_025**: Per-head gradient clipping (clip values per head)
- **Exploration**: Epsilon-greedy mixing ratio
- **TASK_042**: ReDo dormancy stats (when integrated)
- **TASK_043**: ATC contrastive loss (when integrated)

### 2. Graph Rendering Issue
Several graphs only show data points on hover, not continuous lines:
- **Policy Crystallization** (`output_delta_norm_h*`)
- **Latent Magnitude Over Training** (`latent_final_norm`)
- **TRM Utilization Score** (`trm_utilization`)

**Root Cause**: Data is only logged every `intermediates_every` epochs (default: 10),
resulting in sparse data that Chart.js renders as disconnected points instead of smooth lines.

## Solution

### Phase 1: Add Missing Feature Logging to DS24

**File: `tools/DS24/metrics_logger.py`**

Add new columns to `DS24_COLUMNS`:

```python
DS24_COLUMNS = [
    # ... existing columns ...

    # TASK_022: GAE Returns Config
    'ds24_use_gae', 'ds24_gae_lambda',

    # TASK_025: Per-head Gradient Clipping
    'ds24_per_head_clipping',
    'ds24_clip_value', 'ds24_clip_policy',
    'ds24_clip_dynamics', 'ds24_clip_encoder',
    # Per-head gradient norms AFTER clipping (to verify clipping works)
    'grad_norm_value_clipped', 'grad_norm_policy_clipped',
    'grad_norm_dynamics_clipped', 'grad_norm_encoder_clipped',

    # Exploration
    'ds24_epsilon_greedy',

    # TASK_042: ReDo Dormancy Stats (when integrated)
    'redo_dormant_neurons', 'redo_dormant_pct', 'redo_recycled_count',

    # TASK_043: ATC Contrastive Loss (when integrated)
    'atc_loss', 'atc_pos_sim', 'atc_neg_sim', 'atc_temperature',
]
```

**File: `tools/DS24/training_hooks.py`**

Add new logging methods:

```python
def _log_training_config(self, args):
    """Log training configuration flags."""
    self.logger.log_training_config(
        use_gae=getattr(args, 'use_gae', False),
        gae_lambda=getattr(args, 'gae_lambda', 0.95),
        per_head_clipping=getattr(args, 'per_head_clipping', False),
        clip_grad_value=getattr(args, 'clip_grad_value', 1.0),
        clip_grad_policy=getattr(args, 'clip_grad_policy', 1.0),
        clip_grad_dynamics=getattr(args, 'clip_grad_dynamics', 1.0),
        clip_grad_encoder=getattr(args, 'clip_grad_encoder', 1.0),
        epsilon_greedy=getattr(args, 'epsilon_greedy', 0.0),
    )

def _log_redo_stats(self, redo_stats: Optional[Dict] = None):
    """Log ReDo dormancy statistics."""
    if redo_stats:
        self.logger.log_redo_stats(
            dormant_neurons=redo_stats.get('dormant_neurons', 0),
            dormant_pct=redo_stats.get('dormancy_percent', 0.0),
            recycled_count=redo_stats.get('recycled_count', 0),
        )

def _log_atc_stats(self, atc_stats: Optional[Dict] = None):
    """Log ATC contrastive loss statistics."""
    if atc_stats:
        self.logger.log_atc_stats(
            loss=atc_stats.get('loss', 0.0),
            pos_sim=atc_stats.get('pos_similarity', 0.0),
            neg_sim=atc_stats.get('neg_similarity', 0.0),
            temperature=atc_stats.get('temperature', 0.1),
        )
```

### Phase 2: Fix Graph Rendering

**Issue**: Charts show only hover points because data is logged every N epochs.

**Solution A: Interpolate Missing Data Points**

**File: `tools/DS24/dashboard.js`**

Add interpolation function:

```javascript
function interpolateData(data, epochs) {
    // Fill gaps in sparse data for smooth line rendering
    if (!data || data.length === 0) return data;

    const result = [];
    let lastValidValue = null;
    let lastValidIdx = -1;

    for (let i = 0; i < epochs.length; i++) {
        if (data[i] !== null && data[i] !== undefined) {
            // Fill gaps with linear interpolation
            if (lastValidIdx >= 0 && lastValidIdx < i - 1) {
                const step = (data[i] - lastValidValue) / (i - lastValidIdx);
                for (let j = lastValidIdx + 1; j < i; j++) {
                    result[j] = lastValidValue + step * (j - lastValidIdx);
                }
            }
            result[i] = data[i];
            lastValidValue = data[i];
            lastValidIdx = i;
        } else {
            // Carry forward last valid value (for real-time display)
            result[i] = lastValidValue;
        }
    }

    return result;
}
```

Apply to TRM charts:

```javascript
// In updateTRMCharts function
if (state.charts.crystallization) {
    state.charts.crystallization.data.labels = epochs;
    for (let h = 0; h < 4; h++) {
        const key = `output_delta_norm_h${h}`;
        // Interpolate sparse data for smooth line
        state.charts.crystallization.data.datasets[h].data =
            interpolateData(data[key] || [], epochs);
    }
    state.charts.crystallization.update('none');
}

if (state.charts.latentNorm && data.latent_final_norm) {
    state.charts.latentNorm.data.labels = epochs;
    // Interpolate sparse data
    state.charts.latentNorm.data.datasets[0].data =
        interpolateData(data.latent_final_norm, epochs);
    state.charts.latentNorm.update('none');
}

if (state.charts.trmUtilization && data.trm_utilization) {
    state.charts.trmUtilization.data.labels = epochs;
    // Interpolate sparse data
    state.charts.trmUtilization.data.datasets[0].data =
        interpolateData(data.trm_utilization, epochs);
    state.charts.trmUtilization.update('none');
}
```

**Solution B: Chart.js spanGaps Option**

Alternative simpler fix - enable spanGaps:

```javascript
function createLineChart(canvasId, config) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: config.datasets.map(d => ({
                ...d,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true,  // ADD THIS - connects line across null values
            }))
        },
        // ... rest of options
    });
}

function createAreaChart(canvasId, config) {
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: config.label,
                borderColor: config.borderColor,
                backgroundColor: config.backgroundColor,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2,
                spanGaps: true,  // ADD THIS
            }]
        },
        // ... rest of options
    });
}
```

### Phase 3: Add New Dashboard Panels

**File: `tools/DS24/index.html`**

Add new card for training config:

```html
<!-- Training Config Panel -->
<div class="card">
    <h3>Training Config</h3>
    <div class="config-grid">
        <div class="config-item">
            <span class="label">GAE:</span>
            <span id="config-gae" class="value">-</span>
        </div>
        <div class="config-item">
            <span class="label">GAE λ:</span>
            <span id="config-gae-lambda" class="value">-</span>
        </div>
        <div class="config-item">
            <span class="label">Per-head Clip:</span>
            <span id="config-per-head-clip" class="value">-</span>
        </div>
        <div class="config-item">
            <span class="label">ε-greedy:</span>
            <span id="config-epsilon" class="value">-</span>
        </div>
    </div>
</div>

<!-- ReDo Dormancy Panel (when available) -->
<div class="card" id="redo-panel" style="display: none;">
    <h3>ReDo: Dormant Neurons</h3>
    <canvas id="redo-chart" height="200"></canvas>
    <div class="metric-row">
        <span>Dormant:</span>
        <span id="redo-dormant-count">-</span>
        <span id="redo-dormant-pct">(-)</span>
    </div>
</div>

<!-- ATC Contrastive Panel (when available) -->
<div class="card" id="atc-panel" style="display: none;">
    <h3>ATC: Contrastive Loss</h3>
    <canvas id="atc-chart" height="200"></canvas>
    <div class="metric-row">
        <span>Pos Sim:</span>
        <span id="atc-pos-sim">-</span>
        <span>Neg Sim:</span>
        <span id="atc-neg-sim">-</span>
    </div>
</div>
```

## Files to Modify

1. **`tools/DS24/metrics_logger.py`**
   - Add new DS24_COLUMNS for all missing features
   - Add `log_training_config()` method
   - Add `log_redo_stats()` method
   - Add `log_atc_stats()` method

2. **`tools/DS24/training_hooks.py`**
   - Add `_log_training_config()` method
   - Add `_log_redo_stats()` method
   - Add `_log_atc_stats()` method
   - Call new methods in `on_epoch_end()`

3. **`tools/DS24/dashboard.js`**
   - Add `interpolateData()` function
   - Apply interpolation to TRM charts
   - OR add `spanGaps: true` to all charts
   - Add update functions for new panels

4. **`tools/DS24/index.html`**
   - Add Training Config panel
   - Add ReDo Dormancy panel
   - Add ATC Contrastive panel

5. **`tools/DS24/styles.css`**
   - Add styles for new panels
   - Add config-grid styles

6. **`train_jax_fast.py`**
   - Pass new stats to DS24 hooks in training loop
   - Collect ReDo stats when enabled
   - Collect ATC stats when enabled

## Testing Plan

1. **Unit test**: Verify new columns are added to CSV
2. **Visual test**: Verify graphs show continuous lines with sparse data
3. **Integration test**: Run training with DS24 level 3+, verify all new panels populate
4. **Stress test**: Run with intermediates_every=50 to verify interpolation works

## Success Criteria

1. All new feature flags logged to CSV
2. TRM charts show smooth lines (not just hover points)
3. New dashboard panels appear when relevant features are enabled
4. No performance regression from additional logging

## Estimated Scope

- metrics_logger.py: ~50 lines
- training_hooks.py: ~80 lines
- dashboard.js: ~100 lines
- index.html: ~60 lines
- styles.css: ~30 lines
- train_jax_fast.py: ~20 lines

## Dependencies

- TASK_042 (ReDo) and TASK_043 (ATC) need to be fully integrated first
- DS24 server must be running to test visual changes

## Notes

The graph rendering fix (spanGaps or interpolation) should be implemented first
as it's a quick win that improves existing functionality. New panels can be
added incrementally as features are integrated.
