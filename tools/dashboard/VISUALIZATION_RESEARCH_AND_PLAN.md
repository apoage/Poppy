# Enhanced Training Metrics Visualization
## Research Summary & Implementation Plan

**Date**: 2024-12-30
**System**: TRM+MuZero Trading Agent with Neural ODE Regularization

---

## Executive Summary

This document presents research findings and a refactoring plan for optimizing the dashboard visualization of training metrics. The current dashboard (Chart.js/Flask) provides solid foundational coverage but lacks visualization for:
- Recursive reasoning dynamics (TRM H×L cycles)
- Neural ODE physics components
- Component interaction analysis
- Multi-scale temporal dynamics

---

## Part 1: Research Findings

### 1.1 MuZero-Specific Visualization Best Practices

Based on research from ["Visualizing MuZero Models"](https://arxiv.org/pdf/2102.12924) and ["Demystifying MuZero Planning"](https://arxiv.org/html/2411.04580v2):

| Metric Category | Recommended Visualization | Insight Provided |
|-----------------|---------------------------|------------------|
| **Value Prediction** | Calibration scatter with 45° reference line | Prediction bias detection |
| **Value Prediction** | Error heatmap across planning horizons | Model accuracy vs lookahead |
| **Policy Quality** | KL divergence from MCTS policy | Policy improvement tracking |
| **Policy Quality** | Action probability distribution per state | Decision certainty analysis |
| **Reward Prediction** | Predicted vs actual reward timeline | Dynamics model accuracy |
| **Planning Depth** | Value accuracy vs simulation count | Optimal search depth |

**Key Finding**: MuZero latent spaces benefit from PCA/t-SNE visualization to show how value-relevant information is compressed during training.

### 1.2 Transformer/Recursive Layer Visualization

Based on ["AttentionViz"](https://catherinesyeh.github.io/attn-docs/), ["Clustering Head"](https://arxiv.org/html/2410.24050), and ["Analyzing Transformer Dynamics"](https://arxiv.org/abs/2308.10874):

| Aspect | Visualization Technique | Purpose |
|--------|------------------------|---------|
| **Latent Refinement** | Residual magnitude per H×L cycle | Track reasoning depth utilization |
| **Output Evolution** | Policy entropy per H_cycle | See decision crystallization |
| **Embedding Space** | 2D projection (PCA/UMAP) over training | Representation learning progress |
| **Cycle Contribution** | Stacked bar of delta norms | Identify active vs dormant cycles |

**Key Finding**: The TRM's recursive refinement can be visualized as "movement through embedding space" - tracking how the latent state evolves across H×L iterations shows reasoning patterns.

### 1.3 Neural ODE / Physics Regularization Visualization

Based on ["Smooth and Sparse Latent Dynamics"](https://arxiv.org/html/2402.15636v1), ["PINODE"](https://www.nature.com/articles/s41598-023-36799-6), and ["Latent Space Energy-based Neural ODEs"](https://arxiv.org/abs/2409.03845):

| Component | Visualization | Diagnostic Value |
|-----------|---------------|------------------|
| **G_momentum** | Time series per asset | Trend-following strength |
| **G_reversion** | Deviation from MA overlay | Mean-reversion calibration |
| **G_volatility** | GARCH variance estimates | Volatility regime detection |
| **G_correlation** | Heatmap of Γ matrix | Factor structure emergence |
| **Trajectory Smoothness** | Jerk/acceleration metrics | Regularization effectiveness |
| **Physics Loss** | Component breakdown stacked area | Balance of physical constraints |

**Key Finding**: Jerk regularization (third derivative) is crucial for smooth latent dynamics - visualizing jerk magnitude helps diagnose chaotic vs stable learning.

### 1.4 Multi-Component Interaction Analysis

Based on ["LossLens"](https://arxiv.org/html/2412.13321v1) and ["Multi-Task Learning Using Uncertainty"](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf):

| Analysis Type | Visualization | Insight |
|---------------|---------------|---------|
| **Loss Contribution** | Stacked area (normalized) | Which loss dominates training |
| **Loss Gradients** | Gradient norm per loss head | Gradient competition detection |
| **TRM↔ODE Alignment** | Dual y-axis correlation plot | Co-learning effectiveness |
| **Prediction Correlation** | Correlation matrix heatmap | Head redundancy/complementarity |

**Key Finding**: Loss landscape visualization (merge trees, persistence diagrams) reveals training stability - useful for detecting curriculum transition points.

---

## Part 2: Current Dashboard Gap Analysis

### Currently Implemented (10 Charts)
1. Reward over time (with MA)
2. Loss components (total, value, policy, reward)
3. Action distribution (stacked area)
4. Value calibration (MSE + Std)
5. Value calibration (Scale + Shift)
6. Policy entropy
7. Exploration vs Sells scatter
8. Training speed
9. Value calibration scatter

### Missing Components

#### Critical (High Impact)
| Gap | Impact | Priority |
|-----|--------|----------|
| TRM cycle dynamics | Cannot diagnose reasoning utilization | P0 |
| Physics component breakdown | Cannot verify physical model learning | P0 |
| Planning horizon analysis | Cannot optimize MCTS depth | P1 |
| Latent space evolution | Cannot visualize representation learning | P1 |

#### Important (Medium Impact)
| Gap | Impact | Priority |
|-----|--------|----------|
| Loss gradient analysis | Cannot detect gradient conflicts | P2 |
| Correlation matrix heatmap | Cannot see factor emergence | P2 |
| Trajectory smoothness metrics | Cannot verify ODE regularization | P2 |
| Multi-head prediction correlation | Cannot identify redundancy | P3 |

---

## Part 3: Refactoring Plan

### 3.1 Architecture Changes

```
Current:                          Proposed:

┌─────────────────┐              ┌─────────────────────────────────┐
│ CSV Logging     │              │ Enhanced Logging (JSON-Lines)   │
│ (30 columns)    │              │ - Scalar metrics (CSV compat)   │
└────────┬────────┘              │ - Tensor snapshots (periodic)   │
         │                       │ - Attention maps (on-demand)    │
         ▼                       └────────────────┬────────────────┘
┌─────────────────┐                               │
│ Flask Backend   │              ┌────────────────▼────────────────┐
│ (JSON API)      │              │ Flask Backend (Enhanced)        │
└────────┬────────┘              │ - /api/scalars (existing)       │
         │                       │ - /api/tensors (new)            │
         ▼                       │ - /api/physics (new)            │
┌─────────────────┐              │ - /api/latent (new)             │
│ Chart.js        │              └────────────────┬────────────────┘
│ (10 charts)     │                               │
└─────────────────┘              ┌────────────────▼────────────────┐
                                 │ Visualization Layer             │
                                 │ - Chart.js (scalar metrics)     │
                                 │ - Plotly.js (3D/heatmaps)       │
                                 │ - D3.js (custom interactions)   │
                                 └─────────────────────────────────┘
```

### 3.2 New Metrics to Log (Training Side)

Add to `train_jax_fast.py` CSV columns:

```python
# TRM Dynamics (per epoch, aggregated)
'latent_delta_norm_h0', 'latent_delta_norm_h1', ...  # Per H_cycle
'output_delta_norm_h0', 'output_delta_norm_h1', ...  # Per H_cycle
'latent_final_norm',                                  # Final latent magnitude
'trm_utilization',                                    # % of cycles with delta > threshold

# Physics Components (per epoch, aggregated)
'G_momentum_mean', 'G_momentum_std',
'G_reversion_mean', 'G_reversion_std',
'G_volatility_mean', 'G_volatility_std',
'G_correlation_mean', 'G_correlation_std',
'physics_alpha_mean', 'physics_beta_mean',  # Learned parameters
'garch_persistence',                         # α + β for GARCH

# Planning Metrics
'mcts_root_value', 'mcts_visit_entropy',
'value_horizon_1_error', 'value_horizon_5_error',  # Multi-horizon

# ODE Regularization
'trajectory_jerk', 'latent_smoothness',
'physics_constraint_violation',
```

### 3.3 New Dashboard Tabs/Sections

#### Tab: "Reasoning Dynamics" (New)
| Chart | Type | Data Source |
|-------|------|-------------|
| TRM Cycle Utilization | Heatmap (H×L) | latent_delta_norm per cycle |
| Policy Crystallization | Multi-line | output_delta_norm per H_cycle |
| Latent Magnitude | Area chart | latent_final_norm over epochs |
| Cycle Contribution | Stacked bar | Fraction of total delta per cycle |

#### Tab: "Physics Model" (New)
| Chart | Type | Data Source |
|-------|------|-------------|
| Physics Components | Stacked area | G_* means (4 components) |
| Learned Parameters | Multi-line | alpha, beta, sigma evolution |
| Correlation Heatmap | Heatmap 64×64 | Periodic Γ matrix snapshots |
| GARCH Persistence | Line | α + β over training |
| Volatility Regime | Scatter | G_volatility vs realized vol |

#### Tab: "Planning Analysis" (New)
| Chart | Type | Data Source |
|-------|------|-------------|
| Horizon Accuracy | Line (multi) | Error at horizons 1,2,5,10 |
| MCTS Value vs Actual | Scatter | root_value vs episode_reward |
| Search Utilization | Line | visit_entropy (higher = more exploration) |
| Simulation ROI | Line | Reward improvement vs sim count |

#### Tab: "Component Interactions" (New)
| Chart | Type | Data Source |
|-------|------|-------------|
| Loss Contribution | Normalized stacked area | value/policy/reward/pred fractions |
| Gradient Norms | Multi-line | Per-head gradient magnitudes |
| TRM↔Physics Alignment | Dual y-axis | latent_norm vs physics_reg |
| Prediction Correlation | Heatmap | Correlation matrix of all outputs |

### 3.4 Technology Additions

```javascript
// Add to index.html
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

// Plotly for:
// - 3D latent space visualization
// - Interactive heatmaps
// - Surface plots for loss landscapes
```

---

## Part 4: Implementation Task Breakdown

### Phase 1: Logging Infrastructure (Backend)
**Estimated Files**: `train_jax_fast.py`, `mcts_trader/jax_agent/trainer.py`

| Task ID | Description | Dependencies |
|---------|-------------|--------------|
| V1.1 | Add TRM cycle delta tracking to training loop | None |
| V1.2 | Add physics component logging (G_* means/stds) | None |
| V1.3 | Add MCTS diagnostic metrics | None |
| V1.4 | Implement periodic tensor snapshot saving | V1.1 |
| V1.5 | Update CSV schema with new columns | V1.1-V1.3 |

### Phase 2: Backend API Extensions
**Estimated Files**: `tools/dashboard/server.py`

| Task ID | Description | Dependencies |
|---------|-------------|--------------|
| V2.1 | Add `/api/tensors/<run>` endpoint | V1.4 |
| V2.2 | Add `/api/physics/<run>` for physics params | V1.2 |
| V2.3 | Add `/api/latent/<run>` for PCA projections | V2.1 |
| V2.4 | Add analysis functions for new metrics | V1.5 |
| V2.5 | Update flexible CSV reader for new columns | V1.5 |

### Phase 3: Frontend Visualization
**Estimated Files**: `tools/dashboard/dashboard.js`, `index.html`, `styles.css`

| Task ID | Description | Dependencies |
|---------|-------------|--------------|
| V3.1 | Add "Reasoning Dynamics" tab structure | None |
| V3.2 | Implement TRM cycle heatmap (Plotly) | V2.1 |
| V3.3 | Implement latent magnitude chart | V1.1 |
| V3.4 | Add "Physics Model" tab structure | None |
| V3.5 | Implement physics stacked area chart | V2.2 |
| V3.6 | Implement correlation heatmap | V2.2 |
| V3.7 | Add "Planning Analysis" tab structure | None |
| V3.8 | Implement horizon accuracy chart | V1.3 |
| V3.9 | Add "Component Interactions" tab | None |
| V3.10 | Implement loss contribution chart | V1.5 |
| V3.11 | Implement gradient norms chart | V1.5 |
| V3.12 | Add responsive styling for new tabs | V3.1-V3.9 |

### Phase 4: Analysis Enhancement
**Estimated Files**: `tools/dashboard/server.py`

| Task ID | Description | Dependencies |
|---------|-------------|--------------|
| V4.1 | Add TRM utilization analysis | V1.1 |
| V4.2 | Add physics convergence detection | V1.2 |
| V4.3 | Add planning depth recommendations | V1.3 |
| V4.4 | Update insights/warnings for new metrics | V4.1-V4.3 |

### Phase 5: Documentation & Testing
| Task ID | Description | Dependencies |
|---------|-------------|--------------|
| V5.1 | Update README with new features | All |
| V5.2 | Add sample data for testing | V1.5 |
| V5.3 | Cross-browser testing | V3.* |

---

## Part 5: Recommended Chart Types Summary

| Metric Type | Best Chart | Library | Reason |
|-------------|-----------|---------|--------|
| Scalar time series | Line/Area | Chart.js | Existing, performant |
| Multi-component breakdown | Stacked area | Chart.js | Clear contribution view |
| 2D matrix/heatmap | Heatmap | Plotly.js | Interactive, zoomable |
| 3D latent space | 3D scatter | Plotly.js | Rotation, exploration |
| Correlation structure | Heatmap + dendogram | Plotly.js | Clustering insight |
| Calibration | Scatter + reference | Chart.js | Direct comparison |
| Cycle utilization | Heatmap (small) | Plotly.js | Dense info display |
| Loss landscape | Surface plot | Plotly.js | Advanced diagnostics |

---

## Part 6: Priority Recommendations

### Immediate Value (Implement First)
1. **Physics component breakdown** - Critical for verifying market model learning
2. **TRM cycle utilization** - Shows if recursive reasoning is being used
3. **Loss contribution analysis** - Essential for multi-task balance

### High Value (Implement Second)
4. **Horizon accuracy curves** - Optimize MCTS simulation count
5. **Correlation heatmap** - Verify factor emergence
6. **GARCH persistence tracking** - Volatility model health

### Nice to Have (Implement Later)
7. **3D latent space visualization** - Deep diagnostic
8. **Loss landscape surface** - Research tool
9. **Gradient norm tracking** - Advanced debugging

---

## Sources

- [Visualizing MuZero Models](https://arxiv.org/pdf/2102.12924) - MuZero latent space analysis
- [Demystifying MuZero Planning](https://arxiv.org/html/2411.04580v2) - Planning depth effectiveness
- [AttentionViz](https://catherinesyeh.github.io/attn-docs/) - Transformer attention visualization
- [Clustering Head Training Dynamics](https://arxiv.org/html/2410.24050) - Two-phase learning visualization
- [Analyzing Transformer Dynamics](https://arxiv.org/abs/2308.10874) - Embedding space movement
- [Smooth Latent Dynamics with Jerk Regularization](https://arxiv.org/html/2402.15636v1) - ODE smoothness metrics
- [PINODE](https://www.nature.com/articles/s41598-023-36799-6) - Physics-informed neural ODE
- [Latent Space Energy-based Neural ODEs](https://arxiv.org/abs/2409.03845) - Latent trajectory modeling
- [LossLens](https://arxiv.org/html/2412.13321v1) - Loss landscape visualization
- [Multi-Task Learning Loss Weighting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf) - Loss contribution analysis
- [W&B Observability for RL](https://wandb.ai/wandb_fc/genai-research/reports/Observability-tools-for-reinforcement-learning--VmlldzoxNDE3MzExMw) - RL dashboard best practices
