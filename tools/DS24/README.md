# DS24 - Enhanced Training Metrics Dashboard

Advanced visualization dashboard for the TRM+MuZero Trading Agent, featuring:
- **MuZero Metrics**: Value/policy/reward prediction analysis
- **TRM Dynamics**: Recursive reasoning cycle visualization
- **Neural ODE Physics**: Market physics component breakdown
- **Component Interactions**: Multi-task learning diagnostics

## Quick Start

```bash
# Start the dashboard
./start_ds24.sh

# Or with custom port
./start_ds24.sh 8080

# Or directly with Python
python server.py --port 8051
```

Then open http://localhost:8051 in your browser.

## Features

### Metrics Tab (Standard)
- Reward over time with moving average
- Loss components (value, policy, reward)
- Action distribution (buy/hold/sell/limit orders)
- Value calibration (MSE & standard deviation)
- Policy entropy tracking
- Training speed analysis

### Reasoning Dynamics Tab (New)
Visualizes TRM's recursive refinement process:
- **Cycle Heatmap**: Delta magnitude per (H, L) cycle
- **Policy Crystallization**: How decisions form across H cycles
- **Latent Magnitude**: Representation strength over training
- **TRM Utilization**: Percentage of cycles actively contributing

### Physics Model Tab (New)
Neural ODE market physics visualization:
- **Component Breakdown**: G_momentum, G_reversion, G_volatility, G_correlation
- **Learned Parameters**: α (momentum) and β (reversion) coefficients
- **GARCH Persistence**: Volatility model stationarity check
- **Correlation Heatmap**: Asset correlation structure

### Planning Analysis Tab (New)
MCTS planning diagnostics:
- **Horizon Accuracy**: Value prediction error at different lookaheads
- **MCTS Calibration**: Root value vs actual reward scatter
- **Visit Entropy**: Search exploration indicator
- **Simulation ROI**: Reward per additional MCTS simulation

### Component Interactions Tab (New)
Multi-task learning analysis:
- **Loss Contribution**: Normalized breakdown of loss terms
- **Gradient Norms**: Per-head gradient magnitudes
- **TRM ↔ Physics Alignment**: Representation vs physics correlation
- **Head Correlation**: Prediction output correlations

### Analysis Tab
- Automated insights (positive patterns)
- Warnings (potential issues)
- Training phase detection
- Detailed metrics summary
- Component health status

### Checkpoints Tab
- List available model checkpoints
- Checkpoint size and metadata

### Logs Tab
- Browse training log files
- Real-time log viewing

## Integrating Enhanced Metrics

To enable the new visualization features, add DS24 logging to your training:

```python
# In train_jax_fast.py
from tools.DS24.metrics_logger import DS24MetricsLogger

# Initialize logger
ds24_logger = DS24MetricsLogger(
    log_path='logs/training_myrun.csv',
    tensor_dir='tensor_snapshots',
    snapshot_every=100
)

# In training loop, after each epoch:
ds24_logger.reset_row()

# Log TRM dynamics
ds24_logger.log_trm_dynamics(
    latent_deltas={'h0': delta_h0, 'h1': delta_h1, ...},
    output_deltas={'h0': out_delta_h0, 'h1': out_delta_h1, ...},
    latent_final=final_latent
)

# Log physics (if using prediction model)
ds24_logger.log_physics_components(
    G_momentum=physics_out.G_momentum,
    G_reversion=physics_out.G_reversion,
    G_volatility=physics_out.G_volatility,
    G_correlation=physics_out.G_correlation
)

# Log planning metrics
ds24_logger.log_planning_metrics(
    root_values=mcts_output.root_value,
    visit_counts=mcts_output.visit_counts
)

# Get values to append to CSV
ds24_values = ds24_logger.get_current_values()
```

## CSV Columns (DS24 Enhanced)

### Standard Columns
`epoch, reward, reward_ma20, steps, fee_bps, n_buy, n_hold, n_sell, n_limit_buy, n_limit_sell, total_loss, value_loss, policy_loss, reward_loss, value_mse, value_std, policy_entropy, ...`

### DS24 Additional Columns
```
# TRM Dynamics
latent_delta_norm_h0, latent_delta_norm_h1, latent_delta_norm_h2, latent_delta_norm_h3
output_delta_norm_h0, output_delta_norm_h1, output_delta_norm_h2, output_delta_norm_h3
latent_final_norm, trm_utilization

# Physics Components
G_momentum_mean, G_momentum_std
G_reversion_mean, G_reversion_std
G_volatility_mean, G_volatility_std
G_correlation_mean, G_correlation_std
physics_alpha_mean, physics_beta_mean
garch_persistence

# Planning Metrics
mcts_root_value, mcts_visit_entropy
value_horizon_1_error, value_horizon_5_error, value_horizon_10_error

# Gradient Analysis
grad_norm_value, grad_norm_policy, grad_norm_dynamics
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | List training CSV files |
| `GET /api/training/<path>` | Get training metrics |
| `GET /api/analysis/<path>` | Get comprehensive analysis |
| `GET /api/trm/<path>` | Get TRM dynamics data |
| `GET /api/physics/<path>` | Get physics component data |
| `GET /api/planning/<path>` | Get planning metrics |
| `GET /api/loss-components/<path>` | Get normalized loss breakdown |
| `GET /api/tensors/<run>` | List tensor snapshots |
| `GET /api/tensor/<path>` | Get tensor data (with PCA) |
| `GET /api/checkpoints` | List checkpoints |
| `GET /api/log-files` | List log files |
| `GET /api/logs/<path>` | Get log content |
| `GET /api/status` | GPU/training status |

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5 + CSS3 + JavaScript
- **Charts**: Chart.js (line/area), Plotly.js (heatmaps/3D)
- **Data**: CSV logs, NPZ tensor snapshots

## File Structure

```
tools/DS24/
├── server.py           # Flask backend (REST API)
├── index.html          # Dashboard HTML structure
├── dashboard.js        # Chart.js + Plotly.js visualization
├── styles.css          # Dark theme styling
├── metrics_logger.py   # Training integration helper
├── start_ds24.sh       # Startup script
└── README.md           # This file
```

## Research Basis

This dashboard implements visualization best practices from:
- [Visualizing MuZero Models](https://arxiv.org/pdf/2102.12924)
- [AttentionViz](https://catherinesyeh.github.io/attn-docs/)
- [PINODE Physics-Informed Neural ODE](https://www.nature.com/articles/s41598-023-36799-6)
- [LossLens Loss Landscape Visualization](https://arxiv.org/html/2412.13321v1)

## Comparison with Original Dashboard

| Feature | Original | DS24 |
|---------|----------|------|
| Scalar metrics | Yes | Yes |
| TRM dynamics | No | Yes |
| Physics components | No | Yes |
| Planning analysis | No | Yes |
| Loss contribution | Partial | Full |
| Heatmaps | No | Yes (Plotly) |
| 3D visualization | No | Yes (Plotly) |
| Tensor snapshots | No | Yes |
| Auto-analysis | Basic | Enhanced |

## Requirements

- Python 3.8+
- Flask, flask-cors
- pandas, numpy
- Modern browser (Chrome, Firefox, Safari, Edge)

## Troubleshooting

**Dashboard shows "No data"**
- Ensure training has generated CSV files in `logs/` directory
- Check that CSV format matches expected columns

**Physics/TRM tabs show "Not Available"**
- Add DS24 metrics logging to your training script
- See integration example above

**Charts not updating**
- Enable auto-refresh checkbox
- Check browser console for errors
- Verify Flask server is running
