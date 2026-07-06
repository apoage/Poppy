#!/usr/bin/env python3
"""
DS24 - Enhanced Training Metrics Dashboard Server

Advanced visualization for TRM+MuZero Trading Agent with:
- MuZero-specific metrics (value/policy/reward prediction)
- TRM recursive reasoning dynamics (H×L cycles)
- Neural ODE physics components
- Multi-component interaction analysis

Based on research from:
- Visualizing MuZero Models (arXiv:2102.12924)
- AttentionViz transformer visualization
- PINODE physics-informed neural ODE
"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, static_folder='.')
CORS(app)

# Paths
LOGS_DIR = PROJECT_ROOT / 'logs'
CHECKPOINTS_DIR = PROJECT_ROOT / 'jax_ckpt_fast'
TENSORS_DIR = PROJECT_ROOT / 'tensor_snapshots'  # For periodic tensor saves


# =============================================================================
# CSV Reading with Schema Evolution Support
# =============================================================================

def read_csv_flexible(csv_path: str) -> pd.DataFrame:
    """
    Read CSV with flexible schema handling.
    Handles mid-training column additions gracefully.
    """
    try:
        # First pass: detect header
        with open(csv_path, 'r') as f:
            header_line = f.readline().strip()

        columns = header_line.split(',')
        n_cols = len(columns)

        # Handle duplicate column names by appending suffix
        seen = {}
        unique_columns = []
        for col in columns:
            if col in seen:
                seen[col] += 1
                unique_columns.append(f"{col}_dup{seen[col]}")
            else:
                seen[col] = 0
                unique_columns.append(col)
        columns = unique_columns

        # Read with flexible parsing
        rows = []
        with open(csv_path, 'r') as f:
            next(f)  # Skip header
            for line_num, line in enumerate(f, start=2):
                parts = line.strip().split(',')
                if len(parts) < n_cols:
                    # Pad short rows
                    parts.extend([''] * (n_cols - len(parts)))
                elif len(parts) > n_cols:
                    # Truncate long rows (schema expanded mid-training)
                    parts = parts[:n_cols]
                rows.append(parts)

        df = pd.DataFrame(rows, columns=columns)

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        # Replace NaN with None for clean JSON serialization
        df = df.where(pd.notna(df), None)

        return df

    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return pd.DataFrame()


def safe_json(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import math

    # Handle None first
    if obj is None:
        return None

    # Handle numpy integers
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)

    # Handle numpy floats
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    # Handle Python floats (including NaN/Inf)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return [safe_json(x) for x in obj.tolist()]

    # Handle pandas Series
    elif isinstance(obj, pd.Series):
        return [safe_json(x) for x in obj.tolist()]

    # Handle dicts
    elif isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}

    # Handle lists/tuples
    elif isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]

    # Handle empty strings (from CSV empty cells)
    elif obj == '':
        return None

    return obj


# =============================================================================
# API Endpoints - Training Runs
# =============================================================================

@app.route('/api/runs')
def get_runs():
    """List available training runs (CSV files)."""
    runs = []

    if LOGS_DIR.exists():
        csv_files = list(LOGS_DIR.glob('training_*.csv'))
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for csv_path in csv_files[:50]:  # Limit to 50 most recent
            runs.append({
                'path': str(csv_path.relative_to(PROJECT_ROOT)),
                'name': csv_path.stem.replace('training_', ''),
                'size': csv_path.stat().st_size,
                'modified': datetime.fromtimestamp(csv_path.stat().st_mtime).isoformat()
            })

    return jsonify(runs)


@app.route('/api/training/<path:csv_path>')
def get_training_data(csv_path: str):
    """Get training metrics from CSV file."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    if df.empty:
        return jsonify({'error': 'Empty or invalid CSV'}), 400

    # Convert to dict with safe JSON conversion
    data = df.to_dict(orient='list')
    data = safe_json(data)

    return jsonify(data)


# =============================================================================
# API Endpoints - Analysis
# =============================================================================

@app.route('/api/analysis/<path:csv_path>')
def get_analysis(csv_path: str):
    """Perform comprehensive analysis on training data."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    if df.empty or len(df) < 5:
        return jsonify({'error': 'Insufficient data for analysis'}), 400

    analysis = perform_comprehensive_analysis(df)
    return jsonify(safe_json(analysis))


def perform_comprehensive_analysis(df: pd.DataFrame) -> Dict:
    """
    Comprehensive analysis including:
    - Basic metrics
    - Phase detection
    - TRM dynamics (if available)
    - Physics metrics (if available)
    - Insights and warnings
    """
    analysis = {
        'metrics': {},
        'phases': [],
        'insights': [],
        'warnings': [],
        'trm_dynamics': {},
        'physics_analysis': {},
        'planning_analysis': {},
        'market_baseline': {}
    }

    n_epochs = len(df)

    # === Basic Metrics ===
    if 'reward' in df.columns:
        rewards = df['reward'].dropna()
        if len(rewards) > 0:
            analysis['metrics']['reward_current'] = float(rewards.iloc[-1])
            analysis['metrics']['reward_mean'] = float(rewards.mean())
            analysis['metrics']['reward_std'] = float(rewards.std())
            analysis['metrics']['reward_max'] = float(rewards.max())

            # Trend
            if len(rewards) >= 20:
                recent = rewards.iloc[-20:].mean()
                early = rewards.iloc[:20].mean()
                analysis['metrics']['reward_trend'] = float(recent - early)

    # === TASK_058: Effective Reward (difficulty-adjusted) ===
    if 'reward' in df.columns and 'difficulty_level' in df.columns:
        rewards = df['reward'].dropna()
        difficulty = df['difficulty_level'].dropna()
        if len(rewards) > 0 and len(difficulty) > 0:
            # effective_reward = reward * (1 + difficulty)
            # Aligns indices in case of any NaN gaps
            effective = rewards * (1 + difficulty.reindex(rewards.index, fill_value=0))
            analysis['metrics']['effective_reward_current'] = float(effective.iloc[-1])
            analysis['metrics']['effective_reward_mean'] = float(effective.mean())
            analysis['metrics']['effective_reward_max'] = float(effective.max())
            analysis['metrics']['difficulty_current'] = float(difficulty.iloc[-1])
            analysis['metrics']['difficulty_mean'] = float(difficulty.mean())

    # Value calibration
    if 'value_mse' in df.columns:
        mse = df['value_mse'].dropna()
        if len(mse) > 0:
            analysis['metrics']['value_mse_current'] = float(mse.iloc[-1])
            analysis['metrics']['value_mse_improvement'] = float(mse.iloc[0] - mse.iloc[-1]) if len(mse) > 1 else 0

    if 'value_std' in df.columns:
        vstd = df['value_std'].dropna()
        if len(vstd) > 0:
            analysis['metrics']['value_std_current'] = float(vstd.iloc[-1])

    # Policy entropy
    if 'policy_entropy' in df.columns:
        entropy = df['policy_entropy'].dropna()
        if len(entropy) > 0:
            analysis['metrics']['entropy_current'] = float(entropy.iloc[-1])
            analysis['metrics']['entropy_trend'] = float(entropy.iloc[-1] - entropy.iloc[0]) if len(entropy) > 1 else 0

    # Action distribution
    for action in ['n_buy', 'n_hold', 'n_sell', 'n_limit_buy', 'n_limit_sell']:
        if action in df.columns:
            recent = df[action].iloc[-10:] if len(df) >= 10 else df[action]
            analysis['metrics'][f'{action}_recent_mean'] = float(recent.mean())

    # === TRM Dynamics Analysis ===
    trm_cols = [c for c in df.columns if c.startswith('latent_delta_') or c.startswith('output_delta_')]
    if trm_cols:
        analysis['trm_dynamics']['available'] = True
        analysis['trm_dynamics']['cycles'] = []

        for col in sorted(trm_cols):
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    analysis['trm_dynamics']['cycles'].append({
                        'name': col,
                        'mean': float(vals.mean()),
                        'std': float(vals.std()),
                        'trend': float(vals.iloc[-1] - vals.iloc[0]) if len(vals) > 1 else 0
                    })

        # TRM utilization metric
        if 'trm_utilization' in df.columns:
            util = df['trm_utilization'].dropna()
            if len(util) > 0:
                analysis['trm_dynamics']['utilization_current'] = float(util.iloc[-1])
                analysis['trm_dynamics']['utilization_mean'] = float(util.mean())
    else:
        analysis['trm_dynamics']['available'] = False

    # === Physics Analysis ===
    physics_cols = ['G_momentum_mean', 'G_reversion_mean', 'G_volatility_mean', 'G_correlation_mean']
    available_physics = [c for c in physics_cols if c in df.columns]

    if available_physics:
        analysis['physics_analysis']['available'] = True
        analysis['physics_analysis']['components'] = {}

        for col in available_physics:
            vals = df[col].dropna()
            if len(vals) > 0:
                component = col.replace('_mean', '')
                analysis['physics_analysis']['components'][component] = {
                    'current': float(vals.iloc[-1]),
                    'mean': float(vals.mean()),
                    'std': float(vals.std())
                }

        # GARCH persistence
        if 'garch_persistence' in df.columns:
            garch = df['garch_persistence'].dropna()
            if len(garch) > 0:
                analysis['physics_analysis']['garch_persistence'] = float(garch.iloc[-1])
                if garch.iloc[-1] > 0.99:
                    analysis['warnings'].append('GARCH near non-stationarity (persistence > 0.99)')
    else:
        analysis['physics_analysis']['available'] = False

    # === Planning Analysis ===
    if 'mcts_root_value' in df.columns:
        analysis['planning_analysis']['available'] = True
        root_val = df['mcts_root_value'].dropna()
        if len(root_val) > 0:
            analysis['planning_analysis']['root_value_mean'] = float(root_val.mean())

        # Horizon errors
        for horizon in [1, 5, 10]:
            col = f'value_horizon_{horizon}_error'
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    analysis['planning_analysis'][f'horizon_{horizon}_error'] = float(vals.iloc[-1])
    else:
        analysis['planning_analysis']['available'] = False

    # === Market Baseline Analysis ===
    if 'alpha' in df.columns:
        alpha = df['alpha'].dropna()
        if len(alpha) > 0:
            analysis['market_baseline']['available'] = True
            analysis['market_baseline']['alpha_current'] = float(alpha.iloc[-1])
            analysis['market_baseline']['alpha_mean'] = float(alpha.mean())
            analysis['market_baseline']['alpha_std'] = float(alpha.std())

            # Positive alpha rate
            positive_rate = (alpha > 0).sum() / len(alpha) * 100
            analysis['market_baseline']['positive_alpha_rate'] = float(positive_rate)

            # Recent alpha trend
            if len(alpha) >= 20:
                recent = alpha.iloc[-20:].mean()
                early = alpha.iloc[:20].mean()
                analysis['market_baseline']['alpha_trend'] = float(recent - early)
        else:
            analysis['market_baseline']['available'] = False
    else:
        analysis['market_baseline']['available'] = False

    # === Phase Detection (uses effective reward if difficulty available) ===
    if 'reward' in df.columns:
        rewards = df['reward'].values
        difficulty = df['difficulty_level'].values if 'difficulty_level' in df.columns else None
        analysis['phases'] = detect_training_phases(rewards, difficulty)

    # === Generate Insights ===
    analysis['insights'], analysis['warnings'] = generate_insights(df, analysis['metrics'])

    return analysis


def detect_training_phases(rewards: np.ndarray, difficulty: np.ndarray = None) -> List[Dict]:
    """Detect training phases: exploring, improving, declining, plateau.

    Uses effective reward (reward × (1 + difficulty)) if difficulty is available.
    """
    phases = []
    n = len(rewards)

    if n < 10:
        return phases

    # Compute effective reward if difficulty available
    if difficulty is not None and len(difficulty) == len(rewards):
        # Handle NaN values
        diff_clean = np.nan_to_num(difficulty, nan=0.0)
        rew_clean = np.nan_to_num(rewards, nan=0.0)
        effective_rewards = rew_clean * (1 + diff_clean)
    else:
        effective_rewards = np.nan_to_num(rewards, nan=0.0)

    window = max(5, n // 10)

    i = 0
    while i < n - window:
        segment = effective_rewards[i:i + window]

        # Calculate trend
        x = np.arange(len(segment))
        slope = np.polyfit(x, segment, 1)[0] if len(segment) > 1 else 0

        if slope > 0.5:
            phase_type = 'improving'
        elif slope < -0.5:
            phase_type = 'declining'
        else:
            phase_type = 'plateau'

        # Extend phase while trend continues
        end = i + window
        while end < n - 1:
            next_segment = effective_rewards[end:min(end + window, n)]
            if len(next_segment) < 3:
                break
            next_slope = np.polyfit(np.arange(len(next_segment)), next_segment, 1)[0]

            if phase_type == 'improving' and next_slope <= 0.3:
                break
            elif phase_type == 'declining' and next_slope >= -0.3:
                break
            elif phase_type == 'plateau' and abs(next_slope) > 0.5:
                break
            end += window // 2

        phases.append({
            'type': phase_type,
            'start': int(i),
            'end': int(min(end, n)),
            'mean_reward': float(np.mean(effective_rewards[i:end])),
            'mean_raw_reward': float(np.mean(np.nan_to_num(rewards[i:end], nan=0.0)))
        })

        i = end

    return phases


def generate_insights(df: pd.DataFrame, metrics: Dict) -> Tuple[List[str], List[str]]:
    """Generate actionable insights and warnings."""
    insights = []
    warnings = []

    # Reward insights
    if 'reward_trend' in metrics:
        if metrics['reward_trend'] > 1.0:
            insights.append(f"Strong reward improvement: +{metrics['reward_trend']:.2f} over training")
        elif metrics['reward_trend'] < -1.0:
            warnings.append(f"Reward regression: {metrics['reward_trend']:.2f} decline")

    # Value calibration
    if 'value_mse_current' in metrics:
        if metrics['value_mse_current'] < 1.0:
            insights.append("Excellent value calibration (MSE < 1.0)")
        elif metrics['value_mse_current'] > 10.0:
            warnings.append(f"High value MSE ({metrics['value_mse_current']:.1f}) - consider adjusting value loss weight")

    if 'value_std_current' in metrics:
        if metrics['value_std_current'] < 0.5:
            warnings.append("Low value diversity - predictions may be too uniform")
        elif metrics['value_std_current'] > 5.0:
            insights.append("Good value diversity detected")

    # Policy entropy
    if 'entropy_current' in metrics:
        if metrics['entropy_current'] < 1.5:
            warnings.append(f"Low entropy ({metrics['entropy_current']:.2f}) - policy may have collapsed")
        elif metrics['entropy_current'] > 4.0:
            insights.append("High exploration maintained")

        if 'entropy_trend' in metrics and metrics['entropy_trend'] < -1.0:
            warnings.append("Entropy declining rapidly - monitor for collapse")

    # Action distribution
    if 'n_hold_recent_mean' in metrics:
        total = sum(metrics.get(f'n_{a}_recent_mean', 0) for a in ['buy', 'hold', 'sell', 'limit_buy', 'limit_sell'])
        if total > 0:
            hold_pct = metrics['n_hold_recent_mean'] / total * 100
            if hold_pct > 70:
                warnings.append(f"High HOLD rate ({hold_pct:.0f}%) - may be too conservative")

            sell_pct = metrics.get('n_sell_recent_mean', 0) / total * 100
            if sell_pct < 5:
                warnings.append("Very low sell rate - agent not learning to exit positions")

    # Loss components
    if 'total_loss' in df.columns and 'value_loss' in df.columns:
        recent_total = df['total_loss'].iloc[-10:].mean() if len(df) >= 10 else df['total_loss'].mean()
        recent_value = df['value_loss'].iloc[-10:].mean() if len(df) >= 10 else df['value_loss'].mean()

        if recent_total > 0 and recent_value / recent_total > 0.8:
            warnings.append("Value loss dominates total loss - consider rebalancing loss weights")

    # Market baseline (alpha)
    if 'alpha' in df.columns:
        alpha = df['alpha'].dropna()
        if len(alpha) > 10:
            recent_alpha = alpha.iloc[-10:].mean()
            overall_alpha = alpha.mean()

            if recent_alpha > 1.0:
                insights.append(f"Positive alpha ({recent_alpha:+.2f}%) - agent beating market")
            elif recent_alpha < -1.0:
                warnings.append(f"Negative alpha ({recent_alpha:+.2f}%) - agent underperforming market")

            positive_rate = (alpha > 0).sum() / len(alpha) * 100
            if positive_rate > 60:
                insights.append(f"Consistent outperformance: {positive_rate:.0f}% of epochs have positive alpha")
            elif positive_rate < 40:
                warnings.append(f"Frequent underperformance: only {positive_rate:.0f}% of epochs beat market")

    return insights, warnings


# =============================================================================
# API Endpoints - Physics Model
# =============================================================================

@app.route('/api/physics/<path:csv_path>')
def get_physics_data(csv_path: str):
    """Get physics component data for visualization."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    # Extract physics columns
    physics_data = {
        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df))),
        'components': {}
    }

    # G components
    for component in ['G_momentum', 'G_reversion', 'G_volatility', 'G_correlation']:
        mean_col = f'{component}_mean'
        std_col = f'{component}_std'

        if mean_col in df.columns:
            physics_data['components'][component] = {
                'mean': safe_json(df[mean_col].tolist()),
                'std': safe_json(df[std_col].tolist()) if std_col in df.columns else None
            }

    # Learned parameters
    for param in ['physics_alpha_mean', 'physics_beta_mean', 'garch_persistence']:
        if param in df.columns:
            physics_data[param] = safe_json(df[param].tolist())

    return jsonify(physics_data)


# =============================================================================
# API Endpoints - TRM Dynamics
# =============================================================================

@app.route('/api/trm/<path:csv_path>')
def get_trm_data(csv_path: str):
    """Get TRM cycle dynamics data."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    trm_data = {
        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df))),
        'latent_deltas': {},
        'output_deltas': {},
        'summary': {}
    }

    # Extract cycle data
    for col in df.columns:
        if col.startswith('latent_delta_norm_h'):
            cycle = col.replace('latent_delta_norm_', '')
            trm_data['latent_deltas'][cycle] = safe_json(df[col].tolist())
        elif col.startswith('output_delta_norm_h'):
            cycle = col.replace('output_delta_norm_', '')
            trm_data['output_deltas'][cycle] = safe_json(df[col].tolist())

    # Summary metrics
    if 'latent_final_norm' in df.columns:
        trm_data['summary']['latent_final_norm'] = safe_json(df['latent_final_norm'].tolist())

    if 'trm_utilization' in df.columns:
        trm_data['summary']['utilization'] = safe_json(df['trm_utilization'].tolist())

    return jsonify(trm_data)


# =============================================================================
# API Endpoints - Planning
# =============================================================================

@app.route('/api/planning/<path:csv_path>')
def get_planning_data(csv_path: str):
    """Get MCTS planning metrics."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    planning_data = {
        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df))),
        'horizon_errors': {},
        'mcts_metrics': {}
    }

    # Horizon errors
    for horizon in [1, 2, 5, 10]:
        col = f'value_horizon_{horizon}_error'
        if col in df.columns:
            planning_data['horizon_errors'][f'h{horizon}'] = safe_json(df[col].tolist())

    # MCTS metrics
    for metric in ['mcts_root_value', 'mcts_visit_entropy']:
        if metric in df.columns:
            planning_data['mcts_metrics'][metric] = safe_json(df[metric].tolist())

    return jsonify(planning_data)


# =============================================================================
# API Endpoints - Market Baseline (Alpha)
# =============================================================================

@app.route('/api/market-baseline/<path:csv_path>')
def get_market_baseline(csv_path: str):
    """Get market baseline metrics (alpha, market_return, agent_return)."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    baseline_data = {
        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df))),
        'alpha': safe_json(df['alpha'].tolist()) if 'alpha' in df.columns else [],
        'market_return': safe_json(df['market_return'].tolist()) if 'market_return' in df.columns else [],
        'agent_return': safe_json(df['agent_return'].tolist()) if 'agent_return' in df.columns else [],
        'summary': {}
    }

    # Compute summary statistics
    if 'alpha' in df.columns and len(df['alpha'].dropna()) > 0:
        alpha = df['alpha'].dropna()
        baseline_data['summary']['alpha_current'] = float(alpha.iloc[-1]) if len(alpha) > 0 else None
        baseline_data['summary']['alpha_mean'] = float(alpha.mean())
        baseline_data['summary']['alpha_std'] = float(alpha.std())
        baseline_data['summary']['alpha_max'] = float(alpha.max())
        baseline_data['summary']['alpha_min'] = float(alpha.min())

        # Positive alpha rate (% of epochs beating market)
        positive_alpha = (alpha > 0).sum()
        baseline_data['summary']['positive_alpha_rate'] = float(positive_alpha / len(alpha) * 100)

    if 'market_return' in df.columns and len(df['market_return'].dropna()) > 0:
        mkt = df['market_return'].dropna()
        baseline_data['summary']['market_return_current'] = float(mkt.iloc[-1]) if len(mkt) > 0 else None
        baseline_data['summary']['market_return_mean'] = float(mkt.mean())

    if 'agent_return' in df.columns and len(df['agent_return'].dropna()) > 0:
        agent = df['agent_return'].dropna()
        baseline_data['summary']['agent_return_current'] = float(agent.iloc[-1]) if len(agent) > 0 else None
        baseline_data['summary']['agent_return_mean'] = float(agent.mean())

    return jsonify(baseline_data)


# =============================================================================
# API Endpoints - Tensors (Periodic Snapshots)
# =============================================================================

@app.route('/api/tensors/<run_name>')
def get_tensor_snapshots(run_name: str):
    """List available tensor snapshots for a run."""
    run_dir = TENSORS_DIR / run_name

    if not run_dir.exists():
        return jsonify({'snapshots': [], 'message': 'No tensor snapshots available'})

    snapshots = []
    for snap_path in sorted(run_dir.glob('epoch_*.npz')):
        epoch = int(snap_path.stem.split('_')[1])
        snapshots.append({
            'epoch': epoch,
            'path': str(snap_path.relative_to(PROJECT_ROOT)),
            'size': snap_path.stat().st_size
        })

    return jsonify({'snapshots': snapshots})


@app.route('/api/tensor/<path:tensor_path>')
def get_tensor_data(tensor_path: str):
    """Load and return tensor data (with dimensionality reduction if needed)."""
    full_path = PROJECT_ROOT / tensor_path

    if not full_path.exists():
        return jsonify({'error': 'Tensor file not found'}), 404

    try:
        data = np.load(str(full_path), allow_pickle=True)

        result = {}
        for key in data.files:
            arr = data[key]

            # Apply PCA for high-dimensional data
            if arr.ndim >= 2 and arr.shape[-1] > 3:
                from sklearn.decomposition import PCA
                flat = arr.reshape(-1, arr.shape[-1])
                if len(flat) > 3:
                    pca = PCA(n_components=min(3, flat.shape[1]))
                    reduced = pca.fit_transform(flat)
                    result[key] = {
                        'data': safe_json(reduced.tolist()),
                        'explained_variance': safe_json(pca.explained_variance_ratio_.tolist()),
                        'original_shape': list(arr.shape)
                    }
                else:
                    result[key] = {'data': safe_json(arr.tolist()), 'original_shape': list(arr.shape)}
            else:
                result[key] = {'data': safe_json(arr.tolist()), 'original_shape': list(arr.shape)}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Checkpoints
# =============================================================================

@app.route('/api/checkpoints')
def get_checkpoints():
    """List available checkpoints."""
    checkpoints = []

    if CHECKPOINTS_DIR.exists():
        for exp_dir in CHECKPOINTS_DIR.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                for ckpt_dir in exp_dir.glob('checkpoint_*'):
                    if ckpt_dir.is_dir():
                        step = ckpt_dir.name.replace('checkpoint_', '')
                        size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())

                        checkpoints.append({
                            'experiment': exp_dir.name,
                            'step': int(step) if step.isdigit() else 0,
                            'path': str(ckpt_dir.relative_to(PROJECT_ROOT)),
                            'size_mb': round(size / 1024 / 1024, 2),
                            'modified': datetime.fromtimestamp(ckpt_dir.stat().st_mtime).isoformat()
                        })

        checkpoints.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify(checkpoints[:50])


# =============================================================================
# API Endpoints - System Status
# =============================================================================

@app.route('/api/status')
def get_status():
    """Get system status (GPU, training processes)."""
    status = {
        'gpu': None,
        'training_active': False,
        'processes': []
    }

    # GPU status via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 4:
                status['gpu'] = {
                    'utilization': int(parts[0]),
                    'memory_used': int(parts[1]),
                    'memory_total': int(parts[2]),
                    'temperature': int(parts[3])
                }
    except Exception:
        pass

    # Training processes
    try:
        result = subprocess.run(
            ['pgrep', '-af', 'python.*train'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        status['processes'].append({
                            'pid': parts[0],
                            'command': parts[1][:100]
                        })
            status['training_active'] = True
    except Exception:
        pass

    # Check GPU utilization as backup
    if not status['training_active'] and status['gpu'] and status['gpu']['utilization'] > 50:
        status['training_active'] = True

    return jsonify(status)


# =============================================================================
# API Endpoints - Logs
# =============================================================================

@app.route('/api/log-files')
def get_log_files():
    """List available log files."""
    logs = []

    if LOGS_DIR.exists():
        for pattern in ['*.log', '*.txt']:
            for log_path in LOGS_DIR.glob(pattern):
                logs.append({
                    'path': str(log_path.relative_to(PROJECT_ROOT)),
                    'name': log_path.name,
                    'size': log_path.stat().st_size,
                    'modified': datetime.fromtimestamp(log_path.stat().st_mtime).isoformat()
                })

        logs.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify(logs[:30])


@app.route('/api/logs/<path:log_path>')
def get_log_content(log_path: str):
    """Get tail of log file."""
    full_path = PROJECT_ROOT / log_path
    lines = request.args.get('lines', 200, type=int)

    if not full_path.exists():
        return jsonify({'error': 'Log file not found'}), 404

    try:
        result = subprocess.run(
            ['tail', '-n', str(lines), str(full_path)],
            capture_output=True, text=True, timeout=10
        )
        return jsonify({'content': result.stdout})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# API Endpoints - Loss Landscape (Advanced)
# =============================================================================

@app.route('/api/loss-components/<path:csv_path>')
def get_loss_components(csv_path: str):
    """Get normalized loss component breakdown."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    df = read_csv_flexible(str(full_path))

    loss_data = {
        'epochs': df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df))),
        'components': {},
        'normalized': {}
    }

    # Raw loss values
    loss_cols = ['value_loss', 'policy_loss', 'reward_loss', 'pred_loss', 'physics_reg']
    available = [c for c in loss_cols if c in df.columns]

    for col in available:
        loss_data['components'][col] = safe_json(df[col].tolist())

    # Normalized (percentage of total)
    if 'total_loss' in df.columns and available:
        total = df['total_loss'].values
        total = np.where(total > 0, total, 1)  # Avoid division by zero

        for col in available:
            normalized = df[col].values / total * 100
            loss_data['normalized'][col] = safe_json(normalized.tolist())

    return jsonify(loss_data)


# =============================================================================
# Benchmark Analytics API
# =============================================================================

@app.route('/api/benchmarks')
def get_benchmarks():
    """Get benchmark data from summary files, with metrics from training CSVs."""
    import glob
    import re
    from flask import make_response
    import pandas as pd
    import csv

    benchmark_dir = PROJECT_ROOT / 'logs' / 'benchmarks'
    logs_dir = PROJECT_ROOT / 'logs'
    all_periods = ['bull', 'bear', 'sideways', 'ftx_crash', 'etf_approval', 'boring_week']

    # Find summary files for model names: logs/benchmarks/*_benchmark.csv
    summary_files = sorted(glob.glob(str(benchmark_dir / '*_benchmark.csv'))) if benchmark_dir.exists() else []

    benchmarks = {}

    for csv_path in summary_files:
        filename = os.path.basename(csv_path)
        # Parse: poppy_v2_L05_benchmark.csv or poppy_v2_H04_benchmark.csv
        parts = filename.replace('_benchmark.csv', '').split('_')
        if len(parts) >= 3:
            model_name = '_'.join(parts[:-1])  # poppy_v2
            level_str = parts[-1]  # L05 or H04
            level_prefix = level_str[0]

            try:
                level_num = int(level_str[1:])
                level = 100 + level_num if level_prefix == 'H' else level_num
            except ValueError:
                continue

            if model_name not in benchmarks:
                benchmarks[model_name] = {
                    'levels': [],
                    'bull': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'bear': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'sideways': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'ftx_crash': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'etf_approval': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'boring_week': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                    'avg': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                }

            benchmarks[model_name]['levels'].append(level)

            # Read summary file first
            period_data = {}
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        period = row.get('period', '').lower()
                        if period in all_periods:
                            hold_pct = float(row.get('pct_hold', 0) or row.get('hold_pct', 0) or 0)
                            loss = float(row.get('total_loss', 0) or 0)
                            period_data[period] = {
                                'alpha': float(row.get('alpha', 0) or 0),
                                'reward': float(row.get('reward', 0) or 0),
                                'hold_pct': hold_pct,
                                'entropy': float(row.get('entropy', 0) or 0),
                                'loss': loss,
                            }
            except Exception as e:
                logging.warning(f"Error reading summary {csv_path}: {e}")

            # If metrics incomplete, try training CSVs
            # Pattern: training_{model_base}_benchmark_{level}_{period}.csv
            model_base = model_name.rsplit('_', 1)[0] if '_v' in model_name else model_name
            for period in all_periods:
                if period not in period_data or (period_data[period]['hold_pct'] == 0 and period_data[period]['loss'] == 0):
                    training_csv = logs_dir / f'training_{model_base}_benchmark_{level_str}_{period}.csv'
                    if training_csv.exists():
                        try:
                            df = pd.read_csv(training_csv, on_bad_lines='skip')
                            if len(df) > 0:
                                last_rows = df.tail(min(7, len(df)))
                                period_data[period] = {
                                    'alpha': last_rows['alpha'].mean() if 'alpha' in df.columns else period_data.get(period, {}).get('alpha', 0),
                                    'reward': last_rows['reward'].mean() if 'reward' in df.columns else period_data.get(period, {}).get('reward', 0),
                                    'hold_pct': last_rows['pct_hold'].mean() if 'pct_hold' in df.columns else 0,
                                    'entropy': last_rows['policy_entropy'].mean() if 'policy_entropy' in df.columns else 0,
                                    'loss': last_rows['total_loss'].mean() if 'total_loss' in df.columns else 0,
                                }
                        except Exception as e:
                            logging.warning(f"Error reading training CSV {training_csv}: {e}")

        # Append data for each period
        for period in all_periods:
            if period in period_data:
                for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
                    benchmarks[model_name][period][metric].append(period_data[period][metric])
            else:
                for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
                    benchmarks[model_name][period][metric].append(None)

        # Compute averages across all available periods
        for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
            values = [period_data[p][metric] for p in all_periods if p in period_data]
            avg_val = sum(values) / len(values) if values else 0
            benchmarks[model_name]['avg'][metric].append(avg_val)

    # Sort levels and corresponding data
    for model_name in benchmarks:
        levels = benchmarks[model_name]['levels']
        if not levels:
            continue

        # Get sort indices
        sorted_indices = sorted(range(len(levels)), key=lambda i: levels[i])

        # Sort levels
        benchmarks[model_name]['levels'] = [levels[i] for i in sorted_indices]

        # Sort all period data
        for period in ['bull', 'bear', 'sideways', 'ftx_crash', 'etf_approval', 'boring_week', 'avg']:
            for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
                data = benchmarks[model_name][period][metric]
                benchmarks[model_name][period][metric] = [data[i] for i in sorted_indices]

    # =========================================================================
    # DOJO V4 FORMAT: Also read from benchmark_results.csv (new unified format)
    # CSV columns: timestamp, model, version, phase, period, reward, alpha, hold_pct, entropy, avg_reward, passed
    # =========================================================================
    dojo_csv = benchmark_dir / 'benchmark_results.csv'
    if dojo_csv.exists():
        try:
            df = pd.read_csv(dojo_csv)
            for _, row in df.iterrows():
                model_name = f"{row.get('model', 'unknown')}_v{row.get('version', '0')}"
                phase = str(row.get('phase', 'unknown'))
                period = str(row.get('period', '')).lower()

                # Determine level from phase name
                # Phase names like: phase1, phase2_r0, phase2_r1, phase3_H01, etc.
                if 'phase1' in phase:
                    level = -1  # Before L0
                elif 'phase2' in phase:
                    # Extract retry count if present
                    level = -2 if '_r0' in phase else -2 - int(phase.split('_r')[-1]) if '_r' in phase else -2
                elif 'H' in phase:
                    # Historical blocks
                    level = 100 + int(phase.split('H')[-1].split('_')[0]) if 'H' in phase else 200
                else:
                    level = 200  # Unknown

                if model_name not in benchmarks:
                    benchmarks[model_name] = {
                        'levels': [],
                        'bull': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'bear': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'sideways': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'ftx_crash': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'etf_approval': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'boring_week': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'avg': {'alpha': [], 'reward': [], 'hold_pct': [], 'entropy': [], 'loss': []},
                        'phases': [],  # New: track phase names
                    }

                # Check if this level/phase already exists
                if level not in benchmarks[model_name]['levels']:
                    benchmarks[model_name]['levels'].append(level)
                    benchmarks[model_name].setdefault('phases', []).append(phase)
                    # Initialize empty slots for all periods
                    for p in all_periods:
                        for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
                            benchmarks[model_name][p][metric].append(None)
                    for metric in ['alpha', 'reward', 'hold_pct', 'entropy', 'loss']:
                        benchmarks[model_name]['avg'][metric].append(None)

                # Find index for this level
                level_idx = benchmarks[model_name]['levels'].index(level)

                # Update metrics for this period
                if period in all_periods:
                    benchmarks[model_name][period]['reward'][level_idx] = float(row.get('reward', 0) or 0)
                    benchmarks[model_name][period]['alpha'][level_idx] = float(row.get('alpha', 0) or 0)
                    benchmarks[model_name][period]['hold_pct'][level_idx] = float(row.get('hold_pct', 0) or 0)
                    benchmarks[model_name][period]['entropy'][level_idx] = float(row.get('entropy', 0) or 0)

                # Update average from avg_reward column
                avg_reward = float(row.get('avg_reward', 0) or 0)
                benchmarks[model_name]['avg']['reward'][level_idx] = avg_reward

            # Compute avg metrics from period data for Dojo V4 entries
            for model_name, data in benchmarks.items():
                for level_idx in range(len(data['levels'])):
                    # Compute averages across periods for this level
                    for metric in ['alpha', 'hold_pct', 'entropy']:
                        values = []
                        for period in all_periods:
                            val = data[period][metric][level_idx] if level_idx < len(data[period][metric]) else None
                            if val is not None:
                                values.append(val)
                        if values:
                            data['avg'][metric][level_idx] = sum(values) / len(values)

        except Exception as e:
            logging.warning(f"Error reading Dojo V4 benchmark CSV {dojo_csv}: {e}")

    response = make_response(jsonify(benchmarks))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# =============================================================================
# Static Files
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DS24 Enhanced Training Dashboard')
    parser.add_argument('--port', type=int, default=8051, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║  DS24 - Enhanced Training Metrics Dashboard               ║
    ║                                                           ║
    ║  TRM+MuZero Trading Agent Visualization                   ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  - MuZero metrics (value/policy/reward prediction)        ║
    ║  - TRM recursive reasoning dynamics                       ║
    ║  - Neural ODE physics components                          ║
    ║  - Multi-component interaction analysis                   ║
    ╚═══════════════════════════════════════════════════════════╝

    Starting server at http://{args.host}:{args.port}
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)
