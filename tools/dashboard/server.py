#!/usr/bin/env python3
"""
Training Dashboard Server

A Flask-based server that:
1. Serves the static dashboard files
2. Provides REST API for training metrics
3. Includes phase transition analysis
4. Monitors checkpoints and logs

Usage:
    python tools/dashboard/server.py
    # Then open http://localhost:8050
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
LOGS_DIR = PROJECT_ROOT / 'logs'
CHECKPOINTS_DIR = PROJECT_ROOT / 'jax_ckpt_fast'

# Directories to ignore
IGNORE_DIRS = {'_old', 'old', 'archive', 'backup'}


def should_ignore_path(path: Path) -> bool:
    """Check if path contains any ignored directory names."""
    return any(part in IGNORE_DIRS for part in path.parts)


# =============================================================================
# Static File Serving
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# =============================================================================
# API: List Available Training Runs
# =============================================================================

@app.route('/api/runs')
def list_runs():
    """List all available training CSV files (excluding _old directories)."""
    runs = []

    # Scan logs directory
    for csv_path in sorted(LOGS_DIR.glob('**/*.csv'), key=os.path.getmtime, reverse=True):
        # Skip files in ignored directories
        if should_ignore_path(csv_path.relative_to(LOGS_DIR)):
            continue
        if 'training_' in csv_path.name:
            stat = csv_path.stat()
            runs.append({
                'path': str(csv_path.relative_to(PROJECT_ROOT)),
                'name': csv_path.stem.replace('training_', ''),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    return jsonify({'runs': runs})


# =============================================================================
# API: Get Training Data
# =============================================================================

@app.route('/api/training/<path:csv_path>')
def get_training_data(csv_path):
    """Get training data from a CSV file."""
    import json

    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        df = read_csv_flexible(full_path)

        # Convert to JSON-safe format (handle NaN, Inf)
        data = json.loads(df.to_json(orient='records'))

        return jsonify({
            'data': data,
            'columns': list(df.columns),
            'count': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def read_csv_flexible(filepath: Path) -> pd.DataFrame:
    """Read CSV with flexible column handling for mid-training schema changes."""
    from io import StringIO

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("CSV file too short")

    # Parse header
    header = lines[0].strip().split(',')
    header_cols = len(header)

    # Find all column counts
    col_counts = {}
    for line in lines[1:]:
        if line.strip():
            cols = len(line.strip().split(','))
            col_counts[cols] = col_counts.get(cols, 0) + 1

    if not col_counts:
        raise ValueError("No data rows found")

    # Target is the header column count (trust the header)
    target_cols = header_cols

    # Find where value_scale/value_shift should be inserted for short rows
    # They go after value_std (index 21, so insert at position 22)
    # New columns (n_limit_sell, n_limit_buy, pct_limit_sell, pct_limit_buy) go after pct_sell
    try:
        pct_sell_idx = header.index('pct_sell')
        insert_pos = pct_sell_idx + 1
    except ValueError:
        try:
            value_std_idx = header.index('value_std')
            insert_pos = value_std_idx + 1 # Fallback for older CSVs if pct_sell not found
        except ValueError:
            insert_pos = header_cols - 2  # Fallback: before last 2 cols


    # Build normalized CSV
    filtered_lines = [','.join(header) + '\n']
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split(',')
            cols = len(parts)

            if cols == target_cols:
                # Exact match - use as-is
                filtered_lines.append(line if line.endswith('\n') else line + '\n')
            elif cols < target_cols:
                # Short row - pad with empty values at insert position
                missing = target_cols - cols
                padded = parts[:insert_pos] + [''] * missing + parts[insert_pos:]
                filtered_lines.append(','.join(padded) + '\n')
            else:
                # Extra columns - truncate (shouldn't happen normally)
                filtered_lines.append(','.join(parts[:target_cols]) + '\n')

    # Parse the normalized CSV
    df = pd.read_csv(StringIO(''.join(filtered_lines)))

    return df


# =============================================================================
# API: Phase Transition Analysis
# =============================================================================

@app.route('/api/analysis/<path:csv_path>')
def analyze_training(csv_path):
    """Analyze training data for phase transitions and insights."""
    full_path = PROJECT_ROOT / csv_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    try:
        df = read_csv_flexible(full_path)
        analysis = perform_phase_analysis(df)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def perform_phase_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze training data for phase transitions (from visualize_mcts_tree.py)."""
    analysis = {
        'insights': [],
        'warnings': [],
        'metrics': {},
        'phases': []
    }

    if len(df) < 2:
        return analysis

    # Basic metrics
    analysis['metrics']['total_epochs'] = int(len(df))
    analysis['metrics']['total_steps'] = int(df['steps'].sum()) if 'steps' in df.columns else 0

    # Reward analysis
    if 'reward' in df.columns:
        rewards = df['reward']
        analysis['metrics']['avg_reward'] = float(rewards.mean())
        analysis['metrics']['max_reward'] = float(rewards.max())
        analysis['metrics']['min_reward'] = float(rewards.min())
        analysis['metrics']['final_reward_ma50'] = float(rewards.tail(50).mean()) if len(df) >= 50 else float(rewards.mean())

        # Trend analysis
        if len(df) >= 20:
            early_reward = rewards.head(10).mean()
            recent_reward = rewards.tail(10).mean()
            reward_trend = ((recent_reward - early_reward) / abs(early_reward) * 100) if early_reward != 0 else 0
            analysis['metrics']['reward_trend_pct'] = float(reward_trend)

            if reward_trend > 20:
                analysis['insights'].append(f"Strong improvement: reward up {reward_trend:.1f}% from start")
            elif reward_trend < -20:
                analysis['warnings'].append(f"Regression detected: reward down {abs(reward_trend):.1f}% from start")

    # Entropy analysis (exploration vs exploitation)
    if 'policy_entropy' in df.columns:
        entropy = df['policy_entropy']
        analysis['metrics']['avg_entropy'] = float(entropy.mean())
        analysis['metrics']['final_entropy'] = float(entropy.iloc[-1])

        if len(df) >= 10:
            early_entropy = entropy.head(10).mean()
            recent_entropy = entropy.tail(10).mean()
            entropy_change = ((recent_entropy - early_entropy) / early_entropy * 100) if early_entropy != 0 else 0
            analysis['metrics']['entropy_change_pct'] = float(entropy_change)

            if entropy_change < -30:
                analysis['warnings'].append("Policy becoming deterministic (entropy collapsed >30%)")
            elif entropy_change > 30:
                analysis['insights'].append("Policy exploring more (entropy increased >30%)")

            # Entropy classification
            if recent_entropy > 4.0:
                analysis['metrics']['exploration_state'] = 'High Exploration'
            elif recent_entropy > 2.5:
                analysis['metrics']['exploration_state'] = 'Balanced'
            elif recent_entropy > 1.5:
                analysis['metrics']['exploration_state'] = 'Focused'
            else:
                analysis['metrics']['exploration_state'] = 'Deterministic'

    # Value-Reward correlation (calibration)
    if 'avg_value' in df.columns and 'reward' in df.columns:
        valid_mask = ~(df['avg_value'].isna() | df['reward'].isna())
        if valid_mask.sum() > 5:
            corr = df.loc[valid_mask, 'reward'].corr(df.loc[valid_mask, 'avg_value'])
            analysis['metrics']['value_reward_correlation'] = float(corr)

            if corr < 0.3:
                analysis['warnings'].append(f"Poor value calibration (corr={corr:.2f}) - value head struggling")
            elif corr > 0.7:
                analysis['insights'].append(f"Good value calibration (corr={corr:.2f}) - value head learning well")

    # Action distribution analysis
    # Dynamically check for 5-action space
    has_limit_actions = all(col in df.columns for col in ['n_limit_buy', 'n_limit_sell'])

    if all(col in df.columns for col in ['n_buy', 'n_hold', 'n_sell']) or has_limit_actions:
        n_buy_col = 'n_buy'
        n_hold_col = 'n_hold'
        n_sell_col = 'n_sell'
        n_limit_buy_col = 'n_limit_buy'
        n_limit_sell_col = 'n_limit_sell'

        if has_limit_actions:
            # New 5-action space
            total_actions = df[n_buy_col] + df[n_hold_col] + df[n_sell_col] + df[n_limit_buy_col] + df[n_limit_sell_col]
        else:
            # Old 3-action space
            total_actions = df[n_buy_col] + df[n_hold_col] + df[n_sell_col]

        if len(df) >= 10:
            recent = df.tail(10)
            
            if has_limit_actions:
                recent_total = recent[n_buy_col] + recent[n_hold_col] + recent[n_sell_col] + recent[n_limit_buy_col] + recent[n_limit_sell_col]
            else:
                recent_total = recent[n_buy_col] + recent[n_hold_col] + recent[n_sell_col]

            buy_pct = (recent[n_buy_col].sum() / recent_total.sum() * 100) if recent_total.sum() > 0 else 0
            hold_pct = (recent[n_hold_col].sum() / recent_total.sum() * 100) if recent_total.sum() > 0 else 0
            sell_pct = (recent[n_sell_col].sum() / recent_total.sum() * 100) if recent_total.sum() > 0 else 0
            limit_buy_pct = (recent[n_limit_buy_col].sum() / recent_total.sum() * 100) if has_limit_actions and recent_total.sum() > 0 else 0
            limit_sell_pct = (recent[n_limit_sell_col].sum() / recent_total.sum() * 100) if has_limit_actions and recent_total.sum() > 0 else 0

            analysis['metrics']['recent_buy_pct'] = float(buy_pct)
            analysis['metrics']['recent_hold_pct'] = float(hold_pct)
            analysis['metrics']['recent_sell_pct'] = float(sell_pct)
            if has_limit_actions:
                analysis['metrics']['recent_limit_buy_pct'] = float(limit_buy_pct)
                analysis['metrics']['recent_limit_sell_pct'] = float(limit_sell_pct)

            if sell_pct < 10:
                analysis['warnings'].append(f"Low market sell rate ({sell_pct:.1f}%) - may not be learning to exit positions")
            if buy_pct > 60:
                analysis['warnings'].append(f"High market buy rate ({buy_pct:.1f}%) - may be over-accumulating")
            if has_limit_actions and limit_sell_pct + limit_buy_pct < 5:
                analysis['warnings'].append(f"Low limit order usage ({limit_buy_pct+limit_sell_pct:.1f}%)")

        # Detect when sells first appear (logic remains similar, but consider all sell types)
        if (df[n_sell_col] == 0).all() and (has_limit_actions and df[n_limit_sell_col] == 0).all():
            analysis['insights'].append("No sell actions yet.")
        else:
            first_sell_idx = None
            if (df[n_sell_col] > 0).any():
                first_sell_idx = df[df[n_sell_col] > 0].index.min()
            if has_limit_actions and (df[n_limit_sell_col] > 0).any():
                if first_sell_idx is None:
                    first_sell_idx = df[df[n_limit_sell_col] > 0].index.min()
                else:
                    first_sell_idx = min(first_sell_idx, df[df[n_limit_sell_col] > 0].index.min())

            if first_sell_idx is not None and first_sell_idx > 0:
                first_sell_epoch = df.loc[first_sell_idx, 'epoch'] if 'epoch' in df.columns else first_sell_idx
                analysis['insights'].append(f"First sell actions (market or limit) appeared at epoch {first_sell_epoch}")

    # Loss analysis
    if 'total_loss' in df.columns:
        losses = df['total_loss']
        analysis['metrics']['final_loss'] = float(losses.iloc[-1])
        analysis['metrics']['min_loss'] = float(losses.min())

        if len(df) >= 10:
            recent_loss = losses.tail(10).mean()
            early_loss = losses.head(10).mean()
            if early_loss > 0:
                loss_reduction = (1 - recent_loss / early_loss) * 100
                analysis['metrics']['loss_reduction_pct'] = float(loss_reduction)

    # Value MSE analysis (calibration quality)
    if 'value_mse' in df.columns:
        value_mse = df['value_mse'].dropna()
        if len(value_mse) > 0:
            analysis['metrics']['final_value_mse'] = float(value_mse.iloc[-1])
            analysis['metrics']['avg_value_mse'] = float(value_mse.mean())

            if len(value_mse) >= 10:
                early_mse = value_mse.head(10).mean()
                recent_mse = value_mse.tail(10).mean()

                if early_mse > 0:
                    mse_reduction = (1 - recent_mse / early_mse) * 100
                    analysis['metrics']['mse_reduction_pct'] = float(mse_reduction)

                    if mse_reduction > 30:
                        analysis['insights'].append(f"Value MSE dropped {mse_reduction:.1f}% - value head learning well")
                    elif mse_reduction < -30:
                        analysis['warnings'].append(f"Value MSE increased {abs(mse_reduction):.1f}% - calibration degrading")

            # Good MSE thresholds
            if value_mse.iloc[-1] < 3:
                analysis['insights'].append(f"Excellent value calibration (MSE={value_mse.iloc[-1]:.2f})")
            elif value_mse.iloc[-1] > 10:
                analysis['warnings'].append(f"High value MSE ({value_mse.iloc[-1]:.2f}) - value head struggling")

    # Value Std analysis (prediction diversity)
    if 'value_std' in df.columns:
        value_std = df['value_std'].dropna()
        if len(value_std) > 0:
            analysis['metrics']['final_value_std'] = float(value_std.iloc[-1])

            if value_std.iloc[-1] < 0.5:
                analysis['warnings'].append(f"Low value std ({value_std.iloc[-1]:.2f}) - predictions too uniform")
            elif value_std.iloc[-1] > 1.0:
                analysis['insights'].append(f"Good value diversity (std={value_std.iloc[-1]:.2f})")

    # Value calibration scale/shift analysis (adaptive calibration)
    if 'value_scale' in df.columns:
        value_scale = df['value_scale'].dropna()
        if len(value_scale) > 0:
            analysis['metrics']['final_value_scale'] = float(value_scale.iloc[-1])

            if abs(value_scale.iloc[-1] - 1.0) > 0.3:
                analysis['insights'].append(f"Value scale adjusted to {value_scale.iloc[-1]:.2f} (model {'too conservative' if value_scale.iloc[-1] > 1 else 'too aggressive'})")

    if 'value_shift' in df.columns:
        value_shift = df['value_shift'].dropna()
        if len(value_shift) > 0:
            analysis['metrics']['final_value_shift'] = float(value_shift.iloc[-1])

            if abs(value_shift.iloc[-1]) > 1.0:
                analysis['insights'].append(f"Value shift adjusted to {value_shift.iloc[-1]:.2f} (centering predictions)")

    # Action percentage analysis (use pct_* columns if available)
    if all(col in df.columns for col in ['pct_buy', 'pct_hold', 'pct_sell']):
        if len(df) >= 10:
            recent = df.tail(10)
            if has_limit_actions and all(col in df.columns for col in ['pct_limit_buy', 'pct_limit_sell']):
                # 5-action space analysis
                avg_buy = recent['pct_buy'].mean()
                avg_hold = recent['pct_hold'].mean()
                avg_sell = recent['pct_sell'].mean()
                avg_limit_buy = recent['pct_limit_buy'].mean()
                avg_limit_sell = recent['pct_limit_sell'].mean()

                analysis['metrics']['recent_pct_buy'] = float(avg_buy)
                analysis['metrics']['recent_pct_hold'] = float(avg_hold)
                analysis['metrics']['recent_pct_sell'] = float(avg_sell)
                analysis['metrics']['recent_pct_limit_buy'] = float(avg_limit_buy)
                analysis['metrics']['recent_pct_limit_sell'] = float(avg_limit_sell)

                total_sell_pct = avg_sell + avg_limit_sell
                if total_sell_pct < 10:
                    analysis['warnings'].append(f"Very low total sell % ({total_sell_pct:.1f}%) - not learning to exit")
                elif total_sell_pct < 20:
                    analysis['warnings'].append(f"Low total sell % ({total_sell_pct:.1f}%) - may struggle to exit positions")

                if avg_hold > 60:
                    analysis['warnings'].append(f"High hold % ({avg_hold:.1f}%) - overly conservative policy")

                all_pcts = [avg_buy, avg_hold, avg_sell, avg_limit_buy, avg_limit_sell]
                balance = max(all_pcts) - min(all_pcts)
                if balance < 20: # Wider tolerance for 5 actions
                    analysis['insights'].append("Well-balanced action distribution")
                elif max(all_pcts) > 50:
                    action_names = ['Market Buy', 'Hold', 'Market Sell', 'Limit Buy', 'Limit Sell']
                    dominant_action = action_names[np.argmax(all_pcts)]
                    analysis['insights'].append(f"Policy strongly prefers {dominant_action} actions")

            else:
                # Original 3-action space logic
                avg_buy = recent['pct_buy'].mean()
                avg_hold = recent['pct_hold'].mean()
                avg_sell = recent['pct_sell'].mean()

                analysis['metrics']['recent_pct_buy'] = float(avg_buy)
                analysis['metrics']['recent_pct_hold'] = float(avg_hold)
                analysis['metrics']['recent_pct_sell'] = float(avg_sell)

                if avg_sell < 10:
                    analysis['warnings'].append(f"Very low sell % ({avg_sell:.1f}%) - not learning to exit")
                elif avg_sell < 20:
                    analysis['warnings'].append(f"Low sell % ({avg_sell:.1f}%) - may struggle to exit positions")

                if avg_hold > 60:
                    analysis['warnings'].append(f"High hold % ({avg_hold:.1f}%) - overly conservative policy")

                balance = max(avg_buy, avg_hold, avg_sell) - min(avg_buy, avg_hold, avg_sell)
                if balance < 15:
                    analysis['insights'].append("Well-balanced action distribution")
                elif balance > 40:
                    dominant = 'buy' if avg_buy > max(avg_hold, avg_sell) else ('hold' if avg_hold > avg_sell else 'sell')
                    analysis['insights'].append(f"Policy strongly prefers {dominant} actions")

    # Reward MA20 trend analysis
    if 'reward_ma20' in df.columns:
        ma20 = df['reward_ma20'].dropna()
        if len(ma20) >= 20:
            recent_ma20 = ma20.tail(10).mean()
            early_ma20 = ma20.head(10).mean()

            if early_ma20 != 0:
                ma20_trend = ((recent_ma20 - early_ma20) / abs(early_ma20)) * 100
                analysis['metrics']['ma20_trend_pct'] = float(ma20_trend)

    # Synthetic level and model size info
    if 'synthetic_level' in df.columns:
        syn_level = df['synthetic_level'].iloc[-1]
        analysis['metrics']['synthetic_level'] = int(syn_level) if not pd.isna(syn_level) else None
        level_names = {0: 'Sine Waves', 1: 'Multi-Frequency', 2: 'Trends', 3: 'Regime Changes'}
        if not pd.isna(syn_level):
            analysis['metrics']['curriculum_name'] = level_names.get(int(syn_level), f'Level {int(syn_level)}')

    if 'model_size' in df.columns:
        model_size = df['model_size'].iloc[-1]
        if not pd.isna(model_size):
            analysis['metrics']['model_size'] = str(model_size)

    # Detect phases (simple heuristic)
    if 'epoch' in df.columns and 'reward' in df.columns:
        window = max(5, len(df) // 10)
        rolling_reward = df['reward'].rolling(window=window, min_periods=1).mean()

        phases = []
        current_phase = {'start': 0, 'type': 'exploration'}

        for i in range(1, len(rolling_reward)):
            if i > window:
                trend = rolling_reward.iloc[i] - rolling_reward.iloc[i-window]

                if trend > 0.5 and current_phase['type'] != 'improving':
                    current_phase['end'] = i - 1
                    phases.append(current_phase)
                    current_phase = {'start': i, 'type': 'improving'}
                elif trend < -0.5 and current_phase['type'] != 'declining':
                    current_phase['end'] = i - 1
                    phases.append(current_phase)
                    current_phase = {'start': i, 'type': 'declining'}

        current_phase['end'] = len(df) - 1
        phases.append(current_phase)

        # Filter very short phases
        phases = [p for p in phases if p['end'] - p['start'] >= window // 2]
        analysis['phases'] = phases

    return analysis


# =============================================================================
# API: Checkpoint Information
# =============================================================================

@app.route('/api/checkpoints')
def list_checkpoints():
    """List all available checkpoints (excluding _old directories)."""
    checkpoints = []

    if not CHECKPOINTS_DIR.exists():
        return jsonify({'checkpoints': [], 'error': 'Checkpoint directory not found'})

    for exp_dir in CHECKPOINTS_DIR.iterdir():
        # Skip ignored directories
        if not exp_dir.is_dir() or exp_dir.name in IGNORE_DIRS or exp_dir.name.startswith('_'):
            continue

        for ckpt_dir in sorted(exp_dir.glob('checkpoint_*'), key=os.path.getmtime, reverse=True):
            if ckpt_dir.is_dir():
                stat = ckpt_dir.stat()
                step = ckpt_dir.name.replace('checkpoint_', '')

                # Calculate size
                total_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())

                checkpoints.append({
                    'experiment': exp_dir.name,
                    'step': int(step) if step.isdigit() else step,
                    'path': str(ckpt_dir.relative_to(PROJECT_ROOT)),
                    'size_mb': round(total_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    # Sort by modification time
    checkpoints.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({'checkpoints': checkpoints[:50]})  # Limit to 50 most recent


# =============================================================================
# API: Checkpoint Analysis
# =============================================================================

@app.route('/api/checkpoint-analysis/<path:ckpt_path>')
def analyze_checkpoint(ckpt_path):
    """Analyze a specific checkpoint - read metadata and parameter stats."""
    full_path = PROJECT_ROOT / ckpt_path

    if not full_path.exists():
        return jsonify({'error': 'Checkpoint not found'}), 404

    try:
        analysis = perform_checkpoint_analysis(full_path)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/checkpoint-comparison')
def compare_checkpoints():
    """Compare multiple checkpoints from the same experiment."""
    experiment = request.args.get('experiment')

    if not experiment:
        return jsonify({'error': 'experiment parameter required'}), 400

    exp_dir = CHECKPOINTS_DIR / experiment
    if not exp_dir.exists():
        return jsonify({'error': 'Experiment not found'}), 404

    try:
        comparison = perform_checkpoint_comparison(exp_dir)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def perform_checkpoint_analysis(ckpt_path: Path) -> Dict[str, Any]:
    """Analyze checkpoint structure and metadata."""
    analysis = {
        'path': str(ckpt_path),
        'step': None,
        'files': [],
        'total_size_mb': 0,
        'parameter_count': None,
        'metadata': {},
        'array_info': []
    }

    # Extract step number
    step_str = ckpt_path.name.replace('checkpoint_', '')
    if step_str.isdigit():
        analysis['step'] = int(step_str)

    # List files and calculate sizes
    total_size = 0
    for f in ckpt_path.rglob('*'):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            rel_path = str(f.relative_to(ckpt_path))
            analysis['files'].append({
                'name': rel_path,
                'size_kb': round(size / 1024, 2)
            })

    analysis['total_size_mb'] = round(total_size / 1024 / 1024, 2)

    # Try to read metadata files
    metadata_file = ckpt_path / '_METADATA'
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                content = f.read()
                # Parse if JSON-like
                if content.strip().startswith('{'):
                    analysis['metadata'] = json.loads(content)
                else:
                    analysis['metadata']['raw'] = content[:500]  # First 500 chars
        except:
            pass

    # Try to read checkpoint metadata
    ckpt_metadata_file = ckpt_path / '_CHECKPOINT_METADATA'
    if ckpt_metadata_file.exists():
        try:
            with open(ckpt_metadata_file, 'r') as f:
                content = f.read()
                if content.strip().startswith('{'):
                    analysis['checkpoint_metadata'] = json.loads(content)
        except:
            pass

    # Try to read array metadata (Orbax format)
    array_meta_dir = ckpt_path / 'array_metadatas'
    if array_meta_dir.exists():
        for meta_file in array_meta_dir.glob('*'):
            try:
                # These are typically msgpack or similar binary format
                # Just report file presence and size
                analysis['array_info'].append({
                    'process': meta_file.name,
                    'size_kb': round(meta_file.stat().st_size / 1024, 2)
                })
            except:
                pass

    # Estimate parameter count from file sizes
    # Orbax stores float32 by default (4 bytes per param)
    # Look for the main data files
    data_size = 0
    for f in ckpt_path.rglob('d/*'):
        if f.is_file():
            data_size += f.stat().st_size

    if data_size > 0:
        # Assume float32 (4 bytes) with ~30% overhead for metadata/compression
        estimated_params = int(data_size / 4 * 0.7)
        analysis['parameter_count'] = estimated_params
        analysis['parameter_count_formatted'] = format_param_count(estimated_params)

    return analysis


def perform_checkpoint_comparison(exp_dir: Path) -> Dict[str, Any]:
    """Compare all checkpoints in an experiment directory."""
    comparison = {
        'experiment': exp_dir.name,
        'checkpoints': [],
        'progression': {
            'steps': [],
            'sizes_mb': [],
            'timestamps': []
        }
    }

    # Get all checkpoints sorted by step
    ckpt_dirs = []
    for ckpt_dir in exp_dir.glob('checkpoint_*'):
        if ckpt_dir.is_dir():
            step_str = ckpt_dir.name.replace('checkpoint_', '')
            if step_str.isdigit():
                ckpt_dirs.append((int(step_str), ckpt_dir))

    ckpt_dirs.sort(key=lambda x: x[0])

    for step, ckpt_dir in ckpt_dirs:
        stat = ckpt_dir.stat()
        total_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
        size_mb = round(total_size / 1024 / 1024, 2)

        comparison['checkpoints'].append({
            'step': step,
            'size_mb': size_mb,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

        comparison['progression']['steps'].append(step)
        comparison['progression']['sizes_mb'].append(size_mb)
        comparison['progression']['timestamps'].append(stat.st_mtime)

    # Calculate training speed (steps per hour) if we have multiple checkpoints
    if len(comparison['progression']['timestamps']) >= 2:
        timestamps = comparison['progression']['timestamps']
        steps = comparison['progression']['steps']

        total_time_hours = (timestamps[-1] - timestamps[0]) / 3600
        total_steps = steps[-1] - steps[0]

        if total_time_hours > 0:
            comparison['training_speed'] = {
                'steps_per_hour': round(total_steps / total_time_hours, 1),
                'total_training_hours': round(total_time_hours, 2),
                'total_steps': total_steps
            }

    return comparison


def format_param_count(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


# =============================================================================
# API: Log File Content
# =============================================================================

@app.route('/api/logs/<path:log_path>')
def get_log_content(log_path):
    """Get the tail of a log file."""
    full_path = PROJECT_ROOT / log_path

    if not full_path.exists():
        return jsonify({'error': 'File not found'}), 404

    lines = int(request.args.get('lines', 100))

    try:
        with open(full_path, 'r') as f:
            all_lines = f.readlines()
            content = ''.join(all_lines[-lines:])

        return jsonify({
            'content': content,
            'total_lines': len(all_lines),
            'returned_lines': min(lines, len(all_lines))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/log-files')
def list_log_files():
    """List available log files (excluding _old directories)."""
    log_files = []

    for ext in ['*.log', '*.txt']:
        for log_path in LOGS_DIR.glob(f'**/{ext}'):
            # Skip files in ignored directories
            if should_ignore_path(log_path.relative_to(LOGS_DIR)):
                continue
            stat = log_path.stat()
            log_files.append({
                'path': str(log_path.relative_to(PROJECT_ROOT)),
                'name': log_path.name,
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

    # Sort by modification time
    log_files.sort(key=lambda x: x['modified'], reverse=True)

    return jsonify({'files': log_files[:30]})


# =============================================================================
# API: System Status
# =============================================================================

@app.route('/api/status')
def get_status():
    """Get system status and running processes."""
    import subprocess

    status = {
        'timestamp': datetime.now().isoformat(),
        'training_active': False,
        'gpu': None,
        'processes': []
    }

    # Check GPU status first (we'll use this for training detection too)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 5:
                status['gpu'] = {
                    'name': parts[0].strip(),
                    'memory_used_mb': int(parts[1].strip()),
                    'memory_total_mb': int(parts[2].strip()),
                    'utilization_pct': int(parts[3].strip()),
                    'temperature_c': int(parts[4].strip())
                }
    except:
        pass

    # Check for training processes (multiple patterns)
    try:
        # Look for common training script patterns
        for pattern in ['python.*dojo', 'python.*train', 'python.*main']:
            result = subprocess.run(
                ['pgrep', '-af', pattern],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                status['training_active'] = True
                for line in result.stdout.strip().split('\n'):
                    if line and 'pgrep' not in line:
                        parts = line.split(' ', 1)
                        status['processes'].append({
                            'pid': parts[0],
                            'cmd': parts[1] if len(parts) > 1 else ''
                        })
                break
    except:
        pass

    # Fallback: If GPU utilization is high (>50%), assume training is active
    if not status['training_active'] and status['gpu']:
        if status['gpu']['utilization_pct'] > 50:
            status['training_active'] = True

    return jsonify(status)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Training Dashboard Server')
    parser.add_argument('--port', type=int, default=8050, help='Port to run on (default: 8050)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║           Training Dashboard Server                            ║
╠═══════════════════════════════════════════════════════════════╣
║  Dashboard:  http://localhost:{args.port}                        ║
║  API Docs:   http://localhost:{args.port}/api/runs               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
