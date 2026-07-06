#!/usr/bin/env python3
"""
DSe - Dashboard for Easy Training (e-dojo dual-branch monitor)

Monitors dual-branch training with merge visualization.
Port: 8052 (adjacent to DS24's 8051)
"""

import argparse
import json
import logging
import math
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request, send_from_directory
from flask_cors import CORS

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.')
CORS(app)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / 'logs'
DOJO_LOGS_DIR = LOGS_DIR / 'e-dojo'
DUAL_DOJO_LOGS_DIR = LOGS_DIR / 'dual-dojo'
CKPT_A_DIR = PROJECT_ROOT / 'jax_ckpt_easy_A'
CKPT_B_DIR = PROJECT_ROOT / 'jax_ckpt_easy_B'
CKPT_MERGED_DIR = PROJECT_ROOT / 'jax_ckpt_easy_merged'


def safe_json(obj: Any) -> Any:
    """Convert numpy types and handle NaN/Inf for JSON serialization."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def read_csv_flexible(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV with flexible schema handling."""
    try:
        df = pd.read_csv(path)
        # Convert numeric columns
        for col in df.columns:
            if col not in ['timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None


def get_latest_csv(pattern: str) -> Optional[Path]:
    """Get the most recently modified CSV matching pattern."""
    files = sorted(LOGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def get_all_csvs_concatenated(pattern: str) -> Optional[pd.DataFrame]:
    """Get all CSVs matching pattern, concatenated for full history."""
    files = sorted(LOGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None

    dfs = []
    for f in files:
        df = read_csv_flexible(f)
        if df is not None and len(df) > 0:
            dfs.append(df)

    if not dfs:
        return None

    # Concatenate all and re-index by epoch (assuming epoch column or use index)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def get_branch_b_pattern() -> str:
    """Get the correct pattern for branch B (synth, bella_inv, bella, or 90hx)."""
    # Check which type exists (prefer most recent)
    bella_inv = list(LOGS_DIR.glob('branch_B_bella_inv_*.csv'))
    bella = list(LOGS_DIR.glob('branch_B_bella_*.csv'))
    synth = list(LOGS_DIR.glob('branch_B_synth_*.csv'))
    b_90hx = list(LOGS_DIR.glob('branch_B_90hx_*.csv'))

    # bella glob also matches bella_inv — filter to only exact bella (not bella_inv)
    bella = [f for f in bella if 'bella_inv' not in f.name]

    if b_90hx:
        return 'branch_B_90hx_*.csv'
    elif bella:
        return 'branch_B_bella_*.csv'
    elif bella_inv:
        return 'branch_B_bella_inv_*.csv'
    elif synth:
        return 'branch_B_synth_*.csv'
    else:
        return 'branch_B_synth_*.csv'


def get_latest_branch_b_csv() -> Optional[Path]:
    """Get the most recent branch B CSV (synth, bella_inv, bella, or 90hx)."""
    synth = list(LOGS_DIR.glob('branch_B_synth_*.csv'))
    bella_inv = list(LOGS_DIR.glob('branch_B_bella_inv_*.csv'))
    bella = [f for f in LOGS_DIR.glob('branch_B_bella_*.csv') if 'bella_inv' not in f.name]
    b_90hx = list(LOGS_DIR.glob('branch_B_90hx_*.csv'))

    all_files = synth + bella_inv + bella + b_90hx
    if not all_files:
        return None

    return max(all_files, key=lambda p: p.stat().st_mtime)


def get_all_branch_b_csvs() -> List[Path]:
    """Get all branch B CSVs (synth, bella_inv, bella, or 90hx), sorted by mtime."""
    synth = list(LOGS_DIR.glob('branch_B_synth_*.csv'))
    bella_inv = list(LOGS_DIR.glob('branch_B_bella_inv_*.csv'))
    bella = [f for f in LOGS_DIR.glob('branch_B_bella_*.csv') if 'bella_inv' not in f.name]
    b_90hx = list(LOGS_DIR.glob('branch_B_90hx_*.csv'))

    all_files = synth + bella_inv + bella + b_90hx
    return sorted(all_files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_dojo_state() -> Dict:
    """Read current dojo orchestrator state (e-dojo or dual-dojo)."""
    # Prefer dual-dojo state (most recent), fallback to e-dojo
    for logs_dir in [DUAL_DOJO_LOGS_DIR, DOJO_LOGS_DIR]:
        state_file = logs_dir / ('dual_dojo_state.json' if logs_dir == DUAL_DOJO_LOGS_DIR else 'dojo_state.json')
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                if state:  # Non-empty
                    return state
            except Exception as e:
                logger.error(f"Error reading dojo state from {state_file}: {e}")
    return {}


def get_merge_history() -> List[Dict]:
    """Read merge history (e-dojo or dual-dojo, combined)."""
    all_history = []
    for logs_dir in [DOJO_LOGS_DIR, DUAL_DOJO_LOGS_DIR]:
        history_file = logs_dir / 'merge_history.json'
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                if history:
                    all_history.extend(history)
            except Exception as e:
                logger.error(f"Error reading merge history from {history_file}: {e}")
    # Sort by timestamp if available
    all_history.sort(key=lambda h: h.get('timestamp', ''), reverse=False)
    return all_history


def check_process_running(name: str) -> bool:
    """Check if a training process is running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', f'train_easy.*{name}'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except:
        return False


def _safe_relative_path(file_path: Path) -> str:
    """Get path relative to PROJECT_ROOT, or absolute if outside project."""
    try:
        return str(file_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(file_path)


def get_checkpoint_info(ckpt_dir: Path) -> List[Dict]:
    """Get info about checkpoints in a directory."""
    checkpoints = []
    try:
        for subdir in ckpt_dir.iterdir():
            if subdir.is_dir():
                for pkl_file in sorted(subdir.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True):
                    checkpoints.append({
                        'name': pkl_file.name,
                        'branch': subdir.name,
                        'path': _safe_relative_path(pkl_file),
                        'size_mb': pkl_file.stat().st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat()
                    })
        # Also check for merged checkpoints in root
        for pkl_file in sorted(ckpt_dir.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True):
            checkpoints.append({
                'name': pkl_file.name,
                'branch': 'merged',
                'path': _safe_relative_path(pkl_file),
                'size_mb': pkl_file.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat()
            })
    except Exception as e:
        logger.error(f"Error reading checkpoints from {ckpt_dir}: {e}")
    return checkpoints


def load_checkpoint_stats(ckpt_path: Path) -> Dict:
    """Load checkpoint and extract layer statistics."""
    try:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        params = ckpt.get('params', {})
        metrics = ckpt.get('metrics', {})
        epoch = ckpt.get('epoch', 0)

        layer_stats = []

        def is_array_like(v):
            """Check if value is numpy or JAX array."""
            if isinstance(v, np.ndarray):
                return True
            # Check for JAX array (has shape attribute and is not dict)
            if hasattr(v, 'shape') and hasattr(v, 'size') and not isinstance(v, dict):
                return True
            return False

        def to_numpy(v):
            """Convert to numpy array."""
            if isinstance(v, np.ndarray):
                return v
            # JAX array - convert via numpy
            return np.asarray(v)

        def extract_stats(d: Dict, prefix: str = ''):
            for k, v in d.items():
                full_key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, dict):
                    extract_stats(v, full_key)
                elif is_array_like(v):
                    arr = to_numpy(v)
                    layer_stats.append({
                        'name': full_key,
                        'shape': list(arr.shape),
                        'size': int(arr.size),
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'sparsity': float(np.mean(np.abs(arr) < 1e-6))
                    })

        extract_stats(params)

        return {
            'epoch': epoch,
            'metrics': safe_json(metrics),
            'layer_stats': layer_stats,
            'total_params': sum(s['size'] for s in layer_stats)
        }
    except Exception as e:
        logger.error(f"Error loading checkpoint {ckpt_path}: {e}")
        return {'error': str(e)}


def get_weight_heatmap(ckpt_path: Path, layer_name: str, max_size: int = 100) -> Dict:
    """Extract weight matrix for heatmap visualization."""
    try:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        params = ckpt.get('params', {})

        # Navigate to layer
        parts = layer_name.split('/')
        current = params
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {'error': f'Layer {layer_name} not found'}

        # Check for numpy or JAX array
        if not (isinstance(current, np.ndarray) or (hasattr(current, 'shape') and hasattr(current, 'size'))):
            return {'error': 'Not a weight array'}

        # Convert to numpy if JAX array
        arr = np.asarray(current)

        # Reshape to 2D if needed
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        # Downsample if too large
        if arr.shape[0] > max_size:
            indices = np.linspace(0, arr.shape[0]-1, max_size, dtype=int)
            arr = arr[indices]
        if arr.shape[1] > max_size:
            indices = np.linspace(0, arr.shape[1]-1, max_size, dtype=int)
            arr = arr[:, indices]

        return {
            'data': arr.tolist(),
            'shape': list(current.shape),
            'layer': layer_name
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/status')
def api_status():
    """Get current dojo state and process health."""
    state = get_dojo_state()

    # Check if training processes are running
    branch_a_running = check_process_running('branch_A')
    branch_b_running = check_process_running('branch_B')
    easy_running = check_process_running('train_easy')

    # Get latest epoch from each branch
    # Branch A: try dual dojo, historical, then e-dojo patterns
    csv_a = get_latest_csv('branch_A_3090_*.csv') or get_latest_csv('branch_A_historical_*.csv') or get_latest_csv('branch_A_bella_*.csv')
    csv_b = get_latest_branch_b_csv()
    csv_easy = get_latest_csv('easy_*.csv')

    latest_a = {}
    latest_b = {}
    latest_easy = {}

    if csv_a:
        df = read_csv_flexible(csv_a)
        if df is not None and len(df) > 0:
            latest_a = df.iloc[-1].to_dict()

    if csv_b:
        df = read_csv_flexible(csv_b)
        if df is not None and len(df) > 0:
            latest_b = df.iloc[-1].to_dict()

    if csv_easy:
        df = read_csv_flexible(csv_easy)
        if df is not None and len(df) > 0:
            latest_easy = df.iloc[-1].to_dict()

    return jsonify(safe_json({
        'dojo': state,
        'branch_a': {
            'running': branch_a_running,
            'csv': csv_a.name if csv_a else None,
            'latest': latest_a
        },
        'branch_b': {
            'running': branch_b_running,
            'csv': csv_b.name if csv_b else None,
            'latest': latest_b
        },
        'easy': {
            'running': easy_running,
            'csv': csv_easy.name if csv_easy else None,
            'latest': latest_easy
        },
        'timestamp': datetime.now().isoformat()
    }))


@app.route('/api/branches')
def api_branches():
    """List available branch CSVs."""
    branches = {
        'branch_a': [],
        'branch_b': [],
        'easy': []
    }

    # Discover Branch A CSVs (e-dojo: branch_A_bella, historical: branch_A_historical, dual dojo: branch_A_3090)
    branch_a_files = sorted(
        list(LOGS_DIR.glob('branch_A_bella_*.csv')) +
        list(LOGS_DIR.glob('branch_A_historical_*.csv')) +
        list(LOGS_DIR.glob('branch_A_3090_*.csv')),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    for csv_file in branch_a_files:
        branches['branch_a'].append({
            'name': csv_file.name,
            'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat(),
            'size_kb': csv_file.stat().st_size / 1024
        })

    for csv_file in get_all_branch_b_csvs():
        branches['branch_b'].append({
            'name': csv_file.name,
            'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat(),
            'size_kb': csv_file.stat().st_size / 1024
        })

    for csv_file in sorted(LOGS_DIR.glob('easy_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True):
        branches['easy'].append({
            'name': csv_file.name,
            'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat(),
            'size_kb': csv_file.stat().st_size / 1024
        })

    return jsonify(branches)


@app.route('/api/branch/<name>')
def api_branch(name: str):
    """Get training data for a specific branch CSV."""
    csv_path = LOGS_DIR / name
    if not csv_path.exists():
        return jsonify({'error': 'CSV not found'}), 404

    df = read_csv_flexible(csv_path)
    if df is None:
        return jsonify({'error': 'Failed to read CSV'}), 500

    # Convert to records
    data = df.to_dict(orient='records')

    return jsonify(safe_json({
        'name': name,
        'epochs': len(data),
        'columns': list(df.columns),
        'data': data
    }))


@app.route('/api/merges')
def api_merges():
    """Get merge history with effectiveness metrics."""
    history = get_merge_history()
    state = get_dojo_state()

    # Compute effectiveness for each merge
    for i, merge in enumerate(history):
        # Agreement improvement: compare with previous state
        if i > 0:
            prev = history[i-1]
            merge['agreement_improvement'] = (
                (merge.get('agreement_a', 0) + merge.get('agreement_b', 0)) / 2 -
                (prev.get('agreement_a', 0) + prev.get('agreement_b', 0)) / 2
            )
        else:
            merge['agreement_improvement'] = 0

    return jsonify(safe_json({
        'history': history,
        'state': state,
        'total_merges': len(history)
    }))


@app.route('/api/checkpoints')
def api_checkpoints():
    """List all checkpoints across branches (e-dojo + dual dojo)."""
    # Dual dojo checkpoint dirs on scratch disk
    CKPT_BASE = Path('/mnt/scratch/trader_checkpoints')
    DUAL_A = CKPT_BASE / 'jax_ckpt_dual_A'
    DUAL_B = CKPT_BASE / 'jax_ckpt_dual_B'
    DUAL_MERGED = CKPT_BASE / 'jax_ckpt_dual_merged'

    branch_a = get_checkpoint_info(CKPT_A_DIR) if CKPT_A_DIR.exists() else []
    branch_b = get_checkpoint_info(CKPT_B_DIR) if CKPT_B_DIR.exists() else []
    merged = get_checkpoint_info(CKPT_MERGED_DIR) if CKPT_MERGED_DIR.exists() else []

    # Add dual dojo checkpoints
    if DUAL_A.exists():
        branch_a.extend(get_checkpoint_info(DUAL_A))
    if DUAL_B.exists():
        branch_b.extend(get_checkpoint_info(DUAL_B))
    if DUAL_MERGED.exists():
        merged.extend(get_checkpoint_info(DUAL_MERGED))

    checkpoints = {
        'branch_a': branch_a,
        'branch_b': branch_b,
        'merged': merged,
    }

    return jsonify(checkpoints)


def _resolve_ckpt_path(ckpt_path: str) -> Optional[Path]:
    """Resolve checkpoint path — handles both relative (to PROJECT_ROOT) and absolute.

    Also handles Flask stripping leading '/' from absolute paths via 308 redirect
    (e.g. '/api/checkpoint//media/...' → '/api/checkpoint/media/...' → ckpt_path='media/...').
    """
    p = Path(ckpt_path)
    if p.is_absolute() and p.exists():
        return p
    # Flask strips leading '/' from absolute paths — try re-adding it
    absolute = Path('/' + ckpt_path)
    if absolute.exists():
        return absolute
    relative = PROJECT_ROOT / ckpt_path
    if relative.exists():
        return relative
    return None


@app.route('/api/checkpoint/<path:ckpt_path>')
def api_checkpoint(ckpt_path: str):
    """Load checkpoint and return layer statistics."""
    full_path = _resolve_ckpt_path(ckpt_path)
    if not full_path:
        return jsonify({'error': 'Checkpoint not found'}), 404

    stats = load_checkpoint_stats(full_path)
    return jsonify(safe_json(stats))


@app.route('/api/checkpoint/<path:ckpt_path>/heatmap/<path:layer>')
def api_checkpoint_heatmap(ckpt_path: str, layer: str):
    """Get weight heatmap for a specific layer."""
    full_path = _resolve_ckpt_path(ckpt_path)
    if not full_path:
        return jsonify({'error': 'Checkpoint not found'}), 404

    heatmap = get_weight_heatmap(full_path, layer)
    return jsonify(safe_json(heatmap))


@app.route('/api/logs/<branch>')
def api_logs(branch: str):
    """Get tail of branch log file."""
    # Find the most recent log for this branch
    # Search both e-dojo and dual-dojo log directories
    log_dirs = [DOJO_LOGS_DIR]
    if DUAL_DOJO_LOGS_DIR.exists():
        log_dirs.append(DUAL_DOJO_LOGS_DIR)

    if branch == 'branch_B' or branch == 'branch_B_synth':
        all_logs = []
        for d in log_dirs:
            all_logs.extend(d.glob('branch_B_synth_*.log'))
            all_logs.extend(d.glob('branch_B_bella_inv_*.log'))
            all_logs.extend(d.glob('branch_B_bella_*.log'))
            all_logs.extend(d.glob('branch_B_90hx_*.log'))
        # Filter out bella_inv duplicates from bella glob
        all_logs = list({str(f): f for f in all_logs}.values())
        log_files = sorted(all_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    elif branch in ('branch_A', 'branch_A_bella', 'branch_A_historical'):
        all_logs = []
        for d in log_dirs:
            all_logs.extend(d.glob('branch_A_bella_*.log'))
            all_logs.extend(d.glob('branch_A_historical_*.log'))
            all_logs.extend(d.glob('branch_A_3090_*.log'))
        log_files = sorted(all_logs, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        pattern = f'{branch}_*.log'
        all_logs = []
        for d in log_dirs:
            all_logs.extend(d.glob(pattern))
        log_files = sorted(all_logs, key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        return jsonify({'error': 'No log files found', 'lines': []})

    log_file = log_files[0]
    n_lines = request.args.get('lines', 100, type=int)

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            tail = lines[-n_lines:] if len(lines) > n_lines else lines

        return jsonify({
            'file': log_file.name,
            'total_lines': len(lines),
            'lines': [line.rstrip() for line in tail]
        })
    except Exception as e:
        return jsonify({'error': str(e), 'lines': []})


@app.route('/api/live')
def api_live():
    """Combined endpoint for polling - returns all real-time data."""
    state = get_dojo_state()

    # Get ALL data from both branches (concatenated across restarts)
    # Try dual dojo, historical, then e-dojo patterns
    df_a_dual = get_all_csvs_concatenated('branch_A_3090_*.csv')
    df_a_hist = get_all_csvs_concatenated('branch_A_historical_*.csv')
    df_a_edojo = get_all_csvs_concatenated('branch_A_bella_*.csv')
    # Pick first non-None DataFrame (can't use `or` on DataFrames)
    df_a = df_a_dual if df_a_dual is not None else (df_a_hist if df_a_hist is not None else df_a_edojo)
    # Use get_all_branch_b_csvs() for proper dedup (bella vs bella_inv)
    b_csvs = get_all_branch_b_csvs()
    if b_csvs:
        dfs_b = [read_csv_flexible(f) for f in reversed(b_csvs)]  # oldest first
        dfs_b = [df for df in dfs_b if df is not None and len(df) > 0]
        df_b = pd.concat(dfs_b, ignore_index=True) if dfs_b else None
    else:
        df_b = None
    df_easy = get_all_csvs_concatenated('easy_*.csv')

    data_a = df_a.to_dict(orient='records') if df_a is not None else []
    data_b = df_b.to_dict(orient='records') if df_b is not None else []
    data_easy = df_easy.to_dict(orient='records') if df_easy is not None else []

    csv_a = get_latest_csv('branch_A_3090_*.csv') or get_latest_csv('branch_A_historical_*.csv') or get_latest_csv('branch_A_bella_*.csv')
    csv_b = get_latest_branch_b_csv()
    csv_easy = get_latest_csv('easy_*.csv')

    # Get merge events for timeline
    merges = get_merge_history()

    return jsonify(safe_json({
        'state': state,
        'branch_a': {
            'csv': csv_a.name if csv_a else None,
            'data': data_a,
            'running': check_process_running('branch_A')
        },
        'branch_b': {
            'csv': csv_b.name if csv_b else None,
            'data': data_b,
            'running': check_process_running('branch_B')
        },
        'easy': {
            'csv': csv_easy.name if csv_easy else None,
            'data': data_easy,
            'running': check_process_running('train_easy')
        },
        'merges': merges,
        'timestamp': datetime.now().isoformat()
    }))


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSe - Dashboard for Easy Training')
    parser.add_argument('--port', type=int, default=8052, help='Port to run on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    DSe - Easy Training Dashboard             ║
    ║                                                              ║
    ║  Monitoring: e-dojo dual-branch training                     ║
    ║  Branch A: Bella (CNN diffusion)                             ║
    ║  Branch B: Synthetic (curriculum)                            ║
    ║                                                              ║
    ║  URL: http://localhost:{args.port:<5}                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
