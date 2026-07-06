#!/usr/bin/env python3
"""
DS56 - Advanced Dynamics Visualization Server

Level 5-6 Visualization for TRM+MuZero Agent:
- Level 5: Vector Fields, Phase Space, Energy Landscapes, Divergence
- Level 6: TDA, Multifractal Analysis, Strange Attractors, Lyapunov
"""

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__, static_folder='.')
CORS(app)

# Configuration
TENSOR_SNAPSHOTS_DIR = PROJECT_ROOT / 'tensor_snapshots'
LOGS_DIR = PROJECT_ROOT / 'logs'


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DynamicsMetrics:
    """Container for computed dynamics metrics."""
    hurst_exponent: Optional[float] = None
    hurst_rolling: Optional[List[float]] = None
    lyapunov_exponent: Optional[float] = None
    ftle_grid: Optional[np.ndarray] = None
    divergence_mean: Optional[float] = None
    multifractal_spectrum: Optional[Dict] = None
    vector_field: Optional[Dict] = None
    phase_space: Optional[Dict] = None


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_hurst_exponent(data: np.ndarray, max_lag: int = 100) -> float:
    """
    Compute Hurst exponent using R/S analysis.
    H > 0.5: Persistent (trending)
    H = 0.5: Random walk
    H < 0.5: Anti-persistent (mean reverting)
    """
    if len(data) < max_lag:
        max_lag = len(data) // 2

    if max_lag < 10:
        return 0.5  # Default for insufficient data

    lags = range(10, max_lag)
    rs_values = []

    for lag in lags:
        rs_list = []
        for start in range(0, len(data) - lag, lag):
            segment = data[start:start + lag]
            if len(segment) < 2:
                continue

            mean = np.mean(segment)
            cumsum = np.cumsum(segment - mean)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(segment, ddof=1) if np.std(segment, ddof=1) > 0 else 1e-10
            rs_list.append(R / S)

        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression on log-log scale
    log_lags = np.log([v[0] for v in rs_values])
    # Protect against log(0) or negative values
    log_rs = np.log([max(v[1], 1e-10) for v in rs_values])

    try:
        # Filter out any NaN/Inf values
        valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_mask) < 3:
            return 0.5
        slope, _ = np.polyfit(log_lags[valid_mask], log_rs[valid_mask], 1)
        result = float(np.clip(slope, 0.0, 1.0))
        return result if np.isfinite(result) else 0.5
    except:
        return 0.5


def compute_rolling_hurst(data: np.ndarray, window: int = 100) -> List[float]:
    """Compute rolling Hurst exponent over a sliding window."""
    if len(data) < window + 10:
        return []

    hurst_values = []
    for i in range(len(data) - window + 1):
        segment = data[i:i + window]
        h = compute_hurst_exponent(segment, max_lag=window // 2)
        # Ensure valid value in [0, 1]
        if np.isfinite(h):
            hurst_values.append(float(np.clip(h, 0.0, 1.0)))
        else:
            hurst_values.append(0.5)  # Default for invalid

    return hurst_values


def compute_lyapunov_exponent(trajectory: np.ndarray, dt: float = 1.0) -> float:
    """
    Estimate largest Lyapunov exponent using Rosenstein method.
    Positive: Chaotic, Negative: Stable
    """
    if len(trajectory) < 50:
        return 0.0

    # Simplified estimation
    n = len(trajectory)

    # Find nearest neighbors and track divergence
    divergences = []
    for i in range(0, n - 10, 10):
        # Find closest point (excluding temporal neighbors)
        min_dist = float('inf')
        min_j = -1
        for j in range(n):
            if abs(i - j) > 5:  # Theiler window
                dist = np.linalg.norm(trajectory[i] - trajectory[j])
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    min_j = j

        if min_j > 0 and min_j < n - 1:
            # Track divergence
            for k in range(1, min(10, n - max(i, min_j))):
                new_dist = np.linalg.norm(trajectory[i + k] - trajectory[min_j + k])
                if new_dist > 1e-10 and min_dist > 1e-10:
                    divergences.append(np.log(new_dist / min_dist) / (k * dt))

    return float(np.mean(divergences)) if divergences else 0.0


def compute_vector_field(
    latent_states: np.ndarray,
    dynamics_fn: Optional[callable] = None,
    grid_size: int = 20,
    use_pca: bool = True
) -> Dict:
    """
    Compute vector field for visualization.
    Uses PCA to project high-dim states to 2D.
    """
    if latent_states is None or len(latent_states) < 10:
        return {}

    # PCA projection
    if use_pca and latent_states.shape[-1] > 2:
        centered = latent_states - latent_states.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            pca_projection = Vt[:2]  # Top 2 components
            latent_2d = centered @ pca_projection.T
        except:
            latent_2d = latent_states[:, :2]
    else:
        latent_2d = latent_states[:, :2] if latent_states.shape[-1] >= 2 else latent_states

    # Compute velocity from consecutive states
    velocity = np.diff(latent_2d, axis=0)
    positions = latent_2d[:-1]

    # Create grid
    x_range = (latent_2d[:, 0].min(), latent_2d[:, 0].max())
    y_range = (latent_2d[:, 1].min(), latent_2d[:, 1].max())

    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Interpolate velocity field
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(grid_size):
        for j in range(grid_size):
            # Find nearby points and average their velocities
            distances = np.sqrt((positions[:, 0] - X[i, j])**2 +
                               (positions[:, 1] - Y[i, j])**2)
            nearby = distances < (x_range[1] - x_range[0]) / grid_size * 2

            if nearby.sum() > 0:
                weights = 1.0 / (distances[nearby] + 1e-6)
                weights /= weights.sum()
                U[i, j] = np.sum(velocity[nearby, 0] * weights)
                V[i, j] = np.sum(velocity[nearby, 1] * weights)

    return {
        'X': X.tolist(),
        'Y': Y.tolist(),
        'U': U.tolist(),
        'V': V.tolist(),
        'trajectory_x': latent_2d[:, 0].tolist(),
        'trajectory_y': latent_2d[:, 1].tolist(),
        'explained_variance': [float(S[0]**2 / (S**2).sum()),
                               float(S[1]**2 / (S**2).sum())] if 'S' in dir() else [0.5, 0.5]
    }


def compute_divergence_field(vector_field: Dict) -> np.ndarray:
    """Compute divergence of vector field (∇·F)."""
    U = np.array(vector_field.get('U', []))
    V = np.array(vector_field.get('V', []))

    if U.size == 0 or V.size == 0:
        return np.array([])

    # Numerical gradient
    dU_dx = np.gradient(U, axis=1)
    dV_dy = np.gradient(V, axis=0)

    return dU_dx + dV_dy


def compute_energy_landscape(
    latent_states: np.ndarray,
    grid_size: int = 30
) -> Dict:
    """
    Estimate energy landscape from state density.
    E = -log(density)
    """
    if latent_states is None or len(latent_states) < 20:
        return {}

    # PCA to 2D
    centered = latent_states - latent_states.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        latent_2d = centered @ Vt[:2].T
    except:
        return {}

    # Kernel density estimation (simple histogram-based)
    x_range = (latent_2d[:, 0].min(), latent_2d[:, 0].max())
    y_range = (latent_2d[:, 1].min(), latent_2d[:, 1].max())

    hist, x_edges, y_edges = np.histogram2d(
        latent_2d[:, 0], latent_2d[:, 1],
        bins=grid_size,
        range=[x_range, y_range]
    )

    # Smooth and convert to energy
    from scipy.ndimage import gaussian_filter
    hist_smooth = gaussian_filter(hist, sigma=1.5)
    hist_smooth = np.maximum(hist_smooth, 1e-6)  # Avoid log(0)

    energy = -np.log(hist_smooth / hist_smooth.sum())

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    return {
        'X': X.tolist(),
        'Y': Y.tolist(),
        'Z': energy.T.tolist(),  # Transpose for correct orientation
        'min_energy': float(energy.min()),
        'max_energy': float(energy.max())
    }


def compute_multifractal_spectrum(data: np.ndarray) -> Dict:
    """
    Compute multifractal spectrum using DFA.
    Returns spectrum width (measure of multifractality).
    """
    if len(data) < 100:
        return {'width': 0.0, 'h_values': [], 'q_values': []}

    # Simplified MFDFA
    q_values = np.array([-3, -2, -1, 1, 2, 3, 4, 5])
    scales = np.array([8, 16, 32, 64, 128])
    scales = scales[scales < len(data) // 4]

    if len(scales) < 2:
        return {'width': 0.0, 'h_values': [], 'q_values': []}

    # Cumulative sum (profile)
    profile = np.cumsum(data - np.mean(data))

    h_values = []

    for q in q_values:
        fluctuations = []

        for scale in scales:
            n_segments = len(profile) // scale
            if n_segments < 1:
                continue

            fluct_q = 0.0
            for seg in range(n_segments):
                segment = profile[seg * scale:(seg + 1) * scale]
                # Linear detrend
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms = np.sqrt(np.mean((segment - trend)**2))

                if q == 0:
                    fluct_q += np.log(rms) if rms > 0 else 0
                else:
                    fluct_q += rms**q

            if q == 0:
                fluctuations.append((scale, np.exp(fluct_q / n_segments)))
            else:
                fluctuations.append((scale, (fluct_q / n_segments)**(1/q)))

        if len(fluctuations) >= 2:
            log_scales = np.log([f[0] for f in fluctuations])
            log_fluct = np.log([max(f[1], 1e-10) for f in fluctuations])
            try:
                h, _ = np.polyfit(log_scales, log_fluct, 1)
                h_values.append(float(h))
            except:
                h_values.append(0.5)

    width = max(h_values) - min(h_values) if h_values else 0.0

    return {
        'width': float(width),
        'h_values': h_values,
        'q_values': q_values.tolist()
    }


def create_takens_embedding(
    time_series: np.ndarray,
    embedding_dim: int = 3,
    delay: int = 10
) -> np.ndarray:
    """
    Create Takens delay embedding for phase space reconstruction.
    """
    n = len(time_series)
    if n < embedding_dim * delay:
        return np.array([])

    m = n - (embedding_dim - 1) * delay
    embedded = np.zeros((m, embedding_dim))

    for i in range(embedding_dim):
        embedded[:, i] = time_series[i * delay:i * delay + m]

    return embedded


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    """Serve the main dashboard."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('.', filename)


@app.route('/api/snapshots')
def list_snapshots():
    """List available tensor snapshots."""
    snapshots = []

    if TENSOR_SNAPSHOTS_DIR.exists():
        for run_dir in sorted(TENSOR_SNAPSHOTS_DIR.iterdir()):
            if run_dir.is_dir():
                npz_files = list(run_dir.glob('*.npz'))
                if npz_files:
                    snapshots.append({
                        'run_name': run_dir.name,
                        'path': str(run_dir.relative_to(PROJECT_ROOT)),
                        'num_snapshots': len(npz_files),
                        'epochs': sorted([int(f.stem.split('_')[-1]) for f in npz_files
                                        if f.stem.split('_')[-1].isdigit()])
                    })

    return jsonify(snapshots)


@app.route('/api/snapshot/<path:run_name>/<int:epoch>')
def get_snapshot(run_name: str, epoch: int):
    """Load and return a specific tensor snapshot."""
    snapshot_path = TENSOR_SNAPSHOTS_DIR / run_name / f'epoch_{epoch}.npz'

    if not snapshot_path.exists():
        # Try alternate naming
        alt_paths = list((TENSOR_SNAPSHOTS_DIR / run_name).glob(f'*{epoch}*.npz'))
        if alt_paths:
            snapshot_path = alt_paths[0]
        else:
            return jsonify({'error': f'Snapshot not found: {snapshot_path}'}), 404

    try:
        data = np.load(snapshot_path, allow_pickle=True)
        result = {
            'epoch': epoch,
            'run_name': run_name,
            'keys': list(data.keys()),
            'shapes': {k: list(data[k].shape) for k in data.keys()}
        }

        # Include small arrays directly
        for key in data.keys():
            arr = data[key]
            if arr.size < 10000:  # Only include small arrays
                result[key] = arr.tolist()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dynamics/<path:run_name>')
def compute_dynamics(run_name: str):
    """Compute dynamics metrics for a run."""
    epoch = request.args.get('epoch', type=int)

    # Find snapshots
    run_dir = TENSOR_SNAPSHOTS_DIR / run_name
    if not run_dir.exists():
        return jsonify({'error': f'Run not found: {run_name}'}), 404

    npz_files = sorted(run_dir.glob('*.npz'))
    if not npz_files:
        return jsonify({'error': 'No snapshots found'}), 404

    # Load latest or specified epoch
    if epoch is not None:
        # Handle zero-padded epoch filenames (epoch_000010.npz)
        snapshot_path = run_dir / f'epoch_{epoch:06d}.npz'
        if not snapshot_path.exists():
            # Try without padding as fallback
            snapshot_path = run_dir / f'epoch_{epoch}.npz'
            if not snapshot_path.exists():
                return jsonify({'error': f'Epoch {epoch} not found'}), 404
    else:
        snapshot_path = npz_files[-1]

    try:
        data = np.load(snapshot_path, allow_pickle=True)

        # Extract latent states - prefer trajectory (temporal sequence) over batch
        latent_states = None
        if 'latent_trajectory' in data.keys():
            latent_states = data['latent_trajectory']
        elif 'latent_states' in data.keys():
            latent_states = data['latent_states']
        else:
            # Fallback: find any key with 'latent'
            latent_key = next((k for k in data.keys() if 'latent' in k.lower()), None)
            latent_states = data[latent_key] if latent_key else None

        result = {
            'epoch': epoch or int(snapshot_path.stem.split('_')[-1]),
            'run_name': run_name,
            'available_keys': list(data.keys()),
            'num_observations': int(data.get('num_observations', 0)) if 'num_observations' in data.keys() else 0
        }

        if latent_states is not None and len(latent_states) > 0:
            # Flatten if batched
            if latent_states.ndim > 2:
                latent_states = latent_states.reshape(-1, latent_states.shape[-1])

            # Compute metrics
            result['hurst'] = compute_hurst_exponent(latent_states[:, 0])
            result['hurst_rolling'] = compute_rolling_hurst(latent_states[:, 0], window=50)

            if len(latent_states) >= 50:
                result['lyapunov'] = compute_lyapunov_exponent(latent_states)

            result['vector_field'] = compute_vector_field(latent_states)

            if result['vector_field']:
                divergence = compute_divergence_field(result['vector_field'])
                if divergence.size > 0:
                    result['divergence'] = {
                        'mean': float(np.mean(divergence)),
                        'std': float(np.std(divergence)),
                        'grid': divergence.tolist()
                    }

            result['energy_landscape'] = compute_energy_landscape(latent_states)
            result['multifractal'] = compute_multifractal_spectrum(latent_states[:, 0])

            # Takens embedding for attractor visualization
            embedding = create_takens_embedding(latent_states[:, 0], embedding_dim=3, delay=5)
            if len(embedding) > 0:
                result['attractor'] = {
                    'x': embedding[:, 0].tolist(),
                    'y': embedding[:, 1].tolist(),
                    'z': embedding[:, 2].tolist()
                }

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/training/<path:csv_path>')
def get_training_data(csv_path: str):
    """Load training CSV data (reuse from DS24)."""
    csv_file = LOGS_DIR / csv_path

    if not csv_file.exists():
        return jsonify({'error': f'CSV not found: {csv_path}'}), 404

    try:
        import pandas as pd
        df = pd.read_csv(csv_file)

        # Convert to dict of lists
        data = {col: df[col].tolist() for col in df.columns}
        return jsonify(data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def get_status():
    """Get system status."""
    status = {
        'snapshots_available': TENSOR_SNAPSHOTS_DIR.exists(),
        'num_runs': len(list(TENSOR_SNAPSHOTS_DIR.iterdir())) if TENSOR_SNAPSHOTS_DIR.exists() else 0,
        'timestamp': datetime.now().isoformat()
    }

    # GPU status
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            status['gpu'] = {
                'utilization': int(parts[0].strip()),
                'memory_used': int(parts[1].strip()),
                'memory_total': int(parts[2].strip())
            }
    except:
        status['gpu'] = None

    return jsonify(status)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DS56 - Advanced Dynamics Visualization')
    parser.add_argument('--port', type=int, default=5056, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    print(f"DS56 Server starting on http://{args.host}:{args.port}")
    print(f"Tensor snapshots directory: {TENSOR_SNAPSHOTS_DIR}")

    app.run(host=args.host, port=args.port, debug=args.debug)
