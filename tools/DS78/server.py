#!/usr/bin/env python3
"""
DS78 - Shroom Engine: Level 7-8 Volumetric & Fractal Visualization

Advanced visualization for TRM+MuZero latent dynamics:
- Level 7: NeRF-style volumetric rendering, Gaussian splatting
- Level 8: Fractal dimension, attractor reconstruction, SDF rendering
"""

import os
import sys
import json
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


# =============================================================================
# Level 7: Volumetric Rendering
# =============================================================================

def compute_latent_density_field(
    latent_states: np.ndarray,
    grid_resolution: int = 32,
    bandwidth: float = 0.5
) -> Dict:
    """
    Compute 3D density field from latent states using KDE.
    This is the "Latent NeRF" - volumetric representation of latent space.
    """
    if latent_states is None or len(latent_states) < 10:
        return {}

    # PCA to 3D for volumetric rendering
    centered = latent_states - latent_states.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        latent_3d = centered @ Vt[:3].T
        explained_var = (S[:3]**2 / np.sum(S**2)).tolist()
    except:
        return {}

    # Compute bounds
    mins = latent_3d.min(axis=0) - 1
    maxs = latent_3d.max(axis=0) + 1

    # Create 3D grid
    x = np.linspace(mins[0], maxs[0], grid_resolution)
    y = np.linspace(mins[1], maxs[1], grid_resolution)
    z = np.linspace(mins[2], maxs[2], grid_resolution)

    # KDE-based density estimation (simplified)
    density = np.zeros((grid_resolution, grid_resolution, grid_resolution))

    for point in latent_3d:
        # Gaussian kernel contribution
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    dist_sq = (xi - point[0])**2 + (yj - point[1])**2 + (zk - point[2])**2
                    density[i, j, k] += np.exp(-dist_sq / (2 * bandwidth**2))

    # Normalize
    density = density / (density.max() + 1e-10)

    return {
        'density': density.tolist(),
        'bounds': {
            'min': mins.tolist(),
            'max': maxs.tolist()
        },
        'resolution': grid_resolution,
        'explained_variance': explained_var,
        'trajectory_3d': latent_3d.tolist()
    }


def compute_gaussian_splats(
    latent_states: np.ndarray,
    n_components: int = 100
) -> Dict:
    """
    Compute Gaussian splat representation for fast rendering.
    Each point becomes a colored Gaussian blob.
    """
    if latent_states is None or len(latent_states) < 10:
        return {}

    # PCA to 3D
    centered = latent_states - latent_states.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        latent_3d = centered @ Vt[:3].T
    except:
        return {}

    # Subsample if too many points
    if len(latent_3d) > n_components:
        indices = np.linspace(0, len(latent_3d) - 1, n_components, dtype=int)
        latent_3d = latent_3d[indices]

    n_points = len(latent_3d)

    # Compute local density for splat size
    from scipy.spatial.distance import cdist
    if n_points > 1:
        dists = cdist(latent_3d, latent_3d)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        scales = min_dists * 0.5  # Splat radius
    else:
        scales = np.ones(n_points) * 0.1

    # Time-based coloring
    colors = np.linspace(0, 1, n_points)

    return {
        'positions': latent_3d.tolist(),
        'scales': scales.tolist(),
        'colors': colors.tolist(),
        'n_splats': n_points
    }


# =============================================================================
# Level 8: Fractal Analysis
# =============================================================================

def compute_correlation_dimension(
    trajectory: np.ndarray,
    max_radius: float = None,
    n_radii: int = 20
) -> Dict:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    D2 = lim(r->0) log(C(r)) / log(r)
    """
    if trajectory is None or len(trajectory) < 50:
        return {'dimension': None, 'radii': [], 'correlations': []}

    n = len(trajectory)

    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(trajectory)

    if max_radius is None:
        max_radius = np.percentile(distances, 90)

    min_radius = np.percentile(distances, 5)

    # Compute correlation integral C(r) for various radii
    radii = np.logspace(np.log10(min_radius), np.log10(max_radius), n_radii)
    correlations = []

    for r in radii:
        # C(r) = (2 / N(N-1)) * sum(H(r - |xi - xj|))
        count = np.sum(distances < r)
        C_r = 2 * count / (n * (n - 1))
        correlations.append(max(C_r, 1e-10))

    # Linear regression on log-log scale
    log_r = np.log(radii)
    log_C = np.log(correlations)

    # Use middle portion (avoid edge effects)
    mid_start = n_radii // 4
    mid_end = 3 * n_radii // 4

    try:
        slope, intercept = np.polyfit(log_r[mid_start:mid_end], log_C[mid_start:mid_end], 1)
        dimension = float(slope)
    except:
        dimension = None

    return {
        'dimension': dimension,
        'radii': radii.tolist(),
        'correlations': correlations,
        'log_radii': log_r.tolist(),
        'log_correlations': log_C.tolist()
    }


def compute_box_counting_dimension(
    trajectory: np.ndarray,
    n_scales: int = 15
) -> Dict:
    """
    Estimate fractal dimension using box-counting method.
    D0 = lim(e->0) log(N(e)) / log(1/e)
    """
    if trajectory is None or len(trajectory) < 20:
        return {'dimension': None}

    # Normalize to unit cube
    mins = trajectory.min(axis=0)
    maxs = trajectory.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (trajectory - mins) / ranges

    # Count boxes at different scales
    scales = np.logspace(-2, 0, n_scales)
    counts = []

    for scale in scales:
        # Discretize to grid
        grid_coords = (normalized / scale).astype(int)
        unique_boxes = set(map(tuple, grid_coords))
        counts.append(len(unique_boxes))

    # Linear regression
    log_scale = np.log(1 / scales)
    log_count = np.log(counts)

    try:
        slope, _ = np.polyfit(log_scale, log_count, 1)
        dimension = float(slope)
    except:
        dimension = None

    return {
        'dimension': dimension,
        'scales': scales.tolist(),
        'counts': counts,
        'log_inv_scales': log_scale.tolist(),
        'log_counts': log_count.tolist()
    }


def generate_clifford_attractor(
    a: float = -1.4,
    b: float = 1.6,
    c: float = 1.0,
    d: float = 0.7,
    n_points: int = 10000
) -> Dict:
    """
    Generate Clifford attractor for reference/comparison.
    x_{n+1} = sin(a*y_n) + c*cos(a*x_n)
    y_{n+1} = sin(b*x_n) + d*cos(b*y_n)
    """
    x, y = 0.1, 0.1
    points_x, points_y = [], []

    for _ in range(n_points):
        x_new = np.sin(a * y) + c * np.cos(a * x)
        y_new = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_new, y_new
        points_x.append(x)
        points_y.append(y)

    return {
        'x': points_x,
        'y': points_y,
        'params': {'a': a, 'b': b, 'c': c, 'd': d}
    }


def generate_lorenz_attractor(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8/3,
    dt: float = 0.01,
    n_steps: int = 5000
) -> Dict:
    """
    Generate Lorenz attractor for 3D reference.
    """
    x, y, z = 1.0, 1.0, 1.0
    points = []

    for _ in range(n_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        x += dx * dt
        y += dy * dt
        z += dz * dt
        points.append([x, y, z])

    points = np.array(points)

    return {
        'x': points[:, 0].tolist(),
        'y': points[:, 1].tolist(),
        'z': points[:, 2].tolist(),
        'params': {'sigma': sigma, 'rho': rho, 'beta': beta}
    }


def compute_recurrence_plot(
    trajectory: np.ndarray,
    threshold: float = None,
    max_size: int = 200
) -> Dict:
    """
    Compute recurrence plot - reveals structure in dynamics.
    R[i,j] = 1 if ||x_i - x_j|| < threshold
    """
    if trajectory is None or len(trajectory) < 10:
        return {}

    # Subsample if too large
    if len(trajectory) > max_size:
        indices = np.linspace(0, len(trajectory) - 1, max_size, dtype=int)
        trajectory = trajectory[indices]

    n = len(trajectory)

    # Compute distance matrix
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(trajectory, trajectory)

    if threshold is None:
        threshold = np.percentile(dist_matrix, 10)

    # Binary recurrence matrix
    recurrence = (dist_matrix < threshold).astype(int)

    # Compute recurrence rate
    rr = np.sum(recurrence) / (n * n)

    # Compute determinism (fraction of recurrence points forming lines)
    det = compute_determinism(recurrence)

    return {
        'matrix': recurrence.tolist(),
        'size': n,
        'threshold': float(threshold),
        'recurrence_rate': float(rr),
        'determinism': float(det)
    }


def compute_determinism(recurrence: np.ndarray, min_line: int = 2) -> float:
    """Compute determinism from recurrence plot."""
    n = len(recurrence)
    diagonal_lines = []

    # Check diagonals
    for k in range(1, n):
        diag = np.diag(recurrence, k)
        line_len = 0
        for val in diag:
            if val == 1:
                line_len += 1
            else:
                if line_len >= min_line:
                    diagonal_lines.append(line_len)
                line_len = 0
        if line_len >= min_line:
            diagonal_lines.append(line_len)

    if not diagonal_lines:
        return 0.0

    total_diag_points = sum(diagonal_lines)
    total_recurrence = np.sum(recurrence) - n  # Exclude main diagonal

    return total_diag_points / max(total_recurrence, 1)


# =============================================================================
# API Routes
# =============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/snapshots')
def list_snapshots():
    """List available tensor snapshots."""
    snapshots = []

    if TENSOR_SNAPSHOTS_DIR.exists():
        for run_dir in TENSOR_SNAPSHOTS_DIR.iterdir():
            if run_dir.is_dir():
                npz_files = list(run_dir.glob('*.npz'))
                if npz_files:
                    epochs = sorted([
                        int(f.stem.split('_')[-1])
                        for f in npz_files
                    ])
                    snapshots.append({
                        'run_name': run_dir.name,
                        'num_snapshots': len(npz_files),
                        'epochs': epochs,
                        'path': str(run_dir.relative_to(PROJECT_ROOT))
                    })

    return jsonify(snapshots)


@app.route('/api/volume/<path:run_name>')
def compute_volume(run_name: str):
    """Compute volumetric density field for NeRF-style rendering."""
    epoch = request.args.get('epoch', type=int)
    resolution = request.args.get('resolution', type=int, default=24)

    run_dir = TENSOR_SNAPSHOTS_DIR / run_name
    if not run_dir.exists():
        return jsonify({'error': f'Run not found: {run_name}'}), 404

    npz_files = sorted(run_dir.glob('*.npz'))
    if not npz_files:
        return jsonify({'error': 'No snapshots found'}), 404

    # Load snapshot
    if epoch is not None:
        snapshot_path = run_dir / f'epoch_{epoch:06d}.npz'
        if not snapshot_path.exists():
            return jsonify({'error': f'Epoch {epoch} not found'}), 404
    else:
        snapshot_path = npz_files[-1]

    try:
        data = np.load(snapshot_path, allow_pickle=True)

        # Get latent trajectory
        latent = None
        if 'latent_trajectory' in data.keys():
            latent = data['latent_trajectory']
        elif 'latent_states' in data.keys():
            latent = data['latent_states']

        if latent is None:
            return jsonify({'error': 'No latent data found'}), 404

        result = {
            'run_name': run_name,
            'epoch': epoch or int(snapshot_path.stem.split('_')[-1]),
            'density_field': compute_latent_density_field(latent, resolution),
            'gaussian_splats': compute_gaussian_splats(latent)
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/fractal/<path:run_name>')
def compute_fractal(run_name: str):
    """Compute fractal dimension and related metrics."""
    epoch = request.args.get('epoch', type=int)

    run_dir = TENSOR_SNAPSHOTS_DIR / run_name
    if not run_dir.exists():
        return jsonify({'error': f'Run not found: {run_name}'}), 404

    npz_files = sorted(run_dir.glob('*.npz'))
    if not npz_files:
        return jsonify({'error': 'No snapshots found'}), 404

    if epoch is not None:
        snapshot_path = run_dir / f'epoch_{epoch:06d}.npz'
        if not snapshot_path.exists():
            return jsonify({'error': f'Epoch {epoch} not found'}), 404
    else:
        snapshot_path = npz_files[-1]

    try:
        data = np.load(snapshot_path, allow_pickle=True)

        latent = None
        if 'latent_trajectory' in data.keys():
            latent = data['latent_trajectory']
        elif 'latent_states' in data.keys():
            latent = data['latent_states']

        if latent is None:
            return jsonify({'error': 'No latent data found'}), 404

        result = {
            'run_name': run_name,
            'epoch': epoch or int(snapshot_path.stem.split('_')[-1]),
            'correlation_dimension': compute_correlation_dimension(latent),
            'box_counting_dimension': compute_box_counting_dimension(latent),
            'recurrence_plot': compute_recurrence_plot(latent)
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/reference-attractors')
def get_reference_attractors():
    """Get reference attractors for comparison."""
    clifford_params = {
        'a': request.args.get('a', type=float, default=-1.4),
        'b': request.args.get('b', type=float, default=1.6),
        'c': request.args.get('c', type=float, default=1.0),
        'd': request.args.get('d', type=float, default=0.7)
    }

    return jsonify({
        'clifford': generate_clifford_attractor(**clifford_params),
        'lorenz': generate_lorenz_attractor()
    })


if __name__ == '__main__':
    print("DS78 Shroom Engine starting on http://0.0.0.0:5078")
    print(f"Tensor snapshots directory: {TENSOR_SNAPSHOTS_DIR}")
    app.run(host='0.0.0.0', port=5078, debug=False)
