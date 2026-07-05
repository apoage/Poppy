#!/usr/bin/env python3
"""
e-dojo_v1.py - Easy Mode Dual-Branch Orchestrator

TASK_667: Interleaved Dual-Mastery Training

Monitors two training branches (Bella + Synthetic), automatically merges
based on divergence metrics, and manages curriculum progression.

Research-validated approach:
- Fisher-weighted merging using Adam's v (Squisher shortcut)
- Adaptive merge scheduling based on loss barrier
- Experience replay for anti-forgetting

Usage:
    python e-dojo_v1.py --branch-a jax_ckpt_easy_A --branch-b jax_ckpt_easy_B

    # With auto-launch
    python e-dojo_v1.py --auto-launch --preset large --epochs 500
"""

import os
import sys
import time
import json
import argparse
import subprocess
import signal
import gc
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import threading
from queue import Queue
import pickle

# CRITICAL: Force orchestrator to CPU-only BEFORE any JAX import
# This prevents the orchestrator from claiming GPU VRAM
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from tools.orchestration.merge_utils import (
    fed_avg, fisher_merge, ties_merge, performance_weighted_merge,
    merge_optimizer_state, extract_fisher_from_optimizer,
    load_checkpoint, save_checkpoint
)
from tools.orchestration.divergence_metrics import (
    cosine_similarity, quick_barrier_check,
    compute_divergence_report
)


# =============================================================================
# Tiered Checkpoint Management (TASK_144)
# =============================================================================

def cleanup_merged_checkpoints(merged_dir: Path, keep_recent: int = 5, milestone_interval: int = 10):
    """
    TASK_144: Tiered retention for merged checkpoints.

    Keeps:
    - Last N merges (rolling window)
    - Milestones every 10th merge (permanent)
    - merged_latest.pkl (never deleted, it's a copy)

    Deletes intermediate merges to keep disk usage under control.
    """
    files = list(merged_dir.glob("merged_*.pkl"))
    # Exclude merged_latest.pkl (it's a copy, not a versioned file)
    files = [f for f in files if f.name != "merged_latest.pkl"]

    if not files:
        return

    # Map filename to merge count: merged_0001_timestamp.pkl -> 1
    merge_map = {}
    for f in files:
        parts = f.name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            merge_map[int(parts[1])] = f

    if not merge_map:
        return

    # Identify merges to keep
    sorted_counts = sorted(merge_map.keys())
    latest_counts = set(sorted_counts[-keep_recent:])

    # Delete non-retained files
    deleted_count = 0
    for count, file_path in merge_map.items():
        # Keep if: recent OR milestone (every 10th merge)
        if count not in latest_counts and count % milestone_interval != 0:
            file_path.unlink()
            deleted_count += 1

    if deleted_count > 0:
        print(f"[DOJO] Pruned {deleted_count} intermediate merge checkpoints")


# =============================================================================
# Configuration
# =============================================================================

class DojoConfig:
    """Configuration for e-dojo orchestrator."""

    def __init__(
        self,
        branch_a_dir: str = "jax_ckpt_easy_A",
        branch_b_dir: str = "jax_ckpt_easy_B",
        merged_dir: str = "jax_ckpt_easy_merged",
        log_dir: str = "logs/e-dojo",

        # Merge schedule (in steps, 400 steps = 1 epoch)
        min_steps_between_merge: int = 8000,   # 20 epochs minimum (hardening time)
        max_steps_between_merge: int = 20000,  # 50 epochs maximum
        initial_merge_interval: int = 12000,   # 30 epochs initial
        steps_per_epoch: int = 400,

        # Divergence thresholds
        cosine_safe: float = 0.99,
        cosine_normal: float = 0.95,
        cosine_warning: float = 0.90,
        cosine_brake: float = 0.85,    # Divergence brake - restart instead of merge
        barrier_safe: float = 0.01,
        barrier_normal: float = 0.02,
        barrier_warning: float = 0.05,

        # Progression
        level_advancement_threshold: float = 0.70,
        max_initial_level: int = 4,           # Cap levels at L0-4 during early training
        advanced_levels_after_epoch: int = 250,  # L5+ only after AR stabilization (~100 epochs post-E150)

        # Training phases (fraction of total training)
        phase_early_end: float = 0.10,
        phase_mid_end: float = 0.70,
    ):
        self.branch_a_dir = Path(branch_a_dir).resolve()
        self.branch_b_dir = Path(branch_b_dir).resolve()
        self.merged_dir = Path(merged_dir).resolve()
        self.log_dir = Path(log_dir).resolve()

        self.min_steps_between_merge = min_steps_between_merge
        self.max_steps_between_merge = max_steps_between_merge
        self.initial_merge_interval = initial_merge_interval
        self.steps_per_epoch = steps_per_epoch

        self.cosine_safe = cosine_safe
        self.cosine_normal = cosine_normal
        self.cosine_warning = cosine_warning
        self.cosine_brake = cosine_brake
        self.barrier_safe = barrier_safe
        self.barrier_normal = barrier_normal
        self.barrier_warning = barrier_warning

        self.level_advancement_threshold = level_advancement_threshold
        self.max_initial_level = max_initial_level
        self.advanced_levels_after_epoch = advanced_levels_after_epoch
        self.phase_early_end = phase_early_end
        self.phase_mid_end = phase_mid_end

        # Create directories
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# State Management
# =============================================================================

class DojoState:
    """Mutable state for the orchestrator."""

    def __init__(self):
        self.total_epochs = 0
        self.current_level = 0
        self.steps_since_merge = 0
        self.current_merge_interval = 8000  # 20 epochs
        self.merge_count = 0
        self.epoch_at_last_merge = 0  # TASK_672: Track epoch for proper step counting

        self.last_cosine = 1.0
        self.last_barrier = 0.0
        self.last_agreement_a = 0.0
        self.last_agreement_b = 0.0
        self.ar_max_blend = 1.0  # Full AR from the start

        # Per-branch train/val for adaptive AR check
        self.last_train_agree_a = 0.0
        self.last_val_agree_a = 0.0
        self.last_train_agree_b = 0.0
        self.last_val_agree_b = 0.0

        self.history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_epochs": self.total_epochs,
            "current_level": self.current_level,
            "steps_since_merge": self.steps_since_merge,
            "current_merge_interval": self.current_merge_interval,
            "merge_count": self.merge_count,
            "epoch_at_last_merge": self.epoch_at_last_merge,
            "last_cosine": self.last_cosine,
            "last_barrier": self.last_barrier,
            "last_agreement_a": self.last_agreement_a,
            "last_agreement_b": self.last_agreement_b,
            "ar_max_blend": self.ar_max_blend,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DojoState":
        state = cls()
        for k, v in d.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# =============================================================================
# Core Orchestrator
# =============================================================================

# =============================================================================
# TASK_698: Historical Schedule Manager
# =============================================================================

class HistoricalScheduleManager:
    """
    Manages historical training period cycling for TASK_698.

    Loads training_schedule.json, filters valid periods by DB max date,
    flattens into runs with cyclic asset padding to 64, manages cursor.
    """

    N_ASSETS = 64

    def __init__(self, schedule_path: str, db_path: str, benchmark_periods_path: str = None):
        self.db_path = db_path
        self.runs = []
        self.cursor = 0

        # Load schedule
        with open(schedule_path) as f:
            schedule = json.load(f)

        # Get DB max date for filtering
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT MAX(timestamp) FROM ohlcv").fetchone()
        conn.close()
        db_max_ts = row[0] if row and row[0] else 0
        from datetime import datetime as _dt, timezone as _tz
        db_max_date = _dt.fromtimestamp(db_max_ts, tz=_tz.utc).strftime('%Y-%m-%d')
        print(f"[HIST] DB max date: {db_max_date} (ts={db_max_ts})")

        # Filter and flatten periods into runs
        # Schedule format: periods[] with start/end (timestamps or dates) + available_assets/assets
        periods_list = schedule.get('periods', schedule if isinstance(schedule, list) else [])
        for period in periods_list:
            # Handle both timestamp and date formats
            raw_start = period.get('start_date', period.get('start', ''))
            raw_end = period.get('end_date', period.get('end', ''))

            # Convert timestamps to dates if needed
            if isinstance(raw_start, (int, float)):
                start_date = _dt.fromtimestamp(int(raw_start), tz=_tz.utc).strftime('%Y-%m-%d')
            else:
                start_date = str(raw_start)
            if isinstance(raw_end, (int, float)):
                end_date = _dt.fromtimestamp(int(raw_end), tz=_tz.utc).strftime('%Y-%m-%d')
            else:
                end_date = str(raw_end)

            # Skip periods beyond DB data
            if end_date > db_max_date:
                print(f"[HIST] Skipping period {start_date}→{end_date} (beyond DB)")
                continue

            # Use training_runs if available (pre-computed asset subsets from map_training_data.py)
            training_runs = period.get('training_runs', [])
            all_assets = period.get('available_assets', period.get('assets', period.get('symbols', [])))

            if training_runs:
                for tr in training_runs:
                    run_assets = tr.get('assets', [])
                    if not run_assets:
                        continue
                    # Pad to N_ASSETS if needed (shouldn't be, runs are pre-padded to 64)
                    padded = list(run_assets)
                    while len(padded) < self.N_ASSETS:
                        padded = padded + run_assets[:self.N_ASSETS - len(padded)]
                    padded = padded[:self.N_ASSETS]
                    self.runs.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'assets': padded,
                        'original_assets': len(all_assets) if all_assets else len(run_assets),
                    })
            elif all_assets:
                # Fallback: single run from available_assets
                padded = list(all_assets)
                while len(padded) < self.N_ASSETS:
                    padded = padded + all_assets[:self.N_ASSETS - len(padded)]
                padded = padded[:self.N_ASSETS]
                self.runs.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'assets': padded,
                    'original_assets': len(all_assets),
                })

        print(f"[HIST] Loaded {len(self.runs)} valid runs from {schedule_path}")

        # Validate no overlap with benchmark periods
        if benchmark_periods_path and os.path.exists(benchmark_periods_path):
            with open(benchmark_periods_path) as f:
                bench_data = json.load(f)
            # Handle both list and dict-with-periods-key formats
            if isinstance(bench_data, dict):
                bench_periods = bench_data.get('periods', [])
            else:
                bench_periods = bench_data
            # Compare at day level only (truncate timestamps to YYYY-MM-DD)
            def _to_date(s):
                return str(s)[:10]

            overlaps = []
            for bench in bench_periods:
                bench_start = _to_date(bench.get('start_date', bench.get('start', '')))
                bench_end = _to_date(bench.get('end_date', bench.get('end', '')))
                for run in self.runs:
                    run_start = _to_date(run['start_date'])
                    run_end = _to_date(run['end_date'])
                    # Strict overlap: training period must be strictly inside benchmark window
                    # Boundary touch (run_end == bench_start) is NOT overlap
                    if run_start < bench_end and run_end > bench_start:
                        overlaps.append(
                            f"  benchmark {bench_start}→{bench_end} vs training {run_start}→{run_end}"
                        )
            if overlaps:
                print(f"[HIST] WARNING: {len(overlaps)} training/benchmark overlaps detected:")
                for o in overlaps[:5]:
                    print(f"[HIST]   {o}")
                if len(overlaps) > 5:
                    print(f"[HIST]   ... and {len(overlaps) - 5} more")
                print(f"[HIST]   Skipping overlapping runs from schedule")
                # Remove overlapping runs
                clean_runs = []
                for run in self.runs:
                    run_start = _to_date(run['start_date'])
                    run_end = _to_date(run['end_date'])
                    dominated = False
                    for bench in bench_periods:
                        bs = _to_date(bench.get('start_date', bench.get('start', '')))
                        be = _to_date(bench.get('end_date', bench.get('end', '')))
                        if run_start < be and run_end > bs:
                            dominated = True
                            break
                    if not dominated:
                        clean_runs.append(run)
                removed = len(self.runs) - len(clean_runs)
                self.runs = clean_runs
                print(f"[HIST] Removed {removed} overlapping runs, {len(self.runs)} remaining")
            else:
                print(f"[HIST] No overlap with {len(bench_periods)} benchmark periods ✓")

    def get_current_run(self) -> Optional[Dict]:
        """Get current run (period + padded assets)."""
        if not self.runs:
            return None
        return self.runs[self.cursor % len(self.runs)]

    def advance(self):
        """Advance to next run."""
        self.cursor += 1
        if self.cursor >= len(self.runs):
            print(f"[HIST] Wrapped around — starting schedule from beginning")
            self.cursor = 0

    def get_state(self) -> Dict:
        return {'cursor': self.cursor}

    def load_state(self, state: Dict):
        self.cursor = state.get('cursor', 0)


class EasyDojo:
    """
    Easy Mode Dual-Branch Orchestrator.

    Monitors two training branches and coordinates merging.
    """

    def __init__(self, config: DojoConfig, launcher: 'TrainingLauncher' = None, single_mode: bool = False,
                 schedule_manager: 'HistoricalScheduleManager' = None):
        self.config = config
        self.state = DojoState()
        self.running = False
        self._stop_event = threading.Event()
        self.launcher = launcher  # For active level progression
        self.single_mode = single_mode  # TASK_675: Monitor-only, no merges
        self.schedule_manager = schedule_manager  # TASK_698: Historical period cycling

        # TASK_698: Historical mode state
        self._historical_block_count = 0
        self._runs_since_last_block = 0
        self._runs_per_block = 4  # Configurable via CLI
        self._benchmark_db_path = None  # Set from CLI

        # Load or initialize state
        self._load_state()

    def _load_state(self):
        """Load state from checkpoint if exists."""
        state_path = self.config.log_dir / "dojo_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                saved = json.load(f)
            self.state = DojoState.from_dict(saved)
            ar_msg = f", AR cap {self.state.ar_max_blend:.0%}" if self.state.ar_max_blend < 1.0 else ""
            print(f"[DOJO] Loaded state: {self.state.merge_count} merges, level {self.state.current_level}{ar_msg}")
            # TASK_698: Restore historical state
            hist = saved.get('historical', {})
            if hist and self.schedule_manager is not None:
                self.schedule_manager.cursor = hist.get('schedule_cursor', 0)
                self._historical_block_count = hist.get('block_count', 0)
                self._runs_since_last_block = hist.get('runs_since_last_block', 0)
                print(f"[DOJO] Historical state: block {self._historical_block_count}, "
                      f"cursor {self.schedule_manager.cursor}, "
                      f"runs_since_block {self._runs_since_last_block}")

        # Restore merge history (separate file, accumulates across restarts)
        history_path = self.config.log_dir / "merge_history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    self.state.history = json.load(f)
                print(f"[DOJO] Loaded merge history: {len(self.state.history)} entries")
            except (json.JSONDecodeError, Exception):
                self.state.history = []

    def _save_state(self):
        """Save state to checkpoint."""
        state_dict = self.state.to_dict()
        # TASK_698: Persist historical training state
        if self.schedule_manager is not None:
            state_dict['historical'] = {
                'schedule_cursor': self.schedule_manager.cursor,
                'block_count': self._historical_block_count,
                'runs_since_last_block': self._runs_since_last_block,
            }
        state_path = self.config.log_dir / "dojo_state.json"
        with open(state_path, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def _get_latest_checkpoint(self, branch_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a branch directory."""
        checkpoints = list(branch_dir.glob("**/*.pkl"))
        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def _get_training_phase(self) -> str:
        """Determine current training phase based on progress."""
        # Estimate progress based on merge count or epochs
        # This is a rough heuristic - could be improved with actual step tracking
        progress = min(1.0, self.state.merge_count / 50)  # Assume ~50 merges for full training

        if progress < self.config.phase_early_end:
            return "early"
        elif progress < self.config.phase_mid_end:
            return "mid"
        else:
            return "late"

    def _should_merge(self, cosine: float, barrier: float) -> Tuple[bool, str]:
        """
        Determine if branches should be merged.

        Returns:
            (should_merge, reason)
            Special case: "divergence_brake" means restart worse branch, don't merge
        """
        steps = self.state.steps_since_merge

        # DIVERGENCE BRAKE: If cosine is too low, branches are incompatible
        # Merging will just create "mush" - instead restart worse branch
        if cosine < self.config.cosine_brake:
            # TASK_148 FIX: Bootstrap deadlock - random weights always start diverged (~0.46)
            # Need to force first merge to create baseline, then brake can work
            if self.state.merge_count == 0:
                # If we reached the interval, FORCE a merge to create baseline
                if steps >= self.state.current_merge_interval:
                    return True, "initial_sync_merge"
                # Otherwise wait for interval (don't brake, just wait)
                return False, "fresh_start_wait_for_interval"
            # After first merge, brake is safe (merged_latest.pkl exists)
            return True, "divergence_brake"

        # Minimum interval not reached (skip for divergence checks)
        if steps < self.config.min_steps_between_merge:
            return False, "min_interval_not_reached"

        # Emergency merge (moderate divergence - still mergeable)
        if cosine < self.config.cosine_warning or barrier > self.config.barrier_warning:
            return True, "emergency_divergence"

        # Maximum interval reached
        if steps > self.config.max_steps_between_merge:
            return True, "max_interval_reached"

        # Normal merge (moderate divergence)
        if cosine < self.config.cosine_normal or barrier > self.config.barrier_normal:
            return True, "normal_divergence"

        # Current adaptive interval reached
        if steps >= self.state.current_merge_interval:
            return True, "interval_reached"

        return False, "no_merge_needed"

    def _select_merge_method(self, cosine: float, barrier: float, phase: str) -> str:
        """Select appropriate merge method based on metrics and phase.

        Method selection:
        - early phase: FedAvg (simple, stable)
        - moderate divergence (0.90-0.95): TIES (handles sign conflicts)
        - late phase: Fisher with normalization (importance weighting)
        - emergency: FedAvg (safe fallback, Fisher normalized now)
        """
        if phase == "early":
            return "fedavg"

        # TIES for moderate divergence - handles sign conflicts
        if self.config.cosine_brake < cosine < self.config.cosine_normal:
            return "ties"

        # Fisher for late phase (now normalized, safe to use)
        if phase == "late":
            return "fisher"

        return "fedavg"

    def _estimate_barrier(
        self,
        cosine: float,
        ckpt_a: Dict,
        ckpt_b: Dict
    ) -> float:
        """
        Estimate loss barrier heuristically (TASK_667).

        True barrier = loss(midpoint) - avg(endpoint losses)
        Requires validation data to compute properly.

        This heuristic uses:
        1. Cosine divergence: (1 - cosine) as proxy for weight distance
        2. Performance gap: |agreement_a - agreement_b| as proxy for functional divergence

        Returns approximate barrier in [0, 1] range.
        """
        # Get agreement metrics
        agree_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0.5)
        agree_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0.5)

        # Performance gap (higher gap = higher risk)
        perf_gap = abs(agree_a - agree_b)

        # Weight space divergence (1 - cosine)
        weight_div = 1.0 - cosine

        # Heuristic barrier estimate:
        # - Primarily driven by weight divergence
        # - Amplified by performance gap (different solutions)
        # Scale to approximate real barrier range [0, 0.1]
        barrier = weight_div * (1.0 + perf_gap) * 0.5

        return min(barrier, 1.0)  # Cap at 1.0

    def _adjust_merge_interval(self, cosine: float, barrier: float):
        """Adjust merge interval based on divergence metrics."""
        if cosine > self.config.cosine_safe and barrier < self.config.barrier_safe:
            # Very aligned - increase interval
            self.state.current_merge_interval = min(
                int(self.state.current_merge_interval * 1.5),
                self.config.max_steps_between_merge
            )
        elif cosine < self.config.cosine_normal or barrier > self.config.barrier_normal:
            # Diverging - decrease interval
            self.state.current_merge_interval = max(
                int(self.state.current_merge_interval * 0.75),
                self.config.min_steps_between_merge
            )

    def _perform_merge(self, ckpt_a: Dict, ckpt_b: Dict, method: str) -> Dict:
        """
        Perform the actual merge operation.

        Returns merged checkpoint dict.

        TASK_669: Added cleanup of intermediate arrays to prevent memory leaks.
        """
        params_a = ckpt_a['params']
        params_b = ckpt_b['params']

        score_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0.5)
        score_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0.5)

        # Track intermediates for cleanup
        fisher_a = None
        fisher_b = None

        try:
            if method == "fedavg":
                merged_params = fed_avg(params_a, params_b, 0.5)
            elif method == "fisher":
                # Use Adam's v as Fisher (Squisher shortcut)
                # Fisher is now normalized by default to prevent one branch dominating
                if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
                    try:
                        fisher_a = extract_fisher_from_optimizer(ckpt_a['opt_state'])
                        fisher_b = extract_fisher_from_optimizer(ckpt_b['opt_state'])
                        merged_params = fisher_merge(params_a, params_b, fisher_a, fisher_b,
                                                     normalize=True)  # Normalized!
                    except Exception as e:
                        print(f"[DOJO] Warning: Fisher extraction failed ({e}), using FedAvg")
                        merged_params = fed_avg(params_a, params_b, 0.5)
                else:
                    merged_params = fed_avg(params_a, params_b, 0.5)
            elif method == "ties":
                # TIES merging - handles sign conflicts between branches
                # Need base checkpoint (last merged) as params_init
                prev_merged_path = self._get_latest_checkpoint(self.config.merged_dir)
                if prev_merged_path:
                    try:
                        prev_merged = load_checkpoint(str(prev_merged_path), cpu=True)
                        params_init = prev_merged['params']
                        merged_params = ties_merge(params_a, params_b, params_init,
                                                   trim_fraction=0.2)
                        del prev_merged  # Cleanup
                        print(f"[DOJO] TIES merge using base: {prev_merged_path.name}")
                    except Exception as e:
                        print(f"[DOJO] Warning: TIES failed ({e}), using FedAvg")
                        merged_params = fed_avg(params_a, params_b, 0.5)
                else:
                    print(f"[DOJO] Warning: No base checkpoint for TIES, using FedAvg")
                    merged_params = fed_avg(params_a, params_b, 0.5)
            elif method == "performance":
                merged_params = performance_weighted_merge(params_a, params_b, score_a, score_b)
            else:
                merged_params = fed_avg(params_a, params_b, 0.5)

            # Handle optimizer state
            merged_opt_components = None
            if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
                try:
                    merged_nu, merged_mu = merge_optimizer_state(
                        ckpt_a['opt_state'],
                        ckpt_b['opt_state'],
                        reset_momentum=True
                    )
                    merged_opt_components = {'nu': merged_nu, 'mu': merged_mu}
                except Exception as e:
                    print(f"[DOJO] Warning: Optimizer merge failed ({e})")

            # Create merged checkpoint
            merged_ckpt = {
                'params': merged_params,
                'epoch': max(ckpt_a.get('epoch', 0), ckpt_b.get('epoch', 0)),
                'metrics': {
                    'source': 'merged',
                    'method': method,
                    'merge_count': self.state.merge_count + 1,
                    'a_agreement': score_a,
                    'b_agreement': score_b,
                },
                'config': ckpt_a.get('config', {}),
            }

            if merged_opt_components:
                merged_ckpt['merged_opt_components'] = merged_opt_components

            return merged_ckpt

        finally:
            # TASK_669: Cleanup intermediate Fisher arrays
            if fisher_a is not None:
                del fisher_a
            if fisher_b is not None:
                del fisher_b

    def check_and_merge(self) -> Optional[Dict[str, Any]]:
        """
        Check if merge is needed and perform it.

        Returns merge report if merge was performed, None otherwise.

        TASK_669: Added proper cleanup to prevent memory leaks.
        """
        # Find latest checkpoints
        ckpt_a_path = self._get_latest_checkpoint(self.config.branch_a_dir)
        ckpt_b_path = self._get_latest_checkpoint(self.config.branch_b_dir)

        if not ckpt_a_path or not ckpt_b_path:
            return None

        # Variables for cleanup
        ckpt_a = None
        ckpt_b = None
        params_a = None
        params_b = None
        merged_ckpt = None
        report = None

        try:
            # Load checkpoints
            try:
                ckpt_a = load_checkpoint(str(ckpt_a_path), cpu=True)
                ckpt_b = load_checkpoint(str(ckpt_b_path), cpu=True)
            except Exception as e:
                print(f"[DOJO] Error loading checkpoints: {e}")
                return None

            params_a = ckpt_a['params']
            params_b = ckpt_b['params']

            # TASK_672: Compute actual steps from epochs (not poll count)
            epoch_a = ckpt_a.get('epoch', 0)
            epoch_b = ckpt_b.get('epoch', 0)
            current_epoch = max(epoch_a, epoch_b)
            self.state.total_epochs = current_epoch

            # Compute steps since last merge using actual epoch counts
            epochs_since_merge = current_epoch - self.state.epoch_at_last_merge
            self.state.steps_since_merge = epochs_since_merge * self.config.steps_per_epoch

            # Compute divergence metrics
            cosine = cosine_similarity(params_a, params_b)

            # Estimate barrier from metrics (TASK_667)
            # True barrier requires validation data; this is a heuristic proxy
            barrier = self._estimate_barrier(cosine, ckpt_a, ckpt_b)

            self.state.last_cosine = cosine
            self.state.last_barrier = barrier
            metrics_a = ckpt_a.get('metrics', {})
            metrics_b = ckpt_b.get('metrics', {})
            self.state.last_agreement_a = metrics_a.get('oracle_agreement', 0)
            self.state.last_agreement_b = metrics_b.get('oracle_agreement', 0)
            # Per-branch train/val for adaptive AR
            self.state.last_train_agree_a = metrics_a.get('oracle_agreement', 0)
            self.state.last_val_agree_a = metrics_a.get('val_oracle_agreement', 0)
            self.state.last_train_agree_b = metrics_b.get('oracle_agreement', 0)
            self.state.last_val_agree_b = metrics_b.get('val_oracle_agreement', 0)

            # Check if merge needed
            should_merge, reason = self._should_merge(cosine, barrier)

            if not should_merge:
                return None

            # DIVERGENCE BRAKE: Branches too diverged to merge
            # Restart the worse branch from last merged checkpoint instead
            if reason == "divergence_brake":
                print(f"\n[DOJO] {'='*50}")
                print(f"[DOJO] DIVERGENCE BRAKE ACTIVATED")
                print(f"[DOJO] Cosine {cosine:.3f} < {self.config.cosine_brake} (brake threshold)")
                print(f"[DOJO] Branches incompatible for merging - restarting worse branch")
                print(f"[DOJO] Agreement A: {self.state.last_agreement_a:.1%}, B: {self.state.last_agreement_b:.1%}")

                # Find checkpoint to restart from (prefer merged, fallback to better branch)
                merged_path = self._get_latest_checkpoint(self.config.merged_dir)

                # Restart worse branch from best available checkpoint
                if self.launcher is not None:
                    if self.state.last_agreement_a >= self.state.last_agreement_b:
                        # Branch A is better, restart Branch B
                        better_branch_dir = self.config.branch_a_dir
                        if "branch_B_bella_inv" in self.launcher.launch_params:
                            worse_branch = "branch_B_bella_inv"
                            print(f"[DOJO] Restarting Branch B (bella inverted) - agreement {self.state.last_agreement_b:.1%}")
                        else:
                            worse_branch = "branch_B_synth"
                            print(f"[DOJO] Restarting Branch B (synth) - agreement {self.state.last_agreement_b:.1%}")
                    else:
                        # Branch B is better, restart Branch A
                        better_branch_dir = self.config.branch_b_dir
                        worse_branch = "branch_A_bella"
                        print(f"[DOJO] Restarting Branch A (bella) - agreement {self.state.last_agreement_a:.1%}")

                    # TASK_142 FIX: Use better branch checkpoint if no merged checkpoint exists
                    if merged_path:
                        restart_path = merged_path
                        print(f"[DOJO] Using merged checkpoint: {merged_path.name}")
                    else:
                        restart_path = self._get_latest_checkpoint(better_branch_dir)
                        if restart_path:
                            print(f"[DOJO] No merged checkpoint - using better branch: {restart_path.name}")
                        else:
                            print(f"[DOJO] ERROR: No checkpoint available for restart")
                            return None

                    self.launcher.restart_single_branch(worse_branch, str(restart_path))
                else:
                    print("[DOJO] WARNING: No launcher - cannot restart branch")

                print(f"[DOJO] {'='*50}\n")

                # Log as brake event (not a merge)
                return {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "event": "divergence_brake",
                    "cosine": cosine,
                    "restarted_branch": worse_branch if self.launcher else "none",
                }

            # Determine merge method
            phase = self._get_training_phase()
            method = self._select_merge_method(cosine, barrier, phase)

            print(f"\n[DOJO] {'='*50}")
            print(f"[DOJO] MERGE #{self.state.merge_count + 1}")
            print(f"[DOJO] Reason: {reason}")
            print(f"[DOJO] Phase: {phase}, Method: {method}")
            print(f"[DOJO] Cosine: {cosine:.4f}, Barrier: {barrier:.4f}")
            print(f"[DOJO] Agreement A: {self.state.last_agreement_a:.1%}, B: {self.state.last_agreement_b:.1%}")

            # Perform merge
            merged_ckpt = self._perform_merge(ckpt_a, ckpt_b, method)

            # Free source checkpoints before saving (TASK_669)
            del ckpt_a, ckpt_b, params_a, params_b
            ckpt_a = ckpt_b = params_a = params_b = None

            # Save merged checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_path = self.config.merged_dir / f"merged_{self.state.merge_count + 1:04d}_{timestamp}.pkl"
            save_checkpoint(merged_ckpt, str(merged_path))

            # Copy to latest (avoid double pickle serialization - TASK_669)
            latest_path = self.config.merged_dir / "merged_latest.pkl"
            shutil.copy2(str(merged_path), str(latest_path))

            # TASK_144: Tiered cleanup (keep last 5 + every 10th milestone)
            cleanup_merged_checkpoints(self.config.merged_dir, keep_recent=5, milestone_interval=10)

            # Free merged checkpoint after save (TASK_669)
            del merged_ckpt
            merged_ckpt = None

            # Update state
            self.state.merge_count += 1
            self.state.steps_since_merge = 0
            self.state.epoch_at_last_merge = current_epoch  # TASK_672: Track for step counting
            self._adjust_merge_interval(cosine, barrier)
            self._save_state()

            # Compute cross-validation proxy metrics (TASK_667)
            # Alignment = geometric mean of agreements (both need to be high)
            agree_a = self.state.last_agreement_a
            agree_b = self.state.last_agreement_b
            alignment = (agree_a * agree_b) ** 0.5 if agree_a > 0 and agree_b > 0 else 0.0
            # Generalization gap = difference in agreements (lower is better)
            gen_gap = abs(agree_a - agree_b)

            # Log to history
            report = {
                "timestamp": timestamp,
                "merge_count": self.state.merge_count,
                "reason": reason,
                "phase": phase,
                "method": method,
                "cosine": cosine,
                "barrier": barrier,
                "agreement_a": agree_a,
                "agreement_b": agree_b,
                "alignment": alignment,
                "generalization_gap": gen_gap,
                "merged_path": str(merged_path),
            }
            self.state.history.append(report)

            print(f"[DOJO] Alignment: {alignment:.1%}, Gen Gap: {gen_gap:.1%}")

            # Save history
            history_path = self.config.log_dir / "merge_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.state.history, f, indent=2)

            print(f"[DOJO] Saved to: {merged_path}")
            print(f"[DOJO] Next merge interval: {self.state.current_merge_interval} steps")

            # TASK_672: Restart branches to load merged weights
            if self.launcher is not None:
                # TASK_698: Advance historical period before restart so new period is picked up
                if self.schedule_manager is not None:
                    self.schedule_manager.advance()
                    self._runs_since_last_block += 1
                    run = self.schedule_manager.get_current_run()
                    if run:
                        print(f"[DOJO] Historical: advanced to {run['start_date']}→{run['end_date']} "
                              f"({run['original_assets']} assets)")

                self.launcher.restart_both_with_checkpoint(str(merged_path))

                # TASK_698: Block completion check — benchmark + named checkpoint
                if self.schedule_manager is not None and self._runs_since_last_block >= self._runs_per_block:
                    self._handle_block_completion(str(merged_path))
            else:
                print("[DOJO] WARNING: No launcher - branches will NOT load merged weights!")

            # Check adaptive AR advancement
            self._check_ar_advancement()

            # Check level advancement (TASK_667)
            self._check_level_advancement()

            print(f"[DOJO] {'='*50}\n")

            return report

        finally:
            # TASK_669: Aggressive cleanup to prevent memory leaks
            # Delete any remaining references
            if ckpt_a is not None:
                del ckpt_a
            if ckpt_b is not None:
                del ckpt_b
            if params_a is not None:
                del params_a
            if params_b is not None:
                del params_b
            if merged_ckpt is not None:
                del merged_ckpt

            # Clear JAX caches and force garbage collection
            jax.clear_caches()
            gc.collect()

            # Log memory cleanup
            if report is not None:
                print(f"[DOJO] Memory cleanup complete (gc.collect + jax.clear_caches)")

    def _check_ar_advancement(self):
        """
        Adaptive AR blend: +1% if BOTH branches are stable, +3% if historical > 80%.

        Conditions (must hold for both branches):
        - avg(train_agree, val_agree) >= 70%
        - |train_agree - val_agree| <= 5%

        Accelerated advancement when historical branch agreement > 80%:
        - Overfitting risk on finite historical data requires faster AR ramp
        - +3% per merge instead of +1%
        """
        if self.state.ar_max_blend >= 1.0:
            return  # Already at full AR

        # Check Branch A
        avg_a = (self.state.last_train_agree_a + self.state.last_val_agree_a) / 2
        delta_a = abs(self.state.last_train_agree_a - self.state.last_val_agree_a)
        a_stable = avg_a >= 0.70 and delta_a <= 0.05

        # Check Branch B
        avg_b = (self.state.last_train_agree_b + self.state.last_val_agree_b) / 2
        delta_b = abs(self.state.last_train_agree_b - self.state.last_val_agree_b)
        b_stable = avg_b >= 0.70 and delta_b <= 0.05

        old_cap = self.state.ar_max_blend

        # Accelerate AR when historical agreement is high (overfitting prevention)
        historical_high = (self.schedule_manager is not None and avg_a > 0.80)
        step = 0.03 if historical_high else 0.01

        if a_stable and b_stable:
            new_cap = min(old_cap + step, 1.0)
            self.state.ar_max_blend = new_cap
            accel = " (accelerated — historical >80%)" if historical_high else ""
            print(f"[DOJO] ► AR adaptive: {old_cap:.0%} → {new_cap:.0%}{accel}")
            print(f"[DOJO]   A: avg={avg_a:.1%} delta={delta_a:.1%} ✓  "
                  f"B: avg={avg_b:.1%} delta={delta_b:.1%} ✓")
            if new_cap >= 1.0:
                print(f"[DOJO]   ★ Full AR reached!")
        elif historical_high and a_stable:
            # Historical is high but Bella isn't stable — still advance AR to prevent overfitting
            new_cap = min(old_cap + 0.02, 1.0)
            self.state.ar_max_blend = new_cap
            print(f"[DOJO] ► AR forced advance: {old_cap:.0%} → {new_cap:.0%} (historical >80%, Bella unstable)")
            print(f"[DOJO]   A: avg={avg_a:.1%} delta={delta_a:.1%} ✓  "
                  f"B: avg={avg_b:.1%} delta={delta_b:.1%} ✗")
            if new_cap >= 1.0:
                print(f"[DOJO]   ★ Full AR reached!")
        else:
            reasons = []
            if not a_stable:
                reasons.append(f"A: avg={avg_a:.1%} delta={delta_a:.1%}")
            if not b_stable:
                reasons.append(f"B: avg={avg_b:.1%} delta={delta_b:.1%}")
            print(f"[DOJO] ► AR holding at {old_cap:.0%} — {', '.join(reasons)}")

    def _check_level_advancement(self):
        """
        Check if training should advance to next level (TASK_667).

        Advancement criteria: agreement on current level > 70%
        Levels accumulate as a range (e.g., 0-3 → 0-4) instead of replacing.
        L5+ only unlocked after advanced_levels_after_epoch (AR stabilization).
        """
        avg_agreement = 0.5 * (self.state.last_agreement_a + self.state.last_agreement_b)

        if avg_agreement >= self.config.level_advancement_threshold:
            old_level = self.state.current_level

            # Cap based on epoch: L0-4 initially, L5+ after AR stabilization
            current_epoch = getattr(self.state, 'total_epochs', 0)
            if current_epoch < self.config.advanced_levels_after_epoch:
                max_level = self.config.max_initial_level
            else:
                max_level = 14

            new_level = min(old_level + 1, max_level)

            if new_level > old_level:
                self.state.current_level = new_level
                # Accumulate as range: "0-N" instead of just "N"
                level_range = f"0-{new_level}" if new_level > 0 else "0"
                print(f"[DOJO] ★ LEVEL ADVANCEMENT: L{old_level} → L{new_level} (training on L{level_range})")
                print(f"[DOJO]   Agreement: {avg_agreement:.1%} >= {self.config.level_advancement_threshold:.1%}")
                if new_level == max_level and max_level < 14:
                    print(f"[DOJO]   Capped at L{max_level} until E{self.config.advanced_levels_after_epoch}")

                # Log advancement event
                advancement_log = self.config.log_dir / "level_advancement.json"
                events = []
                if advancement_log.exists():
                    with open(advancement_log, 'r') as f:
                        events = json.load(f)
                events.append({
                    "timestamp": datetime.now().isoformat(),
                    "from_level": old_level,
                    "to_level": new_level,
                    "level_range": level_range,
                    "agreement": avg_agreement,
                    "merge_count": self.state.merge_count,
                    "epoch": current_epoch,
                })
                with open(advancement_log, 'w') as f:
                    json.dump(events, f, indent=2)

                # Active level progression: restart synthetic branch with level range
                if self.launcher is not None:
                    print(f"[DOJO]   Restarting synthetic branch with L{level_range}...")
                    self.launcher.restart_branch(
                        name="branch_B_synth",
                        new_levels=level_range,
                    )
                else:
                    print(f"[DOJO]   NOTE: Restart synthetic branch with --levels {level_range} for next run")
            elif new_level == old_level == max_level and max_level < 14:
                pass  # Already at cap, silently skip

    def _handle_block_completion(self, merged_path: str):
        """TASK_698: Handle historical block completion — save named checkpoint + run benchmark."""
        self._historical_block_count += 1
        self._runs_since_last_block = 0
        block_name = f"foxglove_H{self._historical_block_count:02d}"

        # Copy merged checkpoint to named block checkpoint
        block_ckpt = self.config.merged_dir / f"{block_name}.pkl"
        shutil.copy2(merged_path, str(block_ckpt))
        print(f"[DOJO] ★ BLOCK {block_name} complete — saved {block_ckpt.name}")

        # Run benchmark (async — training continues)
        self._run_block_benchmark(str(block_ckpt), block_name)

    def _run_block_benchmark(self, checkpoint_path: str, block_name: str):
        """TASK_698: Run benchmark_easy.py on block checkpoint."""
        if not self._benchmark_db_path:
            print(f"[DOJO] Skipping benchmark — no --db-path set")
            return

        project_root = Path(__file__).parent.parent.parent.resolve()
        benchmark_script = project_root / "tools" / "benchmark_easy.py"
        if not benchmark_script.exists():
            print(f"[DOJO] Skipping benchmark — {benchmark_script} not found")
            return

        bench_dir = project_root / "logs" / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)
        json_path = bench_dir / f"{block_name}.json"

        cmd = [
            str(project_root / "trader_env" / "bin" / "python"),
            str(benchmark_script),
            "--checkpoint", checkpoint_path,
            "--db", self._benchmark_db_path,
            "--json", str(json_path),
            "--quiet",
        ]

        print(f"[DOJO] Running benchmark for {block_name}...")
        env = os.environ.copy()
        # Use RTX 3090 (CUDA 0) for benchmark — both branches paused during merge
        env['CUDA_VISIBLE_DEVICES'] = '0'
        env.pop('JAX_PLATFORMS', None)

        log_file = self.config.log_dir / f"benchmark_{block_name}.log"
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    cwd=project_root, env=env, timeout=600,  # 10 min max
                )
            if result.returncode == 0 and json_path.exists():
                with open(json_path) as f:
                    bench_results = json.load(f)
                summary = bench_results.get('summary', {})
                print(f"[DOJO] Benchmark {block_name}: "
                      f"bal_acc={summary.get('avg_balanced_accuracy', 0):.1%}, "
                      f"alpha={summary.get('avg_alpha', 0):.2%}, "
                      f"trade_rate={summary.get('avg_trade_rate', 0):.1%}, "
                      f"sharpe={summary.get('avg_sharpe', 0):.2f}")

                # Append to CSV tracker
                csv_path = bench_dir / "foxglove_benchmark.csv"
                csv_exists = csv_path.exists()
                with open(csv_path, 'a') as f:
                    if not csv_exists:
                        f.write("timestamp,block,merge_count,avg_balanced_acc,avg_alpha,"
                                "avg_trade_rate,avg_sharpe,positive_alpha_count\n")
                    f.write(f"{datetime.now().isoformat()},{block_name},"
                            f"{self.state.merge_count},"
                            f"{summary.get('avg_balanced_accuracy', 0):.4f},"
                            f"{summary.get('avg_alpha', 0):.4f},"
                            f"{summary.get('avg_trade_rate', 0):.4f},"
                            f"{summary.get('avg_sharpe', 0):.4f},"
                            f"{summary.get('positive_alpha_count', 0)}\n")
            else:
                print(f"[DOJO] Benchmark failed (code {result.returncode}). See {log_file}")
        except subprocess.TimeoutExpired:
            print(f"[DOJO] Benchmark timed out after 600s")
        except Exception as e:
            print(f"[DOJO] Benchmark error: {e}")

    def run_loop(self, poll_interval: float = 30.0):
        """
        Run the orchestrator loop.

        Polls for new checkpoints and merges when appropriate.
        Automatically stops when all training branches complete.
        In single_mode: monitor only, no merge checks.

        Args:
            poll_interval: Seconds between checkpoint checks
        """
        self.running = True
        mode_str = "SINGLE (monitor-only)" if self.single_mode else "DUAL (merge-enabled)"
        print(f"[DOJO] Starting orchestrator loop — {mode_str} (poll every {poll_interval}s)")
        print(f"[DOJO] Branch A: {self.config.branch_a_dir}")
        if not self.single_mode:
            print(f"[DOJO] Branch B: {self.config.branch_b_dir}")
            print(f"[DOJO] Output: {self.config.merged_dir}")

        try:
            while self.running and not self._stop_event.is_set():
                # Check if training branches have completed
                if self.launcher is not None:
                    health = self.launcher.check_health()
                    all_done = all(status in ("completed", "failed") for status in health.values())

                    if all_done and len(health) > 0:
                        print(f"\n[DOJO] {'='*50}")
                        print("[DOJO] TRAINING COMPLETED")
                        for name, status in health.items():
                            print(f"[DOJO]   {name}: {status}")
                        if not self.single_mode:
                            print(f"[DOJO] Total merges: {self.state.merge_count}")
                        print(f"[DOJO] Final level: L{self.state.current_level}")
                        print(f"[DOJO] {'='*50}")
                        break

                if not self.single_mode:
                    report = self.check_and_merge()

                    if report:
                        # Brief pause after merge
                        time.sleep(5)
                        continue

                # Normal polling (or always in single mode)
                time.sleep(poll_interval)

                # TASK_672: Step counting now computed from epochs in check_and_merge()

        except KeyboardInterrupt:
            print("\n[DOJO] Interrupted by user")
        finally:
            self.running = False
            self._save_state()
            print("[DOJO] Orchestrator stopped")

    def stop(self):
        """Stop the orchestrator loop."""
        self._stop_event.set()
        self.running = False


# =============================================================================
# Training Launcher
# =============================================================================

class TrainingLauncher:
    """Launches and manages training subprocesses."""

    # TASK_153: Single source of truth for training defaults
    DEFAULT_PARAMS = {
        'preset': 'large',
        'epochs': 5000,
        'steps_per_epoch': 400,
        'batch_size': 1024,
        'n_envs': 4,        # TASK_153: Reduced from 32 (limits VRAM spike to ~0.7GB)
        'buffer_size': 16,  # TASK_153: Reduced from 32 (limits baseline to ~2.8GB)
        'vram_fraction': 0.45,
    }

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        self.python_path = str(self.project_root / "trader_env" / "bin" / "python")
        self.train_script = str(self.project_root / "train_easy.py")
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_dir = self.project_root / "logs" / "e-dojo"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Store launch params for restarts (TASK_672)
        self.launch_params: Dict[str, Dict] = {}

    @staticmethod
    def _generate_seed() -> int:
        """Generate a random seed for market diversity (limited range for safety)."""
        import random
        return random.randint(0, 99999)  # 0-99999: readable, safe from overflow

    def launch_branch(
        self,
        name: str,
        output_dir: str,
        use_bella: bool = True,
        synthetic_levels: str = "0",
        checkpoint: str = None,
        auto_resume: bool = True,
        invert_regime: bool = False,
        seed: int = None,  # None = random seed per launch (market diversity)
        gpu_id: int = None,  # None = inherit parent env, int = CUDA_VISIBLE_DEVICES
        **kwargs,  # Training params (preset, epochs, n_envs, buffer_size, etc.)
    ) -> subprocess.Popen:
        """Launch a training branch subprocess."""
        # Merge kwargs with defaults
        params = {**self.DEFAULT_PARAMS, **kwargs}

        # Generate random seed if not specified (breaks recurring market patterns)
        if seed is None:
            seed = self._generate_seed()
            print(f"[DOJO] Generated random seed for {name}: {seed}")

        cmd = [
            self.python_path, self.train_script,
            "--preset", params['preset'],
            "--epochs", str(params['epochs']),
            "--steps-per-epoch", str(params['steps_per_epoch']),
            "--batch-size", str(params['batch_size']),
            "--n-envs", str(params['n_envs']),
            "--buffer-size", str(params['buffer_size']),
            "--name", name,
            "--output-dir", str(self.project_root / output_dir),
            "--no-latent-cache",  # TASK_670B: Disable latent caching to prevent drift
            "--seed", str(seed),
        ]

        # Pass adaptive AR cap from orchestrator state
        if hasattr(self, '_dojo_state') and self._dojo_state is not None:
            ar_cap = self._dojo_state.ar_max_blend
            if ar_cap < 1.0:
                cmd.extend(["--ar-max-blend", f"{ar_cap:.2f}"])

        if use_bella:
            cmd.append("--bella")
            if invert_regime:
                cmd.append("--invert-regime")
        else:
            cmd.extend(["--levels", synthetic_levels])

        # TASK_707: Multi-period cycling (overrides single-period TASK_698)
        periods_file = kwargs.get('periods_file', None)
        if periods_file:
            cmd.extend(["--db-path", kwargs.get('db_path', 'trading_data.db')])
            cmd.extend(["--periods-file", periods_file])
            cmd.extend(["--lr", kwargs.get('lr', '5e-5')])
        else:
            # TASK_698: Single-period historical data training args
            hist_args = kwargs.get('historical_args', None)
            if hist_args:
                cmd.extend(["--db-path", hist_args['db_path']])
                cmd.extend(["--start-date", hist_args['start_date']])
                cmd.extend(["--end-date", hist_args['end_date']])
                cmd.extend(["--assets-file", hist_args['assets_file']])
                cmd.extend(["--lr", hist_args.get('lr', '5e-5')])

        # TASK_673: Multipass oracle for soft policy targets
        if kwargs.get('oracle_multipass', False):
            cmd.append("--oracle-multipass")

        # TASK_706: Periodic benchmark on separate GPU
        if kwargs.get('benchmark_every', 0) > 0:
            cmd.extend(["--benchmark-every", str(kwargs['benchmark_every'])])
            cmd.extend(["--benchmark-gpu", str(kwargs.get('benchmark_gpu', 1))])

        # Checkpoint loading - explicit or auto-resume from latest
        if checkpoint:
            cmd.extend(["--checkpoint", str(checkpoint)])
        elif auto_resume:
            # TASK_707: Single-branch mode — prefer branch's own checkpoint over merged
            is_single_branch = kwargs.get('periods_file') is not None or (
                kwargs.get('historical_args') is not None and not any(
                    k.startswith('branch_B') for k in (getattr(self, 'launch_params', {}) or {})
                )
            )

            best_ckpt = None
            best_source = None

            if is_single_branch:
                # Single-branch: check own output dir first
                own_dir = self.project_root / output_dir / name
                own_ckpts = sorted(own_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime) if own_dir.exists() else []
                if own_ckpts:
                    best_ckpt = own_ckpts[-1]
                    best_source = f"BRANCH ({name})"

            if not best_ckpt:
                # Fallback: merged checkpoint (for dual-branch or first launch)
                merged_dir = self.project_root / "jax_ckpt_easy_merged"
                merged_ckpts = sorted(merged_dir.glob("merged_*.pkl"), key=lambda p: p.stat().st_mtime) if merged_dir.exists() else []

                if merged_ckpts:
                    best_ckpt = merged_ckpts[-1]
                    best_source = "MERGED"

            if not best_ckpt:
                # TASK_142 FIX: Basin synchronization - both branches must use SAME checkpoint
                branch_a_dir = self.project_root / "jax_ckpt_easy_A" / "branch_A_bella"
                branch_b_dir = self.project_root / "jax_ckpt_easy_B"
                branch_b_bella_dir = branch_b_dir / "branch_B_bella_inv"
                branch_b_synth_dir = branch_b_dir / "branch_B_synth"

                branch_a_ckpts = sorted(branch_a_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime) if branch_a_dir.exists() else []
                branch_b_ckpts = []
                if branch_b_bella_dir.exists():
                    branch_b_ckpts = sorted(branch_b_bella_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
                if not branch_b_ckpts and branch_b_synth_dir.exists():
                    branch_b_ckpts = sorted(branch_b_synth_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime)

                all_branch_ckpts = []
                if branch_a_ckpts:
                    all_branch_ckpts.append((branch_a_ckpts[-1], "branch_A"))
                if branch_b_ckpts:
                    all_branch_ckpts.append((branch_b_ckpts[-1], "branch_B"))

                if all_branch_ckpts:
                    all_branch_ckpts.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
                    best_ckpt, best_source = all_branch_ckpts[0]
                    print(f"[DOJO] BASIN SYNC: Using {best_source} checkpoint for all branches")

            if best_ckpt:
                cmd.extend(["--checkpoint", str(best_ckpt)])
                print(f"[DOJO] Auto-resuming from {best_source}: {best_ckpt.name}")

        # VRAM control for dual-branch training (TASK_667)
        env = os.environ.copy()
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(params['vram_fraction'])
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        env['PYTHONUNBUFFERED'] = '1'  # Ensure prints appear immediately in logs
        env['TMPDIR'] = '/home/lukas/tmp'  # JAX PTX on SSD, not tmpfs (shared with other work)
        # Reset CPU-only flags from orchestrator so child sees GPU
        env.pop('JAX_PLATFORMS', None)
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        else:
            env.pop('CUDA_VISIBLE_DEVICES', None)  # Let child see all GPUs

        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        gpu_label = f" GPU {gpu_id}" if gpu_id is not None else ""
        print(f"[DOJO] Launching {name}{gpu_label} (VRAM: {params['vram_fraction']:.0%})")
        print(f"[DOJO] Log: {log_file}")

        # TASK_141 FIX: Keep file handle open for subprocess lifetime
        # The 'with' statement was closing the file immediately after Popen
        f = open(log_file, 'w')
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=self.project_root,
            env=env,
        )
        # Store file handle to keep it open (closed when process terminates)
        proc._log_file = f

        self.processes[name] = proc
        return proc

    # Per-GPU defaults: nvidia-smi 0 = CMP 90HX (10GB), nvidia-smi 1 = RTX 3090 (24GB)
    GPU_CONFIGS = {
        'dual_gpu': {
            # CUDA runtime order ≠ nvidia-smi order: CUDA 0=3090, CUDA 1=90HX
            'branch_a': {'gpu_id': 0, 'vram_fraction': 0.85, 'batch_size': 512, 'buffer_size': 16, 'n_envs': 4},  # 3090 (Bella + generator)
            'branch_b': {'gpu_id': 1, 'vram_fraction': 0.95, 'batch_size': 384, 'buffer_size': 8, 'n_envs': 4},   # 90HX (10GB, tight) — TASK_699: bumped from 2 for batched AR
        },
        'single_gpu': {
            'branch_a': {'gpu_id': None, 'vram_fraction': 0.45, 'batch_size': 1024, 'buffer_size': 16, 'n_envs': 4},
            'branch_b': {'gpu_id': None, 'vram_fraction': 0.45, 'batch_size': 1024, 'buffer_size': 16, 'n_envs': 4},
        },
    }

    def launch_dual_branches(
        self,
        synthetic_levels: str = "0",
        stagger_seconds: float = 14.0,
        auto_resume: bool = True,
        dual_bella: bool = False,
        gpu_mode: str = "dual_gpu",
        **kwargs,  # Training params (preset, epochs, etc.)
    ) -> Tuple[subprocess.Popen, subprocess.Popen]:
        """Launch both training branches with stagger and VRAM control.

        Args:
            dual_bella: If True, both branches use Bella with Branch B inverted.
                       If False (default), Branch A=Bella, Branch B=Synthetic.
            gpu_mode: "dual_gpu" = one branch per GPU, "single_gpu" = both on default GPU.
            **kwargs: Training params - see DEFAULT_PARAMS for available options.
        """
        gpu_cfg = self.GPU_CONFIGS.get(gpu_mode, self.GPU_CONFIGS['dual_gpu'])
        cfg_a = gpu_cfg['branch_a']
        cfg_b = gpu_cfg['branch_b']

        # Merge with defaults, then apply per-branch GPU overrides
        params = {**self.DEFAULT_PARAMS, **kwargs}
        params_a = {**params, **{k: v for k, v in cfg_a.items() if k != 'gpu_id'}}
        params_b = {**params, **{k: v for k, v in cfg_b.items() if k != 'gpu_id'}}

        print(f"\n[DOJO] {'='*50}")
        print(f"[DOJO] LAUNCHING DUAL-BRANCH TRAINING ({gpu_mode})")
        print(f"[DOJO] Preset: {params['preset']}, Epochs: {params['epochs']}")
        if dual_bella:
            print(f"[DOJO] Branch A: Bella (normal), Branch B: Bella (INVERTED)")
        else:
            print(f"[DOJO] Branch A: Bella, Branch B: Synthetic L{synthetic_levels}")
        if gpu_mode == 'dual_gpu':
            print(f"[DOJO] Branch A: GPU {cfg_a['gpu_id']} (3090), batch={params_a['batch_size']}, VRAM={params_a['vram_fraction']:.0%}")
            print(f"[DOJO] Branch B: GPU {cfg_b['gpu_id']} (90HX), batch={params_b['batch_size']}, VRAM={params_b['vram_fraction']:.0%}")
        else:
            print(f"[DOJO] Both on default GPU, batch={params_a['batch_size']}, VRAM={params_a['vram_fraction']:.0%}")
        print(f"[DOJO] Auto-resume: {auto_resume}")
        print(f"[DOJO] {'='*50}\n")

        # Branch A: Bella (always normal)
        # seed=None triggers random seed generation (market diversity per restart)
        proc_a = self.launch_branch(
            name="branch_A_bella",
            output_dir="jax_ckpt_easy_A",
            use_bella=True,
            auto_resume=auto_resume,
            invert_regime=False,
            gpu_id=cfg_a['gpu_id'],
            **params_a,
        )

        # Stagger (shorter when on separate GPUs — no VRAM contention)
        actual_stagger = stagger_seconds if gpu_mode == 'single_gpu' else 3.0
        print(f"[DOJO] Waiting {actual_stagger}s before launching Branch B...")
        time.sleep(actual_stagger)

        if dual_bella:
            # Branch B: Bella with inverted regime
            proc_b = self.launch_branch(
                name="branch_B_bella_inv",
                output_dir="jax_ckpt_easy_B",
                use_bella=True,
                auto_resume=auto_resume,
                invert_regime=True,
                gpu_id=cfg_b['gpu_id'],
                **params_b,
            )
            branch_b_name = "branch_B_bella_inv"
        else:
            # Branch B: Synthetic
            proc_b = self.launch_branch(
                name="branch_B_synth",
                output_dir="jax_ckpt_easy_B",
                use_bella=False,
                synthetic_levels=synthetic_levels,
                auto_resume=auto_resume,
                gpu_id=cfg_b['gpu_id'],
                **params_b,
            )
            branch_b_name = "branch_B_synth"

        print(f"\n[DOJO] Both branches launched. PIDs: A={proc_a.pid}, B={proc_b.pid}")

        # Store launch params for restarts (TASK_672)
        # Note: seed=None means random seed will be generated on each restart
        self.launch_params = {
            "branch_A_bella": {
                "output_dir": "jax_ckpt_easy_A",
                "use_bella": True,
                "invert_regime": False,
                "gpu_id": cfg_a['gpu_id'],
                **params_a,
            },
        }

        if dual_bella:
            self.launch_params["branch_B_bella_inv"] = {
                "output_dir": "jax_ckpt_easy_B",
                "use_bella": True,
                "invert_regime": True,
                "gpu_id": cfg_b['gpu_id'],
                **params_b,
            }
        else:
            self.launch_params["branch_B_synth"] = {
                "output_dir": "jax_ckpt_easy_B",
                "use_bella": False,
                "synthetic_levels": synthetic_levels,
                "gpu_id": cfg_b['gpu_id'],
                **params_b,
            }

        return proc_a, proc_b

    def launch_historical_branches(
        self,
        schedule_manager: 'HistoricalScheduleManager',
        db_path: str,
        stagger_seconds: float = 14.0,
        auto_resume: bool = True,
        gpu_mode: str = "dual_gpu",
        **kwargs,
    ) -> Tuple[subprocess.Popen, subprocess.Popen]:
        """TASK_698: Launch Branch A=Historical, Branch B=Bella.

        Branch A trains on historical data (RTX 3090).
        Branch B trains on Bella synthetic (CMP 90HX).
        """
        gpu_cfg = self.GPU_CONFIGS.get(gpu_mode, self.GPU_CONFIGS['dual_gpu'])
        cfg_a = gpu_cfg['branch_a']
        cfg_b = gpu_cfg['branch_b']

        params = {**self.DEFAULT_PARAMS, **kwargs}
        params_a = {**params, **{k: v for k, v in cfg_a.items() if k != 'gpu_id'}}
        params_b = {**params, **{k: v for k, v in cfg_b.items() if k != 'gpu_id'}}

        # Get current run from schedule
        run = schedule_manager.get_current_run()
        if not run:
            raise ValueError("No valid runs in schedule")

        # Write assets file for this run
        assets_file = self._write_assets_json(run['assets'], "branch_A_historical")

        hist_args = {
            'db_path': db_path,
            'start_date': run['start_date'],
            'end_date': run['end_date'],
            'assets_file': assets_file,
            'lr': '5e-5',
        }

        print(f"\n[DOJO] {'='*50}")
        print(f"[DOJO] LAUNCHING HISTORICAL DUAL-BRANCH TRAINING ({gpu_mode})")
        print(f"[DOJO] Preset: {params['preset']}, Epochs: {params['epochs']}")
        print(f"[DOJO] Branch A: HISTORICAL {run['start_date']}→{run['end_date']} "
              f"({run['original_assets']} assets, padded to 64)")
        print(f"[DOJO] Branch B: Bella (synthetic)")
        if gpu_mode == 'dual_gpu':
            print(f"[DOJO] Branch A: GPU {cfg_a['gpu_id']} (3090), batch={params_a['batch_size']}")
            print(f"[DOJO] Branch B: GPU {cfg_b['gpu_id']} (90HX), batch={params_b['batch_size']}")
        print(f"[DOJO] {'='*50}\n")

        # Branch A: Historical (RTX 3090)
        proc_a = self.launch_branch(
            name="branch_A_historical",
            output_dir="jax_ckpt_easy_A",
            use_bella=False,
            auto_resume=auto_resume,
            gpu_id=cfg_a['gpu_id'],
            historical_args=hist_args,
            **params_a,
        )

        actual_stagger = stagger_seconds if gpu_mode == 'single_gpu' else 3.0
        print(f"[DOJO] Waiting {actual_stagger}s before launching Branch B...")
        time.sleep(actual_stagger)

        # Branch B: Bella (CMP 90HX)
        proc_b = self.launch_branch(
            name="branch_B_bella",
            output_dir="jax_ckpt_easy_B",
            use_bella=True,
            auto_resume=auto_resume,
            invert_regime=False,
            gpu_id=cfg_b['gpu_id'],
            **params_b,
        )

        print(f"\n[DOJO] Both branches launched. PIDs: A={proc_a.pid}, B={proc_b.pid}")

        # Store launch params for restarts
        self.launch_params = {
            "branch_A_historical": {
                "output_dir": "jax_ckpt_easy_A",
                "use_bella": False,
                "gpu_id": cfg_a['gpu_id'],
                "historical_args": hist_args,
                **params_a,
            },
            "branch_B_bella": {
                "output_dir": "jax_ckpt_easy_B",
                "use_bella": True,
                "invert_regime": False,
                "gpu_id": cfg_b['gpu_id'],
                **params_b,
            },
        }
        # Store schedule_manager ref for restarts
        self._schedule_manager = schedule_manager
        self._db_path = db_path

        return proc_a, proc_b

    def _write_assets_json(self, assets: list, prefix: str) -> str:
        """Write asset list to temp JSON file for subprocess."""
        assets_dir = self.project_root / "config" / "temp"
        assets_dir.mkdir(parents=True, exist_ok=True)
        assets_file = assets_dir / f"{prefix}_assets.json"
        with open(assets_file, 'w') as f:
            json.dump(assets, f)
        return str(assets_file)

    def launch_single_branch(
        self,
        auto_resume: bool = True,
        **kwargs,
    ) -> subprocess.Popen:
        """TASK_675: Launch single-branch training with full GPU.

        No merging, no stagger, no VRAM fraction cap.
        Dashboard still works (Branch A CSV pattern).
        """
        params = {**self.DEFAULT_PARAMS, **kwargs}
        params['vram_fraction'] = 0.90  # Full GPU (leave 10% for system/dashboard)

        print(f"\n[DOJO] {'='*50}")
        print(f"[DOJO] LAUNCHING SINGLE-BRANCH TRAINING")
        print(f"[DOJO] Preset: {params['preset']}, Epochs: {params['epochs']}")
        print(f"[DOJO] Branch A: Bella (full GPU)")
        print(f"[DOJO] VRAM: {params['vram_fraction']:.0%}")
        print(f"[DOJO] Auto-resume: {auto_resume}")
        print(f"[DOJO] {'='*50}\n")

        proc = self.launch_branch(
            name="branch_A_bella",
            output_dir="jax_ckpt_easy_A",
            use_bella=True,
            auto_resume=auto_resume,
            invert_regime=False,
            **kwargs,
        )

        self.launch_params = {
            "branch_A_bella": {
                "output_dir": "jax_ckpt_easy_A",
                "use_bella": True,
                "invert_regime": False,
                **params,
            },
        }

        print(f"\n[DOJO] Single branch launched. PID: {proc.pid}")
        return proc

    def restart_both_with_checkpoint(self, checkpoint_path: str, stagger_seconds: float = 5.0):
        """
        Restart both branches with a merged checkpoint (TASK_672).

        Called after merge to ensure both branches load the merged weights.
        """
        if not self.launch_params:
            print("[DOJO] ERROR: No launch params stored, cannot restart branches")
            return None, None

        print(f"\n[DOJO] {'='*50}")
        print(f"[DOJO] RESTARTING BRANCHES WITH MERGED CHECKPOINT")
        print(f"[DOJO] Checkpoint: {Path(checkpoint_path).name}")
        print(f"[DOJO] {'='*50}")

        # Stop both branches
        for name, proc in list(self.processes.items()):
            if proc.poll() is None:
                print(f"[DOJO] Stopping {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

        # Brief pause to ensure cleanup
        time.sleep(2)

        # Detect Branch A name (historical or bella)
        if "branch_A_historical" in self.launch_params:
            branch_a_name = "branch_A_historical"
            # TASK_698: Update historical_args with current run from schedule
            if hasattr(self, '_schedule_manager') and self._schedule_manager is not None:
                run = self._schedule_manager.get_current_run()
                if run:
                    assets_file = self._write_assets_json(run['assets'], "branch_A_historical")
                    self.launch_params[branch_a_name]['historical_args'] = {
                        'db_path': self._db_path,
                        'start_date': run['start_date'],
                        'end_date': run['end_date'],
                        'assets_file': assets_file,
                        'lr': '5e-5',
                    }
                    print(f"[DOJO] Historical: {run['start_date']}→{run['end_date']} ({run['original_assets']} assets)")
        else:
            branch_a_name = "branch_A_bella"

        # Restart Branch A
        params_a = self.launch_params.get(branch_a_name, {})
        proc_a = self.launch_branch(
            name=branch_a_name,
            checkpoint=checkpoint_path,
            auto_resume=False,  # Use explicit checkpoint
            **params_a
        )

        # Stagger
        print(f"[DOJO] Waiting {stagger_seconds}s before restarting Branch B...")
        time.sleep(stagger_seconds)

        # Restart Branch B (detect if synth, bella_inv, or bella)
        if "branch_B_bella_inv" in self.launch_params:
            branch_b_name = "branch_B_bella_inv"
        elif "branch_B_bella" in self.launch_params:
            branch_b_name = "branch_B_bella"
        else:
            branch_b_name = "branch_B_synth"

        params_b = self.launch_params.get(branch_b_name, {})
        proc_b = self.launch_branch(
            name=branch_b_name,
            checkpoint=checkpoint_path,
            auto_resume=False,  # Use explicit checkpoint
            **params_b
        )

        print(f"[DOJO] Both branches restarted. PIDs: A={proc_a.pid}, B={proc_b.pid}\n")
        return proc_a, proc_b

    def restart_single_branch(self, branch_name: str, checkpoint_path: str):
        """
        Restart a single branch from checkpoint (for divergence brake).

        Args:
            branch_name: "branch_A_bella" or "branch_B_synth"
            checkpoint_path: Path to checkpoint to restart from
        """
        if branch_name not in self.processes:
            print(f"[DOJO] Warning: Branch {branch_name} not found")
            return None

        proc = self.processes[branch_name]

        # Stop the branch
        if proc.poll() is None:
            print(f"[DOJO] Stopping {branch_name} (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Close log file if any
        if hasattr(proc, '_log_file') and proc._log_file:
            proc._log_file.close()

        time.sleep(2)

        # Restart with original params but new checkpoint
        params = self.launch_params.get(branch_name, {})
        new_proc = self.launch_branch(
            name=branch_name,
            checkpoint=checkpoint_path,
            auto_resume=False,
            **params
        )

        print(f"[DOJO] {branch_name} restarted with PID {new_proc.pid}")
        return new_proc

    def stop_all(self):
        """Stop all training processes."""
        for name, proc in self.processes.items():
            if proc.poll() is None:  # Still running
                print(f"[DOJO] Stopping {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def check_health(self) -> Dict[str, str]:
        """Check status of training processes."""
        status = {}
        for name, proc in self.processes.items():
            ret = proc.poll()
            if ret is None:
                status[name] = "running"
            elif ret == 0:
                status[name] = "completed"
            else:
                status[name] = f"failed (code {ret})"
        return status

    def restart_branch(
        self,
        name: str,
        new_levels: str,
        checkpoint: str = None,
        **kwargs
    ) -> Optional[subprocess.Popen]:
        """
        Restart a synthetic branch with new levels (for level progression).

        Args:
            name: Branch name to restart
            new_levels: New synthetic levels (e.g., "1" or "1-5")
            checkpoint: Checkpoint to resume from (merged weights)
            **kwargs: Additional args passed to launch_branch

        Returns:
            New subprocess or None if restart failed
        """
        if name not in self.processes:
            print(f"[DOJO] Warning: Branch {name} not found, cannot restart")
            return None

        old_proc = self.processes[name]

        # Stop old process
        if old_proc.poll() is None:
            print(f"[DOJO] Stopping {name} for level progression...")
            old_proc.terminate()
            try:
                old_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                old_proc.kill()
                old_proc.wait()

        # Launch new process with updated levels
        print(f"[DOJO] Restarting {name} with levels={new_levels}")

        # Preserve original launch params (GPU config, batch size, etc.)
        saved_params = self.launch_params.get(name, {})
        merged_kwargs = {**saved_params, **kwargs}
        # Override level but keep everything else (gpu_id, batch_size, vram_fraction, etc.)
        merged_kwargs.pop('synthetic_levels', None)  # Will be passed as positional
        if 'output_dir' not in merged_kwargs:
            merged_kwargs['output_dir'] = 'jax_ckpt_easy_B'
        if 'use_bella' not in merged_kwargs:
            merged_kwargs['use_bella'] = False

        new_proc = self.launch_branch(
            name=name,
            synthetic_levels=new_levels,
            **merged_kwargs,
        )

        return new_proc


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="e-dojo: Easy Mode Dual-Branch Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-branch training (full GPU, no merging, with dashboard)
    python e-dojo_v1.py --run --single --preset large --epochs 2500 --oracle-multipass

    # Launch Bella+Synth dual-branch training
    python e-dojo_v1.py --run --preset large --epochs 2500

    # Launch Dual-Bella training (normal + inverted markets)
    python e-dojo_v1.py --run --preset large --epochs 2500 --dual-bella

    # Monitor existing training runs
    python e-dojo_v1.py --branch-a jax_ckpt_easy_A --branch-b jax_ckpt_easy_B

    # Single merge operation
    python e-dojo_v1.py --merge-once ckpt_a.pkl ckpt_b.pkl -o merged.pkl

    # Check divergence between checkpoints
    python e-dojo_v1.py --check-divergence ckpt_a.pkl ckpt_b.pkl
        """
    )

    parser.add_argument("--branch-a", type=str, default="jax_ckpt_easy_A",
                        help="Directory for Branch A checkpoints")
    parser.add_argument("--branch-b", type=str, default="jax_ckpt_easy_B",
                        help="Directory for Branch B checkpoints")
    parser.add_argument("--merged-dir", type=str, default="jax_ckpt_easy_merged",
                        help="Output directory for merged checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/e-dojo",
                        help="Directory for logs and state")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between checkpoint polls")

    # Single operations
    parser.add_argument("--merge-once", nargs=2, metavar=("CKPT_A", "CKPT_B"),
                        help="Perform single merge and exit")
    parser.add_argument("-o", "--output", type=str, default="merged.pkl",
                        help="Output path for --merge-once")
    parser.add_argument("--method", type=str, default="fisher",
                        choices=["fedavg", "fisher", "performance"],
                        help="Merge method for --merge-once")

    parser.add_argument("--check-divergence", nargs=2, metavar=("CKPT_A", "CKPT_B"),
                        help="Check divergence metrics and exit")

    # Auto-launch mode
    parser.add_argument("--run", action="store_true",
                        help="Launch dual-branch training and orchestrate")
    parser.add_argument("--preset", type=str, default="large",
                        choices=["small", "medium", "large"],
                        help="Model preset for training")
    parser.add_argument("--epochs", type=int, default=5000,
                        help="Epochs per branch")
    parser.add_argument("--steps-per-epoch", type=int, default=400,
                        help="Training steps per epoch")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel environments for data generation (TASK_153: reduced for VRAM)")
    parser.add_argument("--buffer-size", type=int, default=16,
                        help="Episodes in VRAM buffer (TASK_153: reduced from 32)")
    parser.add_argument("--levels", type=str, default="0",
                        help="Synthetic levels for Branch B (e.g., '0' or '0-5')")
    parser.add_argument("--stagger", type=float, default=14.0,
                        help="Seconds between branch launches")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, don't resume from existing checkpoints")
    parser.add_argument("--dual-bella", action="store_true",
                        help="Dual-Bella mode: Branch A=Bella, Branch B=Bella (inverted). "
                             "When A goes bull, B goes bear for diverse alignment.")
    parser.add_argument("--single", action="store_true",
                        help="TASK_675: Single-branch mode. Full GPU, no merging. "
                             "Dashboard still works (Branch B shows empty).")
    parser.add_argument("--oracle-multipass", action="store_true",
                        help="TASK_673: Enable multipass oracle for soft policy targets")
    parser.add_argument("--dual-gpu", action="store_true", default=True,
                        help="One branch per GPU: A on 90HX, B on 3090 (default)")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Both branches on default GPU (shared VRAM)")

    # TASK_698: Historical training mode
    parser.add_argument("--branch-a-historical", action="store_true",
                        help="TASK_698: Branch A trains on historical data (Branch B stays Bella)")
    parser.add_argument("--db-path", type=str, default="trading_data.db",
                        help="SQLite database path for historical training")
    parser.add_argument("--schedule", type=str, default="training_schedule.json",
                        help="Training schedule JSON (periods + assets)")
    parser.add_argument("--runs-per-block", type=int, default=4,
                        help="Merges per block (block = named checkpoint + benchmark)")
    parser.add_argument("--benchmark-periods", type=str, default="config/benchmark_periods.json",
                        help="Benchmark periods JSON (for overlap validation)")
    parser.add_argument("--benchmark-every", type=int, default=0,
                        help="TASK_706: Run benchmark every N epochs on separate GPU (0=disabled)")
    parser.add_argument("--benchmark-gpu", type=int, default=1,
                        help="TASK_706: GPU index for benchmark (default: 1 = CMP 90HX)")

    args = parser.parse_args()

    # Single merge operation
    if args.merge_once:
        from tools.orchestration.merge_utils import merge_checkpoints
        merge_checkpoints(
            args.merge_once[0],
            args.merge_once[1],
            args.output,
            method=args.method
        )
        return

    # Check divergence
    if args.check_divergence:
        ckpt_a = load_checkpoint(args.check_divergence[0])
        ckpt_b = load_checkpoint(args.check_divergence[1])
        report = compute_divergence_report(ckpt_a['params'], ckpt_b['params'])

        print(f"\n{'='*50}")
        print("DIVERGENCE REPORT")
        print(f"{'='*50}")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*50}")
        return

    # Auto-launch mode
    launcher = None
    single_mode = args.single
    if args.run:
        launcher = TrainingLauncher()

        # Resume saved level from dojo state (level progression survives restarts)
        # CLI --levels overrides saved state when explicitly provided
        cli_levels_explicit = args.levels != parser.get_default('levels')
        saved_level = args.levels
        state_path = Path(args.log_dir) / "dojo_state.json"
        if state_path.exists() and not cli_levels_explicit:
            try:
                with open(state_path) as f:
                    saved_state = json.load(f)
                sl = saved_state.get('current_level', 0)
                if sl > 0:
                    saved_level = f"0-{sl}"  # Accumulate levels as range
                    print(f"[DOJO] Resuming synthetic levels from saved state: L0-{sl}")
            except (json.JSONDecodeError, KeyError):
                pass
        elif cli_levels_explicit:
            print(f"[DOJO] Using CLI-specified levels: {args.levels} (overriding saved state)")

        training_kwargs = dict(
            preset=args.preset,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            n_envs=args.n_envs,
            buffer_size=args.buffer_size,
            oracle_multipass=args.oracle_multipass,
            benchmark_every=args.benchmark_every,
            benchmark_gpu=args.benchmark_gpu,
        )
        if single_mode and args.branch_a_historical:
            # TASK_707: Single-branch historical with period cycling — no Bella, no merging
            schedule_manager = HistoricalScheduleManager(
                schedule_path=args.schedule,
                db_path=args.db_path,
                benchmark_periods_path=args.benchmark_periods,
            )
            if not schedule_manager.runs:
                raise ValueError("No valid runs in schedule")

            # Generate periods JSON with all runs for train_easy.py --periods-file
            periods_for_json = []
            for run in schedule_manager.runs:
                periods_for_json.append({
                    'start_date': run['start_date'],
                    'end_date': run['end_date'],
                    'assets': run['assets'],
                })
            periods_file = '/tmp/foxglove_periods.json'
            with open(periods_file, 'w') as f:
                json.dump({'periods': periods_for_json}, f)
            print(f"[DOJO] Generated periods file: {len(periods_for_json)} periods → {periods_file}")

            # Full 3090: higher VRAM fraction + batch size
            single_hist_overrides = {'vram_fraction': 0.90, 'batch_size': 1024, 'buffer_size': 16, 'n_envs': 8}
            proc = launcher.launch_branch(
                name="branch_A_historical",
                output_dir="jax_ckpt_easy_A",
                use_bella=False,
                auto_resume=not args.no_resume,
                gpu_id=0,  # RTX 3090 (CUDA 0)
                periods_file=periods_file,
                db_path=args.db_path,
                lr='5e-5',
                **{**training_kwargs, **single_hist_overrides},
            )
            launcher.launch_params = {
                "branch_A_historical": {
                    "output_dir": "jax_ckpt_easy_A",
                    "use_bella": False,
                    "gpu_id": 0,
                    "periods_file": periods_file,
                    "db_path": args.db_path,
                    "lr": "5e-5",
                    **{**training_kwargs, **single_hist_overrides},
                },
            }
            launcher._schedule_manager = schedule_manager
            n_periods = len(set(f"{r['start_date']}→{r['end_date']}" for r in schedule_manager.runs))
            print(f"\n[DOJO] Single historical branch launched with PERIOD CYCLING. PID: {proc.pid}")
            print(f"[DOJO] {len(schedule_manager.runs)} runs across {n_periods} periods (2018-2025)")
        elif single_mode:
            launcher.launch_single_branch(
                auto_resume=not args.no_resume,
                **training_kwargs,
            )
        elif args.branch_a_historical:
            # TASK_698: Historical training mode
            gpu_mode = "single_gpu" if args.single_gpu else "dual_gpu"
            schedule_manager = HistoricalScheduleManager(
                schedule_path=args.schedule,
                db_path=args.db_path,
                benchmark_periods_path=args.benchmark_periods,
            )
            launcher.launch_historical_branches(
                schedule_manager=schedule_manager,
                db_path=args.db_path,
                stagger_seconds=args.stagger,
                auto_resume=not args.no_resume,
                gpu_mode=gpu_mode,
                **training_kwargs,
            )
        else:
            gpu_mode = "single_gpu" if args.single_gpu else "dual_gpu"
            launcher.launch_dual_branches(
                synthetic_levels=saved_level,
                stagger_seconds=args.stagger,
                auto_resume=not args.no_resume,
                dual_bella=args.dual_bella,
                gpu_mode=gpu_mode,
                **training_kwargs,
            )
        # Override branch dirs to match launched processes
        args.branch_a = "jax_ckpt_easy_A"
        args.branch_b = "jax_ckpt_easy_B"

    # Run orchestrator loop
    config = DojoConfig(
        branch_a_dir=args.branch_a,
        branch_b_dir=args.branch_b,
        merged_dir=args.merged_dir,
        log_dir=args.log_dir,
    )

    dojo = EasyDojo(config, launcher=launcher, single_mode=single_mode,
                    schedule_manager=getattr(launcher, '_schedule_manager', None) if launcher else None)

    # TASK_698: Configure historical mode on dojo
    if hasattr(args, 'branch_a_historical') and args.branch_a_historical:
        dojo._benchmark_db_path = args.db_path
        dojo._runs_per_block = args.runs_per_block

    # Link launcher to dojo state for adaptive AR cap propagation
    if launcher is not None:
        launcher._dojo_state = dojo.state

    # Handle signals
    def signal_handler(sig, frame):
        print("\n[DOJO] Received shutdown signal")
        dojo.stop()
        if launcher:
            launcher.stop_all()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        dojo.run_loop(poll_interval=args.poll_interval)
    finally:
        if launcher:
            launcher.stop_all()


if __name__ == "__main__":
    main()
