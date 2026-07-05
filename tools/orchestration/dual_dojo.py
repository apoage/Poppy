#!/usr/bin/env python3
"""
dual_dojo.py - 2-GPU Parallel Training with Checkpoint Merging

TASK_690: Natural state corruption via dual-GPU training.

- Branch A (90HX, batch=512): noisier gradients → exploration
- Branch B (3090, batch=1024): smooth gradients → exploitation
- Periodic merge = natural corruption (merged model doesn't perfectly
  match either branch's oracle trajectory)

Orchestrator runs CPU-only (no VRAM theft). Each branch gets its own GPU
via CUDA_VISIBLE_DEVICES in subprocess env.

Usage:
    # Launch with initial checkpoint
    python dual_dojo.py --run \\
        --checkpoint path/to/checkpoint.pkl \\
        --preset large --epochs 2500 --merge-interval 30

    # Check divergence between two checkpoints
    python dual_dojo.py --check-divergence ckpt_a.pkl ckpt_b.pkl
"""

import os
import sys

# CRITICAL: Force orchestrator to CPU-only BEFORE any JAX import
# This prevents the orchestrator from claiming GPU VRAM
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'

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
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from tools.orchestration.merge_utils import (
    fed_avg, fisher_merge, ties_merge,
    merge_optimizer_state, extract_fisher_from_optimizer,
    load_checkpoint, save_checkpoint
)
from tools.orchestration.divergence_metrics import (
    cosine_similarity, compute_divergence_report
)


# =============================================================================
# Tiered Checkpoint Management
# =============================================================================

def cleanup_merged_checkpoints(merged_dir: Path, keep_recent: int = 5, milestone_interval: int = 10):
    """Tiered retention for merged checkpoints.

    Keeps: last N merges + milestones every 10th merge + merged_latest.pkl.
    """
    files = list(merged_dir.glob("merged_*.pkl"))
    files = [f for f in files if f.name != "merged_latest.pkl"]

    if not files:
        return

    merge_map = {}
    for f in files:
        parts = f.name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            merge_map[int(parts[1])] = f

    if not merge_map:
        return

    sorted_counts = sorted(merge_map.keys())
    latest_counts = set(sorted_counts[-keep_recent:])

    deleted_count = 0
    for count, file_path in merge_map.items():
        if count not in latest_counts and count % milestone_interval != 0:
            file_path.unlink()
            deleted_count += 1

    if deleted_count > 0:
        print(f"[DUAL] Pruned {deleted_count} intermediate merge checkpoints")


# =============================================================================
# Configuration
# =============================================================================

class DualDojoConfig:
    """Configuration for dual-GPU dojo orchestrator."""

    # Checkpoint base on scratch disk
    CKPT_BASE = "/mnt/scratch/trader_checkpoints"

    def __init__(
        self,
        # GPU mapping (nvidia-smi / CUDA_VISIBLE_DEVICES indices)
        # nvidia-smi 0 = CMP 90HX (10GB), nvidia-smi 1 = RTX 3090 (24GB)
        # NOTE: JAX flips these (JAX id 0 = 3090, JAX id 1 = 90HX)
        gpu_a: int = 0,          # CMP 90HX (nvidia-smi index 0)
        gpu_b: int = 1,          # RTX 3090 (nvidia-smi index 1)

        # Branch A (90HX): 10GB VRAM — smaller batch
        batch_size_a: int = 512,
        buffer_size_a: int = 8,
        n_envs_a: int = 2,
        vram_fraction_a: float = 0.90,

        # Branch B (3090): 24GB VRAM — large batch, smooth gradients
        batch_size_b: int = 1024,
        buffer_size_b: int = 16,
        n_envs_b: int = 4,
        vram_fraction_b: float = 0.45,

        # Merge schedule (epoch-based, not step-based)
        merge_interval: int = 30,      # Merge every N epochs of Branch A
        min_merge_interval: int = 15,   # Minimum epochs between merges
        max_merge_interval: int = 60,   # Maximum epochs between merges

        # Divergence thresholds (same as e-dojo)
        cosine_safe: float = 0.99,
        cosine_normal: float = 0.95,
        cosine_warning: float = 0.90,
        cosine_brake: float = 0.85,

        # Directories
        branch_a_dir: str = None,
        branch_b_dir: str = None,
        merged_dir: str = None,
        log_dir: str = "logs/dual-dojo",
    ):
        self.gpu_a = gpu_a
        self.gpu_b = gpu_b

        self.batch_size_a = batch_size_a
        self.buffer_size_a = buffer_size_a
        self.n_envs_a = n_envs_a
        self.vram_fraction_a = vram_fraction_a

        self.batch_size_b = batch_size_b
        self.buffer_size_b = buffer_size_b
        self.n_envs_b = n_envs_b
        self.vram_fraction_b = vram_fraction_b

        self.merge_interval = merge_interval
        self.min_merge_interval = min_merge_interval
        self.max_merge_interval = max_merge_interval

        self.cosine_safe = cosine_safe
        self.cosine_normal = cosine_normal
        self.cosine_warning = cosine_warning
        self.cosine_brake = cosine_brake

        # Default dirs on scratch disk
        self.branch_a_dir = Path(branch_a_dir or f"{self.CKPT_BASE}/jax_ckpt_dual_A").resolve()
        self.branch_b_dir = Path(branch_b_dir or f"{self.CKPT_BASE}/jax_ckpt_dual_B").resolve()
        self.merged_dir = Path(merged_dir or f"{self.CKPT_BASE}/jax_ckpt_dual_merged").resolve()
        self.log_dir = Path(log_dir).resolve()

        # Create directories
        self.branch_a_dir.mkdir(parents=True, exist_ok=True)
        self.branch_b_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# State Management
# =============================================================================

class DualDojoState:
    """Mutable state for the dual dojo orchestrator."""

    def __init__(self):
        self.total_epochs_a = 0
        self.total_epochs_b = 0
        self.epochs_since_merge = 0
        self.current_merge_interval = 30
        self.merge_count = 0
        self.epoch_at_last_merge = 0
        # Per-branch epoch tracking for synchronized merges
        self.epoch_a_at_last_merge = 0
        self.epoch_b_at_last_merge = 0

        self.last_cosine = 1.0
        self.last_barrier = 0.0
        self.last_agreement_a = 0.0
        self.last_agreement_b = 0.0
        self.paused_branch = None  # transient, not persisted

        self.history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_epochs_a": self.total_epochs_a,
            "total_epochs_b": self.total_epochs_b,
            "epochs_since_merge": self.epochs_since_merge,
            "current_merge_interval": self.current_merge_interval,
            "merge_count": self.merge_count,
            "epoch_at_last_merge": self.epoch_at_last_merge,
            "epoch_a_at_last_merge": self.epoch_a_at_last_merge,
            "epoch_b_at_last_merge": self.epoch_b_at_last_merge,
            "last_cosine": self.last_cosine,
            "last_barrier": self.last_barrier,
            "last_agreement_a": self.last_agreement_a,
            "last_agreement_b": self.last_agreement_b,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DualDojoState":
        state = cls()
        for k, v in d.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


# =============================================================================
# Launcher
# =============================================================================

class DualDojoLauncher:
    """Launches and manages training subprocesses on separate GPUs."""

    def __init__(self, config: DualDojoConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        self.python_path = str(self.project_root / "trader_env" / "bin" / "python")
        self.train_script = str(self.project_root / "train_easy.py")
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files: Dict[str, Any] = {}

    @staticmethod
    def _generate_seed() -> int:
        """Generate a random seed for market diversity."""
        import random
        return random.randint(0, 99999)

    def _build_cmd(
        self,
        name: str,
        output_dir: str,
        batch_size: int,
        buffer_size: int,
        n_envs: int,
        preset: str,
        epochs: int,
        checkpoint: str = None,
        seed: int = None,
        encoder_version: int = 3,
        oracle_mask_head: bool = False,
    ) -> list:
        """Build training command for a branch."""
        if seed is None:
            seed = self._generate_seed()

        cmd = [
            self.python_path, self.train_script,
            "--preset", preset,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--buffer-size", str(buffer_size),
            "--n-envs", str(n_envs),
            "--name", name,
            "--output-dir", str(output_dir),
            "--bella",
            "--no-latent-cache",
            "--seed", str(seed),
            "--encoder-version", str(encoder_version),
            "--target-agreement", "0.99",  # Disable early stop for dual dojo
        ]

        if checkpoint:
            cmd.extend(["--checkpoint", str(checkpoint)])

        if oracle_mask_head:
            cmd.append("--oracle-mask-head")

        return cmd

    def launch_branch(
        self,
        name: str,
        gpu_id: int,
        output_dir: str,
        batch_size: int,
        buffer_size: int,
        n_envs: int,
        vram_fraction: float,
        preset: str = "large",
        epochs: int = 2500,
        checkpoint: str = None,
        encoder_version: int = 3,
        oracle_mask_head: bool = False,
    ) -> subprocess.Popen:
        """Launch a training branch on a specific GPU."""
        seed = self._generate_seed()
        cmd = self._build_cmd(
            name=name,
            output_dir=output_dir,
            batch_size=batch_size,
            buffer_size=buffer_size,
            n_envs=n_envs,
            preset=preset,
            epochs=epochs,
            checkpoint=checkpoint,
            seed=seed,
            encoder_version=encoder_version,
            oracle_mask_head=oracle_mask_head,
        )

        # Per-GPU environment
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['JAX_PLATFORMS'] = ''  # Reset — let JAX find the GPU
        env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(vram_fraction)
        env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        env['PYTHONUNBUFFERED'] = '1'

        log_file_path = self.config.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file = open(log_file_path, 'w')

        print(f"[DUAL] Launching {name} on GPU {gpu_id} (batch={batch_size}, buf={buffer_size})")
        print(f"[DUAL]   Seed: {seed}, VRAM: {vram_fraction:.0%}")
        print(f"[DUAL]   Log: {log_file_path}")
        if checkpoint:
            print(f"[DUAL]   Checkpoint: {Path(checkpoint).name}")

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=self.project_root,
            env=env,
        )

        self.processes[name] = proc
        self.log_files[name] = log_file
        return proc

    def launch_both(
        self,
        preset: str = "large",
        epochs: int = 2500,
        checkpoint: str = None,
        encoder_version: int = 3,
        oracle_mask_head: bool = False,
    ) -> Tuple[subprocess.Popen, subprocess.Popen]:
        """Launch both branches on separate GPUs (no stagger needed)."""
        cfg = self.config

        print(f"\n[DUAL] {'='*50}")
        print(f"[DUAL] LAUNCHING DUAL-GPU TRAINING")
        print(f"[DUAL] Branch A (90HX): GPU {cfg.gpu_a}, batch={cfg.batch_size_a}")
        print(f"[DUAL] Branch B (3090): GPU {cfg.gpu_b}, batch={cfg.batch_size_b}")
        print(f"[DUAL] Preset: {preset}, Epochs: {epochs}")
        if oracle_mask_head:
            print(f"[DUAL] Oracle mask head: ENABLED")
        print(f"[DUAL] {'='*50}\n")

        proc_a = self.launch_branch(
            name="branch_A_3090",
            gpu_id=cfg.gpu_a,
            output_dir=str(cfg.branch_a_dir),
            batch_size=cfg.batch_size_a,
            buffer_size=cfg.buffer_size_a,
            n_envs=cfg.n_envs_a,
            vram_fraction=cfg.vram_fraction_a,
            preset=preset,
            epochs=epochs,
            checkpoint=checkpoint,
            encoder_version=encoder_version,
            oracle_mask_head=oracle_mask_head,
        )

        proc_b = self.launch_branch(
            name="branch_B_90hx",
            gpu_id=cfg.gpu_b,
            output_dir=str(cfg.branch_b_dir),
            batch_size=cfg.batch_size_b,
            buffer_size=cfg.buffer_size_b,
            n_envs=cfg.n_envs_b,
            vram_fraction=cfg.vram_fraction_b,
            preset=preset,
            epochs=epochs,
            checkpoint=checkpoint,
            encoder_version=encoder_version,
            oracle_mask_head=oracle_mask_head,
        )

        print(f"[DUAL] Both branches launched. PIDs: A={proc_a.pid}, B={proc_b.pid}\n")
        return proc_a, proc_b

    def restart_both(self, checkpoint: str, preset: str = "large", epochs: int = 2500,
                     encoder_version: int = 3, oracle_mask_head: bool = False):
        """Stop both branches and restart from merged checkpoint."""
        print(f"\n[DUAL] {'='*50}")
        print(f"[DUAL] RESTARTING BRANCHES WITH MERGED CHECKPOINT")
        print(f"[DUAL] Checkpoint: {Path(checkpoint).name}")
        print(f"[DUAL] {'='*50}")

        self.stop_all()
        time.sleep(3)

        return self.launch_both(
            preset=preset,
            epochs=epochs,
            checkpoint=checkpoint,
            encoder_version=encoder_version,
            oracle_mask_head=oracle_mask_head,
        )

    def restart_single(self, name: str, checkpoint: str, preset: str = "large",
                       epochs: int = 2500, encoder_version: int = 3,
                       oracle_mask_head: bool = False):
        """Restart a single branch from checkpoint (divergence brake)."""
        if name not in self.processes:
            print(f"[DUAL] Warning: Branch {name} not found")
            return None

        proc = self.processes[name]
        if proc.poll() is None:
            print(f"[DUAL] Stopping {name} (PID {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Close log file
        if name in self.log_files:
            self.log_files[name].close()

        time.sleep(2)

        cfg = self.config
        if "3090" in name:
            return self.launch_branch(
                name=name, gpu_id=cfg.gpu_a, output_dir=str(cfg.branch_a_dir),
                batch_size=cfg.batch_size_a, buffer_size=cfg.buffer_size_a,
                n_envs=cfg.n_envs_a, vram_fraction=cfg.vram_fraction_a,
                preset=preset, epochs=epochs, checkpoint=checkpoint,
                encoder_version=encoder_version, oracle_mask_head=oracle_mask_head,
            )
        else:
            return self.launch_branch(
                name=name, gpu_id=cfg.gpu_b, output_dir=str(cfg.branch_b_dir),
                batch_size=cfg.batch_size_b, buffer_size=cfg.buffer_size_b,
                n_envs=cfg.n_envs_b, vram_fraction=cfg.vram_fraction_b,
                preset=preset, epochs=epochs, checkpoint=checkpoint,
                encoder_version=encoder_version, oracle_mask_head=oracle_mask_head,
            )

    def pause_branch(self, name: str):
        """SIGSTOP a branch process (freeze without killing)."""
        if name in self.processes and self.processes[name].poll() is None:
            pid = self.processes[name].pid
            os.kill(pid, signal.SIGSTOP)
            print(f"[DUAL] Paused {name} (PID {pid})")

    def resume_branch(self, name: str):
        """SIGCONT a paused branch process."""
        if name in self.processes and self.processes[name].poll() is None:
            pid = self.processes[name].pid
            os.kill(pid, signal.SIGCONT)
            print(f"[DUAL] Resumed {name} (PID {pid})")

    def stop_all(self):
        """Stop all training processes."""
        for name, proc in self.processes.items():
            if proc.poll() is None:
                print(f"[DUAL] Stopping {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        for name, f in self.log_files.items():
            if f and not f.closed:
                f.close()

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


# =============================================================================
# Core Orchestrator
# =============================================================================

class DualDojo:
    """2-GPU Parallel Training Orchestrator with Checkpoint Merging."""

    def __init__(self, config: DualDojoConfig, launcher: DualDojoLauncher,
                 preset: str = "large", epochs: int = 2500, encoder_version: int = 3,
                 oracle_mask_head: bool = False):
        self.config = config
        self.launcher = launcher
        self.state = DualDojoState()
        self.preset = preset
        self.epochs = epochs
        self.encoder_version = encoder_version
        self.oracle_mask_head = oracle_mask_head
        self.running = False
        self._stop_event = threading.Event()

        self._load_state()

    def _load_state(self):
        """Load state from checkpoint if exists."""
        state_path = self.config.log_dir / "dual_dojo_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                self.state = DualDojoState.from_dict(json.load(f))
            print(f"[DUAL] Loaded state: {self.state.merge_count} merges")

    def _save_state(self):
        """Save state to checkpoint."""
        state_path = self.config.log_dir / "dual_dojo_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def _get_latest_checkpoint(self, branch_dir: Path) -> Optional[Path]:
        """Find the latest checkpoint in a branch directory."""
        checkpoints = list(branch_dir.glob("**/*.pkl"))
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    def _get_epoch_from_checkpoint(self, ckpt_path: Path) -> int:
        """Extract epoch from checkpoint without loading full params."""
        try:
            with open(ckpt_path, 'rb') as f:
                ckpt = pickle.load(f)
            return ckpt.get('epoch', 0)
        except Exception:
            return 0

    def _get_training_phase(self) -> str:
        """Determine current training phase based on merge count."""
        if self.state.merge_count < 3:
            return "early"
        elif self.state.merge_count < 20:
            return "mid"
        else:
            return "late"

    def _select_merge_method(self, cosine: float, phase: str) -> str:
        """Select merge method based on metrics and phase."""
        if phase == "early":
            return "fedavg"
        if self.config.cosine_brake < cosine < self.config.cosine_normal:
            return "ties"
        if phase == "late":
            return "fisher"
        return "fedavg"

    def _perform_merge(self, ckpt_a: Dict, ckpt_b: Dict, method: str) -> Dict:
        """Perform the actual merge operation on CPU."""
        params_a = ckpt_a['params']
        params_b = ckpt_b['params']

        score_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0.5)
        score_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0.5)

        fisher_a = None
        fisher_b = None

        try:
            if method == "fedavg":
                merged_params = fed_avg(params_a, params_b, 0.5)

            elif method == "fisher":
                if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
                    try:
                        fisher_a = extract_fisher_from_optimizer(ckpt_a['opt_state'])
                        fisher_b = extract_fisher_from_optimizer(ckpt_b['opt_state'])
                        merged_params = fisher_merge(params_a, params_b, fisher_a, fisher_b,
                                                     normalize=True)
                    except Exception as e:
                        print(f"[DUAL] Warning: Fisher extraction failed ({e}), using FedAvg")
                        merged_params = fed_avg(params_a, params_b, 0.5)
                else:
                    merged_params = fed_avg(params_a, params_b, 0.5)

            elif method == "ties":
                prev_merged_path = self._get_latest_checkpoint(self.config.merged_dir)
                if prev_merged_path:
                    try:
                        prev_merged = load_checkpoint(str(prev_merged_path))
                        params_init = prev_merged['params']
                        merged_params = ties_merge(params_a, params_b, params_init,
                                                   trim_fraction=0.2)
                        del prev_merged
                        print(f"[DUAL] TIES merge using base: {prev_merged_path.name}")
                    except Exception as e:
                        print(f"[DUAL] Warning: TIES failed ({e}), using FedAvg")
                        merged_params = fed_avg(params_a, params_b, 0.5)
                else:
                    print(f"[DUAL] Warning: No base checkpoint for TIES, using FedAvg")
                    merged_params = fed_avg(params_a, params_b, 0.5)
            else:
                merged_params = fed_avg(params_a, params_b, 0.5)

            # Merge optimizer state
            merged_opt_components = None
            if 'opt_state' in ckpt_a and 'opt_state' in ckpt_b:
                try:
                    merged_nu, merged_mu = merge_optimizer_state(
                        ckpt_a['opt_state'], ckpt_b['opt_state'],
                        reset_momentum=True
                    )
                    merged_opt_components = {'nu': merged_nu, 'mu': merged_mu}
                except Exception as e:
                    print(f"[DUAL] Warning: Optimizer merge failed ({e})")

            merged_ckpt = {
                'params': merged_params,
                'epoch': max(ckpt_a.get('epoch', 0), ckpt_b.get('epoch', 0)),
                'metrics': {
                    'source': 'dual_merged',
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
            if fisher_a is not None:
                del fisher_a
            if fisher_b is not None:
                del fisher_b

    def _estimate_barrier(self, cosine: float, ckpt_a: Dict, ckpt_b: Dict) -> float:
        """Estimate loss barrier heuristically (same as e-dojo)."""
        agree_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0.5)
        agree_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0.5)
        perf_gap = abs(agree_a - agree_b)
        weight_div = 1.0 - cosine
        barrier = weight_div * (1.0 + perf_gap) * 0.5
        return min(barrier, 1.0)

    def _adjust_merge_interval(self, cosine: float):
        """Adjust merge interval based on divergence."""
        if cosine > self.config.cosine_safe:
            self.state.current_merge_interval = min(
                int(self.state.current_merge_interval * 1.3),
                self.config.max_merge_interval
            )
        elif cosine < self.config.cosine_normal:
            self.state.current_merge_interval = max(
                int(self.state.current_merge_interval * 0.8),
                self.config.min_merge_interval
            )

    def check_and_merge(self) -> Optional[Dict[str, Any]]:
        """Check if merge is needed and perform it."""
        ckpt_a_path = self._get_latest_checkpoint(self.config.branch_a_dir)
        ckpt_b_path = self._get_latest_checkpoint(self.config.branch_b_dir)

        if not ckpt_a_path or not ckpt_b_path:
            return None

        ckpt_a = None
        ckpt_b = None
        merged_ckpt = None
        report = None

        try:
            # Load checkpoints
            try:
                ckpt_a = load_checkpoint(str(ckpt_a_path))
                ckpt_b = load_checkpoint(str(ckpt_b_path))
            except Exception as e:
                print(f"[DUAL] Error loading checkpoints: {e}")
                return None

            # Track epochs
            epoch_a = ckpt_a.get('epoch', 0)
            epoch_b = ckpt_b.get('epoch', 0)
            self.state.total_epochs_a = epoch_a
            self.state.total_epochs_b = epoch_b

            # Synchronized merge: require BOTH branches to have done enough epochs
            epochs_a_since = epoch_a - self.state.epoch_a_at_last_merge
            epochs_b_since = epoch_b - self.state.epoch_b_at_last_merge
            self.state.epochs_since_merge = min(epochs_a_since, epochs_b_since)
            current_epoch = max(epoch_a, epoch_b)  # kept for state compat
            interval = self.state.current_merge_interval
            a_ready = epochs_a_since >= interval
            b_ready = epochs_b_since >= interval

            # Pause faster branch while waiting for slower one
            if a_ready and not b_ready:
                if self.state.paused_branch != "branch_A_3090":
                    self.launcher.pause_branch("branch_A_3090")
                    self.state.paused_branch = "branch_A_3090"
                return None
            elif b_ready and not a_ready:
                if self.state.paused_branch != "branch_B_90hx":
                    self.launcher.pause_branch("branch_B_90hx")
                    self.state.paused_branch = "branch_B_90hx"
                return None
            elif not a_ready and not b_ready:
                return None

            # Both ready — resume paused branch before merging
            if self.state.paused_branch:
                self.launcher.resume_branch(self.state.paused_branch)
                self.state.paused_branch = None

            # Compute divergence
            cosine = cosine_similarity(ckpt_a['params'], ckpt_b['params'])
            barrier = self._estimate_barrier(cosine, ckpt_a, ckpt_b)
            self.state.last_cosine = cosine
            self.state.last_barrier = barrier
            self.state.last_agreement_a = ckpt_a.get('metrics', {}).get('oracle_agreement', 0)
            self.state.last_agreement_b = ckpt_b.get('metrics', {}).get('oracle_agreement', 0)

            # DIVERGENCE BRAKE
            if cosine < self.config.cosine_brake:
                # First merge: force sync even if diverged
                if self.state.merge_count == 0:
                    print(f"[DUAL] Initial sync merge (cosine={cosine:.3f}, diverged but first merge)")
                else:
                    print(f"\n[DUAL] {'='*50}")
                    print(f"[DUAL] DIVERGENCE BRAKE ACTIVATED")
                    print(f"[DUAL] Cosine {cosine:.3f} < {self.config.cosine_brake}")
                    print(f"[DUAL] Agreement A: {self.state.last_agreement_a:.1%}, B: {self.state.last_agreement_b:.1%}")

                    # Clear pause + restart worse branch from merged checkpoint
                    if self.state.paused_branch:
                        self.launcher.resume_branch(self.state.paused_branch)
                        self.state.paused_branch = None
                    merged_path = self._get_latest_checkpoint(self.config.merged_dir)
                    if merged_path:
                        if self.state.last_agreement_a >= self.state.last_agreement_b:
                            worse = "branch_B_90hx"
                            print(f"[DUAL] Restarting Branch B (90HX) — agreement {self.state.last_agreement_b:.1%}")
                        else:
                            worse = "branch_A_3090"
                            print(f"[DUAL] Restarting Branch A (3090) — agreement {self.state.last_agreement_a:.1%}")

                        self.launcher.restart_single(
                            worse, str(merged_path),
                            preset=self.preset, epochs=self.epochs,
                            encoder_version=self.encoder_version,
                            oracle_mask_head=self.oracle_mask_head,
                        )
                    else:
                        print(f"[DUAL] No merged checkpoint for brake — skipping")

                    print(f"[DUAL] {'='*50}\n")

                    # Reset epoch tracking
                    self.state.epoch_at_last_merge = current_epoch
                    self.state.epoch_a_at_last_merge = epoch_a
                    self.state.epoch_b_at_last_merge = epoch_b
                    self.state.epochs_since_merge = 0
                    self._save_state()

                    return {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "event": "divergence_brake",
                        "reason": "divergence_brake",
                        "cosine": cosine,
                        "barrier": barrier,
                        "restarted_branch": worse,
                    }

            # Select merge method
            phase = self._get_training_phase()
            method = self._select_merge_method(cosine, phase)

            print(f"\n[DUAL] {'='*50}")
            print(f"[DUAL] MERGE #{self.state.merge_count + 1}")
            print(f"[DUAL] Phase: {phase}, Method: {method}")
            print(f"[DUAL] Cosine: {cosine:.4f}")
            print(f"[DUAL] Epochs A: {epoch_a}, B: {epoch_b}")
            print(f"[DUAL] Agreement A: {self.state.last_agreement_a:.1%}, B: {self.state.last_agreement_b:.1%}")

            # Merge
            merged_ckpt = self._perform_merge(ckpt_a, ckpt_b, method)

            # Free source checkpoints before saving
            del ckpt_a, ckpt_b
            ckpt_a = ckpt_b = None

            # Save merged checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_path = self.config.merged_dir / f"merged_{self.state.merge_count + 1:04d}_{timestamp}.pkl"
            save_checkpoint(merged_ckpt, str(merged_path))

            # Copy to latest
            latest_path = self.config.merged_dir / "merged_latest.pkl"
            shutil.copy2(str(merged_path), str(latest_path))

            # Tiered cleanup
            cleanup_merged_checkpoints(self.config.merged_dir, keep_recent=5, milestone_interval=10)

            del merged_ckpt
            merged_ckpt = None

            # Update state
            self.state.merge_count += 1
            self.state.epochs_since_merge = 0
            self.state.epoch_at_last_merge = current_epoch
            self.state.epoch_a_at_last_merge = epoch_a
            self.state.epoch_b_at_last_merge = epoch_b
            self._adjust_merge_interval(cosine)
            self._save_state()

            # Metrics
            agree_a = self.state.last_agreement_a
            agree_b = self.state.last_agreement_b
            alignment = (agree_a * agree_b) ** 0.5 if agree_a > 0 and agree_b > 0 else 0.0

            report = {
                "timestamp": timestamp,
                "merge_count": self.state.merge_count,
                "reason": "interval_reached",
                "phase": phase,
                "method": method,
                "cosine": cosine,
                "barrier": barrier,
                "agreement_a": agree_a,
                "agreement_b": agree_b,
                "alignment": alignment,
                "merged_path": str(merged_path),
            }
            self.state.history.append(report)

            # Save history
            history_path = self.config.log_dir / "merge_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.state.history, f, indent=2)

            print(f"[DUAL] Alignment: {alignment:.1%}")
            print(f"[DUAL] Saved to: {merged_path}")
            print(f"[DUAL] Next merge in ~{self.state.current_merge_interval} epochs")

            # Clear pause state before restart (new PIDs)
            self.state.paused_branch = None

            # Restart both branches from merged checkpoint
            self.launcher.restart_both(
                str(merged_path),
                preset=self.preset,
                epochs=self.epochs,
                encoder_version=self.encoder_version,
                oracle_mask_head=self.oracle_mask_head,
            )

            print(f"[DUAL] {'='*50}\n")
            return report

        finally:
            if ckpt_a is not None:
                del ckpt_a
            if ckpt_b is not None:
                del ckpt_b
            if merged_ckpt is not None:
                del merged_ckpt

            jax.clear_caches()
            gc.collect()

            if report is not None:
                print(f"[DUAL] Memory cleanup complete (gc.collect + jax.clear_caches)")

    def run_loop(self, poll_interval: float = 30.0):
        """Run the orchestrator loop."""
        self.running = True
        print(f"[DUAL] Starting dual dojo loop (poll every {poll_interval}s)")
        print(f"[DUAL] Branch A dir: {self.config.branch_a_dir}")
        print(f"[DUAL] Branch B dir: {self.config.branch_b_dir}")
        print(f"[DUAL] Merged dir: {self.config.merged_dir}")
        print(f"[DUAL] Merge interval: {self.state.current_merge_interval} epochs")

        try:
            while self.running and not self._stop_event.is_set():
                # Check process health
                health = self.launcher.check_health()
                all_done = all(s in ("completed", "failed") for s in health.values())

                if all_done and len(health) > 0:
                    print(f"\n[DUAL] {'='*50}")
                    print("[DUAL] TRAINING COMPLETED")
                    for name, status in health.items():
                        print(f"[DUAL]   {name}: {status}")
                    print(f"[DUAL] Total merges: {self.state.merge_count}")
                    print(f"[DUAL] {'='*50}")
                    break

                # Check for failures and restart
                for name, status in health.items():
                    if "failed" in status:
                        print(f"[DUAL] WARNING: {name} has {status}")
                        # Try to restart from merged checkpoint
                        merged_path = self._get_latest_checkpoint(self.config.merged_dir)
                        if merged_path:
                            print(f"[DUAL] Auto-restarting {name} from {merged_path.name}")
                            self.launcher.restart_single(
                                name, str(merged_path),
                                preset=self.preset, epochs=self.epochs,
                                encoder_version=self.encoder_version,
                                oracle_mask_head=self.oracle_mask_head,
                            )
                        else:
                            print(f"[DUAL] No merged checkpoint — cannot auto-restart")

                # Check for merge
                report = self.check_and_merge()
                if report:
                    time.sleep(5)
                    continue

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n[DUAL] Interrupted by user")
        finally:
            self.running = False
            self._save_state()
            print("[DUAL] Orchestrator stopped")

    def stop(self):
        """Stop the orchestrator loop."""
        self._stop_event.set()
        self.running = False


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dual Dojo: 2-GPU Parallel Training with Checkpoint Merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Launch dual-GPU training from checkpoint
    python dual_dojo.py --run --checkpoint path/to/checkpoint.pkl

    # Custom merge interval
    python dual_dojo.py --run --checkpoint ckpt.pkl --merge-interval 20

    # Custom GPU mapping
    python dual_dojo.py --run --gpu-a 1 --gpu-b 0

    # Check divergence between checkpoints
    python dual_dojo.py --check-divergence ckpt_a.pkl ckpt_b.pkl
        """
    )

    # Mode
    parser.add_argument("--run", action="store_true",
                        help="Launch dual-GPU training and orchestrate")
    parser.add_argument("--check-divergence", nargs=2, metavar=("CKPT_A", "CKPT_B"),
                        help="Check divergence metrics and exit")

    # Training
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Initial checkpoint to start from")
    parser.add_argument("--preset", type=str, default="large",
                        choices=["small", "medium", "large"])
    parser.add_argument("--epochs", type=int, default=2500,
                        help="Epochs per branch")
    parser.add_argument("--encoder-version", type=int, default=3, choices=[2, 3],
                        help="Encoder version: 2=V2, 3=V3 (minGRU)")
    parser.add_argument("--oracle-mask-head", action="store_true",
                        help="TASK_695: Enable oracle mask prediction auxiliary head")

    # Merge schedule
    parser.add_argument("--merge-interval", type=int, default=30,
                        help="Merge every N epochs (default: 30)")
    parser.add_argument("--poll-interval", type=float, default=30.0,
                        help="Seconds between checkpoint polls")

    # GPU mapping
    parser.add_argument("--gpu-a", type=int, default=0,
                        help="CUDA index for Branch A / 3090 (default: 0)")
    parser.add_argument("--gpu-b", type=int, default=1,
                        help="CUDA index for Branch B / 90HX (default: 1)")

    # Branch configs
    parser.add_argument("--batch-a", type=int, default=512,
                        help="Batch size for Branch A / 3090 (default: 512)")
    parser.add_argument("--batch-b", type=int, default=128,
                        help="Batch size for Branch B / 90HX (default: 128)")

    args = parser.parse_args()

    # Check divergence mode
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

    if not args.run:
        parser.print_help()
        return

    # Build config
    config = DualDojoConfig(
        gpu_a=args.gpu_a,
        gpu_b=args.gpu_b,
        batch_size_a=args.batch_a,
        batch_size_b=args.batch_b,
        merge_interval=args.merge_interval,
    )

    # Create launcher
    launcher = DualDojoLauncher(config)

    # Launch both branches
    launcher.launch_both(
        preset=args.preset,
        epochs=args.epochs,
        checkpoint=args.checkpoint,
        encoder_version=args.encoder_version,
        oracle_mask_head=args.oracle_mask_head,
    )

    # Create orchestrator
    dojo = DualDojo(
        config, launcher,
        preset=args.preset,
        epochs=args.epochs,
        encoder_version=args.encoder_version,
        oracle_mask_head=args.oracle_mask_head,
    )
    dojo.state.current_merge_interval = args.merge_interval

    # Handle signals
    def signal_handler(sig, frame):
        print("\n[DUAL] Received shutdown signal")
        dojo.stop()
        launcher.stop_all()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        dojo.run_loop(poll_interval=args.poll_interval)
    finally:
        launcher.stop_all()


if __name__ == "__main__":
    main()
