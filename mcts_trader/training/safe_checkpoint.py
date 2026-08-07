"""
Safe Checkpoint Manager for Curriculum Training.

Provides intelligent checkpoint management that:
- Only saves checkpoints when performance improves
- Detects performance degradation and triggers rollback
- Maintains N best checkpoints for recovery
- Tracks rolling performance metrics

Usage:
    manager = SafeCheckpointManager(
        checkpoint_dir='checkpoints/level_5',
        metric='avg_reward',
        patience=50,
        rollback_threshold=0.5
    )

    for epoch in training:
        metrics = train_epoch()

        if manager.should_save(metrics['avg_reward']):
            manager.save_checkpoint(params, opt_state, epoch, metrics['avg_reward'])

        if manager.should_rollback(metrics['avg_reward']):
            params, opt_state = manager.load_best_checkpoint()
"""

import json
import logging
import math
import os
import shutil
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

# One id per training PROCESS. Written on every score row so a resumed run (which restarts
# `epoch` at 0 while writing into the same scores.jsonl) stays distinguishable in the audit
# trail from the run it resumed — see TASK_876 M8.
_RUN_SESSION_ID = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


def append_checkpoint_score(run_dir, epoch: int, score: float, metric: str,
                            path=None, step: Optional[int] = None,
                            note: Optional[str] = None) -> None:
    """Append one score row for a just-saved checkpoint (TASK_876).

    Writes one JSON object per line to ``<run_dir>/scores.jsonl``. ``run_dir`` is the
    checkpoint's PARENT directory as ``tools/ckpt_gc.py``'s directory walk sees it (the
    directory that holds the ``snapshot_ep<N>`` / ``checkpoint_<step>`` siblings), not the
    checkpoint itself.

    **What the CURRENTLY SHIPPED reader can actually do with these rows — measured, not
    assumed.** ``tools/ckpt_gc.py`` is UNCHANGED by TASK_876 and is EPOCH-KEYED: its
    ``_load_scores`` builds ``{int(row["epoch"]): float(row["score"])}`` and ignores ``path``
    and ``step`` entirely, while ``plan()`` looks a checkpoint up by ``_epoch_of(dir)``, which
    parses the digits out of the DIRECTORY BASENAME. So of the three names this codebase
    writes:

    * ``snapshot_ep<N>``      — round-trips. ``N`` really is the epoch, so the row resolves.
    * ``checkpoint_<step>``   — does NOT resolve. flax names the dir by
      ``state.step_count`` (= iterations_per_epoch x epoch), so ``_epoch_of`` yields the STEP
      and the epoch-keyed lookup misses. Reads as UNSCORED.
    * ``checkpoint_epoch_<N>`` (``SafeCheckpointManager``, incl. ``checkpoint_golden``) — does
      NOT resolve either: ``_epoch_of``'s regex needs digits immediately after
      ``checkpoint_`` and returns ``None``, and ``_is_ckpt_dir`` does not match the name.
      Reads as UNSCORED.

    UNSCORED is the FAIL-SAFE direction: ``plan()`` never score-prunes an unscored checkpoint,
    so an unresolved row cannot itself cause a deletion. Precisely (do not round this up to
    "unscored is immortal"): unscored dirs stay recency-prunable beyond ``--keep-unscored``
    when the disk is under ``--min-free-gb``, exactly as they were before TASK_876. Making the
    reader resolve by ``path``/``step`` (a path-first reader) is **TASK_881**, the checkpoint
    protection model. This row deliberately ships ZERO deleter surface.

    ``path`` and ``step`` are therefore recorded for that FUTURE path-first reader, and for
    the audit trail today — the row format is forward-compatible, not yet fully consumed.
    Always pass ``path`` when the on-disk name is known; ``epoch`` is the key the shipped
    reader uses, and is meaningful for ``snapshot_ep<N>`` only.

    A non-finite score is NOT written as a score (TASK_876 M6): NaN/Inf make a ranking sort
    non-transitive and input-order dependent, and a NaN can take a kept slot from a
    legitimately-scored checkpoint. The row is still written for the audit trail, with
    ``score: null``; ``_load_scores`` drops such a row (``float(None)`` raises and the line is
    skipped), so it reads as UNSCORED — see the fail-safe note above.

    This is pure logging: append-only, best-effort, and MUST NEVER interrupt a training run.
    A failure here (missing dir, disk full, bad permissions) is swallowed and warned — the
    checkpoint save it documents has already happened, and a missing score row only means the
    GC treats that checkpoint as unscored, which is its documented fail-safe.
    """
    try:
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "score": None,
            "metric": str(metric),
            "path": (os.path.abspath(str(path)) if path is not None else None),
            "step": (int(step) if step is not None else None),
            "ts": time.time(),
            "session": _RUN_SESSION_ID,
        }
        if note:
            row["note"] = str(note)
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = float("nan")
        if math.isfinite(s):
            row["score"] = s
        else:
            row["score_invalid"] = repr(score)
            logging.warning(
                f"append_checkpoint_score: non-finite score {score!r} for epoch {epoch} in "
                f"{run_dir} — recording as UNSCORED (the checkpoint is never score-pruned "
                f"and never 'best'; it stays recency-prunable under ckpt_gc's disk floor)")
        with open(d / "scores.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:
        logging.warning(f"append_checkpoint_score: failed to record score for {run_dir}: {e}")


@dataclass
class CheckpointRecord:
    """Record of a saved checkpoint."""
    epoch: int
    metric: float
    path: Path
    timestamp: float = 0.0


class SafeCheckpointManager:
    """
    Intelligent checkpoint management for curriculum training.

    Features:
    - Only saves on performance improvement (rolling average)
    - Auto-rollback on significant degradation
    - Maintains best-N checkpoints
    - Tracks rolling performance metrics
    - Optional golden checkpoint preservation
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = 'avg_reward',
        patience: int = 50,
        rollback_threshold: float = 0.5,
        keep_best_n: int = 3,
        rolling_window: int = 20,
        min_improvement: float = 0.01,
        save_golden: bool = True,
    ):
        """
        Args:
            checkpoint_dir: Directory to store checkpoints
            metric_name: Name of metric to optimize (for logging)
            patience: Epochs to wait before considering rollback
            rollback_threshold: Performance drop to trigger rollback
            keep_best_n: Number of best checkpoints to keep
            rolling_window: Window size for moving average
            min_improvement: Minimum improvement to save checkpoint
            save_golden: Save initial "golden" checkpoint as fallback
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metric_name = metric_name
        self.patience = patience
        self.rollback_threshold = rollback_threshold
        self.keep_best_n = keep_best_n
        self.rolling_window = rolling_window
        self.min_improvement = min_improvement
        self.save_golden = save_golden

        # State tracking
        self.best_metric: float = float('-inf')
        self.best_checkpoints: List[CheckpointRecord] = []
        self.metric_history: List[float] = []
        self.epochs_since_improvement: int = 0
        self.rollback_count: int = 0
        self.golden_checkpoint: Optional[Path] = None
        self.level_start_metric: Optional[float] = None

        # Load existing state if available
        self._load_state()

        logging.info(f"SafeCheckpointManager initialized: {checkpoint_dir}")
        logging.info(f"  patience={patience}, threshold={rollback_threshold}, keep_best_n={keep_best_n}")

    def _load_state(self):
        """Load manager state from checkpoint directory."""
        state_file = self.checkpoint_dir / 'manager_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.best_metric = state.get('best_metric', float('-inf'))
                self.metric_history = state.get('metric_history', [])
                self.epochs_since_improvement = state.get('epochs_since_improvement', 0)
                self.rollback_count = state.get('rollback_count', 0)
                self.level_start_metric = state.get('level_start_metric')

                # Rebuild checkpoint records
                for ckpt in state.get('best_checkpoints', []):
                    path = Path(ckpt['path'])
                    if path.exists():
                        self.best_checkpoints.append(CheckpointRecord(
                            epoch=ckpt['epoch'],
                            metric=ckpt['metric'],
                            path=path,
                            timestamp=ckpt.get('timestamp', 0)
                        ))

                if state.get('golden_checkpoint'):
                    golden_path = Path(state['golden_checkpoint'])
                    if golden_path.exists():
                        self.golden_checkpoint = golden_path

                logging.info(f"Loaded manager state: best_metric={self.best_metric:.3f}, rollbacks={self.rollback_count}")
            except Exception as e:
                logging.warning(f"Failed to load manager state: {e}")

    def _save_state(self):
        """Save manager state to checkpoint directory."""
        state_file = self.checkpoint_dir / 'manager_state.json'
        state = {
            'best_metric': self.best_metric,
            'metric_history': self.metric_history[-200:],  # Keep last 200
            'epochs_since_improvement': self.epochs_since_improvement,
            'rollback_count': self.rollback_count,
            'level_start_metric': self.level_start_metric,
            'best_checkpoints': [
                {
                    'epoch': ckpt.epoch,
                    'metric': ckpt.metric,
                    'path': str(ckpt.path),
                    'timestamp': ckpt.timestamp
                }
                for ckpt in self.best_checkpoints
            ],
            'golden_checkpoint': str(self.golden_checkpoint) if self.golden_checkpoint else None
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_rolling_average(self) -> float:
        """Calculate rolling average of recent metrics."""
        if not self.metric_history:
            return float('-inf')

        window = min(len(self.metric_history), self.rolling_window)
        return float(np.mean(self.metric_history[-window:]))

    def record_metric(self, metric: float):
        """Record a metric value (call every epoch)."""
        self.metric_history.append(metric)

        # Track level start metric for absolute degradation detection
        if self.level_start_metric is None and len(self.metric_history) >= self.rolling_window:
            self.level_start_metric = self.get_rolling_average()

    def should_save(self, current_metric: float) -> bool:
        """
        Determine if we should save a checkpoint.

        Saves when rolling average improves beyond min_improvement.
        """
        self.record_metric(current_metric)

        if len(self.metric_history) < self.rolling_window:
            # Not enough data - save first checkpoint as golden
            if self.save_golden and self.golden_checkpoint is None and len(self.metric_history) == 1:
                return True
            return False

        rolling_avg = self.get_rolling_average()
        improvement = rolling_avg - self.best_metric

        if improvement > self.min_improvement:
            self.best_metric = rolling_avg
            self.epochs_since_improvement = 0
            logging.info(f"Performance improved: {rolling_avg:.3f} (improvement: +{improvement:.3f})")
            return True
        else:
            self.epochs_since_improvement += 1
            return False

    def should_rollback(self, current_metric: float) -> bool:
        """
        Determine if we should rollback to last good checkpoint.

        Triggers when:
        1. Rolling average dropped significantly from best
        2. We've waited at least `patience` epochs
        """
        if len(self.metric_history) < self.rolling_window:
            return False

        if not self.best_checkpoints and not self.golden_checkpoint:
            return False  # Nothing to rollback to

        rolling_avg = self.get_rolling_average()
        degradation = self.best_metric - rolling_avg

        # Also check absolute degradation from level start
        absolute_degradation = 0.0
        if self.level_start_metric is not None:
            absolute_degradation = self.level_start_metric - rolling_avg

        # Rollback conditions:
        should_rollback = (
            degradation > self.rollback_threshold and
            self.epochs_since_improvement >= self.patience
        ) or (
            # Absolute degradation: model is worse than when level started
            absolute_degradation > self.rollback_threshold * 2
        )

        if should_rollback:
            logging.warning("=" * 60)
            logging.warning("ROLLBACK TRIGGERED")
            logging.warning("=" * 60)
            logging.warning(f"  Best metric:     {self.best_metric:.3f}")
            logging.warning(f"  Current rolling: {rolling_avg:.3f}")
            logging.warning(f"  Degradation:     {degradation:.3f} (threshold: {self.rollback_threshold})")
            logging.warning(f"  Epochs waiting:  {self.epochs_since_improvement} (patience: {self.patience})")
            if self.level_start_metric is not None:
                logging.warning(f"  Level start:     {self.level_start_metric:.3f}")
                logging.warning(f"  Absolute drop:   {absolute_degradation:.3f}")

        return should_rollback

    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        epoch: int,
        metric: float
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            checkpoint_data: Dictionary of data to save (params, optimizer, etc.)
            epoch: Current epoch number
            metric: Current metric value

        Returns:
            Path to saved checkpoint
        """
        import time

        # Handle golden checkpoint (first save)
        if self.golden_checkpoint is None and self.save_golden:
            golden_path = self.checkpoint_dir / 'checkpoint_golden'
            self._save_checkpoint_data(golden_path, checkpoint_data, epoch, metric)
            self.golden_checkpoint = golden_path
            logging.info(f"Saved golden checkpoint: {golden_path}")

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        self._save_checkpoint_data(checkpoint_path, checkpoint_data, epoch, metric)

        # Add to best checkpoints list
        record = CheckpointRecord(
            epoch=epoch,
            metric=metric,
            path=checkpoint_path,
            timestamp=time.time()
        )
        self.best_checkpoints.append(record)

        # Sort by metric (descending) and keep only best N
        self.best_checkpoints.sort(key=lambda x: x.metric, reverse=True)

        while len(self.best_checkpoints) > self.keep_best_n:
            old_record = self.best_checkpoints.pop()
            if old_record.path.exists() and old_record.path != checkpoint_path:
                if old_record.path != self.golden_checkpoint:
                    shutil.rmtree(old_record.path)
                    logging.debug(f"Removed old checkpoint: {old_record.path}")

        self._save_state()

        logging.info(f"Saved checkpoint: {checkpoint_path} (metric: {metric:.3f})")
        return checkpoint_path

    def _save_checkpoint_data(
        self,
        path: Path,
        data: Dict[str, Any],
        epoch: int,
        metric: float
    ):
        """Save checkpoint data using orbax or fallback to pickle."""
        path.mkdir(parents=True, exist_ok=True)

        try:
            # Try orbax first (preferred for JAX)
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(path / 'model', data)
        except ImportError:
            # Fallback to pickle
            import pickle
            with open(path / 'checkpoint.pkl', 'wb') as f:
                pickle.dump(data, f)

        # Save metadata
        metadata = {
            'epoch': epoch,
            'metric': metric,
            'metric_history': self.metric_history[-50:],
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_best_checkpoint(self) -> Tuple[Dict[str, Any], int]:
        """
        Load the best available checkpoint for rollback.

        Returns:
            Tuple of (checkpoint_data, epoch)
        """
        # Try best checkpoints first
        for record in self.best_checkpoints:
            if record.path.exists():
                data = self._load_checkpoint_data(record.path)
                if data is not None:
                    self.rollback_count += 1
                    self._save_state()

                    logging.info(f"Loaded checkpoint: {record.path} (metric: {record.metric:.3f})")
                    logging.info(f"Total rollbacks: {self.rollback_count}")

                    return data, record.epoch

        # Fall back to golden checkpoint
        if self.golden_checkpoint and self.golden_checkpoint.exists():
            data = self._load_checkpoint_data(self.golden_checkpoint)
            if data is not None:
                self.rollback_count += 1
                self._save_state()

                logging.warning("Rolled back to GOLDEN checkpoint (last resort)")
                return data, 0

        raise RuntimeError("No valid checkpoint found for rollback!")

    def _load_checkpoint_data(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        try:
            # Try orbax first
            import orbax.checkpoint as ocp
            checkpointer = ocp.StandardCheckpointer()
            return checkpointer.restore(path / 'model')
        except (ImportError, Exception):
            # Fallback to pickle
            pkl_path = path / 'checkpoint.pkl'
            if pkl_path.exists():
                import pickle
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)

        logging.warning(f"Failed to load checkpoint: {path}")
        return None

    def reset_for_new_level(self):
        """Reset manager state for a new curriculum level."""
        self.best_metric = float('-inf')
        self.metric_history = []
        self.epochs_since_improvement = 0
        self.level_start_metric = None
        # Keep rollback count for tracking across levels
        # Keep golden checkpoint as ultimate fallback

        logging.info("SafeCheckpointManager reset for new level")

    def get_stats(self) -> Dict[str, Any]:
        """Get current manager statistics."""
        return {
            'best_metric': self.best_metric,
            'rolling_avg': self.get_rolling_average(),
            'epochs_since_improvement': self.epochs_since_improvement,
            'rollback_count': self.rollback_count,
            'num_checkpoints': len(self.best_checkpoints),
            'has_golden': self.golden_checkpoint is not None,
            'history_length': len(self.metric_history),
        }


# Convenience function for typical usage
def create_level_checkpoint_manager(
    base_dir: str,
    level: int,
    difficulty: str = 'medium'
) -> SafeCheckpointManager:
    """
    Create a SafeCheckpointManager with level-appropriate settings.

    Args:
        base_dir: Base checkpoint directory
        level: Curriculum level (0-14)
        difficulty: 'easy', 'medium', 'hard', or 'expert'
    """
    # Settings based on difficulty
    settings = {
        'easy': {
            'patience': 100,
            'rollback_threshold': 1.0,
            'keep_best_n': 2,
            'rolling_window': 30,
        },
        'medium': {
            'patience': 50,
            'rollback_threshold': 0.5,
            'keep_best_n': 3,
            'rolling_window': 20,
        },
        'hard': {
            'patience': 30,
            'rollback_threshold': 0.3,
            'keep_best_n': 5,
            'rolling_window': 15,
        },
        'expert': {
            'patience': 20,
            'rollback_threshold': 0.2,
            'keep_best_n': 5,
            'rolling_window': 10,
        },
    }

    config = settings.get(difficulty, settings['medium'])

    return SafeCheckpointManager(
        checkpoint_dir=f"{base_dir}/level_{level}",
        **config
    )
