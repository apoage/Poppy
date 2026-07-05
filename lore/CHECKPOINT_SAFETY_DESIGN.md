# Checkpoint Safety System for Advanced Curriculum

## Problem Statement

When training on difficult levels (5+), the model can experience:
1. **Catastrophic forgetting**: Bad epoch sequences cause model to lose learned behaviors
2. **Negative transfer**: New level interferes with previously learned tactics
3. **Oscillation**: Model alternates between good and bad policies

**Solution**: Intelligent checkpoint management that preserves good states and rolls back from bad ones.

---

## Design: Safe Checkpoint Manager

### Core Principles

1. **Save on success, not on schedule**: Only checkpoint when performance improves
2. **Track moving average**: Don't react to single bad epoch - track rolling performance
3. **Auto-rollback**: If performance degrades below threshold, restore last good checkpoint
4. **Preserve history**: Keep N best checkpoints for recovery options

### Architecture

```python
class SafeCheckpointManager:
    """
    Intelligent checkpoint management for curriculum training.

    Features:
    - Only saves on performance improvement
    - Auto-rollback on significant degradation
    - Maintains best-N checkpoints
    - Tracks rolling performance metrics
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric: str = 'avg_reward',        # Metric to optimize
        patience: int = 50,                 # Epochs before rollback
        rollback_threshold: float = 0.5,    # Performance drop to trigger rollback
        keep_best_n: int = 3,               # Number of best checkpoints to keep
        rolling_window: int = 20,           # Window for moving average
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metric = metric
        self.patience = patience
        self.rollback_threshold = rollback_threshold
        self.keep_best_n = keep_best_n
        self.rolling_window = rolling_window

        # State tracking
        self.best_metric = float('-inf')
        self.best_checkpoints = []  # [(metric, path), ...]
        self.metric_history = []
        self.epochs_since_improvement = 0
        self.current_checkpoint = None

    def should_save(self, current_metric: float) -> bool:
        """Determine if we should save a checkpoint."""
        self.metric_history.append(current_metric)

        # Calculate rolling average
        if len(self.metric_history) >= self.rolling_window:
            rolling_avg = np.mean(self.metric_history[-self.rolling_window:])
        else:
            rolling_avg = np.mean(self.metric_history)

        # Check if this is an improvement
        is_improvement = rolling_avg > self.best_metric

        if is_improvement:
            self.best_metric = rolling_avg
            self.epochs_since_improvement = 0
            return True
        else:
            self.epochs_since_improvement += 1
            return False

    def should_rollback(self, current_metric: float) -> bool:
        """Determine if we should rollback to last good checkpoint."""
        if len(self.metric_history) < self.rolling_window:
            return False  # Not enough data

        rolling_avg = np.mean(self.metric_history[-self.rolling_window:])

        # Check for significant degradation
        degradation = self.best_metric - rolling_avg

        # Rollback conditions:
        # 1. Performance dropped significantly AND
        # 2. We've been patient (waited enough epochs)
        should_rollback = (
            degradation > self.rollback_threshold and
            self.epochs_since_improvement >= self.patience
        )

        if should_rollback:
            logging.warning(f"Performance degradation detected!")
            logging.warning(f"  Best: {self.best_metric:.2f}, Current: {rolling_avg:.2f}")
            logging.warning(f"  Degradation: {degradation:.2f} > threshold {self.rollback_threshold}")

        return should_rollback

    def save_checkpoint(self, params, optimizer_state, epoch: int, metric: float):
        """Save checkpoint if it's among the best."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"

        # Save the checkpoint
        orbax_checkpointer.save(checkpoint_path, {
            'params': params,
            'optimizer_state': optimizer_state,
            'epoch': epoch,
            'metric': metric,
            'metric_history': self.metric_history[-100:],  # Keep last 100
        })

        self.current_checkpoint = checkpoint_path

        # Maintain best-N checkpoints
        self.best_checkpoints.append((metric, checkpoint_path))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove old checkpoints beyond keep_best_n
        while len(self.best_checkpoints) > self.keep_best_n:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists() and old_path != checkpoint_path:
                shutil.rmtree(old_path)
                logging.info(f"Removed old checkpoint: {old_path}")

        logging.info(f"Saved checkpoint: {checkpoint_path} (metric: {metric:.2f})")

    def load_best_checkpoint(self):
        """Load the best checkpoint for rollback."""
        if not self.best_checkpoints:
            raise ValueError("No checkpoints available for rollback")

        best_metric, best_path = self.best_checkpoints[0]
        logging.info(f"Rolling back to best checkpoint: {best_path} (metric: {best_metric:.2f})")

        checkpoint = orbax_checkpointer.restore(best_path)

        # Reset tracking state
        self.epochs_since_improvement = 0
        self.metric_history = checkpoint.get('metric_history', [])

        return checkpoint['params'], checkpoint['optimizer_state']
```

---

## Integration with DojoManager

### Modified Training Loop

```python
class DojoManager:
    def _train_level_safe(self, config: Dict, level_idx: int):
        """Train with safe checkpoint management."""

        # Initialize safe checkpoint manager
        safe_ckpt = SafeCheckpointManager(
            checkpoint_dir=f"{self.checkpoint_dir}/level_{level_idx}",
            metric='avg_reward',
            patience=config.get('rollback_patience', 50),
            rollback_threshold=config.get('rollback_threshold', 0.5),
            keep_best_n=config.get('keep_best_n', 3),
        )

        rollback_count = 0
        max_rollbacks = config.get('max_rollbacks', 3)

        for epoch in range(config['epochs']):
            # Train one epoch
            metrics = self._train_epoch()
            current_metric = metrics['avg_reward']

            # Check for checkpoint save
            if safe_ckpt.should_save(current_metric):
                safe_ckpt.save_checkpoint(
                    params=self.params,
                    optimizer_state=self.optimizer_state,
                    epoch=epoch,
                    metric=current_metric
                )

            # Check for rollback
            if safe_ckpt.should_rollback(current_metric):
                if rollback_count >= max_rollbacks:
                    logging.error(f"Max rollbacks ({max_rollbacks}) exceeded!")
                    raise RuntimeError("Training failed: model cannot recover")

                logging.warning(f"Initiating rollback #{rollback_count + 1}")
                self.params, self.optimizer_state = safe_ckpt.load_best_checkpoint()
                rollback_count += 1

                # Optional: reduce learning rate after rollback
                self.reduce_learning_rate(factor=0.5)
```

---

## Curriculum Config with Safety Parameters

```json
{
  "stage": "5.1",
  "level": 5,
  "generator": "pairs_trading",
  "epochs": 300,
  "pass_threshold": 1.5,

  "safety": {
    "enabled": true,
    "rollback_patience": 50,
    "rollback_threshold": 0.5,
    "keep_best_n": 3,
    "max_rollbacks": 3,
    "lr_decay_on_rollback": 0.5
  }
}
```

---

## Level-Specific Safety Settings

| Level | Difficulty | Patience | Threshold | Max Rollbacks |
|-------|------------|----------|-----------|---------------|
| 0-4   | Easy       | 100      | 1.0       | 1 (unlikely)  |
| 5-7   | Medium     | 50       | 0.5       | 3             |
| 8-10  | Hard       | 30       | 0.3       | 5             |
| 11-14 | Expert     | 20       | 0.2       | 10            |

**Rationale**:
- Harder levels = lower patience (detect problems faster)
- Harder levels = lower threshold (more sensitive to degradation)
- Harder levels = more rollbacks allowed (expect more struggles)

---

## Rollback Strategies

### 1. Simple Rollback
Load last good checkpoint, continue training.

### 2. Rollback + LR Decay
Load checkpoint + reduce learning rate by 50%.
Prevents repeating same mistakes.

### 3. Rollback + Curriculum Reset
Load checkpoint + restart current level with easier settings.
```python
if rollback_count > 2:
    # Restart level with lower fees
    config['fee_bps'] = max(0, config['fee_bps'] - 5)
    config['slippage_bps'] = max(0, config['slippage_bps'] - 2)
```

### 4. Rollback + Buffer Clear
Load checkpoint + clear replay buffer.
Removes bad experiences that caused degradation.
```python
if rollback_count > 1:
    replay_buffer.clear()
```

---

## Implementation Files

### 1. `mcts_trader/training/safe_checkpoint.py` (NEW)
```python
class SafeCheckpointManager:
    ...
```

### 2. `dojo_manager.py` (MODIFY)
- Add `SafeCheckpointManager` integration
- Add `safety` config section handling
- Add rollback logic to training loop

### 3. `config/pretrain_curriculum.json` (MODIFY)
- Add `safety` section to each stage
- More aggressive settings for harder levels

---

## Monitoring & Logging

### Log Format
```
[DOJO] Epoch 150/300 | Level 5 | Reward: +1.23% | Best: +1.45% | Rollbacks: 0
[DOJO] Epoch 200/300 | Level 5 | Reward: -0.50% | Best: +1.45% | Rollbacks: 0
[DOJO] ⚠️ Performance degradation detected! Rolling back to epoch 150...
[DOJO] Epoch 200/300 | Level 5 | Reward: +0.80% | Best: +1.45% | Rollbacks: 1
```

### Metrics to Track
- `epochs_since_improvement`: How long since we saved
- `rollback_count`: How many rollbacks this level
- `best_metric`: Best rolling average achieved
- `current_metric`: Current rolling average

---

## Success Criteria for Safe Training

A level is considered "safely passed" when:
1. Pass threshold met for 50+ consecutive epochs
2. Zero rollbacks in final 100 epochs
3. Performance stable (std < 0.5% over last 50 epochs)

---

## Edge Cases

### 1. First Checkpoint Issue
If model degrades before first save:
- Solution: Always save after first epoch (baseline)

### 2. All Checkpoints Bad
If all N best checkpoints are from bad region:
- Solution: Keep one "golden" checkpoint from level start

### 3. Oscillating Performance
Model goes: good → bad → good → bad:
- Solution: Increase rolling_window to smooth detection

### 4. Gradual Decline
Slow degradation doesn't trigger threshold:
- Solution: Also check absolute drop from level start

---

## Next Steps

1. Implement `SafeCheckpointManager` class
2. Integrate with `dojo_manager.py`
3. Add safety configs to curriculum
4. Test on Level 5 (first hard level)
5. Tune parameters based on results
