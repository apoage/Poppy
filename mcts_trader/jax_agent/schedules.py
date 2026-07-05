"""
Training schedules for curriculum-based hyperparameters.

Provides adaptive scheduling for:
- MCTS simulation count (sims warmup)
- Learning rate (TASK_024)
- Other hyperparameters
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional

import optax


@dataclass
class SimsSchedule:
    """
    MCTS simulation count schedule.

    Gradually increases simulations over training epochs to prevent
    early policy lock-in before value function learns meaningful estimates.

    Rationale: With high sims, MCTS quickly converges to deterministic actions.
    Start with low sims for exploration, increase as value function improves.

    Example:
        schedule = SimsSchedule(min_sims=8, max_sims=64, warmup_epochs=100)
        for epoch in range(200):
            sims = schedule.get_sims(epoch)
            # epoch 0: 8 sims, epoch 50: 36 sims, epoch 100+: 64 sims
    """
    min_sims: int = 8
    max_sims: int = 64
    warmup_epochs: int = 100
    schedule_type: Literal['linear', 'step', 'exponential'] = 'linear'

    def get_sims(self, epoch: int) -> int:
        """
        Get simulation count for current epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            Number of MCTS simulations to use
        """
        if self.warmup_epochs <= 0:
            return self.max_sims

        progress = min(1.0, epoch / self.warmup_epochs)

        if self.schedule_type == 'linear':
            # Linear interpolation from min to max
            sims = self.min_sims + progress * (self.max_sims - self.min_sims)

        elif self.schedule_type == 'step':
            # Step schedule: 8 -> 16 -> 32 -> 64
            if progress < 0.25:
                sims = self.min_sims
            elif progress < 0.5:
                sims = max(16, self.min_sims)
            elif progress < 0.75:
                sims = max(32, self.min_sims)
            else:
                sims = self.max_sims

        elif self.schedule_type == 'exponential':
            # Exponential: more time at low sims (log-space interpolation)
            log_min = math.log2(max(1, self.min_sims))
            log_max = math.log2(max(1, self.max_sims))
            sims = 2 ** (log_min + progress * (log_max - log_min))

        else:
            sims = self.max_sims

        return int(sims)

    def __repr__(self) -> str:
        return (f"SimsSchedule(min={self.min_sims}, max={self.max_sims}, "
                f"warmup={self.warmup_epochs}, type={self.schedule_type})")


@dataclass
class LRSchedule:
    """
    Learning rate schedule (for future TASK_024).

    Placeholder for cosine annealing, warmup, etc.
    """
    initial_lr: float = 1e-4
    final_lr: float = 1e-6
    warmup_epochs: int = 10
    total_epochs: int = 1000
    schedule_type: Literal['constant', 'cosine', 'linear'] = 'constant'

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch."""
        if self.schedule_type == 'constant':
            return self.initial_lr

        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs

        # Main schedule
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        if self.schedule_type == 'cosine':
            # Cosine annealing
            return self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'linear':
            return self.initial_lr + progress * (self.final_lr - self.initial_lr)
        else:
            return self.initial_lr

    def to_optax_schedule(self, steps_per_epoch: int = 100) -> optax.Schedule:
        """Convert to optax-compatible schedule.

        TASK_024: Create optax schedule for use with adam/sgd.

        Args:
            steps_per_epoch: Gradient steps per epoch (for step conversion)

        Returns:
            optax.Schedule function mapping step -> learning rate
        """
        # Guard: ensure warmup doesn't exceed total epochs
        effective_warmup_epochs = min(self.warmup_epochs, max(0, self.total_epochs - 1))

        warmup_steps = effective_warmup_epochs * steps_per_epoch
        total_steps = self.total_epochs * steps_per_epoch

        # Guard: total_steps must be > warmup_steps for decay to work
        if total_steps <= warmup_steps:
            total_steps = warmup_steps + steps_per_epoch  # At least 1 epoch of decay

        if self.schedule_type == 'constant':
            return optax.constant_schedule(self.initial_lr)

        elif self.schedule_type == 'cosine':
            if effective_warmup_epochs > 0:
                # Note: optax.warmup_cosine_decay_schedule expects decay_steps to be
                # the TOTAL steps (warmup + decay), not just the decay portion.
                # It internally calculates: decay_steps - warmup_steps for the cosine part.
                return optax.warmup_cosine_decay_schedule(
                    init_value=self.final_lr,
                    peak_value=self.initial_lr,
                    warmup_steps=warmup_steps,
                    decay_steps=total_steps,  # Total steps, not decay portion!
                    end_value=self.final_lr
                )
            else:
                return optax.cosine_decay_schedule(
                    init_value=self.initial_lr,
                    decay_steps=total_steps,
                    alpha=self.final_lr / self.initial_lr
                )

        elif self.schedule_type == 'linear':
            if effective_warmup_epochs > 0:
                # For manually-joined schedules, we need just the decay portion
                decay_portion = total_steps - warmup_steps
                return optax.join_schedules(
                    schedules=[
                        optax.linear_schedule(self.final_lr, self.initial_lr, warmup_steps),
                        optax.linear_schedule(self.initial_lr, self.final_lr, decay_portion),
                    ],
                    boundaries=[warmup_steps]
                )
            else:
                return optax.linear_schedule(self.initial_lr, self.final_lr, total_steps)

        else:
            return optax.constant_schedule(self.initial_lr)


def create_scheduled_optimizer(
    lr_schedule: LRSchedule,
    steps_per_epoch: int = 100,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Create optimizer with learning rate schedule.

    TASK_024: Factory for scheduled Adam optimizer.

    Args:
        lr_schedule: LRSchedule instance
        steps_per_epoch: Gradient steps per epoch
        weight_decay: L2 regularization coefficient
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Optax GradientTransformation chain
    """
    schedule = lr_schedule.to_optax_schedule(steps_per_epoch)

    transforms = [
        optax.clip_by_global_norm(max_grad_norm),
    ]

    if weight_decay > 0:
        transforms.append(optax.add_decayed_weights(weight_decay))

    transforms.append(optax.scale_by_adam())
    transforms.append(optax.scale_by_schedule(schedule))
    transforms.append(optax.scale(-1.0))  # Gradient descent

    return optax.chain(*transforms)
