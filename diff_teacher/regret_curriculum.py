"""
Regret-Based Curriculum Controller.

Adjusts generation difficulty based on agent's regret:
    - Low regret (performing well) -> Increase difficulty
    - High regret (struggling) -> Decrease difficulty

Inspired by PAIRED (Protagonist-Antagonist Induced Regret) algorithm.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class CurriculumState:
    """Current state of the curriculum controller."""

    difficulty: float = 0.5  # Current difficulty [0, 1]
    regret_ema: float = 0.0  # Exponential moving average of regret
    episodes_at_current: int = 0  # Episodes at current difficulty
    total_episodes: int = 0
    difficulty_changes: int = 0

    # History for analysis
    regret_history: List[float] = field(default_factory=list)
    difficulty_history: List[float] = field(default_factory=list)


class RegretCurriculum:
    """
    Curriculum controller based on regret minimization.

    The idea:
        - Generate scenarios at current difficulty
        - Agent tries to maximize PnL
        - Compute regret = optimal_pnl - actual_pnl
        - If regret is low, agent is "solving" current difficulty -> increase
        - If regret is high, agent is struggling -> decrease or maintain

    This creates a natural curriculum where difficulty tracks agent capability.
    """

    def __init__(
        self,
        difficulty_init: float = 0.5,
        difficulty_min: float = 0.1,
        difficulty_max: float = 1.0,
        difficulty_step: float = 0.05,
        regret_low_threshold: float = 0.2,
        regret_high_threshold: float = 0.6,
        ema_alpha: float = 0.1,
        patience: int = 10,  # Episodes before difficulty change
    ):
        """
        Initialize curriculum controller.

        Args:
            difficulty_init: Starting difficulty
            difficulty_min: Minimum difficulty
            difficulty_max: Maximum difficulty
            difficulty_step: How much to adjust per change
            regret_low_threshold: Below this = too easy
            regret_high_threshold: Above this = too hard
            ema_alpha: EMA smoothing factor for regret
            patience: Minimum episodes before changing difficulty
        """
        self.difficulty_min = difficulty_min
        self.difficulty_max = difficulty_max
        self.difficulty_step = difficulty_step
        self.regret_low = regret_low_threshold
        self.regret_high = regret_high_threshold
        self.ema_alpha = ema_alpha
        self.patience = patience

        # State
        self.state = CurriculumState(difficulty=difficulty_init)

        # Recent regrets for variance estimation
        self.recent_regrets = deque(maxlen=100)

    @property
    def difficulty(self) -> float:
        """Current difficulty level."""
        return self.state.difficulty

    @property
    def regret_ema(self) -> float:
        """Current smoothed regret estimate."""
        return self.state.regret_ema

    def update(self, regret: float) -> Tuple[float, str]:
        """
        Update curriculum based on new regret observation.

        Args:
            regret: Regret from latest episode (optimal - actual) / optimal

        Returns:
            new_difficulty: Updated difficulty level
            action: "increase", "decrease", or "maintain"
        """
        # Update EMA
        if self.state.total_episodes == 0:
            self.state.regret_ema = regret
        else:
            self.state.regret_ema = (
                self.ema_alpha * regret + (1 - self.ema_alpha) * self.state.regret_ema
            )

        # Track history
        self.state.regret_history.append(regret)
        self.state.difficulty_history.append(self.state.difficulty)
        self.recent_regrets.append(regret)
        self.state.total_episodes += 1
        self.state.episodes_at_current += 1

        # Determine action
        action = "maintain"

        if self.state.episodes_at_current >= self.patience:
            if self.state.regret_ema < self.regret_low:
                # Performing well -> increase difficulty
                if self.state.difficulty < self.difficulty_max:
                    self.state.difficulty = min(
                        self.state.difficulty + self.difficulty_step,
                        self.difficulty_max,
                    )
                    action = "increase"
                    self.state.episodes_at_current = 0
                    self.state.difficulty_changes += 1

            elif self.state.regret_ema > self.regret_high:
                # Struggling -> decrease difficulty
                if self.state.difficulty > self.difficulty_min:
                    self.state.difficulty = max(
                        self.state.difficulty - self.difficulty_step,
                        self.difficulty_min,
                    )
                    action = "decrease"
                    self.state.episodes_at_current = 0
                    self.state.difficulty_changes += 1

        return self.state.difficulty, action

    def update_batch(self, regrets: np.ndarray) -> Tuple[float, str]:
        """
        Update with a batch of regrets.

        Args:
            regrets: Array of regret values

        Returns:
            new_difficulty, action
        """
        mean_regret = float(np.mean(regrets))
        return self.update(mean_regret)

    def get_conditioning(self) -> Dict[str, float]:
        """
        Get conditioning values for generator based on current state.

        Returns dictionary with:
            - difficulty: Current difficulty level
            - volatility_scale: How much price volatility
            - noise_level: Amount of noise/randomness
            - trend_strength: Strength of trends
            - correlation_breakdown: How much correlations deviate
        """
        d = self.state.difficulty

        return {
            "difficulty": d,
            "volatility_scale": 0.5 + d * 1.5,  # 0.5 to 2.0
            "noise_level": d * 0.5,  # 0 to 0.5
            "trend_strength": 0.3 + d * 0.7,  # 0.3 to 1.0
            "correlation_breakdown": d * 0.3,  # 0 to 0.3
        }

    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get weights for regime sampling based on difficulty.

        Higher difficulty = more volatile/crash regimes.

        Returns:
            Dictionary mapping regime names to sampling weights
        """
        d = self.state.difficulty

        # At low difficulty: mostly sideways/rally
        # At high difficulty: more crash/volatile
        weights = {
            "sideways": max(0.1, 0.4 - d * 0.3),
            "rally": max(0.1, 0.3 - d * 0.2),
            "volatile": 0.1 + d * 0.3,
            "crash": 0.1 + d * 0.3,
        }

        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def sample_regime(self, rng: Optional[np.random.Generator] = None) -> str:
        """
        Sample a regime based on current difficulty.

        Args:
            rng: Random generator

        Returns:
            Regime name ("crash", "rally", "sideways", "volatile")
        """
        if rng is None:
            rng = np.random.default_rng()

        weights = self.get_regime_weights()
        regimes = list(weights.keys())
        probs = list(weights.values())

        return rng.choice(regimes, p=probs)

    def get_metrics(self) -> Dict[str, float]:
        """Get current curriculum metrics."""
        recent = list(self.recent_regrets)

        return {
            "difficulty": self.state.difficulty,
            "regret_ema": self.state.regret_ema,
            "regret_recent_mean": np.mean(recent) if recent else 0.0,
            "regret_recent_std": np.std(recent) if len(recent) > 1 else 0.0,
            "total_episodes": self.state.total_episodes,
            "difficulty_changes": self.state.difficulty_changes,
            "episodes_at_current": self.state.episodes_at_current,
        }

    def save(self, path: str) -> None:
        """Save curriculum state to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "state": {
                "difficulty": self.state.difficulty,
                "regret_ema": self.state.regret_ema,
                "episodes_at_current": self.state.episodes_at_current,
                "total_episodes": self.state.total_episodes,
                "difficulty_changes": self.state.difficulty_changes,
                "regret_history": self.state.regret_history[-1000:],  # Last 1000
                "difficulty_history": self.state.difficulty_history[-1000:],
            },
            "config": {
                "difficulty_min": self.difficulty_min,
                "difficulty_max": self.difficulty_max,
                "difficulty_step": self.difficulty_step,
                "regret_low": self.regret_low,
                "regret_high": self.regret_high,
                "ema_alpha": self.ema_alpha,
                "patience": self.patience,
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load curriculum state from file."""
        with open(path, "r") as f:
            data = json.load(f)

        state = data["state"]
        self.state.difficulty = state["difficulty"]
        self.state.regret_ema = state["regret_ema"]
        self.state.episodes_at_current = state["episodes_at_current"]
        self.state.total_episodes = state["total_episodes"]
        self.state.difficulty_changes = state["difficulty_changes"]
        self.state.regret_history = state.get("regret_history", [])
        self.state.difficulty_history = state.get("difficulty_history", [])

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self.state = CurriculumState(difficulty=self.difficulty_min)
        self.recent_regrets.clear()


class AdaptiveNoiseSchedule:
    """
    Maps curriculum difficulty to noise parameters.

    Used to generate progressively harder market sequences.
    """

    def __init__(self):
        # Noise component ranges (min, max) for difficulty [0, 1]
        self.ranges = {
            "price_noise": (0.0, 1.0),
            "temporal_jitter": (0, 5),
            "time_dilation": (0.0, 0.3),
            "phase_shift_prob": (0.0, 0.5),
        }

    def get_noise_params(self, difficulty: float) -> Dict[str, float]:
        """
        Get noise parameters for given difficulty.

        Args:
            difficulty: [0, 1] difficulty level

        Returns:
            Dictionary of noise parameters
        """
        params = {}
        for name, (min_val, max_val) in self.ranges.items():
            params[name] = min_val + difficulty * (max_val - min_val)
        return params

    def get_progressive_params(
        self, difficulty: float
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Get noise parameters with progressive unlocking.

        At low difficulty, only some noise types are active.
        At higher difficulty, more noise types become active.

        Returns:
            params: Noise parameters
            active: List of active noise types
        """
        active = []
        params = {}

        # Progressive unlocking
        if difficulty >= 0.0:
            active.append("price_noise")
            params["price_noise"] = min(1.0, difficulty * 4)  # Full at 0.25

        if difficulty >= 0.25:
            active.append("temporal_jitter")
            params["temporal_jitter"] = min(5, int((difficulty - 0.25) * 20))

        if difficulty >= 0.5:
            active.append("time_dilation")
            params["time_dilation"] = min(0.3, (difficulty - 0.5) * 0.6)

        if difficulty >= 0.75:
            active.append("phase_shift_prob")
            params["phase_shift_prob"] = min(0.5, (difficulty - 0.75) * 2)

        return params, active


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Regret Curriculum Test")
    print("=" * 60)

    curriculum = RegretCurriculum(
        difficulty_init=0.3,
        difficulty_step=0.1,
        patience=5,
    )

    # Simulate training
    np.random.seed(42)

    print("\nSimulating training episodes...")
    print("-" * 60)

    for episode in range(50):
        # Simulate regret based on difficulty (harder = more regret on average)
        base_regret = curriculum.difficulty * 0.5
        noise = np.random.randn() * 0.1
        regret = np.clip(base_regret + noise, 0, 1)

        # Simulate improvement over time
        regret *= max(0.3, 1 - episode / 100)

        new_diff, action = curriculum.update(regret)

        if action != "maintain" or episode % 10 == 0:
            metrics = curriculum.get_metrics()
            print(
                f"Ep {episode:3d}: regret={regret:.3f}, "
                f"ema={metrics['regret_ema']:.3f}, "
                f"diff={metrics['difficulty']:.2f}, "
                f"action={action}"
            )

    print("\n" + "=" * 60)
    print("Final Metrics:")
    print("=" * 60)
    metrics = curriculum.get_metrics()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test conditioning
    print("\n" + "=" * 60)
    print("Conditioning at various difficulties:")
    print("=" * 60)

    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        curriculum.state.difficulty = d
        cond = curriculum.get_conditioning()
        weights = curriculum.get_regime_weights()
        print(f"\nDifficulty {d:.2f}:")
        print(f"  Conditioning: {cond}")
        print(f"  Regime weights: {weights}")

    # Test noise schedule
    print("\n" + "=" * 60)
    print("Progressive Noise Schedule:")
    print("=" * 60)

    schedule = AdaptiveNoiseSchedule()
    for d in [0.0, 0.25, 0.5, 0.75, 1.0]:
        params, active = schedule.get_progressive_params(d)
        print(f"\nDifficulty {d:.2f}:")
        print(f"  Active: {active}")
        print(f"  Params: {params}")

    # Test save/load
    print("\n" + "=" * 60)
    print("Save/Load Test:")
    print("=" * 60)

    curriculum.save("/tmp/curriculum_test.json")
    print("Saved to /tmp/curriculum_test.json")

    curriculum2 = RegretCurriculum()
    curriculum2.load("/tmp/curriculum_test.json")
    print(f"Loaded difficulty: {curriculum2.difficulty}")
    print(f"Loaded regret_ema: {curriculum2.regret_ema}")
