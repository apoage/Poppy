"""
Early Graduation Module (H4)

Allows training to exit early when the model has mastered the current
difficulty level. Useful for curriculum learning.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyGraduationTracker:
    """
    Tracks consecutive epochs at max difficulty for early graduation.

    When enabled, training exits cleanly after N epochs at max difficulty,
    signaling that the model is ready to advance to a harder level.
    """
    threshold: int = 0  # 0 = disabled
    consecutive_max: int = 0
    difficulty_threshold: float = 0.99  # Consider "max" when >= this

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def update(self, current_difficulty: float) -> bool:
        """
        Update tracker with current difficulty level.

        Args:
            current_difficulty: Current difficulty (0.0 to 1.0)

        Returns:
            True if should graduate (exit training)
        """
        if not self.enabled:
            return False

        if current_difficulty >= self.difficulty_threshold:
            self.consecutive_max += 1
            if self.consecutive_max >= self.threshold:
                return True
        else:
            self.consecutive_max = 0  # Reset if difficulty drops

        return False

    def should_graduate(self) -> bool:
        """Check if graduation threshold has been met."""
        if not self.enabled:
            return False
        return self.consecutive_max >= self.threshold

    def get_status_message(self) -> Optional[str]:
        """Get graduation status message if graduating."""
        if self.should_graduate():
            return (
                f"\n🎓 EARLY GRADUATION: Difficulty at max for {self.consecutive_max} epochs\n"
                f"   Model has mastered this level - saving checkpoint and exiting"
            )
        return None


def create_early_graduation_tracker(threshold: int = 0) -> EarlyGraduationTracker:
    """Factory function for creating tracker."""
    tracker = EarlyGraduationTracker(threshold=threshold)
    if tracker.enabled:
        print(f"🎓 Early graduation enabled: will graduate after {threshold} epochs at max difficulty")
    return tracker
