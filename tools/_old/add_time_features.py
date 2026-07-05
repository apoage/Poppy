#!/usr/bin/env python3
"""
Patch to add time-of-day features to MultiAssetStateEncoder.
This adds 4 channels: sin/cos for hour-of-day and day-of-week.
"""

import torch
import math
from datetime import datetime

def encode_time_features_cyclic(timestamp: int, device='cpu') -> torch.Tensor:
    """
    Cyclic encoding of time features (preserves periodicity).

    Args:
        timestamp: Unix timestamp (seconds)
        device: torch device

    Returns:
        Tensor of shape (4, 64, 64):
        - Channel 0: sin(hour of day)
        - Channel 1: cos(hour of day)
        - Channel 2: sin(day of week)
        - Channel 3: cos(day of week)
    """
    dt = datetime.fromtimestamp(timestamp)

    # Hour (24-hour cycle)
    hour_radians = (dt.hour + dt.minute / 60.0) / 24.0 * 2 * math.pi
    hour_sin = math.sin(hour_radians)
    hour_cos = math.cos(hour_radians)

    # Weekday (7-day cycle, 0=Monday, 6=Sunday)
    weekday_radians = dt.weekday() / 7.0 * 2 * math.pi
    weekday_sin = math.sin(weekday_radians)
    weekday_cos = math.cos(weekday_radians)

    # Create spatial channels (broadcast to 64x64)
    H, W = 64, 64
    channels = torch.tensor([
        [[hour_sin] * W] * H,
        [[hour_cos] * W] * H,
        [[weekday_sin] * W] * H,
        [[weekday_cos] * W] * H,
    ], dtype=torch.float32, device=device)

    return channels


# Monkey-patch MultiAssetStateEncoder
def patch_encoder():
    """Add time features to encoder.encode() method"""
    from mcts_trader.encoder.state import MultiAssetStateEncoder

    # Save original encode method
    original_encode = MultiAssetStateEncoder.encode

    def encode_with_time(self):
        """Enhanced encode that adds time features"""
        # Call original encode
        result = original_encode(self)
        combined = result['combined']  # (27, 64, 64)

        # Add time features if timestamp available
        if hasattr(self, 'current_timestamp') and self.current_timestamp is not None:
            time_features = encode_time_features_cyclic(
                self.current_timestamp,
                device=self.device
            )
            combined = torch.cat([combined, time_features], dim=0)  # (31, 64, 64)

        return {'combined': combined}

    # Apply patch
    MultiAssetStateEncoder.encode = encode_with_time

    # Also patch __init__ to add current_timestamp attribute
    original_init = MultiAssetStateEncoder.__init__

    def init_with_timestamp(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.current_timestamp = None  # Will be set by update()

    MultiAssetStateEncoder.__init__ = init_with_timestamp

    print("✅ Patched MultiAssetStateEncoder to include time features")
    print("   Added 4 channels: sin/cos(hour), sin/cos(weekday)")
    print("   New observation shape: (31, 64, 64)")


# Patch HistoricalMarket to pass timestamp
def patch_historical_market():
    """Add timestamp to market data"""
    from mcts_trader.data.history import HistoricalMarket

    original_get_current_data = HistoricalMarket.get_current_data

    def get_current_data_with_timestamp(self):
        data = original_get_current_data(self)
        data['_timestamp'] = self.current_time
        return data

    HistoricalMarket.get_current_data = get_current_data_with_timestamp

    print("✅ Patched HistoricalMarket to include timestamp in data")


# Patch encoder.update to store timestamp
def patch_encoder_update():
    """Store timestamp when updating encoder"""
    from mcts_trader.encoder.state import MultiAssetStateEncoder

    original_update = MultiAssetStateEncoder.update

    def update_with_timestamp(self, market_data):
        # Extract and store timestamp
        if '_timestamp' in market_data:
            self.current_timestamp = market_data['_timestamp']

        # Call original update
        original_update(self, market_data)

    MultiAssetStateEncoder.update = update_with_timestamp

    print("✅ Patched MultiAssetStateEncoder.update to store timestamp")


# Apply all patches
def apply_all_patches():
    """Apply all time feature patches"""
    print("="*70)
    print("Applying Time Feature Patches")
    print("="*70)

    patch_encoder()
    patch_historical_market()
    patch_encoder_update()

    print()
    print("="*70)
    print("Patches Applied Successfully!")
    print("="*70)
    print()
    print("Changes:")
    print("  - Observation shape: (27, 64, 64) → (31, 64, 64)")
    print("  - Added channels:")
    print("    • sin(hour of day)")
    print("    • cos(hour of day)")
    print("    • sin(day of week)")
    print("    • cos(day of week)")
    print()
    print("Benefits:")
    print("  - Model can learn time-of-day patterns")
    print("  - Captures trading session effects (Asia/Europe/US)")
    print("  - Learns weekday patterns (Monday volatility, etc.)")
    print()
    print("Next steps:")
    print("  1. Update JAX model to accept (31, 64, 64) input")
    print("  2. Update observation space in TradingEnv")
    print("  3. Retrain model with new features")
    print()


# Test function
def test_time_features():
    """Test time feature encoding"""
    print("="*70)
    print("Testing Time Feature Encoding")
    print("="*70)

    # Test various times
    test_cases = [
        (datetime(2024, 1, 1, 0, 0).timestamp(), "Monday 00:00"),    # Start of week
        (datetime(2024, 1, 1, 12, 0).timestamp(), "Monday 12:00"),   # Midday
        (datetime(2024, 1, 5, 18, 0).timestamp(), "Friday 18:00"),   # End of week
        (datetime(2024, 1, 7, 0, 0).timestamp(), "Sunday 00:00"),    # Weekend
    ]

    for timestamp, description in test_cases:
        features = encode_time_features_cyclic(int(timestamp), device='cpu')

        print(f"\n{description}:")
        print(f"  Hour sin/cos: {features[0, 0, 0]:.4f}, {features[1, 0, 0]:.4f}")
        print(f"  Weekday sin/cos: {features[2, 0, 0]:.4f}, {features[3, 0, 0]:.4f}")

        # Verify shape
        assert features.shape == (4, 64, 64), f"Wrong shape: {features.shape}"

    print("\n✅ Time feature encoding test passed!")


# Main function for standalone use
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Add time-of-day features to TradingEnv'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run tests only')
    parser.add_argument('--apply', action='store_true',
                        help='Apply patches (call this in training script)')

    args = parser.parse_args()

    if args.test:
        test_time_features()
    elif args.apply:
        apply_all_patches()
    else:
        print("Usage:")
        print("  python add_time_features.py --test    # Test encoding")
        print("  python add_time_features.py --apply   # Apply patches")
        print()
        print("Or import in training script:")
        print("  from add_time_features import apply_all_patches")
        print("  apply_all_patches()  # Call before creating env")


if __name__ == "__main__":
    main()
