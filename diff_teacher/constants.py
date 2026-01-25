"""Central constants for diff_teacher module.

MED-002: Standardized OHLCV indices to prevent inconsistent usage.
"""

# Standard OHLCV ordering (same as most data sources)
OHLCV_OPEN = 0
OHLCV_HIGH = 1
OHLCV_LOW = 2
OHLCV_CLOSE = 3
OHLCV_VOLUME = 4

# Convenience dict for named access
OHLCV_IDX = {
    'open': OHLCV_OPEN,
    'high': OHLCV_HIGH,
    'low': OHLCV_LOW,
    'close': OHLCV_CLOSE,
    'volume': OHLCV_VOLUME,
}
