import os

# --- Authentication ---
# Keys should be loaded from environment variables for security
API_KEY = os.environ.get('KRAKEN_API_KEY', '')
API_SECRET = os.environ.get('KRAKEN_API_SECRET', '')

# --- Trading Mode ---
# 'SPOT' or 'MARGIN'
# Spot: Buys assets with cash. Sells assets held.
# Margin: Opens leveraged positions (Long/Short). Requires Kraken Margin enabled.
TRADING_MODE = 'SPOT' 

# For Margin mode only (ignored in Spot)
LEVERAGE = 2 

# --- Execution Parameters ---
# Confidence threshold (0.0 - 1.0) to use Market Order
# Signals below this use Limit Orders
MARKET_ORDER_THRESHOLD = 0.80 

# Seconds to wait for a Limit Order to fill before cancelling
LIMIT_ORDER_TIMEOUT = 60 

# --- Risk Management ---
# Maximum % of Equity to risk per trade
# Note: In Spot, this is % of cash to spend. In Margin, it's position size value.
MAX_POSITION_PCT = 0.10 

# Hard Stop: If drawdown exceeds this % from session start, STOP TRADING.
# 0.0 = Disabled
MAX_DRAWDOWN_HARD_STOP = 0.05 

# --- Session Limits ---
# Auto-close everything and exit after N hours
# 0 = Run indefinitely
RUN_HOURS = 6 

# --- Simulation ---
# If True, connects to Kraken but does NOT send actual orders (logs them only)
# Use this for testing the loop without risking funds.
DRY_RUN = True 

# --- Model & Data ---
CHECKPOINT_DIR = os.path.abspath("jax_ckpt_fast/poppy_v2_L0_phase_A")
N_ASSETS = 64
LOOKBACK = 60
SIMULATIONS = 24

# --- Model Architecture (must match checkpoint) ---
MODEL_SIZE = 'small'  # small=64, medium=96, large=160
STATE_DIM = 64  # Must match MODEL_SIZE
ACTION_SPACE_SIZE = 320  # N_ASSETS * 5 (sell/hold/buy/limit_sell/limit_buy)
SUPPORT_SIZE = 401  # PopArt: [-10, +10] with 0.05 resolution
USE_PREDICTION_HEAD = True  # Enable prediction head (trained with --use-prediction-loss)
USE_POPART = True  # Enable PopArt normalization
