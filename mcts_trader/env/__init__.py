from ding.utils import ENV_REGISTRY
from .trading_env import TradingEnv
# Import the wrapper to ensure it registers 'wrapped_trading_env' class
from .wrapper import DICompatibleTradingEnv
