from mcts_trader.env.trading_env import TradingEnv
import logging

logging.basicConfig(level=logging.INFO)

print("Initializing TradingEnv...")
try:
    env = TradingEnv()
    print("TradingEnv initialized.")
    print("Resetting env...")
    obs = env.reset()
    print("Env reset successful.")
except Exception as e:
    print(f"Failed to init env: {e}")
    import traceback
    traceback.print_exc()
