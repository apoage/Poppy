from mcts_trader.config.trading_config import main_config
from mcts_trader.agent.improved_model import ImprovedTRMMuZeroModel
import torch

print("Initializing model...")
try:
    model = ImprovedTRMMuZeroModel(**main_config.policy.model)
    print("Model initialized successfully.")
    print(f"Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"Failed to initialize model: {e}")
    import traceback
    traceback.print_exc()
