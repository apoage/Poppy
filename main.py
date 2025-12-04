import sys
import os
import numpy as np
from lzero.entry import train_muzero
from lzero.policy import utils
import lzero.policy.muzero # Import the module where the function is used
from mcts_trader.config.trading_config import main_config, create_config
from mcts_trader.agent.wrapper import TRMMuZeroModel
from mcts_trader.utils.compat import apply_lightzero_patches
from ding.utils import MODEL_REGISTRY
import torch

# Enable TF32 for faster matmul on Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Apply critical compatibility patches
apply_lightzero_patches()

# Register our custom model
MODEL_REGISTRY.register('trm_muzero', TRMMuZeroModel)

def main():
    """
    Main entry point for TRM+MCTS Trading Agent.
    """
    print("Starting TRM+MCTS Trading Agent Training...")
    
    # Instantiate the custom model
    model = TRMMuZeroModel(**main_config.policy.model)

    # Run the pipeline
    # This will:
    # 1. Create the environment (TradingEnv)
    # 2. Create the policy (MuZeroPolicy with TRMMuZeroModel)
    # 3. Run the collection/training loop
    train_muzero(
        [main_config, create_config],
        seed=0,
        max_env_step=int(1e7),
        model=model
    )

if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    # Add project root to path
    sys.path.append(os.getcwd())
    main()
