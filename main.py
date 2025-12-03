import sys
import os
import numpy as np
from lzero.entry import train_muzero
from lzero.policy import utils
import lzero.policy.muzero # Import the module where the function is used
from mcts_trader.config.trading_config import main_config, create_config
from mcts_trader.agent.wrapper import TRMMuZeroModel
from ding.utils import MODEL_REGISTRY

# Monkey-patch mz_network_output_unpack to ensure reward is list (fix for C++ binding error)
original_unpack = utils.mz_network_output_unpack
def patched_unpack(network_output):
    latent_state, reward, value, policy_logits = original_unpack(network_output)
    
    try:
        # Handle Tensor -> Numpy
        if hasattr(reward, 'detach'): 
            reward = reward.detach().cpu().numpy()
            
        # Handle Numpy
        if isinstance(reward, np.ndarray):
            if reward.ndim == 2 and reward.shape[1] == 1:
                reward = reward.squeeze(1)
            elif reward.ndim == 0: # Scalar
                reward = np.array([reward])
            reward = reward.tolist()
            
        # Handle List
        elif isinstance(reward, list):
            if len(reward) > 0 and isinstance(reward[0], list) and len(reward[0]) == 1:
                 reward = [x[0] for x in reward]
        
        # Fallback ensure it is a list of scalars
        if isinstance(reward, list) and len(reward) > 0 and isinstance(reward[0], list):
             # If still list of lists (e.g. [ [0.5], [0.3] ] but len > 1 check failed above?)
             # flatten blindly
             flat = []
             for x in reward:
                 if isinstance(x, list): flat.append(x[0])
                 else: flat.append(x)
             reward = flat

    except Exception as e:
        print(f"Error in patched_unpack: {e}")

    return latent_state, reward, value, policy_logits

# Apply patch to both locations to be safe
utils.mz_network_output_unpack = patched_unpack
lzero.policy.muzero.mz_network_output_unpack = patched_unpack

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
