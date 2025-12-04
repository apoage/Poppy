import numpy as np
from lzero.policy import utils
import lzero.policy.muzero
import logging

def apply_lightzero_patches():
    """
    Apply critical patches for LightZero compatibility.
    Fixes C++ binding errors where reward is expected as list.
    """
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
                 flat = []
                 for x in reward:
                     if isinstance(x, list): flat.append(x[0])
                     else: flat.append(x)
                 reward = flat

        except Exception as e:
            logging.error(f"Error in patched_unpack: {e}")

        return latent_state, reward, value, policy_logits

    # Apply patch to both locations to be safe
    utils.mz_network_output_unpack = patched_unpack
    lzero.policy.muzero.mz_network_output_unpack = patched_unpack
    logging.info("Applied LightZero monkey-patches successfully.")
