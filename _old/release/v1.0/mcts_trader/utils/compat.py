import numpy as np
from lzero.policy import utils
import lzero.policy.muzero
import logging
import gymnasium as gym

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

def patch_lightzero_gym_compat():
    """
    Patch LightZero to handle Gymnasium 1.x reset() signature.
    Gymnasium returns (obs, info), old Gym returned obs only.
    DI-engine expects just obs.
    """
    try:
        from ding.envs import BaseEnvManager

        # Avoid double patching
        if getattr(BaseEnvManager, '_patched_for_gymnasium', False):
            return

        original_reset = BaseEnvManager.reset

        def patched_reset(self, *args, **kwargs):
            # Call original reset
            result = original_reset(self, *args, **kwargs)
            
            # Handle list of tuples (from vector env)
            if isinstance(result, list):
                new_result = []
                for item in result:
                    if isinstance(item, tuple) and len(item) == 2:
                        new_result.append(item[0])
                    else:
                        new_result.append(item)
                return new_result
            
            # Handle single tuple
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]
                
            return result

    except ImportError:
        logging.warning("Could not import ding.envs.BaseEnvManager to apply patch")

