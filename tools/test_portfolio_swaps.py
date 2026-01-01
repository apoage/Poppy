from train_jax_fast import FastTradingEnv
import numpy as np
import logging

# Configure logging to suppress debug noise
logging.basicConfig(level=logging.INFO)

def test_swap():
    """Test direct asset swap"""
    print("Testing Swap Mechanics...")
    
    # Init env with swaps enabled
    env = FastTradingEnv(
        synthetic_level=0, 
        n_assets=64, 
        enable_swaps=True,
        max_steps=200 # Note: code uses max_steps internally but constructor might map it if I didn't fix signature fully. 
        # I fixed the call site in main(), but let's check init signature
    )
    # In train_jax_fast.py __init__ signature is:
    # def __init__(..., max_steps: int = 200, ...):
    # But I might have passed steps_per_episode in main() which I fixed.
    # Here I should pass max_steps.
    
    env.reset()

    # Manually buy asset 0 (Action: 0*3 + 2 = 2)
    # Asset 0 is SYN_00
    print("Buying Asset 0...")
    obs, reward, done, truncated, info = env.step(2)
    
    # Access positions via dict
    def get_shares(p_env, idx):
        if idx in p_env.portfolio.state.positions:
            return p_env.portfolio.state.positions[idx].size
        return 0.0

    shares_0 = get_shares(env, 0)
    print(f"Holdings after buy: {shares_0:.4f} shares of Asset 0")
    
    if shares_0 <= 0:
        print("❌ Failed to buy asset 0. Cannot test swap.")
        return

    # Swap asset 0 -> asset 1
    # Basic actions = 64*3 = 192
    # Swap source=0, target=1
    # Swap Index = 0*64 + 1 = 1
    # Action = 192 + 1 = 193
    swap_action = 192 + 0 * 64 + 1
    
    print("Swapping Asset 0 -> Asset 1...")
    obs, reward, done, truncated, info = env.step(swap_action)

    shares_0_after = get_shares(env, 0)
    shares_1_after = get_shares(env, 1)
    
    print(f"Holdings after swap:")
    print(f"  Asset 0: {shares_0_after:.4f}")
    print(f"  Asset 1: {shares_1_after:.4f}")

    assert shares_0_after == 0, "Source asset should be empty after swap"
    assert shares_1_after > 0, "Target asset should have shares after swap"

    print("✅ Swap mechanics working")

if __name__ == "__main__":
    test_swap()
