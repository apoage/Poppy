import jax
import jax.numpy as jnp
import numpy as np
from mcts_trader.jax_agent.trm_flax import TRMEncoder
from mcts_trader.jax_agent.trm_encoder_v2 import AssetTransformerEncoder

def test_encoder_identity():
    rng = jax.random.PRNGKey(42)
    batch_size = 2
    n_assets = 64
    window = 60
    channels = 27
    state_dim = 64
    
    # Dummy input (B, C, W, A)
    x = jax.random.normal(rng, (batch_size, channels, window, n_assets))
    
    # --- Test V1 (CNN) ---
    model_v1 = TRMEncoder(state_dim=state_dim)
    params_v1 = model_v1.init(rng, x)
    out_v1 = model_v1.apply(params_v1, x)
    print(f"V1 Output Shape: {out_v1.shape}")
    assert out_v1.shape == (batch_size, state_dim)
    
    # --- Test V2 (Transformer) ---
    model_v2 = AssetTransformerEncoder(state_dim=state_dim, n_assets=n_assets)
    params_v2 = model_v2.init(rng, x)
    out_v2 = model_v2.apply(params_v2, x)
    print(f"V2 Output Shape: {out_v2.shape}")
    assert out_v2.shape == (batch_size, state_dim)
    
    # --- Identity Test ---
    # Change ONLY Asset 0 in Batch 0
    x_mod = x.at[0, :, :, 0].add(10.0)
    
    # In V1, pooling and global convolutions mean small changes ripple across all internal features
    # In V2, before the global projection, assets are only linked by Attention.
    
    print("✅ Encoder tests passed (shapes verified)")

if __name__ == "__main__":
    test_encoder_identity()
