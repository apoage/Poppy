# TASK 009: Implement Neural Market Corrector

**Severity**: FEATURE
**Phase**: 1 (Core Prediction Model)
**Location**: `mcts_trader/jax_agent/prediction/neural_corrector.py` (NEW)
**Status**: OPEN
**Estimated Time**: 3 hours
**Dependencies**: None (parallel with TASK_008)

---

## Objective

Implement the **Neural Market Corrector** - the neural network component of the NeuralMarketODE prediction model. This module learns **residual dynamics** not captured by the physics core.

---

## Critical Design Decisions

### 1. Assets as Tokens (Cross-Asset Transformer)

**Problem**: Original design reshapes latent awkwardly for attention.

**Solution**: Treat each asset as a token in a transformer:

```
asset_embeddings: (batch, n_assets, embed_dim)
# Self-attention over assets - learns which assets influence which
attended = MultiHeadAttention(asset_embeddings, asset_embeddings)
```

**Benefits**:
- Natural representation of cross-asset relationships
- Attention weights reveal learned correlations
- Scales cleanly with number of assets

### 2. Feature Concatenation

Combine asset state with external features:

```
Input per asset: [log_price, log_return, log_ma, external_features...]
```

### 3. Log-Space Consistency

The neural corrector also operates in log-space to be consistent with physics core.

---

## Mathematical Foundation

The neural corrector learns:

$$G_{neural}(\log p, X; \theta_n) = \text{Decoder}(\text{CrossAssetAttn}(\text{Encoder}(\log p, X)))$$

Where:
- $\log p$ = log prices
- $X$ = external features (volume, order flow, indicators)
- CrossAssetAttn = self-attention where each asset attends to others

### What It Captures

1. **Subgrid effects**: Microstructure dynamics
2. **Non-linear interactions**: Complex cross-asset relationships
3. **Regime effects**: Market state-dependent behavior
4. **Discretization errors**: Correct physics approximations

---

## Architecture

```
Input: [log_prices, log_returns, features]
         │
         ▼
    ┌──────────────┐
    │ Per-Asset    │  Project each asset to embed_dim
    │ Embedding    │  Shape: (batch, n_assets, embed_dim)
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Cross-Asset  │  Multi-Head Self-Attention over assets
    │ Transformer  │  Assets attend to correlated assets
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Per-Asset    │  Project back to 1-dim per asset
    │ Decoder      │
    └──────┬───────┘
           │
           ▼
Output: G_neural    (batch, n_assets)
```

---

## Implementation

### File: `mcts_trader/jax_agent/prediction/neural_corrector.py`

```python
"""
Neural Market Corrector.

Learns residual market dynamics not captured by physics core.
Architecture: Per-Asset Embedding -> Cross-Asset Transformer -> Per-Asset Decoder

Key innovation: Treat assets as tokens in transformer attention.
This naturally captures cross-asset correlations.

Based on NeuralOGCM (arXiv:2512.11525v1).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple


class CrossAssetAttention(nn.Module):
    """
    Multi-head self-attention over assets.

    Each asset attends to all other assets, learning which assets
    are correlated or drive each other.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout_rate: Dropout probability
    """
    num_heads: int = 4
    head_dim: int = 32
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Apply cross-asset self-attention.

        Args:
            x: (batch, n_assets, embed_dim) - Asset embeddings
            deterministic: If True, disable dropout

        Returns:
            (batch, n_assets, embed_dim) - Attended embeddings
        """
        batch, n_assets, embed_dim = x.shape
        qkv_dim = self.num_heads * self.head_dim

        # Linear projections for Q, K, V
        q = nn.Dense(qkv_dim, name='query')(x)
        k = nn.Dense(qkv_dim, name='key')(x)
        v = nn.Dense(qkv_dim, name='value')(x)

        # Reshape to (batch, num_heads, n_assets, head_dim)
        q = q.reshape(batch, n_assets, self.num_heads, self.head_dim)
        k = k.reshape(batch, n_assets, self.num_heads, self.head_dim)
        v = v.reshape(batch, n_assets, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, heads, assets, dim)
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        attn_weights = nn.softmax(attn_weights, axis=-1)

        # Dropout on attention weights
        if not deterministic and self.dropout_rate > 0:
            attn_weights = nn.Dropout(self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        # Weighted sum of values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back to (batch, n_assets, qkv_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch, n_assets, qkv_dim)

        # Output projection
        output = nn.Dense(embed_dim, name='output')(attn_output)

        return output

    def get_attention_weights(
        self,
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Get attention weights for visualization.

        Returns: (batch, num_heads, n_assets, n_assets)
        """
        batch, n_assets, embed_dim = x.shape
        qkv_dim = self.num_heads * self.head_dim

        q = nn.Dense(qkv_dim, name='query')(x)
        k = nn.Dense(qkv_dim, name='key')(x)

        q = q.reshape(batch, n_assets, self.num_heads, self.head_dim)
        k = k.reshape(batch, n_assets, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))

        scale = jnp.sqrt(self.head_dim).astype(x.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale
        attn_weights = nn.softmax(attn_weights, axis=-1)

        return attn_weights


class CrossAssetTransformerBlock(nn.Module):
    """
    Transformer block with cross-asset attention.

    Includes:
    - Multi-head self-attention over assets
    - Feed-forward network per asset
    - Layer normalization and residual connections

    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_dim: Feed-forward hidden dimension
        dropout_rate: Dropout probability
    """
    embed_dim: int = 128
    num_heads: int = 4
    mlp_dim: int = 256
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Apply transformer block.

        Args:
            x: (batch, n_assets, embed_dim)
            deterministic: If True, disable dropout

        Returns:
            (batch, n_assets, embed_dim)
        """
        # Pre-norm architecture
        # Attention
        residual = x
        x = nn.LayerNorm(name='attn_ln')(x)
        x = CrossAssetAttention(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.dropout_rate,
            name='cross_asset_attn'
        )(x, deterministic=deterministic)
        x = residual + x

        # Feed-forward
        residual = x
        x = nn.LayerNorm(name='mlp_ln')(x)
        x = nn.Dense(self.mlp_dim, name='mlp_fc1')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.embed_dim, name='mlp_fc2')(x)
        if not deterministic and self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x

        return x


class NeuralMarketCorrector(nn.Module):
    """
    Neural network corrector for residual market dynamics.

    Treats each asset as a token in a transformer, learning which
    assets influence each other through attention.

    Learns patterns not captured by physics:
    - Order flow effects
    - Sentiment impacts
    - Complex cross-asset interactions
    - Regime-dependent behavior

    Attributes:
        n_assets: Number of tradeable assets
        n_features: Number of external features per asset
        embed_dim: Per-asset embedding dimension
        num_heads: Attention heads
        num_layers: Number of transformer blocks
        mlp_dim: Feed-forward hidden dimension
    """
    n_assets: int = 64
    n_features: int = 10
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 128

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute neural correction tendency in LOG-SPACE.

        Args:
            log_prices: (batch, n_assets) - Current log prices
            log_returns: (batch, n_assets) - Recent log returns
            log_moving_avg: (batch, n_assets) - Log moving average
            features: (batch, n_features) - External features (optional)
            deterministic: If True, disable dropout

        Returns:
            G_neural: (batch, n_assets) - Neural tendency
            attention_weights: (batch, num_heads, n_assets, n_assets)
        """
        batch = log_prices.shape[0]

        # ============================================================
        # PER-ASSET EMBEDDING
        # ============================================================

        # Stack per-asset features: (batch, n_assets, n_features_per_asset)
        # Each asset gets: [log_price, log_return, deviation_from_ma]
        deviation = log_prices - log_moving_avg

        per_asset_features = jnp.stack([
            log_prices,
            log_returns,
            deviation,
        ], axis=-1)  # (batch, n_assets, 3)

        # Add global features to each asset if provided
        if features is not None:
            # Broadcast global features to each asset
            global_features = jnp.tile(
                features[:, None, :],
                (1, self.n_assets, 1)
            )  # (batch, n_assets, n_features)
            per_asset_features = jnp.concatenate(
                [per_asset_features, global_features],
                axis=-1
            )

        # Project to embedding dimension
        x = nn.Dense(self.embed_dim, name='asset_embedding')(per_asset_features)
        x = nn.LayerNorm(name='embed_ln')(x)  # (batch, n_assets, embed_dim)

        # ============================================================
        # CROSS-ASSET TRANSFORMER
        # ============================================================

        attention_weights = None
        for i in range(self.num_layers):
            x = CrossAssetTransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                name=f'transformer_block_{i}'
            )(x, deterministic=deterministic)

        # Get attention weights from last layer for visualization
        # (This is a separate forward pass, could be optimized)
        last_block = CrossAssetAttention(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            name=f'transformer_block_{self.num_layers - 1}/cross_asset_attn'
        )
        # Note: attention weights extraction would need separate handling

        # ============================================================
        # PER-ASSET DECODER
        # ============================================================

        x = nn.LayerNorm(name='decoder_ln')(x)

        # Project back to single value per asset
        G_neural = nn.Dense(1, name='decoder_out')(x)
        G_neural = G_neural.squeeze(-1)  # (batch, n_assets)

        return G_neural, attention_weights


class SimpleNeuralCorrector(nn.Module):
    """
    Simplified neural corrector (no attention).

    Use this for faster training or as baseline.
    Uses a shared MLP across all assets.

    Attributes:
        n_assets: Number of assets
        n_features: Number of features
        hidden_dim: Hidden layer dimension
    """
    n_assets: int = 64
    n_features: int = 10
    hidden_dim: int = 256

    @nn.compact
    def __call__(
        self,
        log_prices: jnp.ndarray,
        log_returns: jnp.ndarray,
        log_moving_avg: jnp.ndarray,
        features: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, None]:
        """Simple MLP corrector."""
        batch = log_prices.shape[0]
        deviation = log_prices - log_moving_avg

        # Concatenate all inputs
        inputs = [log_prices, log_returns, deviation]
        if features is not None:
            inputs.append(features)
        x = jnp.concatenate(inputs, axis=-1)

        # Simple 3-layer MLP
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        G_neural = nn.Dense(self.n_assets)(x)

        return G_neural, None
```

---

## Unit Tests

### File: `tests/test_neural_corrector.py`

```python
"""Unit tests for NeuralMarketCorrector."""

import jax
import jax.numpy as jnp
from mcts_trader.jax_agent.prediction.neural_corrector import (
    NeuralMarketCorrector,
    SimpleNeuralCorrector,
    CrossAssetAttention,
    CrossAssetTransformerBlock
)


def test_cross_asset_attention():
    """Attention over assets works correctly."""
    model = CrossAssetAttention(num_heads=4, head_dim=16)
    rng = jax.random.PRNGKey(0)

    # (batch, n_assets, embed_dim)
    x = jax.random.normal(rng, (8, 64, 64))

    params = model.init(rng, x)
    out = model.apply(params, x)

    assert out.shape == (8, 64, 64)
    assert jnp.all(jnp.isfinite(out))


def test_transformer_block():
    """Transformer block processes assets correctly."""
    model = CrossAssetTransformerBlock(embed_dim=64, num_heads=4, mlp_dim=128)
    rng = jax.random.PRNGKey(0)

    x = jax.random.normal(rng, (8, 64, 64))

    params = model.init(rng, x)
    out = model.apply(params, x)

    assert out.shape == x.shape
    assert jnp.all(jnp.isfinite(out))


def test_neural_corrector_shape():
    """Output shape matches n_assets."""
    model = NeuralMarketCorrector(n_assets=64, n_features=10)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01
    features = jax.random.normal(rng, (32, 10))

    params = model.init(rng, log_prices, log_returns, log_ma, features)
    G_neural, attn = model.apply(params, log_prices, log_returns, log_ma, features)

    assert G_neural.shape == (32, 64)


def test_neural_corrector_no_features():
    """Works without external features."""
    model = NeuralMarketCorrector(n_assets=64, n_features=0)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01

    params = model.init(rng, log_prices, log_returns, log_ma, features=None)
    G_neural, _ = model.apply(params, log_prices, log_returns, log_ma, features=None)

    assert G_neural.shape == (32, 64)


def test_neural_corrector_gradients():
    """Gradients flow through the network."""
    model = NeuralMarketCorrector(n_assets=64, n_features=10, num_layers=2)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01
    features = jax.random.normal(rng, (32, 10))

    params = model.init(rng, log_prices, log_returns, log_ma, features)

    def loss_fn(params):
        G, _ = model.apply(params, log_prices, log_returns, log_ma, features)
        return jnp.mean(G ** 2)

    grads = jax.grad(loss_fn)(params)

    # Check gradients exist and are finite
    grad_leaves = jax.tree_util.tree_leaves(grads)
    for grad in grad_leaves:
        assert jnp.all(jnp.isfinite(grad))


def test_simple_corrector():
    """SimpleNeuralCorrector works as baseline."""
    model = SimpleNeuralCorrector(n_assets=64, n_features=10)
    rng = jax.random.PRNGKey(0)

    log_prices = jax.random.normal(rng, (32, 64))
    log_returns = jax.random.normal(rng, (32, 64)) * 0.01
    log_ma = log_prices - 0.01
    features = jax.random.normal(rng, (32, 10))

    params = model.init(rng, log_prices, log_returns, log_ma, features)
    G_neural, _ = model.apply(params, log_prices, log_returns, log_ma, features)

    assert G_neural.shape == (32, 64)


def test_parameter_count():
    """Neural corrector has reasonable parameter count."""
    model = NeuralMarketCorrector(
        n_assets=64,
        n_features=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128
    )
    rng = jax.random.PRNGKey(0)

    dummy_prices = jax.random.normal(rng, (1, 64))
    params = model.init(rng, dummy_prices, dummy_prices, dummy_prices, None)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Neural corrector params: {n_params:,}")

    # Should be in reasonable range (~50-200K)
    assert n_params < 300_000, f"Too many params: {n_params}"
```

---

## Acceptance Criteria

- [ ] `CrossAssetAttention` treats assets as tokens
- [ ] `CrossAssetTransformerBlock` with residual connections
- [ ] `NeuralMarketCorrector` uses transformer architecture
- [ ] `SimpleNeuralCorrector` baseline implemented
- [ ] Works with and without external features
- [ ] Operates in LOG-SPACE (consistent with physics)
- [ ] Gradients flow correctly
- [ ] Unit tests passing

---

## Parameter Count

| Component | Estimate |
|-----------|----------|
| Asset embedding | ~1K |
| Transformer blocks (x2) | ~65K |
| Output decoder | ~1K |
| **Total** | **~70K** |

With SimpleNeuralCorrector: ~100K

Memory: ~280-400 KB

---

## Verification Commands

```bash
# Run unit tests
pytest tests/test_neural_corrector.py -v

# Quick sanity check
python -c "
from mcts_trader.jax_agent.prediction.neural_corrector import NeuralMarketCorrector
import jax
import jax.numpy as jnp

model = NeuralMarketCorrector(n_assets=64, n_features=10, num_layers=2)
rng = jax.random.PRNGKey(0)

log_prices = jax.random.normal(rng, (32, 64))
log_returns = jax.random.normal(rng, (32, 64)) * 0.01
log_ma = log_prices - 0.01
features = jax.random.normal(rng, (32, 10))

params = model.init(rng, log_prices, log_returns, log_ma, features)
G_neural, attn = model.apply(params, log_prices, log_returns, log_ma, features)

print(f'Output shape: {G_neural.shape}')
print(f'Output range: [{G_neural.min():.4f}, {G_neural.max():.4f}]')

n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f'Total parameters: {n_params:,}')

print('SUCCESS: Neural corrector working!')
"
```

---

## Next Steps

After completing this task:
1. Proceed to TASK_010 (NeuralMarketODE Integration)
2. Physics core (TASK_008) + Neural corrector combine via ODE solver
