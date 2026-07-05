# Bug: Dtype Mismatch in Action Embedding

## Priority: MEDIUM
**Status:** ALREADY FIXED

## Verification (2026-01-01)

Line 130 in muzero_model.py shows:
`action_emb = self.action_embedding(action).astype(self.dtype)`

The embedding is properly cast to model dtype after lookup.
**Component:** `mcts_trader/jax_agent/muzero_model.py`

## Problem Description

The action embedding layer may produce wrong dtype when mixed precision (bfloat16) is enabled. Flax embeddings default to float32, causing dtype mismatch errors.

### Location

```python
# muzero_model.py
class MuZeroNetwork(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.action_embedding = nn.Embed(
            num_embeddings=self.n_actions,
            features=self.state_dim,
            # dtype not specified - defaults to float32
        )

    def dynamics(self, state, action):
        action_emb = self.action_embedding(action)  # Returns float32
        combined = jnp.concatenate([state, action_emb])  # state may be bfloat16!
```

### Impact

- Type promotion warnings with bfloat16
- Potential precision loss or errors
- XLA compilation warnings

## Solution

Already partially fixed (per session notes), but verify complete:

```python
# muzero_model.py
def dynamics(self, state, action):
    action_emb = self.action_embedding(action)
    action_emb = action_emb.astype(self.dtype)  # Cast to model dtype
    combined = jnp.concatenate([state, action_emb])
```

## Files to Modify

| File | Change |
|------|--------|
| `muzero_model.py` | Ensure embedding output cast to model dtype |

## Testing

1. Run with `--bf16` flag
2. Check for dtype warnings
3. Verify embedding and state dtypes match
