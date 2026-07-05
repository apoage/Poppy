# BUG #5: invalid_actions Not Passed to Gumbel MuZero

**Severity:** LOW
**File:** `mcts_trader/jax_agent/mctx_search.py:126`
**Status:** FIXED (2026-01-04)

## Code
```python
if self.use_gumbel:
    # Gumbel MuZero (recommended)
    policy_output = mctx.gumbel_muzero_policy(
        ...,
        # invalid_actions argument is missing here!
    )
```

## Issue
action_mask is used to mask root logits, but invalid_actions is not passed to gumbel_muzero_policy. While masking logits to -inf works for the root, explicitly passing invalid_actions is safer and ensures mctx internals handle it correctly (e.g. for completed Q-values).

## Impact
Minor. Masking logits usually suffices for Gumbel, but explicit masking is more robust.

## Expected Fix
Pass `invalid_actions=~action_mask` to `gumbel_muzero_policy`.

## Investigation Checklist
- [ ] Check mctx.gumbel_muzero_policy signature for invalid_actions
- [ ] Verify action_mask semantics (1=valid, 0=invalid)
- [ ] Add invalid_actions parameter if supported
