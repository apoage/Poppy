# BUG #16: TRM Value Support Size Mismatch in Docstring

**Severity:** LOW
**File:** `mcts_trader/jax_agent/trm_flax.py:219`
**Status:** FIXED

## Code
```python
value_support_size: int = VALUE_SUPPORT_SIZE  # Central config: 101 bins over [-50, 50]
# ...
value_support_size: int = VALUE_SUPPORT_SIZE  # Central config: 101 bins over [-50, 50]
```

But in `__init__.py`:
```python
VALUE_SUPPORT_SIZE = 751  # Number of bins (2 unit resolution)
```

## Issue
Docstrings claim 101 bins over [-50, 50] but actual value is 751 bins over [-750, 750]. Comments are stale.

## Impact
Developer confusion. No runtime impact.

## Expected Fix
Update docstrings to reflect actual configuration: 751 bins over [-750, 750].

## Investigation Checklist
- [x] Find all stale docstrings/comments about value support - Found at line 219
- [x] Update to match __init__.py configuration - Changed to "751 bins over [-750, 750]"

## Fix Applied
Changed line 219 comment from "101 bins over [-50, 50]" to "751 bins over [-750, 750]" to match actual VALUE_SUPPORT_SIZE configuration.
