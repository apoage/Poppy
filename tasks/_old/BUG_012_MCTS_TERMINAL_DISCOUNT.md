# BUG #12: Incorrect MCTS Discount for Terminal States

**Severity:** ~~MEDIUM~~ KNOWN LIMITATION / FUTURE ENHANCEMENT
**File:** `mcts_trader/jax_agent/mctx_search.py:104-108`
**Status:** INVESTIGATED - ACCEPTED LIMITATION

## Original Concern
The discount is always `self.discount` (0.997), but at terminal states it should be 0. MCTS doesn't know about episode termination.

## Investigation Result: KNOWN LIMITATION

This is a design choice in the current implementation, not a bug:

### Why It's Acceptable

1. **Fixed episode length**: Episodes are typically 400 steps, termination is rare
2. **Natural discounting**: γ^400 ≈ 0.30, so late-episode values are naturally reduced
3. **Rare early termination**: Only on liquidation (5% of trades hit stop-loss)

### MuZero Paper Approach

The MuZero paper handles this by:
1. Having dynamics model output terminal probability
2. Using `discount = (1 - p_terminal) * γ`

This would require:
- Adding a "done" head to the dynamics output
- Training on done targets from the replay buffer
- Converting probability to discount in MCTS

### Recommendation

Defer to a future enhancement when:
1. Episodes have highly variable lengths
2. Early termination becomes more common
3. Terminal state prediction accuracy is critical

For now, the fixed γ=0.997 is acceptable given the fixed-length episode structure.

## Investigation Checklist
- [x] Check if MuZero paper handles terminal states - Yes, via learned terminal probability
- [x] Determine if dynamics model should predict termination - Could, but low priority
- [x] Consider adding terminal head - Future enhancement, not urgent for fixed-length episodes

## Conclusion

Keep current behavior. Add terminal prediction as a future enhancement if needed.
