# TASK 07: Advanced Order Types & Action Space Extension

**Priority**: HIGH (after core training stable)
**Status**: Design phase
**Complexity**: Complex (action space change + order management)
**Reference**: `lore/ADVANCED_ORDERS_DESIGN.md`

---

## Objective

Extend action space with:
1. **Limit Orders**: Buy/Sell at specified price (maker orders)
2. **Stop Orders**: Trigger on price breakout/breakdown
3. **Zero Action**: Global "do nothing" / skip step

---

## Current State

```
Action Space: 64 assets × 3 actions = 192 discrete
- action % 3 == 0: Sell (taker)
- action % 3 == 1: Hold
- action % 3 == 2: Buy (taker)
```

**Limitations**:
- All trades are **taker** orders (pay fees)
- No way to set price targets
- Must predict exact timing (hard for longer horizons)
- No "wait for all" option

---

## Proposed Extension

### New Action Space (Option A - Unified)

```
Action Space: 64 assets × 6 actions + 1 = 385 discrete

Per-asset actions (0-383):
- action % 6 == 0: SELL_MARKET (taker, pay fee)
- action % 6 == 1: HOLD
- action % 6 == 2: BUY_MARKET (taker, pay fee)
- action % 6 == 3: LIMIT_BUY (maker, earn rebate)
- action % 6 == 4: LIMIT_SELL (maker, earn rebate)
- action % 6 == 5: STOP_MARKET (trigger on breakout)

Global action:
- action == 384: ZERO_ACTION (skip this step entirely)
```

### Price Target Output

For limit/stop orders, model outputs additional **price offset**:

```python
class MuZeroNetworks:
    def prediction(self, latent):
        policy_logits = self.policy_head(latent)  # (batch, 385)
        value = self.value_head(latent)           # (batch, 101)
        price_offset = self.price_head(latent)    # (batch, 1) in [-1, 1]

        # price_offset * 20 = percentage offset from current price
        # e.g., 0.5 → +10% from current price
        return policy_logits, value, price_offset
```

---

## Fee Structure (Kraken Reference)

| Order Type | Fee | Model Incentive |
|------------|-----|-----------------|
| Market (Taker) | +0.16% to +0.26% | Pay fee |
| Limit (Maker) | -0.00% to +0.16% | **Earn rebate** at high volume |
| Stop-Market | +0.16% to +0.26% | Pay fee (but protects capital) |

**Key Insight**: Limit orders can have **negative fees** (rebates) for providing liquidity!

Training signal: reward += fee_saved (or rebate earned)

---

## Zero Action Design

### Purpose
- Skip trading when uncertain
- Reduce over-trading (fee savings)
- Wait for better entry points
- Express "no good opportunities"

### Training Challenge
**Problem**: Model might learn to always skip (lazy policy)

**Solutions**:

1. **Opportunity Cost Penalty**
   ```python
   # If zero_action chosen but market moved favorably
   if action == 384:  # ZERO_ACTION
       best_possible_gain = compute_best_hindsight_trade()
       if best_possible_gain > threshold:
           reward -= opportunity_cost_penalty
   ```

2. **Curriculum Introduction**
   - Levels 0-3: No zero action (must trade)
   - Level 4+: Introduce zero action with high penalty for overuse
   - Gradually reduce penalty as model matures

3. **Action Mask Based on Uncertainty**
   ```python
   # Only allow zero_action when policy entropy is high
   if policy_entropy > 4.0:  # Uncertain
       action_mask[384] = 1.0  # Allow skip
   else:
       action_mask[384] = 0.0  # Must decide
   ```

4. **Skip Budget**
   ```python
   # Limit consecutive skips
   if consecutive_skips >= 3:
       action_mask[384] = 0.0  # Force action
   ```

---

## Implementation Steps

### Phase 1: Limit Orders (Enable Rebates)

1. **Extend action space to 384** (64 × 6)
2. **Add price_offset head** to model
3. **Implement order book simulation**:
   ```python
   class PendingOrder:
       asset_idx: int
       order_type: str  # 'limit_buy', 'limit_sell', 'stop'
       target_price: float
       placed_step: int
       expires_after: int = 100  # Auto-cancel after N steps
   ```
4. **Order execution logic**:
   ```python
   def check_pending_orders(self):
       for order in self.pending_orders:
           if order.type == 'limit_buy' and current_price <= order.target_price:
               self.execute_buy(order.asset_idx, fee=-MAKER_REBATE)
           # ... similar for limit_sell, stop
   ```
5. **Training**: Include price_offset in trajectory, add MSE loss

### Phase 2: Zero Action (Careful Introduction)

1. **Add action 384** (global skip)
2. **Implement skip budget** (max 3 consecutive)
3. **Add opportunity cost tracking** (hindsight analysis)
4. **Curriculum gate**: Only enable at Level 4+
5. **Entropy-based masking**: Only allow when uncertain

---

## Synthetic Training Considerations

### Level 0-3: No Advanced Orders
- Keep simple action space (192)
- Focus on learning market patterns

### Level 4+: Introduce Limit Orders
- Clear trend patterns → easy limit placement
- Reward structure: explicit rebate bonus
- Start with wide price offsets (±10%), narrow over time

### Level 5+: Introduce Zero Action
- Chaotic markets where waiting is optimal
- Heavy penalty for over-skipping
- Track skip rate in metrics

---

## Metrics to Track

```python
# In CSV logging
'limit_orders_placed': count of limit orders
'limit_orders_filled': count executed at target
'limit_orders_expired': count auto-cancelled
'maker_rebates_earned': total rebate value
'zero_actions_taken': count of skips
'skip_rate': zero_actions / total_actions
'opportunity_cost': missed gains from skipping
```

---

## Risk Considerations

1. **Action space explosion**: 385 vs 192 (2× increase)
   - May need larger model
   - MCTS might need more simulations

2. **Order management complexity**:
   - Pending orders need state tracking
   - Order expiration logic
   - Partial fills (future consideration)

3. **Zero action abuse**:
   - Must have strong anti-skip incentives
   - Monitor skip_rate closely
   - Consider removing if >30% skip rate

---

## Acceptance Criteria

- [ ] Limit orders execute at target price
- [ ] Maker rebates properly credited
- [ ] Price offset predictions within ±5% of optimal (hindsight)
- [ ] Zero action usage <20% of steps
- [ ] No regression on Level 0-3 performance
- [ ] Order fill rate >50% for limit orders

---

## References

- `lore/ADVANCED_ORDERS_DESIGN.md` - Original design doc
- `lore/DOUBLE_SAFEGUARD_STOP_LOSS.md` - Stop loss patterns
- Kraken fee schedule: https://www.kraken.com/features/fee-schedule
