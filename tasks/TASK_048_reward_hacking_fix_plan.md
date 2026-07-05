# TASK_048: Reward Hacking Fix Plan (Consolidated)

## Status: PLANNING

## Source Reports

- `docs/REWARD_HACKING_AUDIT.md` (Claude) - 18 vulnerabilities
- `reports/audit_reward_hacking.md` (Gemini) - 6 vulnerabilities

## Consolidated Findings

### Already Addressed

| Issue | Solution | Status |
|-------|----------|--------|
| Probabilistic limit fills | TASK_047 | PENDING |
| HOLD collapse | `--differential-reward` flag | EXISTS |
| Entropy gaming | `--sac-entropy` flag | EXISTS |
| Value self-fulfilling | `--epsilon-greedy` flag | EXISTS |
| No order queue/depth | TASK_047 | PENDING |

### Needs New Tasks

| Priority | Issue | Exploit ID | New Task |
|----------|-------|------------|----------|
| **CRITICAL** | Synthetic data determinism | 008,009,010,011 | TASK_049 |
| **CRITICAL** | Fill at limit price (not market) | 001 | TASK_050 |
| **HIGH** | Marketable limit = taker fee | Gemini #3 | TASK_050 |
| **HIGH** | HOLD has no cost | 005 | TASK_051 (burning money) |
| **MEDIUM** | Fill priority deterministic | Gemini #4 | TASK_052 |
| **MEDIUM** | Episode length fixed | 012 | TASK_053 |
| **MEDIUM** | Pending orders vs position limit | 015 | TASK_054 |
| **MEDIUM** | No borrowing cost for shorts | 014 | TASK_055 |
| **LOW** | Simple encoder missing volume | Gemini #6 | TASK_056 |

### Won't Fix (Acceptable)

| Issue | Reason |
|-------|--------|
| Symlog asymmetry (006) | PopArt handles this, symlog is intentional |
| Division amplification (007) | Liquidation terminates, edge case |
| Step counter leakage (013) | Low impact, noise would hurt learning |
| Lookback leakage (018) | Duplicate of synthetic determinism |

---

## Implementation Order

### Phase 1: Critical (Before Next Training)

1. **TASK_049**: Add noise to synthetic generators
2. **TASK_050**: Fix limit order fill mechanics
3. **TASK_051**: Burning money (HOLD cost)

### Phase 2: High Priority (This Week)

4. **TASK_047**: Probabilistic limit fills (already created)
5. **TASK_052**: Randomize fill priority
6. **TASK_053**: Randomize episode length

### Phase 3: Medium Priority (Next Sprint)

7. **TASK_054**: Position limits include pending orders
8. **TASK_055**: Borrowing cost for shorts
9. **TASK_056**: 27-channel encoder default

---

## Quick Wins (One-Line Fixes)

```python
# 1. Enable differential reward by default
parser.add_argument('--differential-reward', action='store_true', default=True)

# 2. Fix fill price (jax_trading_env.py:473)
buy_fill_price = jnp.minimum(state.order_price, current_prices)  # was: state.order_price
sell_fill_price = jnp.maximum(state.order_price, current_prices)

# 3. Episode length randomization
max_steps = config.max_steps + np.random.randint(-50, 50)
```

---

## Validation Checklist

After fixes, verify:

- [ ] Synthetic Level 0: Different seeds give different optimal policies
- [ ] Limit orders: Fill at market price, not limit
- [ ] HOLD: Has opportunity cost in trending markets
- [ ] Episode end: Agent doesn't pile into positions at step 395
- [ ] Position limits: Can't bypass with pending orders
- [ ] Shorts: Have associated borrowing cost

---

## Dependencies

```
TASK_049 (synthetic noise) ─┐
TASK_050 (fill mechanics) ──┼──> Retrain with fixes
TASK_051 (burning money) ───┘

TASK_047 (probabilistic fills) ──> Can be added later
TASK_052-056 ──> Lower priority, incremental
```
