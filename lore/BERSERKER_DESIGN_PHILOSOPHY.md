# The Berserker: Design Philosophy

## What This System Is

This architecture is a **front-line warrior** - a berserker.

Like a Viking berserker in battle:
- **Strong**: TRM recursive reasoning + MuZero planning
- **Clever**: Learns patterns, adapts tactics, plans ahead
- **Fast**: Short-horizon decisions, immediate execution
- **Focused**: Single battle, single objective, total commitment
- **Limited**: Doesn't command armies, doesn't plan campaigns

## What This System Is NOT

- **Not a general**: Doesn't manage multiple portfolios
- **Not a strategist**: Doesn't plan long-term campaigns
- **Not self-aware**: Doesn't know its own size or market impact
- **Not scalable**: Designed for front-line, not headquarters

## The Battlefield Metaphor

```
HEADQUARTERS (Human / Fund Manager)
│
├── Strategy: "Attack crypto market, capture alpha"
├── Resources: Allocate capital to front-line units
├── Intelligence: Monitor regime shifts
└── Command: Deploy, reinforce, or withdraw berserkers
    │
    ▼
FRONT LINE (This System - The Berserker)
│
├── Tactics: 5-action combat (S/H/B/LS/LB)
├── Planning: MCTS 32-sim lookahead (seconds to minutes)
├── Adaptation: TRM reasoning cycles (immediate)
└── Scope: 64 assets, 400 steps, one battle at a time
```

## Design Constraints

### Horizon: Short

| Aspect | Scope |
|--------|-------|
| Planning depth | 32 MCTS simulations |
| Episode length | 400 steps (~hours to days) |
| Memory | Current episode, no cross-episode state |
| Adaptation | Within-episode learning only |

The berserker fights the battle in front of it. It doesn't remember yesterday's battles or plan tomorrow's.

### Scale: Small

| Portfolio Size | Role |
|----------------|------|
| < $1M | Appropriate for single berserker |
| $1M+ | Too large - berserker becomes a liability |
| $10M+ | Need a general, not a berserker |

A berserker in command of an army would be a disaster. Wrong tool for the job.

### Awareness: None

The system has **no meta-cognition**:
- Doesn't know it's a trading agent
- Doesn't know its portfolio size
- Doesn't know it's affecting markets
- Doesn't know when it's out of its depth

This is by design. A berserker doesn't philosophize mid-battle.

External systems must handle:
- Portfolio scaling decisions
- Market impact awareness
- Regime change detection
- Strategic allocation

## When to Deploy a Berserker

**Good deployment:**
- Clear market regime (trending, ranging, volatile)
- Appropriate position sizing (can't move markets)
- Supervised by higher-level systems
- Stop-loss layers active (5-layer defense)

**Bad deployment:**
- Unknown market regime
- Position sizes that impact prices
- Unsupervised, autonomous operation
- Single point of failure for large capital

## Multi-Berserker Formations

When capital grows, don't give one berserker a bigger sword.
Deploy multiple berserkers with different assignments:

```
Capital: $3M

WRONG:
└── Single Berserker ($3M) ← Will fail, wrong scale

RIGHT:
├── Berserker Alpha ($750K) → BTC, ETH, majors
├── Berserker Beta ($750K) → Mid-cap alts
├── Berserker Gamma ($750K) → Small caps
└── Berserker Delta ($750K) → Different exchange/market
```

Each berserker:
- Independent model instance
- Independent stop-loss
- Independent failure mode
- Appropriate scale for its battlefield

## The General's Role (Not This System)

The general (human or higher-level system) handles:
- Capital allocation across berserkers
- Deployment decisions (which markets, when)
- Strategic withdrawal (regime change detected)
- Reinforcement (add berserkers to winning fronts)
- Overall risk management

This system is the weapon. The general wields it.

## Safety Through Limitation

The berserker's limitations ARE its safety features:

| Limitation | Safety Benefit |
|------------|----------------|
| Short horizon | Can't make catastrophic long-term bets |
| No self-awareness | Can't convince itself it's special |
| Small scale | Can't move markets, limited blast radius |
| Episode-bounded | Fresh start each episode, no grudges |

A berserker that knew it was a berserker might start having opinions about strategy. That's not its job.

## Summary

> **This architecture is designed for short horizons and front-line trading.**
>
> Like a berserker: strong, clever, can move on the battlefield.
>
> Not like a general: doesn't command armies, doesn't plan campaigns.
>
> Deploy appropriately. One berserker per battle. Multiple berserkers for multiple fronts. Never one berserker for an entire war.

## The Five-Layer Oath

Every berserker goes into battle with five guardians watching:

1. **Layer 1 (Self)**: "I will try to survive" (advisory)
2. **Layer 2 (Wrapper)**: "I will stop you if you won't stop yourself"
3. **Layer 3 (Monitor)**: "I watch from outside, I will intervene"
4. **Layer 4 (Exchange)**: "Your stop-loss lives with me, not you"
5. **Layer 5 (External)**: "I watch from another realm entirely"

The berserker may fall. The layers ensure the army survives.
