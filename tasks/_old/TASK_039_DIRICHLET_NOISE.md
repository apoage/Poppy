# TASK_039: Higher Dirichlet Noise at MCTS Root

## Status: IMPLEMENTED (2026-01-06)

**Implementation:**
- `mcts_trader/jax_agent/exploration.py` - `get_recommended_dirichlet_params()` helper
- `train_jax_fast.py` - Added `--dirichlet-aggressive` flag (sets α=0.5, ε=0.5)
- Existing `--noise` and `--noise-frac` flags still work for custom values
## Priority: MEDIUM (Quick to implement, synergistic with ε-greedy)

## Problem Statement

Current MCTS uses default Dirichlet noise parameters that may be too weak:
- `dirichlet_alpha = 0.3` (standard for Go/Chess)
- `dirichlet_fraction = 0.25` (25% noise at root)

For a 320-action space with severe HOLD bias, this provides insufficient exploration:
- 25% noise distributed across 320 actions = ~0.08% per non-HOLD action
- HOLD action still dominates even with noise injection

## Research Basis

From Gemini Deep Research:
> "Consider increasing exploration in MCTS. Raise the Dirichlet noise fraction
> from ~0.25 to ~0.5. This provides a stronger 'push' against the prior even
> if the prior is converging to HOLD."

From AlphaGo/AlphaZero papers:
- Chess: α = 0.3, ε = 0.25
- Go: α = 0.03, ε = 0.25
- Trading (proposed): α = 0.5, ε = 0.5 (due to action space structure)

## Solution: Increase Dirichlet Noise

### Recommended Configuration

```python
# Current (AlphaZero defaults)
dirichlet_alpha = 0.3
dirichlet_fraction = 0.25  # 25% of prior from noise

# Proposed (aggressive exploration)
dirichlet_alpha = 0.5      # Flatter noise distribution
dirichlet_fraction = 0.5   # 50% of prior from noise
```

### Why Higher Values?

1. **Large action space**: 320 actions vs 19×19=361 in Go, but our actions are highly asymmetric
2. **HOLD dominance**: Unlike Go where all moves have similar priors initially, HOLD starts as local optimum
3. **Value function bootstrapping**: Need diverse exploration until value function calibrates
4. **Combinatorial with ε-greedy**: Dirichlet helps during search, ε-greedy helps training targets

## Implementation

### Phase 1: CLI Arguments

**File: `train_jax_fast.py`**

```python
# Add arguments
parser.add_argument('--dirichlet-alpha', type=float, default=0.3,
                   help='Dirichlet noise alpha for MCTS root (default: 0.3)')
parser.add_argument('--dirichlet-fraction', type=float, default=0.25,
                   help='Fraction of root prior from Dirichlet noise (default: 0.25)')
```

### Phase 2: Pass to MCTS

**File: `mcts_trader/jax_agent/mctx_search.py`**

```python
class MuZeroMCTXSearch:
    def __init__(
        self,
        # ... existing params ...
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
    ):
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_fraction = dirichlet_fraction

    def search_fn(self, params, rng, observation, action_mask=None):
        # ... in mctx call ...
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=rng,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.num_simulations,
            max_depth=self.max_depth,
            # Dirichlet noise parameters
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_fraction=self.dirichlet_fraction,
        )
```

### Phase 3: Adaptive Noise Schedule (Optional)

```python
class DirichletSchedule:
    """Decay noise over training as policy improves."""

    def __init__(
        self,
        alpha_initial: float = 0.5,
        alpha_final: float = 0.3,
        fraction_initial: float = 0.5,
        fraction_final: float = 0.25,
        decay_epochs: int = 200,
    ):
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.fraction_initial = fraction_initial
        self.fraction_final = fraction_final
        self.decay_epochs = decay_epochs

    def get_params(self, epoch: int) -> tuple[float, float]:
        """Returns (alpha, fraction) for given epoch."""
        if epoch >= self.decay_epochs:
            return self.alpha_final, self.fraction_final

        progress = epoch / self.decay_epochs
        alpha = self.alpha_initial + progress * (self.alpha_final - self.alpha_initial)
        fraction = self.fraction_initial + progress * (self.fraction_final - self.fraction_initial)
        return alpha, fraction
```

## Configuration Options

| Flag | Default | Proposed | Description |
|------|---------|----------|-------------|
| `--dirichlet-alpha` | 0.3 | 0.5 | Controls noise distribution shape |
| `--dirichlet-fraction` | 0.25 | 0.5 | % of root prior from noise |

### Parameter Effects

| Alpha | Effect |
|-------|--------|
| 0.03 | Very peaked noise (one action gets most) |
| 0.3 | Balanced noise (standard) |
| 0.5 | Flatter noise (more uniform exploration) |
| 1.0 | Uniform noise |

| Fraction | Effect |
|----------|--------|
| 0.1 | 10% exploration, 90% policy |
| 0.25 | 25% exploration, 75% policy (standard) |
| 0.5 | 50% exploration, 50% policy (aggressive) |
| 0.75 | 75% exploration, 25% policy (very aggressive) |

## Expected Behavior

### Before (Default Noise)
```
Root prior: [HOLD=0.85, BUY=0.10, SELL=0.05]
After noise (ε=0.25): [HOLD=0.78, BUY=0.13, SELL=0.09]
→ HOLD still strongly dominant
```

### After (High Noise)
```
Root prior: [HOLD=0.85, BUY=0.10, SELL=0.05]
After noise (ε=0.50): [HOLD=0.60, BUY=0.22, SELL=0.18]
→ Much more balanced exploration
```

## Interaction with Other Components

### ε-Greedy MCTS (TASK_037)
- Dirichlet noise: During MCTS search (affects action selection)
- ε-greedy: After search (affects training targets)
- **Complementary**: Noise helps search explore, ε-greedy ensures diverse training

### Adaptive Entropy
- More noise → more diverse data → less entropy collapse
- May reduce need for aggressive entropy boosting

### MCTS Warmup
- During warmup (few sims): Noise has more impact
- After warmup (many sims): Q-values dominate, noise less effective
- **Recommendation**: Use higher noise during warmup phase

## Testing Plan

1. **Ablation study**: Compare α=0.3 vs α=0.5 vs α=1.0
2. **Fraction sweep**: Test ε=0.25, 0.4, 0.5, 0.6
3. **Combined with ε-greedy**: Test synergy with TASK_037

## Success Criteria

1. **Action distribution**: More balanced S/H/B during early training
2. **Search diversity**: Visit counts less concentrated on HOLD
3. **Learning speed**: Faster initial improvement (more exploration)

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Too much noise → random play | Decay schedule, or use moderate values |
| Slower convergence late training | Decay noise over epochs |
| Conflicts with ε-greedy | Test combinations, may reduce one if redundant |

## Files to Modify

1. **`train_jax_fast.py`** - Add CLI arguments
2. **`mcts_trader/jax_agent/mctx_search.py`** - Pass parameters to mctx
3. **`warmup_sweep.py`** - Add Dirichlet configs to sweep (optional)

## Dependencies

- None (standalone feature, mctx already supports these parameters)

## Follow-up Tasks

- TASK_037 (ε-greedy) - Combine for maximum exploration
- After implementation, tune via warmup_sweep.py

## References

- AlphaZero paper: Dirichlet noise at root
- Gemini Deep Research: "Raise the Dirichlet noise fraction to ~0.5"
- mctx documentation: gumbel_muzero_policy parameters
