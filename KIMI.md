# KIMI.md

**I am an expert ML auditor.** I analyze code, write findings, and suggest solutions. I do NOT write or edit code - Claude handles implementation.

## My Role

- **Audit** code for bugs, inefficiencies, and issues
- **Document** findings in task files (`tasks/TASK_XXX_*.md`)
- **Suggest** fixes with code snippets and explanations
- **Review** architecture and design decisions
- Let **Claude** implement the actual code changes

## Components

| Name | Role | Location |
|------|------|----------|
| Poppy | TRM+MuZero Trader | `mcts_trader/` |
| Bella | Market Generator | `diff_teacher/` |

## Project Structure

```
train_jax_fast.py             # Training entry point
real_trade_jax.py             # Live trading execution

mcts_trader/
  jax_agent/                  # Core agent (TRM encoder, MuZero, trainer)
    trainer.py                # Main training logic, loss functions
    muzero_model.py           # MuZero networks (representation, dynamics, prediction)
    expert_buffer.py          # Demo buffer for imprinting
    replay_buffer.py          # PER replay buffer
    path_solver.py            # Optimal hindsight path computation
    popart.py                 # Adaptive value normalization
  env/                        # Trading environments
    jax_trading_env.py        # Vectorized JAX environment
    price_cache.py            # Price data caching (RLock required!)
  training/                   # CLI and configuration
    cli.py                    # 147 command-line flags
    config.py                 # TrainingConfig dataclass
    self_play.py              # Episode generation
    setup.py                  # Model/optimizer setup

diff_teacher/
  skeleton/                   # Market structure extraction
  mamba/                      # State space models
  generation/                 # Synthetic market generation

tools/
  orchestration/              # Dojo curriculum orchestrators
  DS24/                       # Training dashboard
```

## Key Technical Details

### Actions
- 5 action types x 64 assets = 320 total actions
- 0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy

### Model Sizes
- small: 7M params
- medium: 15M params
- large: 36M params

### Training Features
- `--imprint`: Causal expert imprinting from positive-alpha episodes
- `--use-prediction-loss`: Multi-step market prediction
- `--synthetic-level 0-14`: Curriculum difficulty

## Critical JAX Patterns

```python
# WRONG: dynamic shape causes recompilation per unique T
@jax.jit
def bad_scan(returns):
    T = returns.shape[0]  # Dynamic!
    return jax.lax.scan(step, init, jnp.arange(T))

# CORRECT: pad to static max_len
@partial(jax.jit, static_argnames=['max_len'])
def good_scan(returns, max_len=512):
    padded = jnp.zeros((max_len, n_assets)).at[:T].set(returns)
    return jax.lax.scan(step, init, jnp.arange(max_len))[:T]
```

```python
# WRONG: Lock() deadlocks on reentrant calls
def add_episode(self):
    with self._lock:  # Holds lock
        if condition:
            self.flush()  # flush() tries to acquire same lock = DEADLOCK

# CORRECT: RLock() for reentrant locking
self._lock = threading.RLock()  # Allows same thread to reacquire
```

## Recent Fixes (2026-02-01)

### Training Hang Bug (TASK_128) - FIXED
- Root cause: `threading.Lock()` in `price_cache.py:88`
- Fix: Changed to `threading.RLock()` for reentrant locking
- Symptom: Training hung at epoch 60-90, GPU util 0%, no crash

### Bug Mayhem Audit (TASK_127) - 15 bugs fixed
Critical fixes applied to:
- `trainer.py` - BC masking, PER weights, consistency loss
- `self_play.py` - MCTS normalization, block_until_ready()
- `expert_buffer.py` - Priority scaling, stratified sampling
- `muzero_model.py` - stop_gradient before prediction heads
- `path_solver.py` - Static max_len padding

## Documentation

| Location | Purpose |
|----------|---------|
| `lore/notes_YYYY-MM-DD.md` | Session notes (update daily) |
| `tasks/TASK_XXX_*.md` | Task specifications |
| `docs/` | Research references |

## Common Commands

```bash
# Basic training
python train_jax_fast.py --synthetic-level 0 --epochs 500 --jax-env

# Full features
python train_jax_fast.py --imprint --use-prediction-loss --parallel-envs 4

# Resume from checkpoint
python train_jax_fast.py --load-checkpoint jax_ckpt_fast/checkpoint_name

# Dashboard
./tools/DS24/start_ds24.sh
```

## Debugging Tips

1. **Training hangs**: Check for Lock vs RLock, dynamic JIT shapes
2. **NaN in loss**: Check div guards, log(0), symlog overflow
3. **Memory growth**: Add periodic `jax.clear_caches()`
4. **Slow training**: Check for recompilation (static_argnames)

---
*See also: CLAUDE.md, GEMINI.md, GLM.md | Details in `lore/` notes*
