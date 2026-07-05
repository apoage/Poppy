# TASK 013: Training Configurator Tool

**Severity**: QUALITY OF LIFE
**Phase**: Tooling
**Location**: `train_configurator.py` (new file)
**Status**: OPEN
**Estimated Time**: 2-3 hours
**Dependencies**: None

---

## Problem Statement

`train_jax_fast.py` has grown to **40+ CLI parameters** across multiple categories:
- Data source (synthetic level, db-path, schedule-path, dates)
- Model architecture (model-size, state-dim, h-cycles, l-cycles)
- MCTS settings (sims, noise, gumbel-scale)
- Training (epochs, steps, batch-size)
- Costs (fee, slippage, mode)
- Entropy/exploration (entropy-coef, adaptive-entropy, thresholds)
- Prediction loss (6+ parameters)
- Checkpoint management (load, clear-buffer, reset-optimizer)
- Value calibration (scale, shift, adaptive)

This makes it difficult to:
1. Remember which parameters exist
2. Know which combinations make sense
3. Reproduce experiments
4. Share configurations with others

---

## Solution: Interactive Training Configurator

A wizard-style tool that guides through configuration and outputs the command.

### Option A: TUI Wizard (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│           JAX MuZero Training Configurator                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Quick Start Presets                                         │
│  [2] Build Custom Command                                        │
│  [3] Load Previous Configuration                                 │
│  [4] Compare Two Configs                                         │
│  [q] Quit                                                        │
│                                                                  │
│  Select: _                                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Option B: Question-Based Wizard

```bash
$ python train_configurator.py

=== JAX Training Configurator ===

[Data Source]
? What data do you want to use?
  > Synthetic curriculum (recommended for learning)
    Real data from database
    Training schedule JSON

? Which synthetic level? (0=easiest, 4=hardest)
  > Level 0: Sine waves
    Level 1: Lagged correlations
    Level 2: Dip buying
    Level 3: Noisy trends
    Level 4: Sector rotation

[Model]
? Model size?
  > small (~7M params, fastest)
    medium (~15M params)
    large (~36M params)

[Training]
? How many epochs? [500]:
? Steps per episode? [400]:

[Costs]
? Include trading costs?
  > No costs (pure learning)
    Light costs (5 bps fee)
    Realistic costs (16 bps + 5 bps slip)

[Advanced]
? Configure advanced options?
  > No (use defaults)
    Yes (MCTS, entropy, prediction loss)

=== Generated Command ===

python train_jax_fast.py \
    --synthetic-level 0 \
    --model-size small \
    --epochs 500 \
    --steps 400 \
    --name curriculum_L0

Copy to clipboard? [Y/n]:
Save as preset? [name]:
```

---

## Implementation

### Core Features

1. **Preset Configurations**
   ```python
   PRESETS = {
       "curriculum_L0": {
           "description": "Start learning on sine waves",
           "args": {
               "synthetic_level": 0,
               "epochs": 500,
               "model_size": "small",
               "adaptive_entropy": True,
           }
       },
       "curriculum_L0_to_L1": {
           "description": "Continue from L0 checkpoint to L1",
           "args": {
               "synthetic_level": 1,
               "load_checkpoint": "jax_ckpt_fast/curriculum_L0",
               "clear_buffer": True,
               "reset_optimizer": True,
           }
       },
       "real_data_test": {
           "description": "Test on real data with realistic costs",
           "args": {
               "db_path": "trading_data.db",
               "fee": 16,
               "slippage": 5,
           }
       },
       "prediction_loss_experiment": {
           "description": "Train with auxiliary prediction loss",
           "args": {
               "synthetic_level": 0,
               "use_prediction_loss": True,
               "prediction_loss_weight": 0.2,
           }
       },
   }
   ```

2. **Parameter Groups with Explanations**
   ```python
   PARAM_GROUPS = {
       "data_source": {
           "title": "Data Source",
           "description": "Where to get training data",
           "params": ["synthetic_level", "db_path", "schedule_path", "start_date", "end_date"],
           "mutually_exclusive": ["synthetic_level", "db_path", "schedule_path"],
       },
       "model": {
           "title": "Model Architecture",
           "description": "Neural network configuration",
           "params": ["model_size", "state_dim", "h_cycles", "l_cycles"],
       },
       "mcts": {
           "title": "MCTS Search",
           "description": "Monte Carlo Tree Search settings",
           "params": ["sims", "gumbel_scale", "noise", "noise_frac"],
       },
       "training": {
           "title": "Training Loop",
           "description": "How long and how much to train",
           "params": ["epochs", "steps", "batch_size"],
       },
       "costs": {
           "title": "Trading Costs",
           "description": "Fees and slippage simulation",
           "params": ["fee", "slippage", "mode"],
       },
       "exploration": {
           "title": "Exploration/Entropy",
           "description": "Encourage diverse actions",
           "params": ["entropy_coef", "adaptive_entropy", "entropy_hold_threshold"],
       },
       "prediction": {
           "title": "Prediction Loss (Phase 2)",
           "description": "Auxiliary price prediction",
           "params": ["use_prediction_loss", "prediction_loss_weight", "multi_step_k"],
       },
       "checkpoint": {
           "title": "Checkpoint Management",
           "description": "Resume and transfer learning",
           "params": ["load_checkpoint", "clear_buffer", "reset_optimizer"],
       },
   }
   ```

3. **Smart Defaults Based on Context**
   ```python
   def get_smart_defaults(context):
       """Suggest parameters based on what user selected."""
       if context.get("synthetic_level") == 0:
           return {
               "epochs": 500,
               "fee": 0,
               "slippage": 0,
               "adaptive_entropy": True,
           }
       elif context.get("load_checkpoint"):
           return {
               "clear_buffer": True,  # Usually want fresh buffer
               "reset_optimizer": True,  # Fresh momentum
           }
       elif context.get("db_path"):
           return {
               "fee": 16,
               "slippage": 5,
               "epochs": 1000,
           }
   ```

4. **Validation Rules**
   ```python
   VALIDATION_RULES = [
       {
           "check": lambda c: not (c.get("synthetic_level") and c.get("db_path")),
           "error": "Cannot use both synthetic data and database",
       },
       {
           "check": lambda c: c.get("batch_size", 256) <= 512 or c.get("model_size") != "large",
           "warning": "Large model with batch_size > 512 may OOM",
       },
       {
           "check": lambda c: not c.get("use_prediction_loss") or c.get("synthetic_level") is not None,
           "warning": "Prediction loss works best with synthetic data initially",
       },
   ]
   ```

5. **Command Generation**
   ```python
   def generate_command(config: dict, multiline: bool = True) -> str:
       """Generate the train_jax_fast.py command."""
       parts = ["python train_jax_fast.py"]

       for key, value in config.items():
           if value is None or value is False:
               continue

           arg_name = f"--{key.replace('_', '-')}"

           if value is True:
               parts.append(arg_name)
           else:
               parts.append(f"{arg_name} {value}")

       if multiline:
           return " \\\n    ".join(parts)
       return " ".join(parts)
   ```

---

## User Interface Options

### Option 1: Rich TUI (Best UX)
```python
# Dependencies: pip install rich questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import questionary
```

### Option 2: Simple CLI Wizard
```python
# No dependencies, pure Python
def ask_choice(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    return options[int(input("Select: ")) - 1]
```

### Option 3: Config File Generator
```python
# Generate JSON configs that can be loaded with --config
def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
```

---

## File Structure

```
train_configurator.py      # Main tool
config/
├── presets/
│   ├── curriculum_L0.json
│   ├── curriculum_L1.json
│   ├── real_data_test.json
│   └── prediction_experiment.json
└── user_configs/          # User-saved configurations
    └── .gitkeep
```

---

## Usage Examples

```bash
# Interactive mode
python train_configurator.py

# Quick preset
python train_configurator.py --preset curriculum_L0

# Show all presets
python train_configurator.py --list-presets

# Explain a parameter
python train_configurator.py --explain entropy-coef

# Diff two configs
python train_configurator.py --diff config1.json config2.json

# Export current args as preset
python train_configurator.py --export my_experiment
```

---

## Sample Output

```
=== Generated Configuration ===

┌─ Data Source ─────────────────────────────────────────┐
│ synthetic_level: 0 (Sine waves - easiest)             │
└───────────────────────────────────────────────────────┘

┌─ Model ───────────────────────────────────────────────┐
│ model_size: small (~7.3M parameters)                  │
│ h_cycles: 4, l_cycles: 6                              │
└───────────────────────────────────────────────────────┘

┌─ Training ────────────────────────────────────────────┐
│ epochs: 500                                           │
│ steps: 400 (per episode)                              │
│ batch_size: 256                                       │
└───────────────────────────────────────────────────────┘

┌─ Exploration ─────────────────────────────────────────┐
│ adaptive_entropy: enabled                             │
│ entropy_hold_threshold: 0.70                          │
└───────────────────────────────────────────────────────┘

=== Command ===

python train_jax_fast.py \
    --synthetic-level 0 \
    --model-size small \
    --epochs 500 \
    --steps 400 \
    --batch-size 256 \
    --adaptive-entropy \
    --name curriculum_L0

[C]opy to clipboard  [S]ave as preset  [R]un now  [Q]uit
```

---

## Acceptance Criteria

- [ ] Interactive wizard guides through all major parameter groups
- [ ] Presets for common scenarios (curriculum L0-L4, real data, prediction)
- [ ] Validation catches incompatible parameter combinations
- [ ] Smart defaults based on context
- [ ] Generated commands are correct and copy-pasteable
- [ ] Can save/load user configurations
- [ ] `--explain <param>` shows detailed help for any parameter
- [ ] Works without external dependencies (rich/questionary optional)

---

## Dependencies

**Required**: None (pure Python fallback)

**Optional** (better UX):
- `rich` - Beautiful terminal output
- `questionary` - Interactive prompts
- `pyperclip` - Clipboard support

```bash
pip install rich questionary pyperclip
```

---

## Priority

MEDIUM - Quality of life improvement. Doesn't block training but significantly improves developer experience as parameter count grows.

---

## Future Enhancements

1. **Web UI** - Flask/Streamlit interface for non-CLI users
2. **Experiment Tracker** - Log all runs with their configs
3. **Hyperparameter Sweeps** - Generate multiple configs for grid search
4. **Config Inheritance** - Base configs with overrides
5. **Integration with dojo_manager.py** - Multi-GPU preset generation

---

## Phase 2: Python Config Language for Dojo (Advanced)

**Added**: 2026-01-11
**Location**: Integrate into `tools/orchestration/dojo_manager.py`

### The Problem

As training becomes more sophisticated, static configs aren't enough. We need:
- **Phases** with different settings (bootstrap → refine → exploit)
- **Triggers** that react to training state (entropy collapse, loss plateau)
- **Conditional logic** (if X then Y)
- **Schedules** (warmup X over N epochs)

YAML can't express this cleanly. A DSL is overkill. **Python IS the config language.**

### Config Complexity Evolution

```
CLI args     → Too many flags
    ↓
YAML file    → Flat, no logic
    ↓
Nested YAML  → Still no conditionals
    ↓
Python Config → Full expressiveness, IDE support, type hints
```

### Python Config Design

```python
# configs/kitchen_sink_v3.py
from dojo import Config, Phase, Trigger, Warmup, Schedule

config = Config(
    name="kitchen_sink_v3",

    # Base settings (always active)
    base=dict(
        model_size="small",
        batch_size=384,
        parallel_envs=50,
        jax_env=True,
        bf16=True,
        use_popart=True,
        popart_beta=0.0001,
    ),

    # Training phases (sequential)
    phases=[
        Phase(
            name="bootstrap",
            until=lambda state: state.entropy > 4.0 and state.epoch > 20,
            config=dict(
                sims=8,
                epsilon_greedy=0.3,
                entropy_coef=0.2,
                synthetic_level=0,
            ),
        ),
        Phase(
            name="refine",
            until=lambda state: state.hold_pct < 0.70 or state.epoch > 200,
            config=dict(
                sims=Warmup(start=8, end=32, over_epochs=100),
                epsilon_greedy=Warmup(start=0.3, end=0.1, over_epochs=50),
                entropy_coef=0.1,
            ),
        ),
        Phase(
            name="exploit",
            until=None,  # Run until end
            config=dict(
                sims=64,
                epsilon_greedy=0.0,
                entropy_coef=0.05,
            ),
        ),
    ],

    # Reactive triggers (checked every epoch)
    triggers=[
        Trigger(
            name="entropy_emergency",
            when=lambda state: state.entropy < 3.0,
            action=lambda cfg: cfg.update(entropy_boost=0.5, sims=8),
            cooldown_epochs=10,
        ),
        Trigger(
            name="loss_plateau",
            when=lambda state: state.loss_plateau(window=20, threshold=0.01),
            action=lambda cfg: cfg.update(lr=cfg.lr * 0.5),
            cooldown_epochs=50,
        ),
        Trigger(
            name="hold_collapse",
            when=lambda state: state.hold_pct > 0.90,
            action=lambda cfg: cfg.update(
                epsilon_greedy=0.4,
                dirichlet_aggressive=True,
            ),
            cooldown_epochs=20,
        ),
    ],

    # Curriculum progression
    curriculum=Schedule([
        (0, dict(synthetic_level=0, fee=0, slippage=0)),
        (100, dict(synthetic_level=1, fee=5, slippage=0)),
        (200, dict(synthetic_level=2, fee=10, slippage=2)),
        (300, dict(synthetic_level=3, fee=16, slippage=5)),
    ]),
)
```

### Core Classes

```python
# dojo/config_language.py

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, List

@dataclass
class Phase:
    """Training phase with termination condition."""
    name: str
    config: Dict[str, Any]
    until: Optional[Callable[['TrainingState'], bool]] = None

    def is_complete(self, state: 'TrainingState') -> bool:
        if self.until is None:
            return False
        return self.until(state)

@dataclass
class Trigger:
    """Reactive config change based on training state."""
    name: str
    when: Callable[['TrainingState'], bool]
    action: Callable[[Dict], None]
    cooldown_epochs: int = 10
    _last_fired: int = -999

    def check(self, state: 'TrainingState', config: Dict) -> bool:
        if state.epoch - self._last_fired < self.cooldown_epochs:
            return False
        if self.when(state):
            self.action(config)
            self._last_fired = state.epoch
            return True
        return False

@dataclass
class Warmup:
    """Linear interpolation over epochs."""
    start: float
    end: float
    over_epochs: int

    def get_value(self, epoch: int, phase_start_epoch: int) -> float:
        progress = min(1.0, (epoch - phase_start_epoch) / self.over_epochs)
        return self.start + (self.end - self.start) * progress

@dataclass
class Schedule:
    """Epoch-based config changes."""
    changes: List[tuple]  # [(epoch, config_dict), ...]

    def get_config(self, epoch: int) -> Dict[str, Any]:
        result = {}
        for trigger_epoch, config in self.changes:
            if epoch >= trigger_epoch:
                result.update(config)
        return result

@dataclass
class TrainingState:
    """Snapshot of training state for condition evaluation."""
    epoch: int
    entropy: float
    hold_pct: float
    loss: float
    reward_ma: float
    alpha: float
    lr: float

    _loss_history: List[float] = None

    def loss_plateau(self, window: int = 20, threshold: float = 0.01) -> bool:
        if self._loss_history is None or len(self._loss_history) < window:
            return False
        recent = self._loss_history[-window:]
        return (max(recent) - min(recent)) / (max(recent) + 1e-8) < threshold

class Config:
    """Main configuration object."""
    def __init__(
        self,
        name: str,
        base: Dict[str, Any],
        phases: List[Phase] = None,
        triggers: List[Trigger] = None,
        curriculum: Schedule = None,
    ):
        self.name = name
        self.base = base
        self.phases = phases or []
        self.triggers = triggers or []
        self.curriculum = curriculum

        self._current_phase_idx = 0
        self._phase_start_epoch = 0

    def get_config(self, state: TrainingState) -> Dict[str, Any]:
        """Compute current config based on state."""
        # Start with base
        config = self.base.copy()

        # Apply curriculum
        if self.curriculum:
            config.update(self.curriculum.get_config(state.epoch))

        # Apply current phase
        if self.phases:
            phase = self.phases[self._current_phase_idx]

            # Check phase transition
            if phase.is_complete(state):
                if self._current_phase_idx < len(self.phases) - 1:
                    self._current_phase_idx += 1
                    self._phase_start_epoch = state.epoch
                    phase = self.phases[self._current_phase_idx]

            # Apply phase config (resolve Warmups)
            for key, value in phase.config.items():
                if isinstance(value, Warmup):
                    config[key] = value.get_value(state.epoch, self._phase_start_epoch)
                else:
                    config[key] = value

        # Check triggers
        for trigger in self.triggers:
            trigger.check(state, config)

        return config
```

### Dojo Integration

```python
# tools/orchestration/dojo_manager.py

from configs.kitchen_sink_v3 import config

class DojoManager:
    def __init__(self, config: Config):
        self.config = config
        self.state = TrainingState(...)

    def run_epoch(self):
        # Get current config based on state
        current_config = self.config.get_config(self.state)

        # Apply to training
        self.trainer.update_config(current_config)

        # Run epoch
        metrics = self.trainer.train_epoch()

        # Update state
        self.state = TrainingState(
            epoch=self.state.epoch + 1,
            entropy=metrics['entropy'],
            hold_pct=metrics['hold_pct'],
            loss=metrics['loss'],
            reward_ma=metrics['reward_ma50'],
            alpha=metrics['alpha'],
            lr=self.trainer.current_lr,
        )

        # Log phase/trigger events
        self.log_config_state()
```

### Why Python > YAML/DSL

| Aspect | YAML | Custom DSL | Python Config |
|--------|------|------------|---------------|
| **Conditionals** | Hacks only | Must implement | Native |
| **Type checking** | No | Manual | MyPy/IDE |
| **Autocomplete** | No | No | Full IDE support |
| **Functions** | No | Limited | Full power |
| **Debugging** | Hard | Hard | Standard tools |
| **Learning curve** | Low | Medium | Zero (already know it) |
| **Expressiveness** | Low | Medium | Unlimited |

### Example Configs to Create

```
configs/
├── __init__.py
├── base.py              # Base class, utilities
├── presets/
│   ├── curriculum_L0.py
│   ├── curriculum_progressive.py
│   ├── kitchen_sink_v3.py
│   ├── stability_focused.py
│   └── max_exploration.py
└── experiments/
    ├── ablation_entropy.py
    └── ablation_sims.py
```

### Usage

```bash
# Run with Python config
python dojo_manager.py --config configs.presets.kitchen_sink_v3

# Or import and use programmatically
from configs.presets.kitchen_sink_v3 import config
dojo = DojoManager(config)
dojo.run()
```

### Acceptance Criteria (Phase 2)

- [ ] `Config`, `Phase`, `Trigger`, `Warmup`, `Schedule` classes implemented
- [ ] `TrainingState` captures all needed metrics
- [ ] `dojo_manager.py` supports Python config objects
- [ ] At least 3 preset configs created
- [ ] Phase transitions logged clearly
- [ ] Trigger activations logged with cooldown tracking
- [ ] Backward compatible (old CLI still works)
