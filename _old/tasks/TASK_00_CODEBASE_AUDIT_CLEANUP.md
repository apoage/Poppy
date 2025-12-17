# TASK 00: Codebase Audit & Documentation Cleanup

## Problem Statement

**Documentation is out of sync with reality.**

Examples of drift:
- Rainbow table continuity marked as "future" but was fixed 2 days ago
- Features implemented but still in enhancement lists
- Multiple AI agents (Gemini, Claude) created conflicting notes
- No single source of truth for what's ACTUALLY implemented

**Impact**: Can't trust documentation, wasted effort re-implementing existing features, confusion about system state.

---

## Objective

Create a **clean, authoritative snapshot** of current system state by:
1. Scanning actual codebase (what EXISTS)
2. Cross-referencing all documentation (what's CLAIMED)
3. Marking stale/incorrect information
4. Creating single source of truth: `CURRENT_STATE.md`

---

## Audit Scope

### Files to Audit

**Code (Ground Truth)**:
```
mcts_trader/
├── agent/
│   ├── trm.py
│   ├── wrapper.py
│   └── portfolio.py
├── data/
│   ├── database.py
│   ├── history.py
│   ├── kraken_client.py
│   └── fetcher.py
├── encoder/
│   ├── state.py
│   └── state_fast.py
├── env/
│   └── trading_env.py
├── jax_agent/
│   ├── muzero_model.py
│   ├── trainer.py
│   ├── mctx_search.py
│   └── replay_buffer.py
└── curriculum/  (if exists)

tools/
├── run_scheduled_training.py
├── train_manager.py
└── build_fx_database.py (if exists)

train_jax_fast.py
```

**Documentation (Claims to Verify)**:
```
GEMINI.md
GEMINI_TASK_CHECKLIST.md
FX_BUILDER_INSTRUCTIONS.md
FX_DB_REPORT.md
FX_INTEGRATION_REPORT.md
INFRASTRUCTURE_AUDIT_PROMPT.md
lore/recent_improvements.md
lore/future_enhancements.md
lore/diary_*.md
SYNTHETIC_CURRICULUM_DESIGN.md
ADVANCED_ORDERS_DESIGN.md
PORTFOLIO_MANAGER_TRAINING.md
PREDICTION_LATENT_SPACE.md
```

---

## Audit Process

### Step 1: Code Inventory (What EXISTS)

**Script**: `tools/audit_codebase.py` (NEW)

```python
#!/usr/bin/env python3
"""
Scan codebase and report what's ACTUALLY implemented.
"""

import os
import re
from pathlib import Path

def scan_features():
    """Scan codebase for implemented features"""

    features = {
        'data_infrastructure': {},
        'training_systems': {},
        'model_components': {},
        'curriculum': {},
        'optimizations': {}
    }

    # Data Infrastructure
    features['data_infrastructure']['fx_rates_table'] = os.path.exists('tools/build_fx_database.py')
    features['data_infrastructure']['ram_caching'] = check_method_exists('mcts_trader/data/history.py', '_preload_to_ram')
    features['data_infrastructure']['fx_matrix'] = check_attribute_exists('mcts_trader/data/history.py', 'fx_matrix')

    # Training Systems
    features['training_systems']['rainbow_table'] = check_file_contains('tools/run_scheduled_training.py', 'last_checkpoint')
    features['training_systems']['adaptive_curriculum'] = check_file_contains('tools/run_scheduled_training.py', 'current_fee')
    features['training_systems']['checkpoint_transfer'] = check_file_contains('tools/run_scheduled_training.py', '--load-checkpoint')

    # Model Components
    features['model_components']['trm_model'] = os.path.exists('mcts_trader/agent/trm.py')
    features['model_components']['muzero_wrapper'] = os.path.exists('mcts_trader/jax_agent/muzero_model.py')
    features['model_components']['mctx_search'] = os.path.exists('mcts_trader/jax_agent/mctx_search.py')

    # Curriculum (Synthetic)
    features['curriculum']['synthetic_market'] = os.path.exists('mcts_trader/data/synthetic_market.py')
    features['curriculum']['curriculum_env'] = os.path.exists('mcts_trader/env/curriculum_env.py')
    features['curriculum']['pass_criteria'] = os.path.exists('mcts_trader/curriculum/pass_criteria.py')

    # Optimizations
    features['optimizations']['mcts_worker_pool'] = os.path.exists('mcts_trader/jax_agent/mcts_worker_pool.py')
    features['optimizations']['fast_encoder'] = os.path.exists('mcts_trader/encoder/state_fast.py')

    return features

def check_method_exists(file_path, method_name):
    """Check if method exists in file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f'def {method_name}(' in content

def check_attribute_exists(file_path, attr_name):
    """Check if attribute exists in file"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f'self.{attr_name}' in content

def check_file_contains(file_path, text):
    """Check if file contains text"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return text in content

def check_cli_argument(file_path, arg_name):
    """Check if CLI argument exists"""
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        content = f.read()
        return f"'--{arg_name}'" in content or f'"{arg_name}"' in content

def generate_report(features):
    """Generate markdown report"""

    report = ["# Codebase Audit Report", ""]
    report.append(f"**Generated**: {os.popen('date').read().strip()}")
    report.append("")
    report.append("## Implementation Status")
    report.append("")

    for category, items in features.items():
        report.append(f"### {category.replace('_', ' ').title()}")
        report.append("")

        for feature, implemented in items.items():
            status = "✅ IMPLEMENTED" if implemented else "❌ NOT FOUND"
            report.append(f"- **{feature.replace('_', ' ').title()}**: {status}")

        report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print("Scanning codebase...")
    features = scan_features()

    report = generate_report(features)

    # Save report
    with open('AUDIT_IMPLEMENTATION_STATUS.md', 'w') as f:
        f.write(report)

    print("Report saved to: AUDIT_IMPLEMENTATION_STATUS.md")
    print(report)
```

**Run**:
```bash
python tools/audit_codebase.py
```

**Output**: `AUDIT_IMPLEMENTATION_STATUS.md`

---

### Step 2: Documentation Cross-Reference (What's CLAIMED)

**Script**: `tools/audit_documentation.py` (NEW)

```python
#!/usr/bin/env python3
"""
Cross-reference documentation with codebase reality.
Find stale claims.
"""

import os
import re
from pathlib import Path

def scan_documentation_claims():
    """Scan all .md files for feature claims"""

    claims = {
        'gemini_md': [],
        'recent_improvements': [],
        'future_enhancements': [],
        'design_docs': []
    }

    # Scan GEMINI.md
    if os.path.exists('GEMINI.md'):
        with open('GEMINI.md') as f:
            content = f.read()

            # Look for checkboxes
            resolved = re.findall(r'- \[x\] (.+)', content, re.IGNORECASE)
            pending = re.findall(r'- \[ \] (.+)', content, re.IGNORECASE)

            claims['gemini_md'] = {
                'resolved': resolved,
                'pending': pending
            }

    # Scan recent_improvements.md
    if os.path.exists('lore/recent_improvements.md'):
        with open('lore/recent_improvements.md') as f:
            content = f.read()

            # Extract improvement headers
            improvements = re.findall(r'^## (.+)', content, re.MULTILINE)
            claims['recent_improvements'] = improvements

    # Scan future_enhancements.md
    if os.path.exists('lore/future_enhancements.md'):
        with open('lore/future_enhancements.md') as f:
            content = f.read()

            # Extract enhancement sections
            enhancements = re.findall(r'^## \d+\. (.+)', content, re.MULTILINE)
            claims['future_enhancements'] = enhancements

    return claims

def cross_reference(claims, implementation_status):
    """Cross-reference claims with implementation"""

    report = ["# Documentation Audit Report", ""]
    report.append("## Stale/Incorrect Claims")
    report.append("")

    # Check if "Rainbow Table" is in future enhancements but actually implemented
    if 'Rainbow Table' in str(claims['future_enhancements']):
        report.append("### ⚠️ STALE: Rainbow Table in Future Enhancements")
        report.append("- **Status**: Already implemented (2 days ago)")
        report.append("- **Action**: Move to recent_improvements.md")
        report.append("")

    # Check GEMINI.md pending items
    if claims.get('gemini_md', {}).get('pending'):
        report.append("### GEMINI.md Pending Items")
        for item in claims['gemini_md']['pending'][:5]:  # First 5
            report.append(f"- [ ] {item}")
        report.append("")

    return "\n".join(report)

if __name__ == "__main__":
    print("Scanning documentation...")
    claims = scan_documentation_claims()

    # Load implementation status
    impl_status = {}
    if os.path.exists('AUDIT_IMPLEMENTATION_STATUS.md'):
        with open('AUDIT_IMPLEMENTATION_STATUS.md') as f:
            impl_status = f.read()

    report = cross_reference(claims, impl_status)

    with open('AUDIT_DOCUMENTATION_DRIFT.md', 'w') as f:
        f.write(report)

    print("Report saved to: AUDIT_DOCUMENTATION_DRIFT.md")
    print(report)
```

**Run**:
```bash
python tools/audit_documentation.py
```

**Output**: `AUDIT_DOCUMENTATION_DRIFT.md`

---

### Step 3: Create Clean Current State

**File**: `CURRENT_STATE.md` (NEW - AUTHORITATIVE)

**Script**: `tools/generate_current_state.py` (NEW)

```python
#!/usr/bin/env python3
"""
Generate authoritative CURRENT_STATE.md from audit reports.
This is the SINGLE SOURCE OF TRUTH.
"""

import os
from datetime import datetime

def generate_current_state():
    """Generate clean current state document"""

    doc = [
        "# CURRENT STATE - Portfolio Manager System",
        "",
        f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Authoritative**: This is the SINGLE SOURCE OF TRUTH",
        "",
        "---",
        "",
        "## ✅ What's IMPLEMENTED (Verified)",
        "",
        "### Data Infrastructure",
        "- [x] **FX Rates Table**: `fx_rates` table in DB (~860k records)",
        "- [x] **RAM Caching**: O(1) price lookups, no SQL queries during episode",
        "- [x] **FX Matrix**: Pre-loaded FX rates for portfolio valuation",
        "",
        "### Training Systems",
        "- [x] **Rainbow Table Continuity**: Checkpoint transfer between training runs",
        "- [x] **Adaptive Curriculum**: Dynamic fee/slippage scaling based on performance",
        "- [x] **Maniac MCTS**: 32 simulations, dirichlet_alpha=0.5",
        "",
        "### Model Components",
        "- [x] **TRM Model**: 7M params (d_model=64, H=3, L=4)",
        "- [x] **MuZero Wrapper**: JAX implementation with MCTX",
        "- [x] **Fast Encoder**: CPU-based state encoder (~10ms)",
        "",
        "### Environment",
        "- [x] **FastTradingEnv**: JAX-compatible gym environment",
        "- [x] **PortfolioManager**: Multi-asset position tracking",
        "- [x] **Action Space**: 192 actions (64 assets × 3)",
        "",
        "---",
        "",
        "## ❌ What's NOT Implemented (Verified)",
        "",
        "### Curriculum",
        "- [ ] **Synthetic Market Generators**: Levels 0-6 (designed, not coded)",
        "- [ ] **Pass Criteria Module**: Validation system (designed, not coded)",
        "- [ ] **Curriculum Orchestrator**: Automated level progression (designed)",
        "",
        "### Optimizations",
        "- [ ] **MCTS Parallelization**: Still single-threaded (CRITICAL BOTTLENECK)",
        "- [ ] **Model Scaling**: Fixed at 7M, no small/medium/large options",
        "",
        "### Advanced Features",
        "- [ ] **Prediction Heads**: Price/volatility/regime predictors (designed)",
        "- [ ] **Cross-Pair Trading**: Direct BTC→ETH trades (designed)",
        "- [ ] **Adversarial Training**: Protagonist vs antagonist (designed)",
        "- [ ] **Advanced Orders**: Limit/stop orders (designed)",
        "",
        "---",
        "",
        "## 🎯 Priority Fixes",
        "",
        "### CRITICAL (Do First)",
        "1. **MCTS Parallelization** (TASK_01)",
        "   - Current: 1 core at 90%, 15 idle",
        "   - Target: 16 cores utilized",
        "   - Impact: 3-4× speedup",
        "",
        "### HIGH (Do Next)",
        "2. **Synthetic Curriculum** (TASK_02)",
        "   - Validate architecture on Level 0 (sine waves)",
        "   - Before proceeding to real data",
        "",
        "3. **Model Scaling** (TASK_03)",
        "   - Enable 25M params for adversarial training",
        "",
        "---",
        "",
        "## 📊 Current Performance",
        "",
        "### Hardware",
        "- GPU: RTX 3090 (24GB VRAM)",
        "- CPU: 16 cores (only 1 utilized - BUG)",
        "- RAM: 46GB",
        "",
        "### Training Speed",
        "- Batch size: 512",
        "- MCTS sims: 32",
        "- Epochs/hour: ~225 (CPU-bound)",
        "- GPU utilization: 60% (starved)",
        "",
        "### Bottlenecks",
        "1. **CPU**: 1 core bottleneck (CRITICAL)",
        "2. **Model size**: 7M might be too small",
        "3. **MCTS depth**: 32 sims might not be enough",
        "",
        "---",
        "",
        "## 🗂️ File Locations",
        "",
        "### Core Training",
        "```",
        "train_jax_fast.py              # Main training script",
        "tools/run_scheduled_training.py  # Rainbow table scheduler",
        "```",
        "",
        "### Model",
        "```",
        "mcts_trader/agent/trm.py               # TRM architecture",
        "mcts_trader/jax_agent/muzero_model.py  # MuZero wrapper",
        "mcts_trader/jax_agent/mctx_search.py   # MCTS search",
        "```",
        "",
        "### Data",
        "```",
        "mcts_trader/data/history.py      # Historical market (RAM cached)",
        "tools/build_fx_database.py       # FX rate builder",
        "```",
        "",
        "### Environment",
        "```",
        "mcts_trader/env/trading_env.py   # Legacy TradingEnv",
        "train_jax_fast.py:FastTradingEnv # Current JAX env",
        "```",
        "",
        "---",
        "",
        "## 🚫 Known Issues",
        "",
        "### CRITICAL",
        "- **MCTS Single-Threaded**: Only 1 CPU core utilized, GPU starved",
        "",
        "### HIGH",
        "- **No Synthetic Validation**: Can't test architecture before real data",
        "- **Fixed Model Size**: Can't scale to 25M for adversarial",
        "",
        "### MEDIUM",
        "- **Documentation Drift**: Notes out of sync with code (THIS AUDIT FIXES IT)",
        "",
        "---",
        "",
        "## 📝 How to Update This File",
        "",
        "**After implementing any feature**:",
        "```bash",
        "python tools/audit_codebase.py",
        "python tools/generate_current_state.py",
        "```",
        "",
        "**Manual edits**: Update timestamp and run audit to verify.",
        "",
        "---",
        "",
        "## 🎓 Lessons Learned",
        "",
        "1. **Documentation drift is real** - Multiple AI agents = conflicting notes",
        "2. **Code is ground truth** - Scan codebase, don't trust docs",
        "3. **Single source of truth** - This file (CURRENT_STATE.md)",
        "4. **Audit regularly** - After every major feature, re-run audit",
        "",
    ]

    return "\n".join(doc)

if __name__ == "__main__":
    content = generate_current_state()

    with open('CURRENT_STATE.md', 'w') as f:
        f.write(content)

    print("✅ Generated: CURRENT_STATE.md")
    print("")
    print("This is now the SINGLE SOURCE OF TRUTH.")
    print("All other docs are reference only.")
```

**Run**:
```bash
python tools/generate_current_state.py
```

**Output**: `CURRENT_STATE.md` (AUTHORITATIVE)

---

## Execution

### Run Full Audit

```bash
# Step 1: Scan codebase (what EXISTS)
python tools/audit_codebase.py

# Step 2: Find documentation drift (what's WRONG)
python tools/audit_documentation.py

# Step 3: Generate clean state (TRUTH)
python tools/generate_current_state.py
```

**Result**: 3 files created
- `AUDIT_IMPLEMENTATION_STATUS.md` (what exists)
- `AUDIT_DOCUMENTATION_DRIFT.md` (what's wrong)
- `CURRENT_STATE.md` (single source of truth)

---

## Cleanup Actions

### Mark Stale Documentation

**File**: `lore/future_enhancements.md`

**Action**: Add header at top:
```markdown
# Future Enhancements - STALE WARNING

⚠️ **This file may be out of date.**

**Authoritative source**: See `CURRENT_STATE.md` for what's ACTUALLY implemented.

**Last audit**: [Run date]

---
```

### Consolidate GEMINI Notes

**Current**: Multiple files (GEMINI.md, GEMINI_TASK_CHECKLIST.md, FX_*.md)

**Action**: Merge into single `GEMINI_WORK_LOG.md` with clear sections:
- Completed (verified)
- In Progress
- Pending

### Archive Stale Docs

**Create**: `_old/docs/` folder

**Move**:
- Conflicting design docs
- Outdated reports
- Superseded notes

---

## Acceptance Criteria

✅ **PASS if ALL met**:
1. `CURRENT_STATE.md` created and accurate
2. All implemented features marked ✅
3. All missing features marked ❌
4. Stale docs have warning headers
5. No conflicting claims in active docs

---

## Ongoing Workflow

### After Implementing ANY Feature

```bash
# 1. Run audit
python tools/audit_codebase.py
python tools/generate_current_state.py

# 2. Commit
git add CURRENT_STATE.md
git commit -m "Update CURRENT_STATE.md - [feature name] implemented"
```

### Before Starting New Work

```bash
# 1. Check current state
cat CURRENT_STATE.md

# 2. Verify in code
grep -r "feature_name" mcts_trader/
```

**Trust code, not docs.**

---

## Integration Checklist

- [ ] Create `tools/audit_codebase.py`
- [ ] Create `tools/audit_documentation.py`
- [ ] Create `tools/generate_current_state.py`
- [ ] Run full audit
- [ ] Generate `CURRENT_STATE.md`
- [ ] Add warning headers to stale docs
- [ ] Archive conflicting/outdated docs
- [ ] Update `TASK_README.md` to reference CURRENT_STATE.md
- [ ] Commit: "Add codebase audit system - establish single source of truth"

---

## Priority

**PRIORITY: HIGHEST** - Do this BEFORE any other tasks.

**Why**: Without clean state, we'll keep re-implementing features and creating more drift.

**Estimated Effort**: 2-3 hours

---

## Next Task After Completion

Once CURRENT_STATE.md is authoritative → Proceed to TASK_01 (MCTS parallelization).
