# Claude Code Debug Log

## Open Issues

### Issue #1: Monkey-Patch Dependency
**Location**: `main.py:12-51`
**Severity**: High
**Description**: Core training relies on monkey-patching LightZero's `mz_network_output_unpack` function to fix C++ binding error.

**Details**:
- C++ MCTS tree expects `reward` as Python list
- LightZero returns tensors/arrays that cause type mismatch
- Patch converts all reward formats → list of scalars

**Risks**:
- Fragile: breaks if LightZero updates this function
- Hard to debug: error occurs deep in C++ binding
- Maintenance burden: must verify patch on LightZero upgrades

**Recommendations**:
1. File issue with LightZero project (if not already fixed in v0.2.0+)
2. Add version pinning: `lightzero==0.2.0` in requirements.txt
3. Add unit test that validates patch behavior
4. Document this in CLAUDE.md (already done)

---

### Issue #2: Experiment Folder Proliferation
**Location**: Root directory
**Severity**: Medium
**Description**: Training creates timestamped folders (`default_experiment_YYMMDD_HHMMSS`) that accumulate over time.

**Observed**:
```
default_experiment_251203_000048/
default_experiment_251203_000136/
default_experiment_251203_000345/
... (30+ folders)
```

**Impact**:
- Disk space usage (each contains checkpoints)
- Hard to identify "current" training run
- Clutters root directory

**Recommendations**:
1. Configure LightZero to use single experiment dir with internal versioning
2. Create cleanup script: `scripts/cleanup_experiments.sh` (keep last N)
3. Add to .gitignore (already done)

---

### Issue #3: Action Decoding Logic
**Location**: `env/trading_env.py:142-145`, CLAUDE.md
**Severity**: Low (documentation inconsistency)
**Description**: Code comment may have error in action decoding explanation.

**Code**:
```python
asset_idx = action // self.n_actions_per_asset  # [0, 63]
action_type = action % self.n_actions_per_asset  # 0=Sell, 1=Hold, 2=Buy
```

**In CLAUDE.md**:
> Example: action=130 → asset 43, Buy (130 = 43*3 + 1... wait that's wrong in the code comments - verify this)

**Verification Needed**:
- action=130: 130 // 3 = 43, 130 % 3 = 1 → Asset 43, Hold (not Buy)
- Buy would be action=131 (43*3 + 2)

**Action**: Update CLAUDE.md with correct example.

---

## Resolved Issues

*(None yet)*

---

## Investigation Queue

- [ ] Why does TradingEnv.step() print `.` to stdout? (Line 194)
- [ ] Is `to_play` field used in single-agent RL? (observation dict)
- [ ] Verify HistoricalMarket handles edge cases (start/end of data)
- [ ] Check if encoder handles missing asset data gracefully

---

*Last Updated: 2025-12-03 22:40 UTC*
