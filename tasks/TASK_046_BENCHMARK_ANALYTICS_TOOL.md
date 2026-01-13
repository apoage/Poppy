# TASK_046: Benchmark Analytics Tool

## Status: PLANNING

## Priority: MEDIUM

## Overview

Create a comprehensive analytics tool for analyzing benchmark results from Dojo v2 curriculum training. The tool should provide insights into model progression, market adaptation, and identify areas for improvement.

---

## Data Sources

### Benchmark Files (per level)
```
logs/benchmarks/
├── level_00_benchmark.csv
├── level_00_benchmark.json
├── level_01_benchmark.csv
├── level_01_benchmark.json
├── ...
└── level_14_benchmark.json
```

### CSV Columns
```
period, reward, reward_std, alpha, market_return, agent_return,
n_sell, n_hold, n_buy, n_limit_sell, n_limit_buy,
pct_sell, pct_hold, pct_buy, pct_limit_sell, pct_limit_buy,
entropy, total_loss, self_play_time, epochs_completed
```

### Training Logs
```
logs/training_dojo_v2_L*_phase_*.csv
```

---

## Analytics Features

### 1. Curriculum Progression Dashboard

**Purpose**: Visualize how model performance evolves across levels

**Metrics to track**:
- Reward progression (Level 0 → 14)
- Alpha progression per market condition
- Action distribution evolution
- Entropy trend (policy crystallization)

**Visualizations**:
```
┌─────────────────────────────────────────────────────┐
│  Reward Progression Across Levels                  │
│  ───────────────────────────────────────────────── │
│  50 │                                    ●──●      │
│     │                              ●──●            │
│  40 │                        ●──●                  │
│     │                  ●──●                        │
│  30 │            ●──●                              │
│     │      ●──●                                    │
│  20 │●──●                                          │
│     └────────────────────────────────────────────  │
│       0   2   4   6   8  10  12  14   Level        │
└─────────────────────────────────────────────────────┘
```

### 2. Market Condition Analysis

**Purpose**: Compare performance across bull/bear/sideways markets

**Metrics**:
- Alpha by market condition
- Optimal action distribution per condition
- Reward stability (std deviation)

**Output**:
```
Level 5 - Market Condition Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Condition   Reward   Alpha    Hold%   Strategy
─────────────────────────────────────────────
Bull        42.50   +23.1%   45.2%   Trend-following
Bear        28.30   -15.2%   72.1%   Defensive
Sideways    35.10    +5.8%   58.3%   Mixed
```

### 3. Strategy Evolution Report

**Purpose**: Track how agent's trading strategy evolves

**Metrics**:
- Limit order usage over levels
- Buy/Sell balance
- Market vs limit order ratio
- Position sizing patterns

**Analysis**:
- Does agent learn to use limit orders?
- Does agent become more/less aggressive?
- Does agent adapt strategy to market condition?

### 4. Generalization Score

**Purpose**: Quantify how well synthetic training transfers to real markets

**Formula**:
```python
generalization_score = (
    0.4 * normalize(bull_alpha) +
    0.4 * normalize(bear_alpha) +
    0.2 * normalize(sideways_alpha)
) * stability_factor

stability_factor = 1 - (reward_std / reward_mean)
```

**Output per level**:
```
Level  Gen.Score  Bull    Bear    Side    Stability
─────────────────────────────────────────────────────
  0      0.42    +15.2%  -8.3%   +3.1%    0.85
  5      0.58    +22.1%  +2.3%   +8.5%    0.78
 10      0.71    +28.5%  +12.1%  +15.2%   0.82
 14      0.79    +35.2%  +18.3%  +22.1%   0.88
```

### 5. Anomaly Detection

**Purpose**: Identify concerning patterns

**Flags**:
- Policy collapse (Hold% > 90%)
- Reward regression (Level N < Level N-1)
- Action imbalance (one action dominates)
- High variance (reward_std > reward_mean)
- Bear market failure (alpha < -20%)

### 6. Comparative Analysis

**Purpose**: Compare different training runs

**Features**:
- Side-by-side level comparison
- Training config diff
- Performance delta analysis

---

## Implementation

### File Structure
```
tools/analysis/
├── benchmark_analytics.py    # Main analytics tool
├── benchmark_visualizer.py   # Chart generation
├── benchmark_report.py       # Report generation
└── templates/
    ├── report_template.html  # HTML report template
    └── summary_template.md   # Markdown summary
```

### CLI Interface
```bash
# Full analytics report
python tools/analysis/benchmark_analytics.py --all

# Specific level analysis
python tools/analysis/benchmark_analytics.py --level 5

# Compare two runs
python tools/analysis/benchmark_analytics.py --compare run1 run2

# Generate HTML dashboard
python tools/analysis/benchmark_analytics.py --dashboard

# Export to PDF
python tools/analysis/benchmark_analytics.py --export pdf
```

### Output Formats

1. **Console**: Pretty-printed tables and summaries
2. **Markdown**: `logs/benchmarks/analysis_report.md`
3. **HTML Dashboard**: `logs/benchmarks/dashboard.html`
4. **JSON**: Machine-readable analytics data

---

## Report Sections

### Executive Summary
```markdown
# Dojo v2 Training Report
Generated: 2026-01-14 10:30:00

## Key Findings
- Best Level: 10 (Breakout) - Generalization Score: 0.78
- Weakest: Level 7 (Index Arb) - Bear Market Struggle
- Policy Health: Stable (no collapse detected)
- Recommendation: Focus on bear market adaptation

## Performance Summary
| Metric           | Start (L0) | End (L14) | Change |
|------------------|------------|-----------|--------|
| Avg Reward       | 22.5       | 45.3      | +101%  |
| Bull Alpha       | +12%       | +35%      | +23pp  |
| Bear Alpha       | -15%       | +18%      | +33pp  |
| Generalization   | 0.35       | 0.79      | +126%  |
```

### Detailed Level Analysis
```markdown
## Level 5: Pairs Trading

### Performance
- Bull: +22.1% alpha (Strong)
- Bear: +2.3% alpha (Weak)
- Sideways: +8.5% alpha (Moderate)

### Strategy Analysis
- Shifted to more limit orders vs L4
- Increased sell activity in bear markets
- Hold% dropped from 68% to 52%

### Recommendations
- Bear market performance needs attention
- Consider increasing training epochs in Phase C
```

---

## Visualization Examples

### 1. Heatmap: Alpha by Level × Market
```
         Bull    Bear   Sideways
L0      ████░░  ░░░░░░  ██░░░░
L5      █████░  ██░░░░  ███░░░
L10     ██████  ████░░  █████░
L14     ██████  █████░  ██████
        (darker = higher alpha)
```

### 2. Action Distribution Stack Chart
```
100% ┤███████████████████████████████████
     │ HOLD ████████████████████████████
 50% ┤═══════════════════════════════════
     │ BUY  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
     │ SELL ░░░░░░░░░░░░░░░░░
  0% └───────────────────────────────────
      L0  L2  L4  L6  L8  L10 L12 L14
```

### 3. Generalization Score Trend
```
Score
1.0 ┤                              ●
    │                          ●
0.8 ┤                      ●
    │                  ●
0.6 ┤              ●
    │          ●
0.4 ┤      ●
    │  ●
0.2 ┤
    └────────────────────────────────
     L0  L2  L4  L6  L8 L10 L12 L14
```

---

## Integration with DS24

The analytics tool should be accessible from DS24 dashboard:

```javascript
// In dashboard.js - Add benchmark tab
function loadBenchmarkAnalytics() {
    fetch('/api/benchmark/summary')
        .then(r => r.json())
        .then(data => renderBenchmarkCharts(data));
}
```

---

## Dependencies

```python
# Required packages
pandas>=2.0.0        # Data manipulation
matplotlib>=3.7.0    # Visualization
seaborn>=0.12.0      # Statistical viz
tabulate>=0.9.0      # Pretty tables
jinja2>=3.1.0        # HTML templates
weasyprint>=60.0     # PDF export (optional)
```

---

## Implementation Phases

### Phase 1: Core Analytics (MVP)
- [ ] Load and parse benchmark CSV/JSON files
- [ ] Calculate basic statistics
- [ ] Generate console report
- [ ] Markdown export

### Phase 2: Visualization
- [ ] Progression charts
- [ ] Market condition heatmaps
- [ ] Action distribution plots
- [ ] HTML dashboard

### Phase 3: Advanced Features
- [ ] Anomaly detection
- [ ] Comparative analysis
- [ ] DS24 integration
- [ ] PDF export

---

## Success Criteria

1. **Automated**: Runs after each benchmark without manual intervention
2. **Actionable**: Reports provide clear improvement recommendations
3. **Visual**: Charts make patterns immediately obvious
4. **Tracked**: Historical comparison across training runs
5. **Integrated**: Accessible from DS24 dashboard

---

## Notes

- Start simple, add features based on actual training insights
- Focus on identifying failure modes early
- Make it easy to share reports with team
- Consider adding Slack/Discord notifications for key milestones
