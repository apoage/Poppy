# Multi-Asset Correlation Trading Agent - Master Implementation Guide

## Project Overview
Build an autonomous trading agent that monitors 64+ volatile crypto assets, learns cross-asset correlations, and executes trades based on lead-lag patterns. Uses TRM (Tiny Recursive Models) + MCTS for decision-making.

## Core Strategy
- **Lead-lag arbitrage**: When Asset A moves, predict and trade Asset B before it follows
- **Correlation exploitation**: Learn all correlation types (positive, negative, delayed)
- **Volatility banking**: Higher volatility = more opportunities

## Technical Stack
- Python 3.10+
- PyTorch (CUDA 12.x)
- Kraken API (REST + WebSocket V2)
- SQLite/PostgreSQL for data cache
- LightZero 0.2.0 for MCTS

## Architecture Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INFRASTRUCTURE                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Kraken API   │→ │ Rate Limiter │→ │ Local DB Cache       │  │
│  │ (REST+WS)    │  │ (be nice)    │  │ (SQLite/Postgres)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STATE ENCODER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ 64-Asset     │  │ Correlation  │  │ Technical            │  │
│  │ Price/Volume │+ │ Matrix       │+ │ Indicators           │  │
│  │ (per asset)  │  │ (64x64)      │  │ (RSI, MACD, etc)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                              ↓                                  │
│              Multi-Channel Tensor (ready for CNN/TRM)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    TRM + MCTS AGENT                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ TRM Recursive Reasoner (~7M params)                      │  │
│  │ - H_cycles: output refinement (trading decisions)        │  │
│  │ - L_cycles: latent refinement (correlation understanding)│  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ MCTS Planner (50-100 simulations)                        │  │
│  │ - Plans hours ahead                                      │  │
│  │ - Epistemic uncertainty for unknown correlations         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                  │
│              Action: [asset_idx, direction, size]               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EXECUTION LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Position     │  │ Risk         │  │ Order                │  │
│  │ Manager      │→ │ Manager      │→ │ Executor             │  │
│  │ (portfolio)  │  │ (limits)     │  │ (Kraken WS)          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Data Infrastructure
- Kraken data fetcher with rate limiting
- Local database schema for OHLCV + trades
- Incremental historical data collection
- Real-time WebSocket feed integration

### Phase 2: Multi-Asset State Encoder  
- 64-asset parallel processing
- Rolling correlation matrix computation
- Technical indicator generation
- GPU-accelerated encoding

### Phase 3: TRM+MCTS Agent
- TRM architecture (~7M params)
- Multi-asset action space
- MCTS with correlation-aware simulations
- Portfolio-level value estimation

### Phase 4: Training Loop
- Historical backtesting environment
- Paper trading mode
- Self-play with synthetic market scenarios
- Live deployment pipeline

### Phase 5: Monitoring & Analysis
- Correlation visualization
- Trade attribution (which signals drove decisions)
- Performance dashboards
- Alert system

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Asset pool | 64 (expandable) | Top volatile, exclude stablecoins |
| React time | 1-5 minutes | Execution granularity |
| Plan horizon | 1-4 hours | MCTS lookahead |
| Trades/day | 16-256 | Based on signal quality |
| Position sizing | 0.1-1.0 | Per-asset allocation |
| Max portfolio exposure | 2.0 | Can be 200% with leverage |

## File Structure
```
mcts_trader/
├── data/
│   ├── fetcher.py          # Kraken API client
│   ├── cache.py            # Local DB operations
│   └── stream.py           # WebSocket handler
├── encoder/
│   ├── correlation.py      # Correlation matrix
│   ├── indicators.py       # Technical indicators
│   └── state.py            # Combined encoder
├── agent/
│   ├── trm.py              # TRM model
│   ├── mcts.py             # MCTS planner
│   └── portfolio.py        # Position management
├── training/
│   ├── environment.py      # Backtesting env
│   ├── trainer.py          # Training loop
│   └── paper_trade.py      # Paper trading
├── utils/
│   ├── config.py           # Configuration
│   └── logging.py          # Logging setup
└── main.py                 # Entry point
```

## Success Metrics
- Sharpe > 2.0 (paper trading)
- Max drawdown < 15%
- Profit factor > 1.5
- Win rate > 52%
- Correlation alpha: outperform single-asset strategies
