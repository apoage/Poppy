**PROMPT 6: Monitoring & Analysis**

```markdown
# Monitoring & Analysis - Real-time Dashboard and Performance Attribution

## Objective
Complete monitoring infrastructure: live dashboards, correlation visualization, performance attribution, and alerting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Metrics     │  │  Visualizer  │  │  Alert Manager   │  │
│  │  Collector   │──│  (Dash/Web)  │──│  (Discord/Email) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  TimescaleDB │  │  Correlation │  │  Attribution     │  │
│  │  (Metrics)   │  │  Heatmaps    │  │  Analysis        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Metrics Collector

```python
"""
metrics.py - Performance metrics collection and storage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import sqlite3
import json

@dataclass
class TradeMetrics:
    """Single trade metrics"""
    timestamp: datetime
    asset_idx: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    hold_time: timedelta
    correlation_signal: Optional[float] = None
    lead_asset: Optional[str] = None

@dataclass
class PortfolioMetrics:
    """Portfolio snapshot metrics"""
    timestamp: datetime
    equity: float
    cash: float
    n_positions: int
    gross_exposure: float
    net_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown: float
    sharpe_1h: float
    sharpe_24h: float
    sortino_24h: float
    win_rate: float
    profit_factor: float

class MetricsCollector:
    """
    Collects and stores trading metrics.
    """
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self._init_db()
        
        # In-memory buffers for real-time
        self.equity_buffer = deque(maxlen=1440)  # 24h of minute data
        self.trades_buffer = deque(maxlen=1000)
        self.returns_buffer = deque(maxlen=1440)
        
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TEXT PRIMARY KEY,
                equity REAL,
                cash REAL,
                n_positions INTEGER,
                gross_exposure REAL,
                net_exposure REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                drawdown REAL,
                sharpe_1h REAL,
                sharpe_24h REAL,
                sortino_24h REAL,
                win_rate REAL,
                profit_factor REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                asset_idx INTEGER,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                size REAL,
                pnl REAL,
                pnl_pct REAL,
                hold_time_seconds INTEGER,
                correlation_signal REAL,
                lead_asset TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlation_snapshots (
                timestamp TEXT,
                asset_i INTEGER,
                asset_j INTEGER,
                correlation_15m REAL,
                correlation_1h REAL,
                correlation_4h REAL,
                lead_lag REAL,
                PRIMARY KEY (timestamp, asset_i, asset_j)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_portfolio_snapshot(self, metrics: PortfolioMetrics):
        """Record portfolio snapshot"""
        self.equity_buffer.append((metrics.timestamp, metrics.equity))
        
        if len(self.equity_buffer) > 1:
            prev_equity = self.equity_buffer[-2][1]
            ret = (metrics.equity - prev_equity) / prev_equity
            self.returns_buffer.append(ret)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.timestamp.isoformat(),
            metrics.equity, metrics.cash, metrics.n_positions,
            metrics.gross_exposure, metrics.net_exposure,
            metrics.unrealized_pnl, metrics.realized_pnl,
            metrics.drawdown, metrics.sharpe_1h, metrics.sharpe_24h,
            metrics.sortino_24h, metrics.win_rate, metrics.profit_factor
        ))
        conn.commit()
        conn.close()
    
    def record_trade(self, trade: TradeMetrics):
        """Record completed trade"""
        self.trades_buffer.append(trade)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (timestamp, asset_idx, symbol, side, entry_price, 
                               exit_price, size, pnl, pnl_pct, hold_time_seconds,
                               correlation_signal, lead_asset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp.isoformat(), trade.asset_idx, trade.symbol,
            trade.side, trade.entry_price, trade.exit_price, trade.size,
            trade.pnl, trade.pnl_pct, int(trade.hold_time.total_seconds()),
            trade.correlation_signal, trade.lead_asset
        ))
        conn.commit()
        conn.close()
    
    def record_correlations(self, timestamp: datetime, correlations: np.ndarray):
        """Record correlation matrix snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        n = correlations.shape[1]
        for i in range(n):
            for j in range(i+1, n):
                cursor.execute("""
                    INSERT OR REPLACE INTO correlation_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp.isoformat(), i, j,
                    float(correlations[0, i, j]),  # 15m
                    float(correlations[1, i, j]),  # 1h
                    float(correlations[2, i, j]),  # 4h
                    float(correlations[3, i, j])   # lead-lag
                ))
        
        conn.commit()
        conn.close()
    
    def get_realtime_metrics(self) -> Dict:
        """Get current real-time metrics"""
        if len(self.equity_buffer) < 2:
            return {}
        
        equity = [e[1] for e in self.equity_buffer]
        returns = list(self.returns_buffer)
        
        # Calculate metrics
        current = equity[-1]
        peak = max(equity)
        
        metrics = {
            'current_equity': current,
            'drawdown': (peak - current) / peak if peak > 0 else 0,
            'return_1h': (equity[-1] / equity[-60] - 1) if len(equity) >= 60 else 0,
            'return_24h': (equity[-1] / equity[0] - 1) if len(equity) > 1 else 0,
        }
        
        if len(returns) >= 60:
            ret_arr = np.array(returns[-60:])
            metrics['sharpe_1h'] = ret_arr.mean() / (ret_arr.std() + 1e-8) * np.sqrt(60)
        
        if len(returns) >= 1440:
            ret_arr = np.array(returns)
            metrics['sharpe_24h'] = ret_arr.mean() / (ret_arr.std() + 1e-8) * np.sqrt(1440)
            neg_ret = ret_arr[ret_arr < 0]
            metrics['sortino_24h'] = ret_arr.mean() / (neg_ret.std() + 1e-8) * np.sqrt(1440) if len(neg_ret) > 0 else 0
        
        # Trade stats
        recent_trades = [t for t in self.trades_buffer if t.timestamp > datetime.now() - timedelta(hours=24)]
        if recent_trades:
            wins = [t for t in recent_trades if t.pnl > 0]
            losses = [t for t in recent_trades if t.pnl < 0]
            metrics['win_rate'] = len(wins) / len(recent_trades)
            metrics['trades_24h'] = len(recent_trades)
            
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return metrics
```

### 2. Correlation Visualizer

```python
"""
correlation_viz.py - Real-time correlation heatmaps and network graphs
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional
import json

class CorrelationVisualizer:
    """
    Visualize cross-asset correlations and lead-lag relationships.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.n_assets = len(symbols)
    
    def create_correlation_heatmap(
        self,
        correlations: np.ndarray,  # (4, n, n) - 3 timeframes + lead-lag
        title: str = "Cross-Asset Correlations"
    ) -> go.Figure:
        """Create interactive correlation heatmap"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['15-min Correlation', '1-hour Correlation', 
                          '4-hour Correlation', 'Lead-Lag Matrix']
        )
        
        timeframe_names = ['15m', '1h', '4h', 'Lead-Lag']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for idx, (row, col) in enumerate(positions):
            heatmap = go.Heatmap(
                z=correlations[idx],
                x=self.symbols[:20],  # Limit for readability
                y=self.symbols[:20],
                colorscale='RdBu_r' if idx < 3 else 'Viridis',
                zmin=-1 if idx < 3 else -10,
                zmax=1 if idx < 3 else 10,
                name=timeframe_names[idx],
                showscale=True
            )
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title=title,
            height=800,
            width=1000
        )
        
        return fig
    
    def create_lead_lag_network(
        self,
        lead_lag_matrix: np.ndarray,
        threshold: float = 3.0  # Only show strong lead-lag
    ) -> go.Figure:
        """Create network graph of lead-lag relationships"""
        
        G = nx.DiGraph()
        
        # Add nodes
        for i, symbol in enumerate(self.symbols):
            G.add_node(i, label=symbol)
        
        # Add edges for significant lead-lag
        edges = []
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i != j and abs(lead_lag_matrix[i, j]) > threshold:
                    weight = lead_lag_matrix[i, j]
                    if weight > 0:  # i leads j
                        edges.append((i, j, weight))
        
        for src, dst, weight in edges:
            G.add_edge(src, dst, weight=weight)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=abs(weight) / 2, color='red' if weight > 0 else 'blue'),
                hoverinfo='none'
            ))
        
        # Node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [self.symbols[node] for node in G.nodes()]
        
        # Size by connectivity
        node_sizes = [10 + G.degree(node) * 3 for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_sizes,
                color=[G.degree(node) for node in G.nodes()],
                colorscale='YlOrRd',
                showscale=True,
                colorbar=dict(title='Connections')
            ),
            hoverinfo='text'
        )
        
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title='Lead-Lag Network (Red: Leader → Follower)',
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            width=900
        )
        
        return fig
    
    def create_regime_chart(
        self,
        correlations_history: List[Tuple[datetime, np.ndarray]],
    ) -> go.Figure:
        """Show correlation regime changes over time"""
        
        timestamps = [t[0] for t in correlations_history]
        
        # Calculate average correlation at each timestamp
        avg_corr = []
        for _, corr in correlations_history:
            # Average of absolute correlations (excluding diagonal)
            mask = ~np.eye(corr.shape[1], dtype=bool)
            avg_corr.append(np.abs(corr[1][mask]).mean())  # 1h timeframe
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=avg_corr,
            mode='lines',
            name='Avg Correlation',
            line=dict(color='blue', width=2)
        ))
        
        # Add regime bands
        high_corr = np.percentile(avg_corr, 75)
        low_corr = np.percentile(avg_corr, 25)
        
        fig.add_hrect(y0=high_corr, y1=1.0, fillcolor='red', opacity=0.1,
                     annotation_text='High Correlation Regime')
        fig.add_hrect(y0=0, y1=low_corr, fillcolor='green', opacity=0.1,
                     annotation_text='Low Correlation Regime')
        
        fig.update_layout(
            title='Market Correlation Regime',
            xaxis_title='Time',
            yaxis_title='Average Cross-Asset Correlation',
            height=400
        )
        
        return fig
```

### 3. Performance Attribution

```python
"""
attribution.py - Performance attribution analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import sqlite3

class PerformanceAttributor:
    """
    Attribute returns to different factors and strategies.
    """
    
    def __init__(self, metrics_db: str = "metrics.db"):
        self.db_path = metrics_db
    
    def attribute_returns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Decompose returns by attribution factors.
        
        Returns breakdown:
        - By asset
        - By trade type (long/short)
        - By correlation signal strength
        - By hold time
        - By time of day
        """
        conn = sqlite3.connect(self.db_path)
        
        trades_df = pd.read_sql_query(f"""
            SELECT * FROM trades 
            WHERE timestamp >= '{start_date.isoformat()}'
            AND timestamp <= '{end_date.isoformat()}'
        """, conn)
        
        conn.close()
        
        if trades_df.empty:
            return {}
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        
        attribution = {}
        
        # By asset
        by_asset = trades_df.groupby('symbol').agg({
            'pnl': ['sum', 'count', 'mean'],
            'pnl_pct': 'mean'
        }).round(4)
        by_asset.columns = ['total_pnl', 'n_trades', 'avg_pnl', 'avg_pnl_pct']
        attribution['by_asset'] = by_asset.to_dict('index')
        
        # By side
        by_side = trades_df.groupby('side').agg({
            'pnl': ['sum', 'count', 'mean']
        }).round(4)
        by_side.columns = ['total_pnl', 'n_trades', 'avg_pnl']
        attribution['by_side'] = by_side.to_dict('index')
        
        # By correlation signal strength
        trades_df['corr_bucket'] = pd.cut(
            trades_df['correlation_signal'].fillna(0),
            bins=[-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf],
            labels=['Strong Neg', 'Weak Neg', 'Neutral', 'Weak Pos', 'Strong Pos']
        )
        by_corr = trades_df.groupby('corr_bucket').agg({
            'pnl': ['sum', 'count', 'mean']
        }).round(4)
        by_corr.columns = ['total_pnl', 'n_trades', 'avg_pnl']
        attribution['by_correlation'] = by_corr.to_dict('index')
        
        # By hold time
        trades_df['hold_bucket'] = pd.cut(
            trades_df['hold_time_seconds'] / 60,  # Convert to minutes
            bins=[0, 5, 15, 30, 60, np.inf],
            labels=['<5m', '5-15m', '15-30m', '30-60m', '>60m']
        )
        by_hold = trades_df.groupby('hold_bucket').agg({
            'pnl': ['sum', 'count', 'mean']
        }).round(4)
        by_hold.columns = ['total_pnl', 'n_trades', 'avg_pnl']
        attribution['by_hold_time'] = by_hold.to_dict('index')
        
        # By hour of day
        by_hour = trades_df.groupby('hour').agg({
            'pnl': ['sum', 'count', 'mean']
        }).round(4)
        by_hour.columns = ['total_pnl', 'n_trades', 'avg_pnl']
        attribution['by_hour'] = by_hour.to_dict('index')
        
        # Lead-lag effectiveness
        if trades_df['lead_asset'].notna().any():
            lead_trades = trades_df[trades_df['lead_asset'].notna()]
            attribution['lead_lag_stats'] = {
                'n_lead_lag_trades': len(lead_trades),
                'total_pnl': float(lead_trades['pnl'].sum()),
                'avg_pnl': float(lead_trades['pnl'].mean()),
                'win_rate': float((lead_trades['pnl'] > 0).mean()),
                'top_leads': lead_trades.groupby('lead_asset')['pnl'].sum().nlargest(5).to_dict()
            }
        
        # Summary stats
        attribution['summary'] = {
            'total_pnl': float(trades_df['pnl'].sum()),
            'n_trades': len(trades_df),
            'win_rate': float((trades_df['pnl'] > 0).mean()),
            'avg_pnl': float(trades_df['pnl'].mean()),
            'best_trade': float(trades_df['pnl'].max()),
            'worst_trade': float(trades_df['pnl'].min()),
            'sharpe': float(trades_df['pnl'].mean() / trades_df['pnl'].std()) if trades_df['pnl'].std() > 0 else 0
        }
        
        return attribution
    
    def get_risk_metrics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Calculate risk metrics"""
        conn = sqlite3.connect(self.db_path)
        
        snapshots_df = pd.read_sql_query(f"""
            SELECT * FROM portfolio_snapshots 
            WHERE timestamp >= '{start_date.isoformat()}'
            AND timestamp <= '{end_date.isoformat()}'
            ORDER BY timestamp
        """, conn)
        
        conn.close()
        
        if snapshots_df.empty:
            return {}
        
        equity = snapshots_df['equity'].values
        returns = np.diff(equity) / equity[:-1]
        
        risk = {}
        
        # VaR and CVaR
        risk['var_95'] = float(np.percentile(returns, 5))
        risk['var_99'] = float(np.percentile(returns, 1))
        risk['cvar_95'] = float(returns[returns <= np.percentile(returns, 5)].mean())
        
        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        risk['max_drawdown'] = float(drawdown.max())
        risk['avg_drawdown'] = float(drawdown.mean())
        
        # Calmar ratio
        annual_return = (equity[-1] / equity[0]) ** (365 * 24 * 60 / len(equity)) - 1
        risk['calmar_ratio'] = float(annual_return / risk['max_drawdown']) if risk['max_drawdown'] > 0 else 0
        
        # Volatility
        risk['daily_vol'] = float(returns.std() * np.sqrt(1440))
        risk['annual_vol'] = float(returns.std() * np.sqrt(1440 * 365))
        
        return risk
```

### 4. Alert Manager

```python
"""
alerts.py - Alerting and notifications
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = None
    data: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AlertManager:
    """
    Manages alerts and notifications via Discord/Email.
    """
    
    def __init__(
        self,
        discord_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None
    ):
        self.discord_webhook = discord_webhook
        self.email_config = email_config
        
        self.alert_history: List[Alert] = []
        self.rules: List[Callable] = []
        
        # Default thresholds
        self.thresholds = {
            'max_drawdown': 0.10,      # 10%
            'min_sharpe': 0.5,
            'max_position_loss': 0.05, # 5%
            'min_trades_per_hour': 1,
            'max_correlation': 0.95,   # Regime warning
        }
    
    def add_rule(self, rule: Callable):
        """Add custom alert rule"""
        self.rules.append(rule)
    
    async def check_and_alert(self, metrics: Dict):
        """Check metrics against rules and send alerts"""
        alerts = []
        
        # Drawdown alert
        if metrics.get('drawdown', 0) > self.thresholds['max_drawdown']:
            alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                title="⚠️ High Drawdown Alert",
                message=f"Drawdown at {metrics['drawdown']*100:.1f}% (threshold: {self.thresholds['max_drawdown']*100}%)",
                data={'drawdown': metrics['drawdown']}
            ))
        
        # Sharpe alert
        if metrics.get('sharpe_24h', 999) < self.thresholds['min_sharpe']:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                title="📉 Low Sharpe Ratio",
                message=f"24h Sharpe at {metrics['sharpe_24h']:.2f} (threshold: {self.thresholds['min_sharpe']})",
                data={'sharpe': metrics['sharpe_24h']}
            ))
        
        # Custom rules
        for rule in self.rules:
            alert = rule(metrics)
            if alert:
                alerts.append(alert)
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
            self.alert_history.append(alert)
    
    async def _send_alert(self, alert: Alert):
        """Send alert to configured channels"""
        if self.discord_webhook:
            await self._send_discord(alert)
    
    async def _send_discord(self, alert: Alert):
        """Send alert to Discord"""
        colors = {
            AlertLevel.INFO: 0x3498db,
            AlertLevel.WARNING: 0xf1c40f,
            AlertLevel.CRITICAL: 0xe74c3c
        }
        
        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": colors[alert.level],
            "timestamp": alert.timestamp.isoformat(),
            "fields": []
        }
        
        if alert.data:
            for key, value in alert.data.items():
                embed["fields"].append({
                    "name": key,
                    "value": str(value),
                    "inline": True
                })
        
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as resp:
                    if resp.status != 204:
                        logger.error(f"Discord alert failed: {resp.status}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    async def send_daily_report(self, attribution: Dict, risk: Dict):
        """Send daily performance report"""
        summary = attribution.get('summary', {})
        
        report = f"""
📊 **Daily Trading Report** - {datetime.now().strftime('%Y-%m-%d')}

**Performance:**
• Total PnL: ${summary.get('total_pnl', 0):.2f}
• Trades: {summary.get('n_trades', 0)}
• Win Rate: {summary.get('win_rate', 0)*100:.1f}%
• Sharpe: {summary.get('sharpe', 0):.2f}

**Risk:**
• Max Drawdown: {risk.get('max_drawdown', 0)*100:.1f}%
• VaR 95%: {risk.get('var_95', 0)*100:.2f}%
• Calmar: {risk.get('calmar_ratio', 0):.2f}

**Top Performers:**
{self._format_top_assets(attribution.get('by_asset', {}))}
        """
        
        alert = Alert(
            level=AlertLevel.INFO,
            title="📊 Daily Report",
            message=report
        )
        
        await self._send_alert(alert)
    
    def _format_top_assets(self, by_asset: Dict) -> str:
        if not by_asset:
            return "No trades"
        
        sorted_assets = sorted(by_asset.items(), key=lambda x: x[1].get('total_pnl', 0), reverse=True)[:5]
        lines = []
        for symbol, stats in sorted_assets:
            pnl = stats.get('total_pnl', 0)
            emoji = "🟢" if pnl > 0 else "🔴"
            lines.append(f"{emoji} {symbol}: ${pnl:.2f}")
        return "\n".join(lines)
```

### 5. Dashboard Server

```python
"""
dashboard.py - Real-time monitoring dashboard using Dash
"""

from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from threading import Thread

class TradingDashboard:
    """
    Real-time monitoring dashboard.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        correlation_viz: CorrelationVisualizer,
        attributor: PerformanceAttributor,
        port: int = 8050
    ):
        self.metrics = metrics_collector
        self.corr_viz = correlation_viz
        self.attributor = attributor
        self.port = port
        
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        self.app.layout = html.Div([
            html.H1("🤖 MCTS Trading Agent Monitor", style={'textAlign': 'center'}),
            
            # Refresh interval
            dcc.Interval(id='interval', interval=60000, n_intervals=0),  # 1 min
            
            # Top metrics row
            html.Div([
                html.Div([
                    html.H3("Portfolio Value"),
                    html.H2(id='equity-value', style={'color': '#2ecc71'})
                ], className='metric-card'),
                html.Div([
                    html.H3("24h Return"),
                    html.H2(id='return-24h')
                ], className='metric-card'),
                html.Div([
                    html.H3("Sharpe Ratio"),
                    html.H2(id='sharpe-value')
                ], className='metric-card'),
                html.Div([
                    html.H3("Drawdown"),
                    html.H2(id='drawdown-value', style={'color': '#e74c3c'})
                ], className='metric-card'),
            ], className='metrics-row'),
            
            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(id='equity-chart')
                ], style={'width': '60%'}),
                html.Div([
                    dcc.Graph(id='trades-chart')
                ], style={'width': '40%'})
            ], style={'display': 'flex'}),
            
            # Correlation section
            html.Div([
                html.H2("Cross-Asset Correlations"),
                dcc.Graph(id='correlation-heatmap')
            ]),
            
            # Lead-lag network
            html.Div([
                html.H2("Lead-Lag Network"),
                dcc.Graph(id='lead-lag-network')
            ]),
            
            # Attribution table
            html.Div([
                html.H2("Performance Attribution"),
                html.Div(id='attribution-table')
            ])
        ])
    
    def _setup_callbacks(self):
        @self.app.callback(
            [Output('equity-value', 'children'),
             Output('return-24h', 'children'),
             Output('sharpe-value', 'children'),
             Output('drawdown-value', 'children')],
            [Input('interval', 'n_intervals')]
        )
        def update_metrics(n):
            m = self.metrics.get_realtime_metrics()
            return (
                f"${m.get('current_equity', 0):,.2f}",
                f"{m.get('return_24h', 0)*100:+.2f}%",
                f"{m.get('sharpe_24h', 0):.2f}",
                f"{m.get('drawdown', 0)*100:.1f}%"
            )
        
        @self.app.callback(
            Output('equity-chart', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_equity_chart(n):
            equity_data = list(self.metrics.equity_buffer)
            if not equity_data:
                return go.Figure()
            
            df = pd.DataFrame(equity_data, columns=['timestamp', 'equity'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['equity'],
                mode='lines', name='Equity',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig.update_layout(
                title='Portfolio Equity',
                xaxis_title='Time',
                yaxis_title='Value ($)',
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('correlation-heatmap', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_correlation(n):
            # Get latest correlations from encoder
            # This would need to be connected to live data
            return go.Figure()  # Placeholder
    
    def run(self, debug: bool = False):
        """Run dashboard server"""
        self.app.run_server(host='0.0.0.0', port=self.port, debug=debug)
    
    def run_async(self):
        """Run in background thread"""
        thread = Thread(target=self.run, daemon=True)
        thread.start()
        return thread
```

### 6. Main Monitor

```python
"""
monitor.py - Main monitoring orchestration
"""

import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TradingMonitor:
    """
    Orchestrates all monitoring components.
    """
    
    def __init__(
        self,
        agent: 'TRMMCTSAgent',
        encoder: 'MultiAssetStateEncoder',
        discord_webhook: Optional[str] = None
    ):
        self.agent = agent
        self.encoder = encoder
        
        # Components
        self.metrics = MetricsCollector()
        self.corr_viz = CorrelationVisualizer(list(encoder.symbol_to_idx.keys()))
        self.attributor = PerformanceAttributor()
        self.alerts = AlertManager(discord_webhook=discord_webhook)
        self.dashboard = TradingDashboard(
            self.metrics, self.corr_viz, self.attributor
        )
        
        self.running = False
    
    async def start(self):
        """Start monitoring"""
        self.running = True
        
        # Start dashboard in background
        self.dashboard.run_async()
        logger.info(f"Dashboard running at http://localhost:{self.dashboard.port}")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._daily_report_loop())
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                realtime = self.metrics.get_realtime_metrics()
                
                # Check alerts
                await self.alerts.check_and_alert(realtime)
                
                # Record correlation snapshot every 15 min
                if datetime.now().minute % 15 == 0:
                    corr = self.encoder.encode()['correlation']
                    self.metrics.record_correlations(datetime.now(), corr)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _daily_report_loop(self):
        """Send daily reports"""
        while self.running:
            now = datetime.now()
            next_report = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            
            await asyncio.sleep((next_report - now).total_seconds())
            
            # Generate and send report
            end = datetime.now()
            start = end - timedelta(days=1)
            
            attribution = self.attributor.attribute_returns(start, end)
            risk = self.attributor.get_risk_metrics(start, end)
            
            await self.alerts.send_daily_report(attribution, risk)
    
    def stop(self):
        self.running = False
```

## Usage

```python
# Initialize
monitor = TradingMonitor(
    agent=agent,
    encoder=encoder,
    discord_webhook="https://discord.com/api/webhooks/..."
)

# Start monitoring
await monitor.start()

# In trading loop
while trading:
    # ... execute trades ...
    
    # Record trade
    monitor.metrics.record_trade(TradeMetrics(
        timestamp=datetime.now(),
        asset_idx=trade['asset_idx'],
        symbol=trade['symbol'],
        side=trade['side'],
        entry_price=trade['entry'],
        exit_price=trade['exit'],
        size=trade['size'],
        pnl=trade['pnl'],
        pnl_pct=trade['pnl_pct'],
        hold_time=trade['hold_time'],
        correlation_signal=trade.get('corr_signal'),
        lead_asset=trade.get('lead_asset')
    ))
    
    # Record portfolio snapshot
    monitor.metrics.record_portfolio_snapshot(PortfolioMetrics(...))
```

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Live Equity | Real-time portfolio value chart |
| Correlation Heatmaps | 15m/1h/4h cross-asset correlations |
| Lead-Lag Network | Interactive graph of predictive relationships |
| Attribution Tables | PnL breakdown by asset, time, signal strength |
| Risk Metrics | VaR, CVaR, drawdown, Calmar |
| Alerts | Discord notifications for thresholds |
| Daily Reports | Automated performance summaries |
```

---

All 6 prompts complete. Ready to start implementation?
