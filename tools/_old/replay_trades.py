#!/usr/bin/env python3
"""
Replay trade_log.csv with corrected portfolio calculation to see actual performance.
Now includes proper FX conversion for multi-currency positions.

Usage:
    python replay_trades.py [log_file] [initial_capital] [session_idx] [--legacy]

    --legacy: Use legacy mode (no FX conversion) for logs from before 2025-12-11
              This correctly interprets logs where positions were sized as if all
              quotes were USD.
"""
import csv
import sys
import sqlite3
import asyncio
import time
from datetime import datetime, timezone
import logging
import os
import argparse

sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.agent.portfolio import (
    PortfolioManager, Position,
    extract_quote_currency, get_quote_to_usd_rate
)
from mcts_trader.data.kraken_client import KrakenClient

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'replay_trades.log')),
        logging.StreamHandler()
    ]
)

# Mapping for currency to probable Kraken pair name in DB/API
CURRENCY_TO_PAIR = {
    'EUR': 'EUR/USD',
    'GBP': 'GBP/USD',
    'AUD': 'AUD/USD',
    'CAD': 'USD/CAD', # Note: Inverse
    'CHF': 'USD/CHF', # Note: Inverse
    'BTC': 'BTC/USD',
    'XBT': 'BTC/USD',
    'ETH': 'ETH/USD',
    'USDT': 'USDT/USD',
    'USDC': 'USDC/USD',
}

async def get_historical_fx_rate(currency: str, timestamp_str: str, db_cursor: sqlite3.Cursor) -> float:
    """
    Get FX rate (Currency -> USD) for a specific timestamp.
    Priority: 1. Database, 2. Kraken API (fallback)
    """
    if currency in ['USD', 'USDT', 'USDC']: # Approximate stablecoins as 1.0 for speed, or fetch if strict
        return 1.0

    # Parse timestamp
    # Log format: YYYY-MM-DD HH:MM:SS
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        ts = int(dt.replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        # Fallback if format differs
        logging.warning(f"Could not parse timestamp {timestamp_str}")
        return 1.0

    pair = CURRENCY_TO_PAIR.get(currency)
    if not pair:
        # Try generic construction
        pair = f"{currency}/USD"

    # 1. Try Database
    # We need asset_id for the pair. 
    # This assumes we can lookup ID by symbol.
    # Let's do a quick lookup query.
    db_cursor.execute("SELECT id FROM assets WHERE symbol LIKE ? OR symbol LIKE ?", (f"%{pair}", f"%{pair.replace('BTC','XBT')}"))
    row = db_cursor.fetchone()
    
    if row:
        asset_id = row[0]
        # Find closest candle
        # Search for candle within 5 minutes
        db_cursor.execute(
            "SELECT close FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ? ORDER BY ABS(timestamp - ?) LIMIT 1",
            (asset_id, ts - 300, ts + 300, ts)
        )
        price_row = db_cursor.fetchone()
        if price_row:
            price = price_row[0]
            if "USD/" in pair: # Inverse pair like USD/CAD
                return 1.0 / price
            return price

    # 2. Fallback to Kraken API
    logging.info(f"Fetching API rate for {pair} @ {timestamp_str}...")
    try:
        async with KrakenClient() as client:
            # map pair to API format
            # simple heuristic: remove slash
            query_pair = pair.replace('/', '') # e.g. BTCUSD
            
            # get_ohlc might need mapping logic, but let's try standard pairs
            # Kraken OHLC since parameter
            ohlc = await client.get_ohlc(query_pair, interval=1, since=ts-60)
            if ohlc:
                # Find closest candle
                # ohlc is [time, open, high, low, close, ...]
                closest = min(ohlc, key=lambda x: abs(float(x[0]) - ts))
                price = float(closest[4]) # Close price
                
                if "USD/" in pair:
                    return 1.0 / price
                return price
    except Exception as e:
        logging.warning(f"API Fetch Failed for {pair}: {e}")

    logging.info(f"CRITICAL: Could not find FX rate for {currency} at {timestamp_str}. DB and API failed.")
    raise ValueError(f"Missing FX rate for {currency}")


def detect_sessions(log_file: str) -> list:
    """Detect multiple trading sessions in log file by finding portfolio resets"""
    sessions = []
    current_session_start = 0
    last_portfolio_value = None

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            port_val = float(row.get('portfolio_value', 0))

            # Detect session reset: portfolio jumps back to round number (100, 10000, etc)
            if last_portfolio_value is not None:
                # Check if this looks like a fresh start
                is_round = (abs(port_val - 100) < 1 or abs(port_val - 10000) < 10)
                big_change = abs(port_val - last_portfolio_value) > last_portfolio_value * 0.5

                if is_round and big_change and i > current_session_start + 5:
                    sessions.append((current_session_start, i - 1))
                    current_session_start = i

            last_portfolio_value = port_val

    # Add final session
    sessions.append((current_session_start, None))
    return sessions


async def replay_trades_async(log_file: str = os.path.join('logs', "trade_log.csv"), initial_capital: float = None, # Default to logs folder
                  session_idx: int = -1, legacy_mode: bool = False):
    """Async wrapper for replay logic"""
    
    # Connect to DB
    conn = sqlite3.connect("trading_data.db")
    cursor = conn.cursor()

    try:
        # Detect sessions
        sessions = detect_sessions(log_file)
        logging.info(f"Found {len(sessions)} trading session(s) in log")

        if len(sessions) > 1:
            for i, (start, end) in enumerate(sessions):
                end_str = str(end) if end else "EOF"
                marker = " <-- SELECTED" if (i == session_idx or (session_idx == -1 and i == len(sessions) - 1)) else ""
                logging.info(f"  Session {i}: rows {start}-{end_str}{marker}")

        # Select session
        if session_idx == -1:
            session_idx = len(sessions) - 1
        start_row, end_row = sessions[session_idx]

        # Read session data
        rows = []
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= start_row and (end_row is None or i <= end_row):
                    rows.append(row)

        if not rows:
            logging.info("No data in selected session")
            return

        # Detect initial capital from first row of session
        if initial_capital is None:
            first_row = rows[0]
            port_val = float(first_row.get('portfolio_value', 10000))
            cash_val = float(first_row.get('cash', 10000))
            initial_capital = max(port_val, cash_val)
            # Round to nearest likely starting value
            if 90 <= initial_capital <= 110:
                initial_capital = 100.0
            elif 9000 <= initial_capital <= 11000:
                initial_capital = 10000.0

        mode_str = "LEGACY (No FX)" if legacy_mode else "FX-Adjusted"
        logging.info(f"Replaying session {session_idx} with initial capital: ${initial_capital:.2f}")
        logging.info(f"Mode: {mode_str}")

        # Track positions manually for replay
        positions = {}  # symbol -> {side, size, entry_price, quote_currency}
        cash = initial_capital  # Cash is always in USD
        fx_rates = {}  # symbol -> price for FX lookups

        logging.info("=" * 80)
        logging.info(f"TRADE REPLAY - {mode_str}")
        logging.info("=" * 80)

        trade_count = 0

        for row in rows:
            timestamp = row['timestamp']
            symbol = row['symbol']
            action = row['action']
            price = float(row['price'])
            size = float(row['size'])
            fee = float(row['fee'])
            old_portfolio_value = float(row.get('portfolio_value', 0)) # handle missing key
            old_cash = float(row.get('cash', 0)) # handle missing key

            # Update cache with observed prices (still useful as a first hint)
            if price > 0:
                fx_rates[symbol] = price

            # Process actual trades (not signals)
            if action.startswith('open_') or action.startswith('close_'):
                trade_count += 1

                # Get quote currency
                quote_currency = extract_quote_currency(symbol)
                
                # Fetch FX Rate
                if legacy_mode:
                    fx_rate = 1.0
                else:
                    # Check cache first for direct hit? No, safer to fetch authoritative rate.
                    # But if we have it from previous fetched steps we can cache it.
                    # Let's simple call our helper.
                    fx_rate = await get_historical_fx_rate(quote_currency, timestamp, cursor)
                    # Update cache for portfolio internal lookups if needed
                    # Note: PortfolioManager uses keys like 'USD/EUR' etc.
                    # We can populate fx_rates dict with what we found
                    fx_key = f"{quote_currency}/USD" 
                    fx_rates[fx_key] = fx_rate


                # Value in USD (or legacy "USD-equivalent")
                value_usd = size * price * fx_rate
                fee_usd = fee * fx_rate if fee > 0 else value_usd * 0.0026  # 26 bps

                if action == 'open_long':
                    cash -= (value_usd + fee_usd)
                    positions[symbol] = {
                        'side': 'long',
                        'size': size,
                        'entry_price': price,
                        'quote_currency': quote_currency
                    }

                elif action == 'close_long':
                    if symbol in positions:
                        cash += (value_usd - fee_usd)
                        del positions[symbol]

                elif action == 'open_short':
                    cash += (value_usd - fee_usd)
                    positions[symbol] = {
                        'side': 'short',
                        'size': size,
                        'entry_price': price,
                        'quote_currency': quote_currency
                    }

                elif action == 'close_short':
                    if symbol in positions:
                        cash -= (value_usd + fee_usd)
                        del positions[symbol]

                # Calculate corrected portfolio value
                long_value_usd = 0.0
                short_liability_usd = 0.0

                for sym, pos in positions.items():
                    current_price = fx_rates.get(sym, pos['entry_price'])
                    if legacy_mode:
                        pos_fx_rate = 1.0
                    else:
                        # We need rate at CURRENT timestamp for valuation
                        pos_fx_rate = await get_historical_fx_rate(pos['quote_currency'], timestamp, cursor)

                    if pos['side'] == 'long':
                        long_value_usd += pos['size'] * current_price * pos_fx_rate
                    else:
                        # Liability = current * size * fx
                        liability = current_price * pos['size'] * pos_fx_rate
                        short_liability_usd += liability

                corrected_value = cash + long_value_usd - short_liability_usd

                logging.info(f"[{timestamp}] {action} {symbol} (Quote: {quote_currency}, FX: {fx_rate:.4f})")
                logging.info(f"  Price: {price:.6f} | Size: {size:.6f} | Value: ${value_usd:.2f}")
                logging.info(f"  OLD Value: ${old_portfolio_value:.2f} | NEW Value: ${corrected_value:.2f}")

        # Final summary
        logging.info("=" * 80)
        logging.info(f"FINAL SUMMARY ({mode_str})")
        logging.info("=" * 80)

        # Calculate final value with latest prices
        long_value_usd = 0.0
        short_liability_usd = 0.0
        
        # Use last timestamp for final valuation
        last_ts = rows[-1]['timestamp'] if rows else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logging.info(f"Open Positions ({len(positions)}):")
        for sym, pos in positions.items():
            current_price = fx_rates.get(sym, pos['entry_price'])
            if legacy_mode:
                fx_rate = 1.0
            else:
                fx_rate = await get_historical_fx_rate(pos['quote_currency'], last_ts, cursor)

            if pos['side'] == 'long':
                market_val_usd = pos['size'] * current_price * fx_rate
                pnl_usd = (current_price - pos['entry_price']) * pos['size'] * fx_rate
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                long_value_usd += market_val_usd
                logging.info(f"  LONG  {sym:<12} | Quote: {pos['quote_currency']:<4} | Size: {pos['size']:.4f} | Entry: {pos['entry_price']:.6f} | Now: {current_price:.6f} | USD Val: ${market_val_usd:,.2f} | PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")
            else:
                pnl_quote = (pos['entry_price'] - current_price) * pos['size']
                pnl_usd = pnl_quote * fx_rate
                pnl_pct = (pos['entry_price'] - current_price) / pos['entry_price'] * 100
                
                liability_usd = pos['size'] * current_price * fx_rate
                short_liability_usd += liability_usd
                
                logging.info(f"  SHORT {sym:<12} | Quote: {pos['quote_currency']:<4} | Size: {pos['size']:.4f} | Entry: {pos['entry_price']:.6f} | Now: {current_price:.6f} | Liability: ${liability_usd:,.2f} | PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")

        final_value = cash + long_value_usd - short_liability_usd
        total_return = (final_value - initial_capital) / initial_capital * 100

        logging.info(f"Cash (USD): ${cash:,.2f}")
        logging.info(f"Long Market Value (USD): ${long_value_usd:,.2f}")
        logging.info(f"Short Unrealized PnL (USD): ${short_pnl_usd:+,.2f}")
        logging.info(f"INITIAL CAPITAL: ${initial_capital:,.2f}")
        logging.info(f"FINAL PORTFOLIO VALUE: ${final_value:,.2f}")
        logging.info(f"TOTAL RETURN: {total_return:+.2f}%")
        logging.info(f"Total Trades Executed: {trade_count}")

        if legacy_mode:
            logging.info("⚠️  LEGACY MODE: FX conversion disabled. Values are in 'quote currency units'")
            logging.info("   For logs after 2025-12-11, remove --legacy flag for accurate USD accounting.")

    finally:
        conn.close()

def replay_trades(log_file, init_cap, session, legacy_mode):
    asyncio.run(replay_trades_async(log_file, init_cap, session, legacy_mode))

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Replay trade log CSV with FX-adjusted portfolio calculation.')
    parser.add_argument('log_file', nargs='?', default=os.path.join('logs', "trade_log.csv"), 
                        help='Path to trade log CSV. Defaults to logs/trade_log.csv.')
    parser.add_argument('initial_capital', nargs='?', type=float, default=None, 
                        help='Override initial capital (auto-detect if None).')
    parser.add_argument('session_idx', nargs='?', type=int, default=-1, 
                        help='Which session to replay (-1 = last/most recent).')
    parser.add_argument('--legacy', action='store_true', 
                        help='Use legacy mode (no FX conversion) for logs from before 2025-12-11.')
    args = parser.parse_args()

    replay_trades(args.log_file, args.initial_capital, args.session_idx, legacy_mode=args.legacy)