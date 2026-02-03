# paper_trading.py
"""
Paper Trading System - Must run for 1 month before live trading.

Features:
- Simulates trades with real market data (no real money)
- Tracks all trades and performance
- Validates performance before allowing live trading
- Same logic as live trading but no actual orders
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from config import (
    SYMBOL, INTERVALS, BEST_MODEL_PATH, RISK_PERCENT, DEFAULT_LEVERAGE,
    ENTRY_THRESHOLD, SWITCH_THRESHOLD, TP_MIN, SL_MIN
)

# ============================================================
# PAPER TRADING CONFIGURATION
# ============================================================

PAPER_TRADING_REQUIRED_DAYS = 30       # Must paper trade for 30 days
PAPER_TRADING_MIN_TRADES = 50          # Minimum trades required
PAPER_TRADING_MIN_WIN_RATE = 0.60      # Must achieve 60% win rate
PAPER_TRADING_MIN_PROFIT = 0.0         # Must be profitable (>0%)
PAPER_TRADING_MAX_DRAWDOWN = 0.30      # Max 30% drawdown allowed

# File paths
PAPER_TRADE_LOG = "paper_trades.json"
PAPER_TRADING_STATUS = "paper_trading_status.json"

# ============================================================
# PAPER TRADING TRACKER
# ============================================================

class PaperTradingTracker:
    """
    Tracks paper trading performance and validates readiness for live trading.
    """

    def __init__(self, starting_capital=1000.0):
        self.starting_capital = starting_capital
        self.capital = starting_capital
        self.peak_capital = starting_capital
        self.trades = []
        self.open_position = None
        self.start_date = None
        self.status = self._load_status()
        self._load_trades()

    def _load_status(self):
        """Load paper trading status."""
        if os.path.exists(PAPER_TRADING_STATUS):
            try:
                with open(PAPER_TRADING_STATUS, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'started': None,
            'live_approved': False,
            'approval_date': None,
            'total_paper_days': 0
        }

    def _save_status(self):
        """Save paper trading status."""
        with open(PAPER_TRADING_STATUS, 'w') as f:
            json.dump(self.status, f, indent=2, default=str)

    def _load_trades(self):
        """Load existing paper trades."""
        if os.path.exists(PAPER_TRADE_LOG):
            try:
                with open(PAPER_TRADE_LOG, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.capital = data.get('capital', self.starting_capital)
                    self.peak_capital = data.get('peak_capital', self.starting_capital)
                    self.start_date = data.get('start_date')
                    if self.start_date:
                        self.start_date = datetime.fromisoformat(self.start_date)
            except:
                pass

    def _save_trades(self):
        """Save paper trades."""
        data = {
            'trades': self.trades,
            'capital': self.capital,
            'peak_capital': self.peak_capital,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'last_updated': datetime.now().isoformat()
        }
        with open(PAPER_TRADE_LOG, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def start_paper_trading(self):
        """Initialize paper trading period."""
        if not self.start_date:
            self.start_date = datetime.now()
            self.status['started'] = self.start_date.isoformat()
            self._save_status()
            print(f"[PAPER] Paper trading started: {self.start_date}")
            print(f"[PAPER] Must complete {PAPER_TRADING_REQUIRED_DAYS} days before live trading")

    def open_trade(self, symbol, side, entry_price, qty, confidence, leverage):
        """Open a paper trade."""
        if self.open_position:
            print(f"[PAPER] Already have open position: {self.open_position['side']}")
            return None

        trade_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.open_position = {
            'trade_id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'qty': qty,
            'confidence': confidence,
            'leverage': leverage,
            'entry_time': datetime.now().isoformat(),
            'position_value': qty * entry_price,
            'margin_used': (qty * entry_price) / leverage
        }

        print(f"[PAPER] Opened {side} position:")
        print(f"        Entry: ${entry_price:.2f} | Qty: {qty:.4f}")
        print(f"        Value: ${self.open_position['position_value']:.2f} | Leverage: {leverage}x")

        return trade_id

    def close_trade(self, exit_price, exit_reason='MANUAL'):
        """Close paper trade and calculate PnL."""
        if not self.open_position:
            print("[PAPER] No open position to close")
            return None

        pos = self.open_position
        entry = pos['entry_price']
        qty = pos['qty']
        leverage = pos['leverage']

        # Calculate PnL
        if pos['side'] == 'LONG':
            pnl_percent = (exit_price - entry) / entry
        else:  # SHORT
            pnl_percent = (entry - exit_price) / entry

        # PnL with leverage
        pnl_dollar = pos['margin_used'] * pnl_percent * leverage

        # Update capital
        self.capital += pnl_dollar
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        # Record trade
        trade_record = {
            **pos,
            'exit_price': exit_price,
            'exit_time': datetime.now().isoformat(),
            'exit_reason': exit_reason,
            'pnl_percent': pnl_percent * 100,
            'pnl_dollar': pnl_dollar,
            'capital_after': self.capital,
            'result': 'WIN' if pnl_dollar > 0 else 'LOSS'
        }
        self.trades.append(trade_record)
        self._save_trades()

        # Print result
        result_emoji = "WIN" if pnl_dollar > 0 else "LOSS"
        print(f"[PAPER] Closed {pos['side']} - {result_emoji}")
        print(f"        Exit: ${exit_price:.2f} | Reason: {exit_reason}")
        print(f"        PnL: {pnl_percent*100:+.2f}% (${pnl_dollar:+.2f})")
        print(f"        Capital: ${self.capital:.2f}")

        self.open_position = None
        return trade_record

    def get_unrealized_pnl(self, current_price):
        """Calculate unrealized PnL for open position."""
        if not self.open_position:
            return 0.0

        pos = self.open_position
        entry = pos['entry_price']

        if pos['side'] == 'LONG':
            pnl_percent = (current_price - entry) / entry
        else:
            pnl_percent = (entry - current_price) / entry

        pnl_dollar = pos['margin_used'] * pnl_percent * pos['leverage']
        return pnl_dollar

    def get_metrics(self):
        """Calculate performance metrics."""
        if not self.trades:
            return None

        closed_trades = [t for t in self.trades if 'exit_price' in t]
        if not closed_trades:
            return None

        wins = sum(1 for t in closed_trades if t['result'] == 'WIN')
        losses = len(closed_trades) - wins

        total_profit = sum(t['pnl_dollar'] for t in closed_trades if t['pnl_dollar'] > 0)
        total_loss = abs(sum(t['pnl_dollar'] for t in closed_trades if t['pnl_dollar'] < 0))

        # Calculate drawdown
        capitals = [self.starting_capital]
        for t in closed_trades:
            capitals.append(t['capital_after'])

        peak = capitals[0]
        max_drawdown = 0
        for cap in capitals:
            if cap > peak:
                peak = cap
            dd = (peak - cap) / peak
            if dd > max_drawdown:
                max_drawdown = dd

        # Days trading
        if self.start_date:
            days_trading = (datetime.now() - self.start_date).days
        else:
            days_trading = 0

        return {
            'total_trades': len(closed_trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(closed_trades) if closed_trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'total_pnl': self.capital - self.starting_capital,
            'total_return': (self.capital / self.starting_capital - 1) * 100,
            'max_drawdown': max_drawdown,
            'current_capital': self.capital,
            'starting_capital': self.starting_capital,
            'days_trading': days_trading,
            'avg_trade': np.mean([t['pnl_dollar'] for t in closed_trades]) if closed_trades else 0
        }

    def check_live_ready(self):
        """
        Check if paper trading requirements are met for live trading.
        Returns (ready, reasons)
        """
        if self.status.get('live_approved'):
            return True, ["Already approved for live trading"]

        metrics = self.get_metrics()
        if not metrics:
            return False, ["No trades recorded yet"]

        reasons = []
        ready = True

        # Check duration
        if metrics['days_trading'] < PAPER_TRADING_REQUIRED_DAYS:
            ready = False
            reasons.append(f"Need {PAPER_TRADING_REQUIRED_DAYS - metrics['days_trading']} more days (currently {metrics['days_trading']} days)")

        # Check number of trades
        if metrics['total_trades'] < PAPER_TRADING_MIN_TRADES:
            ready = False
            reasons.append(f"Need {PAPER_TRADING_MIN_TRADES - metrics['total_trades']} more trades (currently {metrics['total_trades']} trades)")

        # Check win rate
        if metrics['win_rate'] < PAPER_TRADING_MIN_WIN_RATE:
            ready = False
            reasons.append(f"Win rate {metrics['win_rate']:.1%} < required {PAPER_TRADING_MIN_WIN_RATE:.1%}")

        # Check profitability
        if metrics['total_return'] < PAPER_TRADING_MIN_PROFIT:
            ready = False
            reasons.append(f"Not profitable: {metrics['total_return']:.1f}% return")

        # Check drawdown
        if metrics['max_drawdown'] > PAPER_TRADING_MAX_DRAWDOWN:
            ready = False
            reasons.append(f"Drawdown {metrics['max_drawdown']:.1%} > max allowed {PAPER_TRADING_MAX_DRAWDOWN:.1%}")

        if ready:
            self.status['live_approved'] = True
            self.status['approval_date'] = datetime.now().isoformat()
            self._save_status()
            reasons = ["All requirements met! Live trading approved."]

        return ready, reasons

    def print_status(self):
        """Print current paper trading status."""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("  PAPER TRADING STATUS")
        print("=" * 60)

        if self.start_date:
            print(f"  Started: {self.start_date.strftime('%Y-%m-%d %H:%M')}")
            days = (datetime.now() - self.start_date).days
            print(f"  Days: {days} / {PAPER_TRADING_REQUIRED_DAYS} required")
        else:
            print("  Not started yet")

        if metrics:
            print(f"\n  Capital: ${metrics['current_capital']:.2f} (started ${metrics['starting_capital']:.2f})")
            print(f"  Return: {metrics['total_return']:+.1f}%")
            print(f"\n  Trades: {metrics['total_trades']} / {PAPER_TRADING_MIN_TRADES} required")
            print(f"  Win Rate: {metrics['win_rate']:.1%} (min {PAPER_TRADING_MIN_WIN_RATE:.1%})")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.1%} (max {PAPER_TRADING_MAX_DRAWDOWN:.1%})")

        print("\n" + "-" * 60)
        ready, reasons = self.check_live_ready()
        if ready:
            print("  STATUS: READY FOR LIVE TRADING")
        else:
            print("  STATUS: NOT READY")
            print("  Requirements not met:")
            for r in reasons:
                print(f"    - {r}")
        print("=" * 60)

        if self.open_position:
            print(f"\n  Open Position: {self.open_position['side']} @ ${self.open_position['entry_price']:.2f}")


# ============================================================
# PAPER TRADING MODE RUNNER
# ============================================================

def run_paper_trading():
    """
    Run the bot in paper trading mode.
    Uses real market data but simulates trades.
    """
    from trader import get_realtime_price, get_balance
    from feature_pipeline import fetch_features_multi_timeframe
    from strategy_lstm_live import lstm_based_action, select_leverage
    import time

    print("\n" + "=" * 60)
    print("  PAPER TRADING MODE")
    print("  No real money will be used")
    print("=" * 60)

    # Initialize tracker
    tracker = PaperTradingTracker(starting_capital=1000.0)
    tracker.start_paper_trading()
    tracker.print_status()

    print("\n[PAPER] Starting paper trading loop...")
    print("[PAPER] Press Ctrl+C to stop\n")

    consecutive_losses = 0
    ENTRY_THRESHOLD_CURRENT = ENTRY_THRESHOLD

    while True:
        try:
            # Fetch features
            df = fetch_features_multi_timeframe()
            price = get_realtime_price(SYMBOL)

            if not price:
                print("[PAPER] Could not get price, retrying...")
                time.sleep(30)
                continue

            # Get model prediction
            action, confidence = lstm_based_action(df)
            leverage = select_leverage(confidence)

            # Calculate ATR for TP/SL
            atr_cols = [col for col in df.columns if "atr" in col]
            avg_atr = df[atr_cols].iloc[-1].mean()
            base_range = avg_atr / price

            # Dynamic TP/SL
            TP_RATIO = max(TP_MIN, min(0.015, base_range * 1.5))
            SL_RATIO = max(SL_MIN, min(0.02, base_range * 1.2))

            timestamp = datetime.now().strftime('%H:%M:%S')

            # Check open position
            if tracker.open_position:
                pos = tracker.open_position
                unrealized_pnl = tracker.get_unrealized_pnl(price)
                pnl_percent = unrealized_pnl / pos['margin_used'] * 100 if pos['margin_used'] > 0 else 0

                print(f"[{timestamp}] {pos['side']} | Price: ${price:.2f} | PnL: {pnl_percent:+.2f}% (${unrealized_pnl:+.2f})")

                # Check TP
                if pos['side'] == 'LONG':
                    tp_price = pos['entry_price'] * (1 + TP_RATIO)
                    sl_price = pos['entry_price'] * (1 - SL_RATIO)
                else:
                    tp_price = pos['entry_price'] * (1 - TP_RATIO)
                    sl_price = pos['entry_price'] * (1 + SL_RATIO)

                # TP hit
                if (pos['side'] == 'LONG' and price >= tp_price) or \
                   (pos['side'] == 'SHORT' and price <= tp_price):
                    tracker.close_trade(price, 'TAKE_PROFIT')
                    consecutive_losses = 0
                    ENTRY_THRESHOLD_CURRENT = ENTRY_THRESHOLD

                # SL hit
                elif (pos['side'] == 'LONG' and price <= sl_price) or \
                     (pos['side'] == 'SHORT' and price >= sl_price):
                    tracker.close_trade(price, 'STOP_LOSS')
                    consecutive_losses += 1
                    ENTRY_THRESHOLD_CURRENT = min(0.95, ENTRY_THRESHOLD + consecutive_losses * 0.01)

                # Switch position
                elif action != 'HOLD' and action != pos['side'] and confidence >= SWITCH_THRESHOLD:
                    tracker.close_trade(price, f'SWITCH_TO_{action}')
                    # Open new position
                    qty = (tracker.capital * RISK_PERCENT / 100 * leverage) / price
                    tracker.open_trade(SYMBOL, action, price, qty, confidence, leverage)

            else:
                # No open position - look for entry
                print(f"[{timestamp}] Signal: {action} | Conf: {confidence:.2f} | Price: ${price:.2f}")

                if action != 'HOLD' and confidence >= ENTRY_THRESHOLD_CURRENT:
                    qty = (tracker.capital * RISK_PERCENT / 100 * leverage) / price
                    tracker.open_trade(SYMBOL, action, price, qty, confidence, leverage)

            # Periodic status update (every 10 iterations = ~10 minutes)
            if len(tracker.trades) > 0 and len(tracker.trades) % 10 == 0:
                tracker.print_status()

        except KeyboardInterrupt:
            print("\n[PAPER] Stopping paper trading...")
            break
        except Exception as e:
            print(f"[PAPER] Error: {e}")

        time.sleep(60)  # Check every minute

    # Final status
    tracker.print_status()
    ready, reasons = tracker.check_live_ready()

    if ready:
        print("\n[PAPER] Congratulations! You can now switch to live trading.")
        print("[PAPER] Edit config.py: PAPER_TRADING_MODE = False")
    else:
        print("\n[PAPER] Continue paper trading to meet requirements.")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def is_live_trading_allowed():
    """Check if live trading is allowed (paper trading completed)."""
    if os.path.exists(PAPER_TRADING_STATUS):
        try:
            with open(PAPER_TRADING_STATUS, 'r') as f:
                status = json.load(f)
                return status.get('live_approved', False)
        except:
            pass
    return False


def get_paper_trading_status():
    """Get current paper trading status."""
    tracker = PaperTradingTracker()
    return tracker.get_metrics(), tracker.check_live_ready()


def reset_paper_trading():
    """Reset paper trading (start over)."""
    if os.path.exists(PAPER_TRADE_LOG):
        os.remove(PAPER_TRADE_LOG)
    if os.path.exists(PAPER_TRADING_STATUS):
        os.remove(PAPER_TRADING_STATUS)
    print("[PAPER] Paper trading reset. Starting fresh.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python paper_trading.py <command>")
        print("Commands:")
        print("  run      - Start paper trading")
        print("  status   - Show current status")
        print("  check    - Check if ready for live trading")
        print("  reset    - Reset paper trading (start over)")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'run':
        run_paper_trading()

    elif command == 'status':
        tracker = PaperTradingTracker()
        tracker.print_status()

    elif command == 'check':
        tracker = PaperTradingTracker()
        ready, reasons = tracker.check_live_ready()
        print("\n" + "=" * 50)
        if ready:
            print("LIVE TRADING: APPROVED")
        else:
            print("LIVE TRADING: NOT APPROVED")
            print("\nRequirements not met:")
            for r in reasons:
                print(f"  - {r}")
        print("=" * 50)

    elif command == 'reset':
        confirm = input("Are you sure you want to reset paper trading? (yes/no): ")
        if confirm.lower() == 'yes':
            reset_paper_trading()
        else:
            print("Cancelled.")

    else:
        print(f"Unknown command: {command}")
