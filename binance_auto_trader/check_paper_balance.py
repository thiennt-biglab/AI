#!/usr/bin/env python3
"""Check paper trading balance and status."""

import os
import json
from datetime import datetime

# Data directory for status files
DATA_DIR = os.environ.get('DATA_DIR', 'data')
PAPER_TRADING_STATUS = os.path.join(DATA_DIR, "paper_trading_status.json")
PAPER_TRADE_LOG = os.path.join(DATA_DIR, "paper_trades.json")

def check_paper_balance():
    print("=" * 50)
    print("PAPER TRADING ACCOUNT")
    print("=" * 50)

    # Check status file
    if not os.path.exists(PAPER_TRADING_STATUS):
        print("No paper trading data found.")
        print("Start paper trading first: python paper_trading.py")
        return

    with open(PAPER_TRADING_STATUS, 'r') as f:
        status = json.load(f)

    # Display balance info
    capital = status.get('capital', 1000)
    start_capital = status.get('start_capital', 1000)
    total_pnl = capital - start_capital
    pnl_percent = (total_pnl / start_capital) * 100 if start_capital > 0 else 0

    print(f"Starting Capital: ${start_capital:.2f}")
    print(f"Current Capital:  ${capital:.2f}")
    print(f"Total PnL:        ${total_pnl:+.2f} ({pnl_percent:+.2f}%)")
    print("=" * 50)

    # Trade stats
    trades = status.get('trades', [])
    wins = status.get('wins', 0)
    losses = status.get('losses', 0)
    total_trades = wins + losses

    if total_trades > 0:
        win_rate = (wins / total_trades) * 100
        print(f"\nTRADE STATISTICS:")
        print(f"-" * 50)
        print(f"Total Trades: {total_trades}")
        print(f"Wins:         {wins}")
        print(f"Losses:       {losses}")
        print(f"Win Rate:     {win_rate:.1f}%")

    # Open position
    open_pos = status.get('open_position')
    if open_pos:
        print(f"\nOPEN POSITION:")
        print(f"-" * 50)
        print(f"Side:   {open_pos.get('side', 'N/A')}")
        print(f"Entry:  ${open_pos.get('entry_price', 0):.2f}")
        print(f"Qty:    {open_pos.get('qty', 0):.6f}")
        print(f"Leverage: {open_pos.get('leverage', 1)}x")
    else:
        print(f"\nNo open position.")

    # Recent trades
    if trades and len(trades) > 0:
        print(f"\nRECENT TRADES (last 5):")
        print(f"-" * 50)
        for trade in trades[-5:]:
            side = trade.get('side', '?')
            entry = trade.get('entry_price', 0)
            exit_p = trade.get('exit_price', 0)
            pnl = trade.get('pnl', 0)
            result = "WIN" if pnl > 0 else "LOSS"
            print(f"{side}: ${entry:.2f} -> ${exit_p:.2f} | {result} ${pnl:+.2f}")

    # Paper trading requirements check
    print(f"\n" + "=" * 50)
    print("LIVE TRADING REQUIREMENTS:")
    print("=" * 50)

    days = status.get('days_trading', 0)
    min_days = 30
    min_trades = 50
    min_win_rate = 60

    print(f"Days Trading:  {days}/{min_days} {'✓' if days >= min_days else '✗'}")
    print(f"Total Trades:  {total_trades}/{min_trades} {'✓' if total_trades >= min_trades else '✗'}")
    if total_trades > 0:
        print(f"Win Rate:      {win_rate:.1f}%/{min_win_rate}% {'✓' if win_rate >= min_win_rate else '✗'}")
    print(f"Profitable:    {'✓' if total_pnl > 0 else '✗'}")

    all_passed = (days >= min_days and total_trades >= min_trades and
                  (total_trades == 0 or win_rate >= min_win_rate) and total_pnl > 0)

    if all_passed:
        print(f"\n>>> READY FOR LIVE TRADING!")
    else:
        print(f"\n>>> Continue paper trading...")

if __name__ == "__main__":
    check_paper_balance()
