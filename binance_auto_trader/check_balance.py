#!/usr/bin/env python3
"""Quick script to check Binance Futures balance."""

from binance.client import Client
from config import API_KEY, API_SECRET

def check_balance():
    client = Client(API_KEY, API_SECRET)

    # Get futures account info
    account = client.futures_account()

    # Extract key info
    total_balance = float(account['totalWalletBalance'])
    available = float(account['availableBalance'])
    unrealized_pnl = float(account['totalUnrealizedProfit'])
    margin_balance = float(account['totalMarginBalance'])

    print("=" * 50)
    print("BINANCE FUTURES ACCOUNT")
    print("=" * 50)
    print(f"Total Balance:    ${total_balance:.2f}")
    print(f"Available:        ${available:.2f}")
    print(f"Unrealized PnL:   ${unrealized_pnl:+.2f}")
    print(f"Margin Balance:   ${margin_balance:.2f}")
    print("=" * 50)

    # Check open positions
    positions = [p for p in account['positions'] if float(p['positionAmt']) != 0]

    if positions:
        print("\nOPEN POSITIONS:")
        print("-" * 50)
        for pos in positions:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            entry = float(pos['entryPrice'])
            pnl = float(pos['unrealizedProfit'])
            side = "LONG" if amt > 0 else "SHORT"
            print(f"{symbol}: {side} {abs(amt):.4f} @ ${entry:.2f} | PnL: ${pnl:+.2f}")
    else:
        print("\nNo open positions.")

    return total_balance

if __name__ == "__main__":
    check_balance()
