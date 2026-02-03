"""
Simulate trading with $100 starting capital.
Shows detailed win/loss progression for both modes.
"""
import numpy as np
from binance.client import Client
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS

client = Client(API_KEY, API_SECRET)

def fetch_data(symbol, interval, limit=10000):
    """Fetch historical data from Binance."""
    print(f"[INFO] Fetching {limit} candles of {symbol} {interval}...")
    all_klines = []
    last_time = None

    while len(all_klines) < limit:
        fetch_limit = min(1000, limit - len(all_klines))
        klines = client.futures_klines(
            symbol=symbol, interval=interval,
            limit=fetch_limit, endTime=last_time
        )
        if not klines:
            break
        all_klines = klines + all_klines
        last_time = klines[0][0] - 1
        if len(klines) < fetch_limit:
            break

    # Convert to arrays
    close = np.array([float(k[4]) for k in all_klines])
    high = np.array([float(k[2]) for k in all_klines])
    low = np.array([float(k[3]) for k in all_klines])

    # Calculate ATR
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr = np.zeros(len(close))
    atr[14:] = np.convolve(tr, np.ones(14)/14, mode='valid')[:len(close)-14]

    # Calculate RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = np.zeros(len(close))
    rsi[14:len(rs)+14] = 100 - (100 / (1 + rs))

    return close, high, low, atr, rsi


def simulate_mode(close, high, low, atr, rsi, mode='accuracy', starting_capital=100):
    """
    Simulate trading for a specific mode.
    Returns detailed trade history.
    """
    # Mode settings
    if mode == 'accuracy':
        CONFIDENCE_MIN = 0.92
        TP = 0.012  # 1.2%
        SL = 0.008  # 0.8%
        LEVERAGE = 3
        RISK_PCT = 0.10  # 10% of capital per trade
        MIN_GAP = 3
        WINDOW = 8
    else:  # profit mode
        CONFIDENCE_MIN = 0.85
        TP = 0.015  # 1.5%
        SL = 0.008  # 0.8%
        LEVERAGE = 5
        RISK_PCT = 0.15  # 15% of capital per trade
        MIN_GAP = 2
        WINDOW = 12

    capital = starting_capital
    trades = []
    last_trade_idx = -MIN_GAP
    np.random.seed(42)

    for i in range(50, len(close) - WINDOW - 1):
        # Generate signal based on RSI
        signal = None
        confidence = 0

        if rsi[i] < 40:  # Oversold -> LONG
            signal = 'LONG'
            confidence = 0.80 + (40 - rsi[i]) / 100 + np.random.uniform(-0.05, 0.12)
        elif rsi[i] > 60:  # Overbought -> SHORT
            signal = 'SHORT'
            confidence = 0.80 + (rsi[i] - 60) / 100 + np.random.uniform(-0.05, 0.12)
        elif rsi[i] < 50 and rsi[i] > rsi[i-1]:  # Momentum up
            signal = 'LONG'
            confidence = 0.75 + np.random.uniform(-0.05, 0.15)
        elif rsi[i] > 50 and rsi[i] < rsi[i-1]:  # Momentum down
            signal = 'SHORT'
            confidence = 0.75 + np.random.uniform(-0.05, 0.15)

        if not signal or confidence < CONFIDENCE_MIN:
            continue

        if i - last_trade_idx < MIN_GAP:
            continue

        last_trade_idx = i
        entry_price = close[i]

        # Calculate position size
        position_value = capital * RISK_PCT * LEVERAGE

        # Simulate trade outcome
        result = None
        pnl = 0

        if signal == 'LONG':
            for j in range(1, WINDOW + 1):
                # Check TP
                if high[i + j] >= entry_price * (1 + TP):
                    pnl = position_value * TP
                    result = 'WIN'
                    break
                # Check SL
                if low[i + j] <= entry_price * (1 - SL):
                    pnl = -position_value * SL
                    result = 'LOSS'
                    break
            else:
                # Timeout - use close price
                change = (close[i + WINDOW] - entry_price) / entry_price
                pnl = position_value * change
                result = 'WIN' if change > 0 else 'LOSS'

        else:  # SHORT
            for j in range(1, WINDOW + 1):
                # Check TP
                if low[i + j] <= entry_price * (1 - TP):
                    pnl = position_value * TP
                    result = 'WIN'
                    break
                # Check SL
                if high[i + j] >= entry_price * (1 + SL):
                    pnl = -position_value * SL
                    result = 'LOSS'
                    break
            else:
                change = (entry_price - close[i + WINDOW]) / entry_price
                pnl = position_value * change
                result = 'WIN' if change > 0 else 'LOSS'

        capital += pnl
        trades.append({
            'trade_num': len(trades) + 1,
            'signal': signal,
            'result': result,
            'pnl': pnl,
            'capital': capital,
            'confidence': confidence
        })

        # Stop if bankrupt
        if capital <= 0:
            capital = 0
            break

    return trades, capital


def print_simulation_report(trades_acc, final_acc, trades_prof, final_prof):
    """Print detailed simulation report."""

    print("\n" + "=" * 70)
    print("  TRADING SIMULATION: $100 Starting Capital")
    print("  Symbol: BTCUSDT | Timeframe: 5m | Data: 10,000 candles")
    print("=" * 70)

    # Summary table
    wins_acc = sum(1 for t in trades_acc if t['result'] == 'WIN')
    losses_acc = sum(1 for t in trades_acc if t['result'] == 'LOSS')
    wins_prof = sum(1 for t in trades_prof if t['result'] == 'WIN')
    losses_prof = sum(1 for t in trades_prof if t['result'] == 'LOSS')

    print("\n" + "-" * 70)
    print(f"{'SUMMARY':<30} {'ACCURACY MODE':>18} {'PROFIT MODE':>18}")
    print("-" * 70)
    print(f"{'Starting Capital':<30} {'$100.00':>18} {'$100.00':>18}")
    print(f"{'Final Capital':<30} {'${:.2f}'.format(final_acc):>18} {'${:.2f}'.format(final_prof):>18}")
    print(f"{'Total P/L':<30} {'${:.2f}'.format(final_acc - 100):>18} {'${:.2f}'.format(final_prof - 100):>18}")
    print(f"{'Return %':<30} {'{:.1f}%'.format((final_acc/100-1)*100):>18} {'{:.1f}%'.format((final_prof/100-1)*100):>18}")
    print("-" * 70)
    print(f"{'Total Trades':<30} {len(trades_acc):>18} {len(trades_prof):>18}")
    print(f"{'Wins':<30} {wins_acc:>18} {wins_prof:>18}")
    print(f"{'Losses':<30} {losses_acc:>18} {losses_prof:>18}")
    print(f"{'Win Rate':<30} {'{:.1f}%'.format(wins_acc/len(trades_acc)*100 if trades_acc else 0):>18} {'{:.1f}%'.format(wins_prof/len(trades_prof)*100 if trades_prof else 0):>18}")
    print("-" * 70)

    # Trade by trade for Accuracy Mode
    print("\n" + "=" * 70)
    print("  ACCURACY MODE - Trade by Trade")
    print("=" * 70)
    print(f"{'#':<5} {'Signal':<8} {'Result':<8} {'P/L':>12} {'Capital':>12} {'Conf':>8}")
    print("-" * 70)

    for t in trades_acc[:50]:  # Show first 50 trades
        pnl_str = '${:.2f}'.format(t['pnl'])
        if t['pnl'] >= 0:
            pnl_str = '+' + pnl_str
        print(f"{t['trade_num']:<5} {t['signal']:<8} {t['result']:<8} {pnl_str:>12} {'${:.2f}'.format(t['capital']):>12} {'{:.0f}%'.format(t['confidence']*100):>8}")

    if len(trades_acc) > 50:
        print(f"... and {len(trades_acc) - 50} more trades")

    # Trade by trade for Profit Mode
    print("\n" + "=" * 70)
    print("  PROFIT MODE - Trade by Trade")
    print("=" * 70)
    print(f"{'#':<5} {'Signal':<8} {'Result':<8} {'P/L':>12} {'Capital':>12} {'Conf':>8}")
    print("-" * 70)

    for t in trades_prof[:50]:  # Show first 50 trades
        pnl_str = '${:.2f}'.format(t['pnl'])
        if t['pnl'] >= 0:
            pnl_str = '+' + pnl_str
        print(f"{t['trade_num']:<5} {t['signal']:<8} {t['result']:<8} {pnl_str:>12} {'${:.2f}'.format(t['capital']):>12} {'{:.0f}%'.format(t['confidence']*100):>8}")

    if len(trades_prof) > 50:
        print(f"... and {len(trades_prof) - 50} more trades")

    # Capital progression
    print("\n" + "=" * 70)
    print("  CAPITAL PROGRESSION")
    print("=" * 70)

    # Show every 10th trade
    print(f"\n{'Trade #':<12} {'Accuracy Mode':>18} {'Profit Mode':>18}")
    print("-" * 50)

    max_trades = max(len(trades_acc), len(trades_prof))
    step = max(1, max_trades // 20)

    for i in range(0, max_trades, step):
        cap_acc = trades_acc[i]['capital'] if i < len(trades_acc) else trades_acc[-1]['capital'] if trades_acc else 100
        cap_prof = trades_prof[i]['capital'] if i < len(trades_prof) else trades_prof[-1]['capital'] if trades_prof else 100
        print(f"{i+1:<12} {'${:.2f}'.format(cap_acc):>18} {'${:.2f}'.format(cap_prof):>18}")

    print(f"{'FINAL':<12} {'${:.2f}'.format(final_acc):>18} {'${:.2f}'.format(final_prof):>18}")

    # Winner
    print("\n" + "=" * 70)
    if final_acc > final_prof:
        print(f"  >>> WINNER: ACCURACY MODE")
        print(f"      Made ${final_acc - final_prof:.2f} more than Profit Mode")
    else:
        print(f"  >>> WINNER: PROFIT MODE")
        print(f"      Made ${final_prof - final_acc:.2f} more than Accuracy Mode")
    print("=" * 70)


def main():
    # Fetch data
    close, high, low, atr, rsi = fetch_data(SYMBOL, INTERVALS[0], limit=10000)
    print(f"[INFO] Loaded {len(close)} candles")

    # Simulate both modes
    print("\n[INFO] Simulating Accuracy Mode...")
    trades_acc, final_acc = simulate_mode(close, high, low, atr, rsi, mode='accuracy', starting_capital=100)

    print("[INFO] Simulating Profit Mode...")
    trades_prof, final_prof = simulate_mode(close, high, low, atr, rsi, mode='profit', starting_capital=100)

    # Print report
    print_simulation_report(trades_acc, final_acc, trades_prof, final_prof)

    return trades_acc, final_acc, trades_prof, final_prof


if __name__ == "__main__":
    main()
