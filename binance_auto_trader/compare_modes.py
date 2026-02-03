"""
Compare ACCURACY vs PROFIT trading modes on real Binance data.
Generates detailed report with metrics comparison.
"""
import numpy as np
import pandas as pd
from binance.client import Client
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import API_KEY, API_SECRET, SYMBOL, INTERVALS

client = Client(API_KEY, API_SECRET)

def fetch_full_klines(symbol, interval, total_limit=10000):
    """Fetch historical klines from Binance."""
    all_klines = []
    last_time = None

    while len(all_klines) < total_limit:
        fetch_limit = min(1000, total_limit - len(all_klines))
        try:
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=fetch_limit,
                endTime=last_time
            )
        except Exception as e:
            print(f"[ERROR] API call failed: {e}")
            break

        if not klines:
            break
        all_klines = klines + all_klines
        last_time = klines[0][0] - 1
        if len(klines) < fetch_limit:
            break

    return all_klines

# ============================================================
# MODE CONFIGURATIONS
# ============================================================
MODES = {
    'accuracy': {
        'name': 'ACCURACY MODE',
        'confidence_threshold': 0.92,
        'entropy_threshold': 0.45,
        'base_tp': 0.012,        # 1.2%
        'base_sl': 0.008,        # 0.8%
        'leverage': 3,
        'risk_percent': 10,
        'use_dynamic_tpsl': False,
        'use_trailing_stop': False,
        'use_compound': False,
        'min_signal_gap': 3,
        'backtest_window': 8,
    },
    'profit': {
        'name': 'PROFIT MODE',
        'confidence_threshold': 0.85,
        'entropy_threshold': 0.55,
        'base_tp': 0.015,        # 1.5%
        'base_sl': 0.008,        # 0.8%
        'leverage': 5,
        'risk_percent': 15,
        'use_dynamic_tpsl': True,
        'use_trailing_stop': True,
        'use_compound': True,
        'min_signal_gap': 2,
        'backtest_window': 12,
    }
}

def fetch_data(symbol, interval, limit=5000):
    """Fetch historical data from Binance."""
    print(f"[INFO] Fetching {limit} candles of {symbol} {interval}...")
    klines = fetch_full_klines(symbol, interval, total_limit=limit)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Calculate ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd_diff'] = ema12 - ema26

    df.dropna(inplace=True)
    return df


def generate_signals(df, num_signals=500):
    """
    Generate simulated trading signals based on technical indicators.
    Returns signals with confidence scores (simulating ML model output).
    More relaxed conditions to generate enough signals for comparison.
    """
    signals = []
    np.random.seed(42)  # For reproducibility

    for i in range(50, len(df) - 20):
        rsi = df['rsi'].iloc[i]
        macd = df['macd_diff'].iloc[i]
        macd_prev = df['macd_diff'].iloc[i-1]
        atr_pct = df['atr'].iloc[i] / df['close'].iloc[i]
        close = df['close'].iloc[i]
        close_prev = df['close'].iloc[i-1]
        price_change = (close - close_prev) / close_prev

        signal = None
        base_confidence = 0

        # LONG signals - more relaxed conditions
        if rsi < 40:  # Oversold
            base_confidence = 0.80 + (40 - rsi) / 100
            signal = 1
        elif rsi < 50 and macd > macd_prev:  # Momentum turning up
            base_confidence = 0.75 + (50 - rsi) / 200
            signal = 1
        elif macd > 0 and macd > macd_prev and price_change > 0:  # Bullish momentum
            base_confidence = 0.78
            signal = 1
        elif rsi < 55 and price_change > 0.001:  # Price moving up
            base_confidence = 0.72
            signal = 1

        # SHORT signals - more relaxed conditions
        elif rsi > 60:  # Overbought
            base_confidence = 0.80 + (rsi - 60) / 100
            signal = 2
        elif rsi > 50 and macd < macd_prev:  # Momentum turning down
            base_confidence = 0.75 + (rsi - 50) / 200
            signal = 2
        elif macd < 0 and macd < macd_prev and price_change < 0:  # Bearish momentum
            base_confidence = 0.78
            signal = 2
        elif rsi > 45 and price_change < -0.001:  # Price moving down
            base_confidence = 0.72
            signal = 2

        if signal and base_confidence > 0.70:
            # Add variance to simulate model predictions
            confidence = min(0.98, base_confidence + np.random.uniform(-0.08, 0.15))
            entropy = max(0.1, min(0.8, 1.1 - confidence + np.random.uniform(-0.15, 0.15)))

            signals.append({
                'idx': i,
                'signal': signal,
                'confidence': confidence,
                'entropy': entropy,
                'atr': df['atr'].iloc[i],
                'close': close,
                'high': df['high'].iloc[i],
                'low': df['low'].iloc[i],
            })

    # Limit and shuffle to avoid bias
    if len(signals) > num_signals:
        signals = signals[:num_signals]

    return signals


def backtest_mode(df, signals, mode_config):
    """Run backtest for a specific mode configuration."""
    config = mode_config

    capital = 100
    peak_capital = 100
    wins = 0
    losses = 0
    profits = []
    trades = []
    last_trade_idx = -config['min_signal_gap']

    for sig in signals:
        i = sig['idx']

        # Apply filters based on mode
        if sig['confidence'] < config['confidence_threshold']:
            continue
        if sig['entropy'] > config['entropy_threshold']:
            continue
        if i - last_trade_idx < config['min_signal_gap']:
            continue

        last_trade_idx = i
        price_entry = sig['close']
        atr = sig['atr']

        # Dynamic TP/SL
        if config['use_dynamic_tpsl'] and atr > 0:
            atr_pct = atr / price_entry
            tp = max(config['base_tp'], min(atr_pct * 2.0, 0.03))
            sl = max(config['base_sl'], min(atr_pct * 1.0, 0.015))
        else:
            tp = config['base_tp']
            sl = config['base_sl']

        # Simulate trade
        trade_result = 0
        window = config['backtest_window']

        if i + window >= len(df):
            continue

        if sig['signal'] == 1:  # LONG
            trailing_stop = price_entry * (1 - sl)
            highest = price_entry

            for j in range(1, window + 1):
                idx = i + j
                price_high = df['high'].iloc[idx]
                price_low = df['low'].iloc[idx]
                price_close = df['close'].iloc[idx]

                # Trailing stop update
                if config['use_trailing_stop'] and price_high > highest:
                    highest = price_high
                    unrealized = (highest - price_entry) / price_entry
                    if unrealized > tp * 0.5:
                        trailing_stop = max(trailing_stop, price_entry * (1 + unrealized * 0.5))

                # Check TP
                if price_high >= price_entry * (1 + tp):
                    trade_result = tp
                    wins += 1
                    break
                # Check SL
                elif price_low <= trailing_stop:
                    trade_result = (trailing_stop - price_entry) / price_entry
                    if trade_result >= 0:
                        wins += 1
                    else:
                        losses += 1
                    break
            else:
                trade_result = (df['close'].iloc[i + window] - price_entry) / price_entry
                if trade_result > 0:
                    wins += 1
                else:
                    losses += 1

        elif sig['signal'] == 2:  # SHORT
            trailing_stop = price_entry * (1 + sl)
            lowest = price_entry

            for j in range(1, window + 1):
                idx = i + j
                price_high = df['high'].iloc[idx]
                price_low = df['low'].iloc[idx]
                price_close = df['close'].iloc[idx]

                if config['use_trailing_stop'] and price_low < lowest:
                    lowest = price_low
                    unrealized = (price_entry - lowest) / price_entry
                    if unrealized > tp * 0.5:
                        trailing_stop = min(trailing_stop, price_entry * (1 - unrealized * 0.5))

                if price_low <= price_entry * (1 - tp):
                    trade_result = tp
                    wins += 1
                    break
                elif price_high >= trailing_stop:
                    trade_result = (price_entry - trailing_stop) / price_entry
                    if trade_result >= 0:
                        wins += 1
                    else:
                        losses += 1
                    break
            else:
                trade_result = (price_entry - df['close'].iloc[i + window]) / price_entry
                if trade_result > 0:
                    wins += 1
                else:
                    losses += 1

        # Apply leverage
        leveraged_result = trade_result * config['leverage']

        # Apply to capital
        if config['use_compound']:
            capital *= (1 + leveraged_result * config['risk_percent'] / 100)
        else:
            capital += 100 * leveraged_result * config['risk_percent'] / 100

        profits.append(trade_result)
        trades.append({
            'type': 'LONG' if sig['signal'] == 1 else 'SHORT',
            'result': trade_result,
            'confidence': sig['confidence']
        })

        peak_capital = max(peak_capital, capital)

    # Calculate metrics
    total_trades = wins + losses
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0

    total_win = sum(p for p in profits if p > 0)
    total_loss = abs(sum(p for p in profits if p < 0))
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

    avg_trade = np.mean(profits) * 100 if profits else 0
    std_trade = np.std(profits) * 100 if profits else 0
    sharpe = avg_trade / std_trade if std_trade > 0 else 0

    # Max drawdown
    if config['use_compound']:
        cumulative = [100]
        for p in profits:
            cumulative.append(cumulative[-1] * (1 + p * config['leverage'] * config['risk_percent'] / 100))
    else:
        cumulative = np.cumsum([100] + [100 * p * config['leverage'] * config['risk_percent'] / 100 for p in profits])

    cumulative = np.array(cumulative)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak * 100
    max_drawdown = np.max(drawdown)

    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': capital - 100,
        'return_pct': (capital / 100 - 1) * 100,
        'max_drawdown': max_drawdown,
        'avg_trade': avg_trade,
        'std_trade': std_trade,
        'sharpe': sharpe,
        'final_capital': capital,
    }


def print_comparison_report(results_accuracy, results_profit, symbol):
    """Print detailed comparison report."""

    print("\n" + "=" * 70)
    print(f"  MODE COMPARISON REPORT - {symbol}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Side by side comparison
    print("\n" + "-" * 70)
    print(f"{'METRIC':<25} {'ACCURACY MODE':>20} {'PROFIT MODE':>20}")
    print("-" * 70)

    metrics = [
        ('Total Trades', 'total_trades', '{:.0f}'),
        ('Wins', 'wins', '{:.0f}'),
        ('Losses', 'losses', '{:.0f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Total Return ($)', 'total_return', '${:.2f}'),
        ('Return (%)', 'return_pct', '{:.1f}%'),
        ('Max Drawdown', 'max_drawdown', '{:.1f}%'),
        ('Avg Trade', 'avg_trade', '{:.3f}%'),
        ('Std Trade', 'std_trade', '{:.3f}%'),
        ('Sharpe Ratio', 'sharpe', '{:.2f}'),
        ('Final Capital', 'final_capital', '${:.2f}'),
    ]

    for label, key, fmt in metrics:
        val_acc = results_accuracy[key]
        val_prof = results_profit[key]

        # Format values
        if '%' in fmt:
            str_acc = fmt.format(val_acc)
            str_prof = fmt.format(val_prof)
        elif '$' in fmt:
            str_acc = fmt.format(val_acc)
            str_prof = fmt.format(val_prof)
        else:
            str_acc = fmt.format(val_acc)
            str_prof = fmt.format(val_prof)

        # Add winner indicator
        if key in ['win_rate', 'profit_factor', 'total_return', 'return_pct', 'sharpe', 'final_capital']:
            if val_prof > val_acc:
                str_prof = str_prof + " <<<"
            elif val_acc > val_prof:
                str_acc = str_acc + " <<<"
        elif key == 'max_drawdown':
            if val_prof < val_acc:
                str_prof = str_prof + " <<<"
            elif val_acc < val_prof:
                str_acc = str_acc + " <<<"

        print(f"{label:<25} {str_acc:>20} {str_prof:>20}")

    print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    acc_score = (results_accuracy['win_rate'] * 2 +
                 results_accuracy['profit_factor'] * 10 +
                 results_accuracy['return_pct'])

    prof_score = (results_profit['win_rate'] * 2 +
                  results_profit['profit_factor'] * 10 +
                  results_profit['return_pct'])

    print(f"\n  Composite Score:")
    print(f"    Accuracy Mode: {acc_score:.1f}")
    print(f"    Profit Mode:   {prof_score:.1f}")

    if prof_score > acc_score:
        winner = "PROFIT MODE"
        diff = results_profit['return_pct'] - results_accuracy['return_pct']
    else:
        winner = "ACCURACY MODE"
        diff = results_accuracy['return_pct'] - results_profit['return_pct']

    print(f"\n  >>> WINNER: {winner}")
    print(f"    Extra return: +{abs(diff):.1f}%")

    # Recommendations
    print("\n" + "-" * 70)
    print("  RECOMMENDATIONS")
    print("-" * 70)

    if results_accuracy['win_rate'] > 60:
        print("  • High win rate achieved - Accuracy mode is reliable")
    if results_profit['return_pct'] > results_accuracy['return_pct'] * 1.5:
        print("  • Profit mode generates significantly higher returns")
    if results_profit['max_drawdown'] > 20:
        print("  • WARNING: Profit mode has high drawdown - consider reducing leverage")
    if results_accuracy['total_trades'] < 10:
        print("  • Accuracy mode has few trades - may need more data")
    if results_profit['sharpe'] > results_accuracy['sharpe']:
        print("  • Profit mode has better risk-adjusted returns")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("  ACCURACY vs PROFIT MODE COMPARISON")
    print("  Using Real Binance Data")
    print("=" * 70)

    # Fetch real data
    df = fetch_data(SYMBOL, INTERVALS[0], limit=10000)
    print(f"[INFO] Loaded {len(df)} candles")
    print(f"[INFO] Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Generate signals (simulating ML model)
    print("\n[INFO] Generating trading signals...")
    signals = generate_signals(df, num_signals=300)
    print(f"[INFO] Generated {len(signals)} signals")

    # Backtest both modes
    print("\n[INFO] Running Accuracy Mode backtest...")
    results_accuracy = backtest_mode(df, signals, MODES['accuracy'])

    print("[INFO] Running Profit Mode backtest...")
    results_profit = backtest_mode(df, signals, MODES['profit'])

    # Print comparison report
    print_comparison_report(results_accuracy, results_profit, SYMBOL)

    return results_accuracy, results_profit


if __name__ == "__main__":
    results = main()
