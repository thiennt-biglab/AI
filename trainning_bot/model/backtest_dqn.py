import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import DQN
from env.trading_env import TradingEnv
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
from config import TIMEFRAME, HISTORY_LIMIT, MODEL_PATH, BEST_MODEL_PATH, SYMBOL, NEWSAPI_KEY, BINANCE_PAIR_K1,BINANCE_PAIR_K2

import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    price_df = fetch_ohlcv(symbol=SYMBOL, timeframe=TIMEFRAME, limit=HISTORY_LIMIT)
    price_df = add_technical_indicators(price_df)

    news_df = fetch_binance_news_from_newsapi(NEWSAPI_KEY)
    sentiment_df = pd.concat([news_df])
    price_df = merge_sentiment_to_price(price_df, sentiment_df)

    for col in ['sentiment', 'geo_sentiment']:
        if col not in price_df.columns:
            price_df[col] = 0.0

    return price_df.dropna()

def backtest():
    df = load_data()
    env = TradingEnv(df)
    model = DQN.load(BEST_MODEL_PATH)

    obs, _ = env.reset()
    history = []
    action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell

    while True:
        action, _ = model.predict(obs)
        action = int(action)
        obs, reward, done, _, info = env.step(action)

        action_counts[action] += 1

        history.append({
            "timestamp": df.index[env.current_step],
            "action": action,
            "net_worth": info["net_worth"],
            "balance": env.balance,
            "coin": env.coin,
            "price": df.loc[df.index[env.current_step]]['close']
        })

        print(f"📊 Step {env.current_step} | Action: {action} | Balance: {env.balance:.2f} | Coin: {env.coin:.4f} | Net Worth: {info['net_worth']:.2f}")

        if done:
            break

    results = pd.DataFrame(history).set_index("timestamp")
    results["net_worth"].plot(figsize=(10, 5), title="Net Worth Over Time")
    plt.ylabel(BINANCE_PAIR_K2)
    plt.grid(True)
    plt.show()

    # Tính PnL từng trade
    trades = []
    holding = False
    entry_price = 0

    for i in range(len(results)):
        row = results.iloc[i]
        if row["action"] == 1 and not holding:
            entry_price = row["price"]
            holding = True
        elif row["action"] == 2 and holding:
            exit_price = row["price"]
            pnl = exit_price - entry_price
            trades.append(pnl)
            holding = False

    num_wins = sum(1 for p in trades if p > 0)
    num_losses = sum(1 for p in trades if p <= 0)
    total_trades = len(trades)
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = sum(trades) / total_trades if total_trades > 0 else 0

    # Tổng kết
    initial = 1000
    final = results["net_worth"].iloc[-1]
    print("\n✅ Backtest Summary:")
    print(f"  • Initial Net Worth : {initial}")
    print(f"  • Final Net Worth   : {final:.2f}")
    print(f"  • Profit/Loss       : {final - initial:.2f}")
    print(f"\n📈 Trade Accuracy:")
    print(f"  • Total Trades      : {total_trades}")
    print(f"  • Win Rate          : {win_rate:.2f}%")
    print(f"  • Avg PnL per Trade : {avg_pnl:.2f} {BINANCE_PAIR_K2}")
    print(f"\n📌 Action Breakdown:")
    print(f"  • Hold : {action_counts[0]}")
    print(f"  • Buy  : {action_counts[1]}")
    print(f"  • Sell : {action_counts[2]}")

if __name__ == "__main__":
    backtest()
