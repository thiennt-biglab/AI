# model/backtest_td3.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from env.trading_env_cont import TradingEnvContinuous
from model.train_td3 import load_data
from config import BEST_MODEL_PATH, MODEL_PATH

def backtest():
    df = load_data()
    env = TradingEnvContinuous(df)

    model_path = BEST_MODEL_PATH
    model = TD3.load(model_path)

    obs, _ = env.reset()
    history = []
    action_bins = {"buy": 0, "sell": 0, "hold": 0}

    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        
        print(f"Action: {action}, Type: {type(action)}")

        act_value = float(action[0])
        if act_value > 0.05:
            action_type = "buy"
        elif act_value < -0.05:
            action_type = "sell"
        else:
            action_type = "hold"

        action_bins[action_type] += 1

        history.append({
            "timestamp": df.index[env.current_step],
            "action_value": act_value,
            "action_type": action_type,
            "net_worth": info["net_worth"],
            "balance": env.balance,
            "coin": env.coin,
            "price": df.loc[df.index[env.current_step]]["close"]
        })

        print(f"📊 Step {env.current_step} | Action: {action_type} ({act_value:.4f}) | Net Worth: {info['net_worth']:.2f}")

        if done:
            break

    results = pd.DataFrame(history).set_index("timestamp")
    results["net_worth"].plot(figsize=(10, 5), title="Net Worth Over Time")
    plt.ylabel("USDT")
    plt.grid(True)
    plt.show()

    # Analyze trades
    trades = []
    holding = False
    entry_price = 0

    for i in range(len(results)):
        row = results.iloc[i]
        if row["action_type"] == "buy" and not holding:
            entry_price = row["price"]
            holding = True
        elif row["action_type"] == "sell" and holding:
            pnl = row["price"] - entry_price
            trades.append(pnl)
            holding = False

    wins = sum(1 for x in trades if x > 0)
    losses = len(trades) - wins
    avg_pnl = sum(trades) / len(trades) if trades else 0
    win_rate = (wins / len(trades)) * 100 if trades else 0

    print("\n✅ Backtest Summary:")
    print(f"  • Initial Net Worth : 1000")
    print(f"  • Final Net Worth   : {results['net_worth'].iloc[-1]:.2f}")
    print(f"  • Profit/Loss       : {results['net_worth'].iloc[-1] - 1000:.2f}")
    print("\n📈 Trade Accuracy:")
    print(f"  • Total Trades      : {len(trades)}")
    print(f"  • Win Rate          : {win_rate:.2f}%")
    print(f"  • Avg PnL per Trade : {avg_pnl:.2f} USDT")
    print("\n📌 Action Breakdown:")
    print(f"  • Hold : {action_bins['hold']}")
    print(f"  • Buy  : {action_bins['buy']}")
    print(f"  • Sell : {action_bins['sell']}")

if __name__ == "__main__":
    backtest()
