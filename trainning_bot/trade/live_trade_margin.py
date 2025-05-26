# trade/live_trade_margin.py
import sys
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import torch
from stable_baselines3 import DQN
from binance.client import Client
from utils.trade_state import load_state, save_state
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    MODEL_PATH, BEST_MODEL_PATH,
    NEWSAPI_KEY, BINANCE_TRADE_SYMBOL,
    BINANCE_PAIR_K1, BINANCE_PAIR_K2
)

LAST_ACTION_FILE = "trade_last_action_margin.txt"
LOG_PATH = "logs/trades_margin.csv"
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


def get_dynamic_cooldown(confidence):
    if confidence > 0.8:
        return 1
    elif confidence > 0.6:
        return 3
    return 5


def too_soon_to_trade(cooldown_minutes):
    if not os.path.exists(LAST_ACTION_FILE):
        return False
    try:
        with open(LAST_ACTION_FILE, "r") as f:
            last_time = pd.to_datetime(f.read().strip())
        return datetime.now(timezone.utc) - last_time < timedelta(minutes=cooldown_minutes)
    except:
        return False


def mark_trade_time():
    with open(LAST_ACTION_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())


def recommend_action(q_values):
    actions = ['HOLD', 'BUY', 'SELL']
    idx = int(np.argmax(q_values))
    return actions[idx], q_values[idx]


def should_force_exit(price, entry_price):
    if entry_price <= 0:
        return False
    pnl = ((price - entry_price) / entry_price) * 100
    return pnl < -4 or pnl > 10


def run_live():
    state = load_state()
    df = fetch_ohlcv(limit=50)
    df = add_technical_indicators(df)
    news_df = fetch_binance_news_from_newsapi(NEWSAPI_KEY)
    sentiment_df = pd.concat([news_df])
    df = merge_sentiment_to_price(df, sentiment_df)
    if 'geo_sentiment' not in df.columns:
        df['geo_sentiment'] = 0.0

    price = df.iloc[-1]['close']
    coin = 1.0 if state.get("holding") else 0.0
    entry_price = state.get("entry_price", 0.0)
    unrealized_pnl = ((price - entry_price) / entry_price) * 100 if coin > 0 and entry_price > 0 else 0

    obs = np.array([
        price,
        df.iloc[-1]['volume'],
        df.iloc[-1]['rsi'],
        df.iloc[-1]['ema'],
        df.iloc[-1]['macd'],
        df.iloc[-1]['sentiment'],
        df.iloc[-1]['geo_sentiment'],
        coin,
        unrealized_pnl
    ], dtype=np.float32)

    model = DQN.load(BEST_MODEL_PATH)
    q_values = model.q_net(torch.tensor(obs).unsqueeze(0)).detach().numpy().flatten()
    recommendation, confidence = recommend_action(q_values)

    cooldown = get_dynamic_cooldown(confidence)
    if too_soon_to_trade(cooldown):
        print(f"⏳ Cooldown {cooldown} min active, skipping trading...")
        return

    try:
        symbol = BINANCE_TRADE_SYMBOL
        usdt_balance = float(client.get_asset_balance(asset=BINANCE_PAIR_K2)["free"])
        base_balance = float(client.get_asset_balance(asset=BINANCE_PAIR_K1)["free"])

        if should_force_exit(price, entry_price):
            print("🚨 Exiting position due to stoploss/takeprofit")
            recommendation = 'SELL'

        final_action = 'HOLD'
        if recommendation == 'BUY' and usdt_balance > 5:
            qty = round((usdt_balance - 1) / price, 6)
            client.create_margin_order(symbol=symbol, side='BUY', type='MARKET', quantity=qty)
            final_action = 'BUY'
            state["entry_price"] = price
            state["holding"] = True
            save_state(state)
            mark_trade_time()

        elif recommendation == 'SELL' and base_balance * price > 5:
            qty = round(base_balance, 6)
            client.create_margin_order(symbol=symbol, side='SELL', type='MARKET', quantity=qty)
            final_action = 'SELL'
            state["entry_price"] = 0
            state["holding"] = False
            save_state(state)
            mark_trade_time()

        print(f"⚙️ Recommended: {recommendation} ({confidence:.2f}), Final: {final_action}")
        print(f"💹 Holding {BINANCE_PAIR_K1}: {base_balance:.6f} × {price:.2f} = {base_balance * price:.2f} {BINANCE_PAIR_K2}")
        print(f"💹 Holding {BINANCE_PAIR_K2}: {usdt_balance:.2f} {BINANCE_PAIR_K2}")

        log = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc),
            "price": price,
            "action": final_action,
            "usdt_balance": usdt_balance,
            "base_balance": base_balance,
            "recommendation": recommendation,
            "q_values": q_values.tolist()
        }])
        log.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

    except Exception as e:
        print(f"⚠️ Trading error: {e}")

if __name__ == "__main__":
    while True:
        print("🚀 Running live trade margin...")
        run_live()
        print("✅ Done. Sleeping 1 mins...\n")
        time.sleep(60)
