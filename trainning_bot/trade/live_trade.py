import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.trade_state import load_state, save_state
from stable_baselines3 import DQN
from binance.client import Client
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
from config import (
    BINANCE_API_KEY, BINANCE_API_SECRET,
    MODEL_PATH, BEST_MODEL_PATH,
    NEWSAPI_KEY, BINANCE_TRADE_SYMBOL,
    BINANCE_PAIR_K1, BINANCE_PAIR_K2
)

COOLDOWN_MINUTES = 5
LAST_ACTION_FILE = "trade_last_action.txt"
LOG_PATH = "logs/trades.csv"

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

def get_lot_size_info(symbol):
    info = client.get_symbol_info(symbol)
    filters = {f["filterType"]: f for f in info["filters"]}
    min_qty = float(filters["LOT_SIZE"]["minQty"])
    step_size = float(filters["LOT_SIZE"]["stepSize"])
    return min_qty, step_size

def round_step(value, step):
    return float(np.floor(value / step) * step)

def too_soon_to_trade():
    if not os.path.exists(LAST_ACTION_FILE):
        return False
    try:
        with open(LAST_ACTION_FILE, "r") as f:
            last_time = pd.to_datetime(f.read().strip())
        return datetime.now(timezone.utc) - last_time < timedelta(minutes=COOLDOWN_MINUTES)
    except:
        return False

def mark_trade_time():
    with open(LAST_ACTION_FILE, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())

def recommend_action(q_values):
    actions = ['HOLD', 'BUY', 'SELL']
    idx = int(np.argmax(q_values))
    return actions[idx], q_values[idx]

def risk_filter(action, usdt_balance, base_balance, price, entry_price):
    min_trade_usd = 5
    if action == 'BUY' and usdt_balance < min_trade_usd:
        return 'HOLD'
    if action == 'SELL' and base_balance * price < min_trade_usd:
        return 'HOLD'
    if action == 'SELL' and entry_price > 0:
        pnl = ((price - entry_price) / entry_price) * 100
        if pnl < -5:
            print(f"⚠️ Skip SELL due to stop-loss ({pnl:.2f}%)")
            return 'HOLD'
    return action

def run_live():
    if too_soon_to_trade():
        print("⏳ Cooldown active, skipping trading...")
        return

    df = fetch_ohlcv(limit=50)
    df = add_technical_indicators(df)
    news_df = fetch_binance_news_from_newsapi(NEWSAPI_KEY)
    df = merge_sentiment_to_price(df, pd.concat([news_df]))
    if 'geo_sentiment' not in df.columns:
        df['geo_sentiment'] = 0.0

    state = load_state()
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

    try:
        symbol = BINANCE_TRADE_SYMBOL
        usdt_balance = float(client.get_asset_balance(asset=BINANCE_PAIR_K2)["free"])
        base_balance = float(client.get_asset_balance(asset=BINANCE_PAIR_K1)["free"])
        min_qty, step_size = get_lot_size_info(symbol)

        final_action = risk_filter(recommendation, usdt_balance, base_balance, price, entry_price)
        print(f"⚙️ Recommended: {recommendation} ({confidence:.2f}), Risk-Filtered: {final_action}")

        if final_action == 'BUY':
            qty = (usdt_balance - 1) / price
            qty = max(min_qty, round_step(qty, step_size))
            client.order_market_buy(symbol=symbol, quantity=qty)
            state["entry_price"] = price
            state["holding"] = True
            save_state(state)
            mark_trade_time()
            print(f"💰 MUA {qty} {BINANCE_PAIR_K1}")

        elif final_action == 'SELL':
            qty = round_step(base_balance, step_size)
            if qty >= min_qty:
                client.order_market_sell(symbol=symbol, quantity=qty)
                state["entry_price"] = 0
                state["holding"] = False
                save_state(state)
                mark_trade_time()
                print(f"📤 BÁN {qty} {BINANCE_PAIR_K1}")
            else:
                print(f"⚠️ Không đủ số lượng để bán: {qty} < minQty {min_qty}")

        else:
            print("🤝 GIỮ")

        estimated_value = base_balance * price
        print(f"💹 Holding {BINANCE_PAIR_K1}: {base_balance:.6f} × {price:.2f} = {estimated_value:.2f} {BINANCE_PAIR_K2}")
        print(f"💹 Holding {BINANCE_PAIR_K2}: {usdt_balance:.2f} {BINANCE_PAIR_K2}")

        log_entry = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc),
            "price": price,
            "action": final_action,
            "usdt_balance": usdt_balance,
            "base_balance": base_balance,
            "recommendation": recommendation,
            "q_values": q_values.tolist()
        }])
        log_entry.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

    except Exception as e:
        print(f"⚠️ Lỗi khi giao dịch hoặc lấy số dư: {e}")


if __name__ == "__main__":
    while True:
        print("🚀 Running live trade prediction...")
        run_live()
        print("✅ Done. Sleeping 1 mins...\n")
        time.sleep(60)
