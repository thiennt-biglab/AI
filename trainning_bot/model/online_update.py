# model/online_update.py
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from env.trading_env import TradingEnv
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi
from config import MODEL_PATH, BEST_MODEL_PATH, NEWSAPI_KEY

def update_model():
    print("🔁 Loading model...")

    # B1: Tải dữ liệu
    price_df = fetch_ohlcv(limit=100)
    price_df = add_technical_indicators(price_df)

    # B2: Gộp dữ liệu cảm xúc
    news_df = fetch_binance_news_from_newsapi(NEWSAPI_KEY)
    sentiment_df = pd.concat([news_df])
    df = merge_sentiment_to_price(price_df, sentiment_df)

    for col in ['sentiment', 'geo_sentiment']:
        if col not in df.columns:
            df[col] = 0.0

    if len(df) < 10:
        print("⚠️ Not enough data to update.")
        return

    env = TradingEnv(df)
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0005,
        buffer_size=10000,
        exploration_fraction=0.4,
        tensorboard_log="./logs/"
    )
    model._setup_learn(total_timesteps=1)
    model.set_parameters(BEST_MODEL_PATH)

    # B3: Add nhiều mẫu vào buffer
    reward_sum = 0.0
    for i in range(10, len(df) - 1):
        env.current_step = i
        obs = env._get_observation()
        action, _ = model.predict(obs)
        obs_, reward, done, info = env.step(action)
        reward_sum += reward
        model.replay_buffer.add(
            obs.reshape(1, -1),
            obs_.reshape(1, -1),
            np.array([action]),
            np.array([float(reward)]),
            np.array([done]),
            [{}]
        )

    model.train(batch_size=64, gradient_steps=10)
    model.save(BEST_MODEL_PATH)

    # B4: Logging
    log_path = "logs/online_updates.csv"
    reward_delta = 0.0
    if os.path.exists(log_path):
        try:
            previous = pd.read_csv(log_path).tail(1)
            if not previous.empty:
                reward_delta = reward_sum - previous.iloc[0]['reward']
        except Exception as e:
            print(f"⚠️ Failed to calculate reward delta: {e}")

    log_entry = pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow(),
        "action": int(action),
        "reward": float(reward_sum),
        "reward_delta": reward_delta,
        "net_worth": float(info["net_worth"])
    }])
    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode='a', header=False, index=False, quoting=1)
    else:
        log_entry.to_csv(log_path, index=False)

    if abs(reward_delta) > 0.1:
        print(f"🚨 Unusual reward change detected: Δ {reward_delta:.4f}")

    if info['net_worth'] < 0.5 * env.initial_balance:
        print("🛑 Stopping: Critical loss detected.")
        exit()

    print(f"✅ Updated: Action={action}, Reward={reward_sum:.2f}, Net worth={info['net_worth']:.2f}")

if __name__ == "__main__":
    while True:
        update_model()
        print("⏳ Sleeping 15 minutes...\n")
        time.sleep(15 * 60)
