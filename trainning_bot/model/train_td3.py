# model/train_td3.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from env.trading_env import TradingEnv
from env.trading_env_cont import TradingEnvContinuous
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi
from utils.feature_engineering import add_technical_indicators, merge_sentiment_to_price
from config import TIMEFRAME, HISTORY_LIMIT, BEST_MODEL_PATH, SYMBOL, NEWSAPI_KEY
import gymnasium as gym
from gymnasium import spaces

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

def train():
    df = load_data()
    env = TradingEnvContinuous(df)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        buffer_size=50000,
        action_noise=action_noise,
        batch_size=128,
        tau=0.005,
        policy_delay=2,
        train_freq=(1, "step"),
        gradient_steps=1,
        tensorboard_log="./logs/td3/"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/best",
        log_path="./logs/td3_eval",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    model.learn(total_timesteps=200_000, callback=eval_callback)
    model.save(BEST_MODEL_PATH)
    print(f"✅ TD3 model saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train()
