# model/train_dqn.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import DQN
from env.trading_env import TradingEnv
from data.fetch_binance import fetch_ohlcv
from data.fetch_news import fetch_binance_news_from_newsapi, fetch_twitter_sentiment
from data.fetch_geopolitical_news import fetch_geopolitical_news
from utils.feature_engineering import (
    add_technical_indicators,
    merge_sentiment_to_price
)
import pandas as pd
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from config import TIMEFRAME, HISTORY_LIMIT, MODEL_PATH, SYMBOL, TWITTER_BEARER_TOKEN, TWITTER_USERS, NEWSAPI_KEY

TRADE_SYMBOL = SYMBOL


def load_data():
    price_df = fetch_ohlcv(symbol=TRADE_SYMBOL, timeframe=TIMEFRAME, limit=HISTORY_LIMIT)
    price_df = add_technical_indicators(price_df)

    news_df = fetch_binance_news_from_newsapi(NEWSAPI_KEY)
    sentiment_df = pd.concat([news_df])
    geo_df = fetch_geopolitical_news(NEWSAPI_KEY, limit=30)

    price_df = merge_sentiment_to_price(price_df, sentiment_df, prefix='sentiment')
    price_df = merge_sentiment_to_price(price_df, geo_df, prefix='geo_sentiment')

    return price_df.dropna()


def train():
    df = load_data()
    env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

    model_path = MODEL_PATH

    if os.path.exists(f"{model_path}.zip"):
        print("📦 Loading existing model...")
        model = DQN.load(model_path, env=env)
        model.set_env(env)  # rebind env in case it's different
    else:
        print("🆕 Creating new model...")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=0.0003,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            tau=0.01,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.3,
            exploration_final_eps=0.01,
            target_update_interval=500,
            tensorboard_log="./logs/dqn"
        )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="dqn_model"
    )

    model.learn(
        total_timesteps=300_000,
        log_interval=10,
        callback=[eval_callback, checkpoint_callback]
    )

    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
