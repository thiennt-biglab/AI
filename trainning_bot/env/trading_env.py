# env/trading_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.coin = 0
        self.current_step = 0
        self.last_buy_price = 0
        self.last_net_worth = initial_balance

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.coin = 0
        self.current_step = 0
        self.last_buy_price = 0
        self.last_net_worth = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.loc[self.current_step]
        unrealized_pnl = 0
        if self.coin > 0 and self.last_buy_price > 0:
            unrealized_pnl = ((row['close'] - self.last_buy_price) / self.last_buy_price) * 100

        return np.array([
            row['close'],
            row['volume'],
            row['rsi'],
            row['ema'],
            row['macd'],
            row['sentiment'],
            row['geo_sentiment'],
            self.coin,
            unrealized_pnl
        ], dtype=np.float32)

    def step(self, action):
        row = self.df.loc[self.current_step]
        price = row['close']

        if action == 1 and self.balance > 0 and self.coin == 0:
            self.coin = self.balance / price
            self.last_buy_price = price
            self.balance = 0

        elif action == 2 and self.coin > 0:
            self.balance = self.coin * price
            self.coin = 0
            self.last_buy_price = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False

        net_worth = self.balance + self.coin * price
        reward = (net_worth - self.last_net_worth) / self.last_net_worth
        self.last_net_worth = net_worth

        obs = self._get_observation()
        info = {'net_worth': net_worth}

        return obs, reward, done, truncated, info
