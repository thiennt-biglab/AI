# env/trading_env.py
import gym
import numpy as np
from gym import spaces

class TradingEnvContinuous(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.coin = 0
        self.current_step = 0
        self.last_net_worth = initial_balance
        self.last_buy_price = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.coin = 0
        self.current_step = 0
        self.last_net_worth = self.initial_balance
        self.last_buy_price = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.loc[self.current_step]
        price = row['close']
        self.unrealized_pnl = ((price - self.last_buy_price) / self.last_buy_price) * 100 if self.coin > 0 and self.last_buy_price > 0 else 0

        return np.array([
            row['close'], row['volume'], row['rsi'], row['ema'], row['macd'],
            row['sentiment'], row['geo_sentiment'], self.coin, self.unrealized_pnl
        ], dtype=np.float32)

    def step(self, action):
        action = float(np.clip(action[0], -1.0, 1.0))
        row = self.df.loc[self.current_step]
        price = row['close']
        fee_rate = 0.001

        if action > 0.1 and self.balance > 0:
            amount = (self.balance * action) / price
            cost = amount * price * (1 + fee_rate)
            if cost <= self.balance:
                self.coin += amount
                self.last_buy_price = price
                self.balance -= cost

        elif action < -0.1 and self.coin > 0:
            amount = self.coin * abs(action)
            self.coin -= amount
            self.balance += amount * price * (1 - fee_rate)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        net_worth = self.balance + self.coin * price
        reward = np.clip((net_worth - self.last_net_worth) / self.last_net_worth + 0.1 * self.unrealized_pnl, -1, 1)
        self.last_net_worth = net_worth

        obs = self._get_observation()
        info = {'net_worth': net_worth}
        return obs, reward, done, False, info