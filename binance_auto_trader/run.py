from config import *
from trader import (
    get_balance, calculate_qty, place_market_order,
    place_sl_tp_order, cancel_open_orders, get_current_position_side,
    close_position, get_unrealized_pnl
)
from feature_pipeline import fetch_features_multi_timeframe
from strategy_lstm_live import (
    lstm_based_action, select_leverage,
    compute_min_leverage, preprocess_for_lstm
)
import time
import numpy as np
from tensorflow.keras.models import load_model
from train_lstm_keras import focal_loss

COOLDOWN_SECONDS = 60
DYNAMIC_TP_MIN = 0.02
DYNAMIC_TP_MAX = 0.12
DYNAMIC_SL_MIN = 0.015
DYNAMIC_SL_MAX = 0.06

import json
import os

STATE_FILE = "state.json"
MAX_CONSECUTIVE_LOSSES = 3  # Bạn có thể tùy chỉnh
SUSPEND_SECONDS = 1800  # Ngừng bot 1 giờ nếu vượt giới hạn

# Load trạng thái trước đó
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"consecutive_losses": 0}

# Lưu trạng thái mới
def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def is_trend_confirmed(df, current_action):
    ema = df.filter(like='ema_20_').iloc[-3:]['ema_20_1m'].values  # Giả sử EMA 1m
    if len(ema) < 3:
        return False
    if current_action == 'LONG':
        return ema[-1] > ema[-2] > ema[-3]
    elif current_action == 'SHORT':
        return ema[-1] < ema[-2] < ema[-3]
    return False

print("[START] Running Binance Futures Auto-Trader with LSTM Strategy")

focal = focal_loss(gamma=1.0, alpha=0.5)
model = load_model(BEST_MODEL_FILE, custom_objects={'loss': focal})

while True:
    try:
        df = fetch_features_multi_timeframe()
        scaled_input = preprocess_for_lstm(df)

        print("\n=== Scaled Input Shape ===")
        print(scaled_input.shape)

        print("\n=== NaN / Inf check ===")
        print("Any NaN:", np.isnan(scaled_input).any())
        print("Any Inf:", np.isinf(scaled_input).any())


        price = df.filter(like='ema_20_').iloc[-1].values[0]
        atr_cols = [col for col in df.columns if "atr" in col]
        avg_atr = df[atr_cols].iloc[-1].mean()
        volatility_ratio = avg_atr / price

        action, confidence = lstm_based_action(df)

        proba = model.predict(scaled_input, verbose=0)
        print("\n=== Prediction Probabilities ===")
        print(proba)

        TAKE_PROFIT_RATIO = max(
            DYNAMIC_TP_MIN,
            min(DYNAMIC_TP_MAX, 0.03 + confidence * 0.03 + volatility_ratio * 0.4)
        )
        LOSS_CUTOFF_RATIO = min(
            DYNAMIC_SL_MAX,
            max(DYNAMIC_SL_MIN, 0.02 + (1 - confidence) * 0.03 + volatility_ratio * 0.3)
        )

        current_position = get_current_position_side(SYMBOL)

        leverage = round(select_leverage(confidence))

        if current_position:
            pnl = get_unrealized_pnl(SYMBOL)
            balance = get_balance()
            profit_ratio = (pnl * leverage) / balance if balance else 0

            print(f"[INFO] Current: {current_position} | PnL: {pnl:.2f} | Profit: {profit_ratio*100:.2f}%")
            state = load_state()

            if profit_ratio >= TAKE_PROFIT_RATIO:
                close_position(SYMBOL, current_position)
                print(f"[AUTO-TP] Closed {current_position} | +{profit_ratio*100:.2f}%")
                state["consecutive_losses"] = 0
                save_state(state)
                time.sleep(COOLDOWN_SECONDS)
                continue

            elif profit_ratio <= -LOSS_CUTOFF_RATIO:
                close_position(SYMBOL, current_position)
                print(f"[AUTO-SL] Closed {current_position} | -{profit_ratio*100:.2f}%")
                state["consecutive_losses"] += 1
                save_state(state)

                if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
                    print(f"[CRITICAL] {state['consecutive_losses']} consecutive losses → pausing for safety.")
                    time.sleep(SUSPEND_SECONDS)
                    state["consecutive_losses"] = 0
                    save_state(state)
                else:
                    time.sleep(COOLDOWN_SECONDS)
                continue

        print(f"[INFO] Action: {action} | Confidence: {confidence:.2f}")
        if action != 'HOLD':
            balance = get_balance()
            qty, leverage = calculate_qty(balance, price, leverage, RISK_PERCENT, SYMBOL)

            if isinstance(qty, (np.ndarray, list)):
                qty = qty[0]

            if qty is None or not np.isscalar(qty) or np.isnan(qty) or qty <= 0:
                print(f"[WARN] Invalid qty: {qty}, skipping trade.")
                continue

            notional = qty * price
            if notional < 5:
                leverage = max(leverage, compute_min_leverage(balance, price, RISK_PERCENT))
                qty = calculate_qty(balance, price, leverage, RISK_PERCENT)
                notional = qty * price
                if notional < 5:
                    print(f"[SKIP] Notional {notional:.2f} < 5 USDT")
                    continue
                print(f"[ADJUST] Increased leverage → {leverage}x")

            action_to_take = 'HOLD'
            trend_confirmed = is_trend_confirmed(df, action)

            if current_position == 'LONG' and action == 'SHORT' and confidence >= SWITCH_THRESHOLD:
                pnl = get_unrealized_pnl(SYMBOL)
                if pnl > 0 or trend_confirmed:
                    close_position(SYMBOL, 'LONG')
                    action_to_take = 'SHORT'
                else:
                    print("[SKIP] Avoided forced SHORT — waiting for trend confirmation or profit.")

            elif current_position == 'SHORT' and action == 'LONG' and confidence >= SWITCH_THRESHOLD:
                pnl = get_unrealized_pnl(SYMBOL)
                if pnl > 0 or trend_confirmed:
                    close_position(SYMBOL, 'SHORT')
                    action_to_take = 'LONG'
                else:
                    print("[SKIP] Avoided forced LONG — waiting for trend confirmation or profit.")

            elif not current_position and confidence >= ENTRY_THRESHOLD:
                action_to_take = action

            if action_to_take != 'HOLD':
                place_market_order(SYMBOL, action_to_take, qty, leverage)
                cancel_open_orders(SYMBOL, action_to_take)
                place_sl_tp_order(SYMBOL, action_to_take, qty, price)
                print(f"[TRADE] {action_to_take} {qty} {SYMBOL} @ {price:.2f} | Lev: {leverage}x")
            else:
                print(f"[INFO] Signal not strong enough to open new position.")
        else:
            print(f"[INFO] HOLD – No trade signal.")

    except Exception as e:
        print(f"[ERROR] {e}")

    time.sleep(COOLDOWN_SECONDS)
