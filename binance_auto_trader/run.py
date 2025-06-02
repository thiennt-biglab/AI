from config import *
from trader import (
    get_latest_klines,
    get_balance,
    calculate_qty,
    place_market_order,
    place_sl_tp_order,
    cancel_open_orders,
    get_current_position_side,
    close_position,
    get_unrealized_pnl
)
from feature_pipeline import fetch_features_multi_timeframe
from strategy_lstm_live import (
    lstm_based_action,
    select_leverage,
    compute_min_leverage,
    preprocess_for_lstm
)
import time
import numpy as np
import traceback

print("[START] Running Binance Futures Auto-Trader with LSTM Strategy")

while True:
    try:
        df = fetch_features_multi_timeframe()
        scaled_input = preprocess_for_lstm(df)

        try:
            klines = get_latest_klines(SYMBOL, interval='1m', limit=1)
            print(f"[DEBUG] Latest kline: {klines}")
            price = float(klines.iloc[0]['close'])
        except Exception as e:
            print(f"[ERROR] When fetching price: {e.__class__.__name__}: {e}")
            traceback.print_exc()


        atr_cols = [col for col in df.columns if "atr" in col]
        avg_atr = df[atr_cols].iloc[-1].mean()

        action, confidence = lstm_based_action(df)

        TAKE_PROFIT_RATIO = max(0.02, min(0.1, confidence * 0.04 + avg_atr / price))
        LOSS_CUTOFF_RATIO = min(0.05, max(0.01, (1 - confidence) * 0.04 + avg_atr / price))

        current_position = get_current_position_side(SYMBOL)
        if current_position:
            pnl = get_unrealized_pnl(SYMBOL)
            balance = get_balance()
            profit_ratio = pnl / balance if balance else 0

            print(f"[INFO] Current position: {current_position} | PnL: {pnl:.2f} | Balance: {balance:.2f} | Profit Ratio: {profit_ratio*100:.2f}%")

            if not hasattr(close_position, "peak_profit"):
                close_position.peak_profit = profit_ratio
            if profit_ratio > close_position.peak_profit:
                close_position.peak_profit = profit_ratio

            trailing_trigger = 0.03
            trailing_drawdown = 0.5

            if close_position.peak_profit >= trailing_trigger:
                stop_threshold = close_position.peak_profit * (1 - trailing_drawdown)
                if profit_ratio <= stop_threshold:
                    print(f"[TRAILING-STOP] Profit dropped from {close_position.peak_profit*100:.2f}% to {profit_ratio*100:.2f}% → Closing")
                    close_position(SYMBOL, current_position)
                    close_position.peak_profit = 0
                    time.sleep(60)
                    continue

            if profit_ratio >= TAKE_PROFIT_RATIO:
                close_position(SYMBOL, current_position)
                close_position.peak_profit = 0
                print(f"[AUTO-PROFIT] Closed {current_position} with profit {profit_ratio*100:.2f}% (TP {TAKE_PROFIT_RATIO*100:.2f}%)")
                time.sleep(60)
                continue
            elif profit_ratio <= -LOSS_CUTOFF_RATIO:
                close_position(SYMBOL, current_position)
                close_position.peak_profit = 0
                print(f"[AUTO-STOP] Closed {current_position} with loss {profit_ratio*100:.2f}% (SL {LOSS_CUTOFF_RATIO*100:.2f}%)")
                time.sleep(60)
                continue
            else:
                print(f"[INFO] Profit {profit_ratio*100:.2f}% (TP {TAKE_PROFIT_RATIO*100:.2f}%, SL {LOSS_CUTOFF_RATIO*100:.2f}%) → Hold")

        print(f"[INFO] Action: {action} | Confidence: {confidence:.2f} | Price: {price:.4f}")
        if action != 'HOLD':
            balance = get_balance()
            init_leverage = select_leverage(confidence)
            leverage = round(init_leverage)
            qty = calculate_qty(balance, price, leverage, RISK_PERCENT, SYMBOL)

            if qty <= 0 or qty is None or np.isnan(qty):
                print(f"[WARN] Invalid qty: {qty}, skipping trade.")
                continue

            notional = qty * price
            if notional < 5:
                min_leverage = compute_min_leverage(balance, price, RISK_PERCENT)
                leverage = max(leverage, min_leverage)
                qty = calculate_qty(balance, price, leverage, RISK_PERCENT)
                notional = qty * price
                if notional < 5:
                    print(f"[WARN] Even after leverage adjust, notional still < 5 USDT → skip")
                    continue
                print(f"[ADJUST] Leverage changed to {leverage} to meet notional ${notional:.2f}")
            else:
                min_leverage = compute_min_leverage(balance, price, RISK_PERCENT)
                print(f"[INFO] Leverage: {leverage} | Required min: {min_leverage}")

            current_position = get_current_position_side(SYMBOL)
            action_to_take = 'HOLD'

            print(f"[INFO] Current: {current_position} | Action: {action} | Confidence: {confidence:.2f}")

            if current_position == 'LONG' and action == 'SHORT' and confidence >= SWITCH_THRESHOLD:
                print(f"[AUTO-CLOSE] Closing LONG → SHORT @ {confidence:.2f}")
                close_position(SYMBOL, 'LONG')
                action_to_take = 'SHORT'
            elif current_position == 'SHORT' and action == 'LONG' and confidence >= SWITCH_THRESHOLD:
                print(f"[AUTO-CLOSE] Closing SHORT → LONG @ {confidence:.2f}")
                close_position(SYMBOL, 'SHORT')
                action_to_take = 'LONG'
            elif current_position is None and confidence >= ENTRY_THRESHOLD:
                action_to_take = action

            if action_to_take != 'HOLD':
                place_market_order(SYMBOL, action_to_take, qty, leverage)
                cancel_open_orders(SYMBOL, action_to_take)
                place_sl_tp_order(SYMBOL, action_to_take, qty, price)
                print(f"[TRADE] {action_to_take} {qty} {SYMBOL} @ {price:.4f} | Confidence: {confidence:.2f} | Leverage: {leverage}x")
            else:
                print("[INFO] Signal not strong enough to act.")
        else:
            print("[INFO] No trade signal.")

    except Exception as e:
        print(f"[ERROR] {e}")

    time.sleep(60)
