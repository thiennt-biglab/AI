from config import *
from trader import get_latest_klines, get_balance, calculate_qty, place_market_order, place_sl_tp_order, cancel_open_orders, get_current_position_side
from feature_pipeline import fetch_features_multi_timeframe  # <=== Sử dụng lại
from strategy_lstm_live import lstm_based_action, select_leverage
import time
import numpy as np

print("[START] Running Binance Futures Auto-Trader with LSTM Strategy")

while True:
    try:
        df = fetch_features_multi_timeframe()
        action, confidence = lstm_based_action(df)

        if action != 'HOLD':
            price = df.filter(like='ema_20_').iloc[-1].values[0]
            balance = get_balance()
            init_leverage = select_leverage(confidence)
            leverage = init_leverage
            qty = calculate_qty(balance, price, leverage, RISK_PERCENT)
            
            if qty <= 0 or qty is None or np.isnan(qty):
                print(f"[WARN] Invalid qty: {qty}, skipping trade.")
                continue

            notional = qty * price
            if notional < 5:
                # Tăng leverage để đạt đúng 5 USDT
                min_leverage = int(np.ceil(5 / ((balance * RISK_PERCENT / 100) / price)))
                leverage = max(leverage, min_leverage)
                qty = calculate_qty(balance, price, leverage, RISK_PERCENT)
                notional = qty * price
                if notional < 5:
                    print(f"[WARN] Even after leverage adjust, notional still < 5 USDT → skip")
                    continue

            print(f"[WARN] Leverage main {leverage} min_leverage {min_leverage}")

            # Lấy vị thế hiện tại
            current_position = get_current_position_side(SYMBOL)

            # Nếu có vị thế đang giữ và tín hiệu đảo chiều mạnh
            if current_position == 'LONG' and action == 'SHORT' and confidence > 0.8:
                print(f"[AUTO-CLOSE] Closing LONG due to SHORT signal @ {confidence:.2f}")
                close_position(SYMBOL, 'LONG')
                continue

            if current_position == 'SHORT' and action == 'LONG' and confidence > 0.8:
                print(f"[AUTO-CLOSE] Closing SHORT due to LONG signal @ {confidence:.2f}")
                close_position(SYMBOL, 'SHORT')
                continue

            
            place_market_order(SYMBOL, action, qty, leverage)
            
            position_side = 'LONG' if action == 'LONG' else 'SHORT'
            cancel_open_orders(SYMBOL, position_side)
            place_sl_tp_order(SYMBOL, action, qty, price)

            print(f"[TRADE] {action} {qty} {SYMBOL} @ {price:.2f} | confidence: {confidence:.2f} | leverage: {leverage}x")
        else:
            print("[INFO] No trade signal.")
    except Exception as e:
        print(f"[ERROR] {e}")
    time.sleep(60)
