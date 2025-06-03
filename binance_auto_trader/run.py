from config import *
from trader import (
    get_balance,
    calculate_qty,
    place_market_order,
    place_sl_tp_order,
    cancel_open_orders,
    get_current_position_side,
    close_position,
    get_unrealized_pnl,
    get_realtime_price,
    get_open_position_qty,
    close_partial_position
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
from datetime import datetime

PARTIAL_TP_1_DONE = False
PARTIAL_TP_2_DONE = False

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def handle_close_position(reason, symbol, position, profit_ratio, reset_peak=True, reset_partial=True, wait_after=60):
    log(f"[CLOSE] Reason: {reason} | PnL: {profit_ratio*100:.2f}%")
    close_position(symbol, position)
    if reset_peak:
        close_position.peak_profit = 0
    if reset_partial:
        global PARTIAL_TP_1_DONE, PARTIAL_TP_2_DONE
        PARTIAL_TP_1_DONE = False
        PARTIAL_TP_2_DONE = False
    time.sleep(wait_after)

log("[START] Running Binance Futures Auto-Trader with LSTM Strategy")
consecutive_losses = 0
ENTRY_THRESHOLD_BASE = ENTRY_THRESHOLD
last_switch_time = 0

while True:
    try:
        df = fetch_features_multi_timeframe()
        scaled_input = preprocess_for_lstm(df)

        price = get_realtime_price(SYMBOL)
        log(f"[DEBUG] Real-time price from Binance: {price}")

        atr_cols = [col for col in df.columns if "atr" in col]
        avg_atr = df[atr_cols].iloc[-1].mean()

        action, confidence = lstm_based_action(df)

        ema_fast = df['ema_10m'].iloc[-1] if 'ema_10m' in df.columns else None
        ema_slow = df['ema_50m'].iloc[-1] if 'ema_50m' in df.columns else None
        trend = 'UP' if ema_fast and ema_slow and ema_fast > ema_slow else 'DOWN'

        base_range = avg_atr / price

        TAKE_PROFIT_RATIO = round(min(0.05, max(0.015, base_range * (1.0 + confidence))) * TP_BUFFER, 4)
        LOSS_CUTOFF_RATIO = round(min(0.03, max(0.006, base_range * (1.0 - confidence + 0.2))) * SL_BUFFER, 4)

        PARTIAL_TP_1 = round(0.25 * TAKE_PROFIT_RATIO, 4)
        PARTIAL_TP_2 = round(0.6 * TAKE_PROFIT_RATIO, 4)

        current_position = get_current_position_side(SYMBOL)
        if current_position:
            pnl = get_unrealized_pnl(SYMBOL)
            balance = get_balance()
            profit_ratio = pnl / balance if balance else 0
            log(f"[INFO] Current position: {current_position} | PnL: {pnl:.2f} | Balance: {balance:.2f} | Profit Ratio: {profit_ratio*100:.2f}%")

            if not hasattr(close_position, "peak_profit"):
                close_position.peak_profit = profit_ratio
            if profit_ratio > close_position.peak_profit:
                close_position.peak_profit = profit_ratio

            trailing_trigger = 0.03
            trailing_drawdown = 0.5

            if close_position.peak_profit >= trailing_trigger:
                stop_threshold = close_position.peak_profit * (1 - trailing_drawdown)
                if profit_ratio <= stop_threshold:
                    handle_close_position("TRAILING-STOP", SYMBOL, current_position, profit_ratio)
                    continue

            qty_open = get_open_position_qty(SYMBOL, current_position)
            if not PARTIAL_TP_1_DONE and profit_ratio >= PARTIAL_TP_1:
                close_partial_position(SYMBOL, current_position, qty_open * 0.3)
                PARTIAL_TP_1_DONE = True
            if not PARTIAL_TP_2_DONE and profit_ratio >= PARTIAL_TP_2:
                close_partial_position(SYMBOL, current_position, qty_open * 0.5)
                PARTIAL_TP_2_DONE = True

            if profit_ratio >= TAKE_PROFIT_RATIO:
                handle_close_position("AUTO-PROFIT", SYMBOL, current_position, profit_ratio)
                consecutive_losses = 0
                ENTRY_THRESHOLD = ENTRY_THRESHOLD_BASE
                continue

            elif profit_ratio <= -LOSS_CUTOFF_RATIO:
                handle_close_position("AUTO-STOP", SYMBOL, current_position, profit_ratio)
                consecutive_losses += 1
                ENTRY_THRESHOLD = min(0.95, ENTRY_THRESHOLD_BASE + consecutive_losses * 0.01)
                continue
            else:
                log(f"[INFO] Profit {profit_ratio*100:.2f}% (TP {TAKE_PROFIT_RATIO*100:.2f}%, SL {LOSS_CUTOFF_RATIO*100:.2f}%) → Hold")

        if consecutive_losses >= 3:
            if trend and ((trend == 'UP' and action != 'LONG') or (trend == 'DOWN' and action != 'SHORT')):
                log(f"[FILTER] Đang lỗ → chỉ đánh theo trend mạnh, bỏ lệnh ngược")
                continue

        log(f"[INFO] Action: {action} | Confidence: {confidence:.2f} | Price: {price:.4f}")

        BREAKOUT_MARGIN_PERCENT = 0.01
        breakout_margin = price * BREAKOUT_MARGIN_PERCENT

        recent_high = df['high_15m'].iloc[-2] if 'high_15m' in df.columns else None
        recent_low = df['low_15m'].iloc[-2] if 'low_15m' in df.columns else None

        breakout_boost = min(0.06, max(0.02, avg_atr / price * 10))
        if action == 'LONG' and recent_high and price > recent_high + breakout_margin and confidence >= ENTRY_THRESHOLD:
            confidence += breakout_boost
            log(f"[BREAKOUT] LONG breakout → Boost confidence lên {confidence:.2f}")
        elif action == 'SHORT' and recent_low and price < recent_low - breakout_margin and confidence >= ENTRY_THRESHOLD:
            confidence += breakout_boost
            log(f"[BREAKOUT] SHORT breakout → Boost confidence lên {confidence:.2f}")

        if action != 'HOLD':
            if confidence < ENTRY_THRESHOLD:
                log(f"[FILTER] Confidence {confidence:.2f} < ENTRY_THRESHOLD → Bỏ qua tín hiệu")
                continue

            action_prev, _ = lstm_based_action(df[:-1])
            if action != action_prev and abs(confidence - ENTRY_THRESHOLD) < 0.02:
                log(f"[FILTER] Hành vi không ổn định + confidence sát ngưỡng → Bỏ")
                continue

            rsi_now = df['rsi_5m'].iloc[-1] if 'rsi_5m' in df.columns else None
            if rsi_now:
                if (action == 'LONG' and rsi_now < 50):
                    confidence -= 0.02
                    log(f"[RSI] LONG nhưng RSI thấp ({rsi_now:.2f}) → giảm confidence: {confidence:.2f}")
                elif (action == 'SHORT' and rsi_now > 40):
                    confidence -= 0.02
                    log(f"[RSI] SHORT nhưng RSI cao ({rsi_now:.2f}) → giảm confidence: {confidence:.2f}")

            if trend:
                if action == 'LONG' and trend != 'UP':
                    log(f"[FILTER] Trend đang DOWN, không nên vào LONG")
                    continue
                if action == 'SHORT' and trend != 'DOWN':
                    log(f"[FILTER] Trend đang UP, không nên vào SHORT")
                    continue

            if avg_atr / price < 0.003:
                log(f"[FILTER] ATR/Price = {avg_atr/price:.4f} < 0.003 → Thị trường quá tĩnh → Bỏ")
                continue
            elif 0.003 <= avg_atr / price <= 0.006:
                log(f"[FILTER] ATR/Price = {avg_atr/price:.4f} → Sideway nguy hiểm, bỏ qua")
                continue

            balance = get_balance()
            init_leverage = select_leverage(confidence)
            leverage = round(init_leverage)
            qty = calculate_qty(balance, price, leverage, RISK_PERCENT, SYMBOL)

            if qty <= 0 or qty is None or np.isnan(qty):
                log(f"[WARN] Invalid qty: {qty}, skipping trade.")
                continue

            notional = qty * price
            if notional < 5:
                min_leverage = compute_min_leverage(balance, price, RISK_PERCENT)
                leverage = max(leverage, min_leverage)
                qty = calculate_qty(balance, price, leverage, RISK_PERCENT)
                notional = qty * price
                if notional < 5:
                    log(f"[WARN] Even after leverage adjust, notional still < 5 USDT → skip")
                    continue
                log(f"[ADJUST] Leverage changed to {leverage} to meet notional ${notional:.2f}")
            else:
                min_leverage = compute_min_leverage(balance, price, RISK_PERCENT)
                log(f"[INFO] Leverage: {leverage} | Required min: {min_leverage}")

            current_position = get_current_position_side(SYMBOL)
            action_to_take = 'HOLD'

            if (current_position in ['LONG', 'SHORT']) and (action != current_position) and confidence >= SWITCH_THRESHOLD:
                if time.time() - last_switch_time < 180:
                    log("[SKIP] Vừa switch gần đây → chờ thêm")
                    continue

            if current_position:
                pnl = get_unrealized_pnl(SYMBOL)
                balance = get_balance()
                profit_ratio = pnl / balance if balance else 0

                if not hasattr(close_position, "peak_profit"):
                    close_position.peak_profit = profit_ratio
                if profit_ratio > close_position.peak_profit:
                    close_position.peak_profit = profit_ratio

                if close_position.peak_profit >= trailing_trigger:
                    stop_threshold = close_position.peak_profit * (1 - trailing_drawdown)
                    if profit_ratio <= stop_threshold:
                        handle_close_position("TRAILING-SWITCH", SYMBOL, current_position, profit_ratio)
                        continue

            if current_position == 'LONG' and action == 'SHORT' and confidence >= SWITCH_THRESHOLD:
                log(f"[AUTO-CLOSE] Closing LONG → SHORT @ {confidence:.2f}")
                close_position(SYMBOL, 'LONG')
                last_switch_time = time.time()
                action_to_take = 'SHORT'
            elif current_position == 'SHORT' and action == 'LONG' and confidence >= SWITCH_THRESHOLD:
                log(f"[AUTO-CLOSE] Closing SHORT → LONG @ {confidence:.2f}")
                close_position(SYMBOL, 'SHORT')
                last_switch_time = time.time()
                action_to_take = 'LONG'
            elif current_position is None and confidence >= ENTRY_THRESHOLD:
                action_to_take = action

            if action_to_take != 'HOLD':
                should_cancel = (
                        current_position and
                        (action_to_take != current_position) and
                        confidence >= SWITCH_THRESHOLD
                )
                if should_cancel:
                    cancel_open_orders(SYMBOL, current_position)

                order, entry_price = place_market_order(SYMBOL, action_to_take, qty, leverage)
                if not entry_price:
                    log("[ERROR] Không lấy được entry price sau khi đặt lệnh!")
                    continue

                sl, tp = place_sl_tp_order(SYMBOL, action_to_take, qty, entry_price, TAKE_PROFIT_RATIO, LOSS_CUTOFF_RATIO)
                log(f"[TRADE] {action_to_take} {qty} {SYMBOL} @ {entry_price:.4f} | Confidence: {confidence:.2f} | Leverage: {leverage}x | TP: {tp} | SL: {sl}")
            else:
                log("[INFO] Signal not strong enough to act.")
        else:
            log("[INFO] No trade signal.")

    except Exception as e:
        log(f"[ERROR] {e}")

    time.sleep(60)
