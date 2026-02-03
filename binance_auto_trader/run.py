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
    close_partial_position,
    get_entry_price
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
import uuid
import json
import os
from datetime import datetime

# Data directory for status files (mounted outside Docker)
DATA_DIR = os.environ.get('DATA_DIR', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

SIGNAL_STATUS_FILE = os.path.join(DATA_DIR, "signal_status.json")

def save_signal_status(signal, price, confidence, tp_price, sl_price, leverage, balance, position):
    """Save current signal and balance to status file for external monitoring."""
    status = {
        'timestamp': datetime.now().isoformat(),
        'signal': signal,
        'price': price,
        'confidence': confidence,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'leverage': leverage,
        'balance': balance,
        'position': position,
        'mode': 'LIVE'
    }
    with open(SIGNAL_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)
    log(f"[STATUS] Saved to {SIGNAL_STATUS_FILE}")

# Import continuous learning system
try:
    from continuous_learning import (
        get_manager as get_cl_manager,
        log_entry as cl_log_entry,
        log_exit as cl_log_exit,
        start_monitoring as cl_start_monitoring
    )
    HAS_CONTINUOUS_LEARNING = USE_CONTINUOUS_LEARNING
    print("[INFO] Continuous learning system loaded")
except ImportError as e:
    HAS_CONTINUOUS_LEARNING = False
    print(f"[WARN] Continuous learning not available: {e}")

PARTIAL_TP_1_DONE = False
PARTIAL_TP_2_DONE = False

# Track active trade for continuous learning
ACTIVE_TRADE_ID = None
ACTIVE_TRADE_ENTRY_PRICE = None

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def generate_trade_id():
    """Generate unique trade ID."""
    return f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def log_trade_entry(symbol, side, entry_price, qty, confidence):
    """Log trade entry for continuous learning."""
    global ACTIVE_TRADE_ID, ACTIVE_TRADE_ENTRY_PRICE
    if HAS_CONTINUOUS_LEARNING:
        trade_id = generate_trade_id()
        cl_log_entry(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            qty=qty,
            confidence=confidence,
            model_version=BEST_MODEL_PATH
        )
        ACTIVE_TRADE_ID = trade_id
        ACTIVE_TRADE_ENTRY_PRICE = entry_price
        return trade_id
    return None

def log_trade_exit(exit_price, exit_reason='UNKNOWN'):
    """Log trade exit for continuous learning."""
    global ACTIVE_TRADE_ID, ACTIVE_TRADE_ENTRY_PRICE
    if HAS_CONTINUOUS_LEARNING and ACTIVE_TRADE_ID:
        cl_log_exit(ACTIVE_TRADE_ID, exit_price, exit_reason)
        ACTIVE_TRADE_ID = None
        ACTIVE_TRADE_ENTRY_PRICE = None

def handle_close_position(reason, symbol, position, profit_ratio, reset_peak=True, reset_partial=True, wait_after=60):
    log(f"[CLOSE] Reason: {reason} | PnL: {profit_ratio*100:.2f}%")

    # Log exit for continuous learning
    current_price = get_realtime_price(symbol)
    if current_price:
        log_trade_exit(current_price, reason)

    close_position(symbol, position)
    if reset_peak:
        close_position.peak_profit = 0
    if reset_partial:
        global PARTIAL_TP_1_DONE, PARTIAL_TP_2_DONE
        PARTIAL_TP_1_DONE = False
        PARTIAL_TP_2_DONE = False
    time.sleep(wait_after)

log("[START] Running Binance Futures Auto-Trader with LSTM Strategy")

# ============================================================
# PAPER TRADING CHECK - Must complete before live trading!
# ============================================================
try:
    from config import PAPER_TRADING_MODE
    from paper_trading import is_live_trading_allowed, run_paper_trading

    if PAPER_TRADING_MODE:
        log("[PAPER] Paper trading mode is ENABLED")
        log("[PAPER] Running in simulation mode (no real money)")
        log("[PAPER] Complete 30 days of paper trading to unlock live trading")
        run_paper_trading()
        exit(0)  # Exit after paper trading
    else:
        # Check if paper trading was completed
        if not is_live_trading_allowed():
            log("[ERROR] Paper trading not completed!")
            log("[ERROR] You must complete 30 days of paper trading first.")
            log("[ERROR] Set PAPER_TRADING_MODE = True in config.py")
            log("[ERROR] Then run: python run.py")
            exit(1)
        else:
            log("[LIVE] Paper trading completed. Live trading enabled.")
except ImportError:
    log("[WARN] Paper trading module not found, proceeding with live trading")

# Start continuous learning monitoring in background
if HAS_CONTINUOUS_LEARNING:
    log("[CL] Starting continuous learning monitoring...")
    cl_start_monitoring()
    cl_manager = get_cl_manager()
    cl_manager.print_status()

consecutive_losses = 0
ENTRY_THRESHOLD_BASE = ENTRY_THRESHOLD
last_switch_time = 0

while True:
    try:
        df = fetch_features_multi_timeframe(use_cache=True)  # Use cache for live trading
        scaled_input = preprocess_for_lstm(df)

        price = get_realtime_price(SYMBOL)
        log(f"[DEBUG] Real-time price from Binance: {price}")

        atr_cols = [col for col in df.columns if "atr" in col]
        avg_atr = df[atr_cols].iloc[-1].mean()

        action, confidence = lstm_based_action(df)

        # Use dynamic EMA columns based on primary timeframe
        primary_tf = INTERVALS[0]
        ema_fast_col = f'ema_9_{primary_tf}'
        ema_slow_col = f'ema_20_{primary_tf}'
        ema_fast = df[ema_fast_col].iloc[-1] if ema_fast_col in df.columns else None
        ema_slow = df[ema_slow_col].iloc[-1] if ema_slow_col in df.columns else None
        trend = 'UP' if ema_fast and ema_slow and ema_fast > ema_slow else 'DOWN'

        base_range = avg_atr / price

        # Điều chỉnh TP/SL buffer để scale theo biến động
        TP_BUFFER = 1.15 if base_range > 0.015 else 1.25
        SL_BUFFER = 1.4 if base_range > 0.015 else 1.5

        # Lấy leverage dựa trên confidence (nếu cần dùng)
        init_leverage = select_leverage(confidence)
        leverage = round(init_leverage)

        # Tính TP khoảng 1.2%
        TAKE_PROFIT_RATIO = round(
            min(0.015, base_range * (0.6 + confidence * 0.4)) * TP_BUFFER, 4
        )

        # Tính SL đảm bảo ít nhất là 1.5%
        LOSS_CUTOFF_RATIO = round(
            max(0.015, min(0.03, base_range * (1.0 - confidence + 0.3)) * SL_BUFFER), 4
        )

        # Chốt lời từng phần
        PARTIAL_TP_1 = round(0.3 * TAKE_PROFIT_RATIO, 4)
        PARTIAL_TP_2 = round(0.7 * TAKE_PROFIT_RATIO, 4)

        print(f"base_range: {base_range} | TP: {TAKE_PROFIT_RATIO} | SL; {LOSS_CUTOFF_RATIO}")
        entry_price = get_entry_price(SYMBOL)
        current_position = get_current_position_side(SYMBOL)

        balance = get_balance()
        qty = calculate_qty(balance, price, leverage, RISK_PERCENT, SYMBOL)

        if current_position:
            pnl = get_unrealized_pnl(SYMBOL)
            profit_ratio = (price - entry_price) / entry_price if entry_price else 0

            log(f"[INFO] Current position: {current_position} | PnL: {pnl:.2f} | Balance: {balance:.2f} | Profit Ratio: {profit_ratio*100:.2f}%")

            if not hasattr(close_position, "peak_profit"):
                close_position.peak_profit = profit_ratio
            if profit_ratio > close_position.peak_profit:
                close_position.peak_profit = profit_ratio

            trailing_trigger = 0.02 + confidence * 0.015  # từ 2% đến 3.5%
            trailing_drawdown = 0.2 + (1 - confidence) * 0.3  # từ 20% đến 50%

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

        if consecutive_losses >= 2:
            if trend and ((trend == 'UP' and action != 'LONG') or (trend == 'DOWN' and action != 'SHORT')):
                log(f"[FILTER] Đang lỗ → chỉ đánh theo trend mạnh, bỏ lệnh ngược")
                continue

        # Calculate TP/SL prices for display
        if action == 'LONG':
            tp_price = price * (1 + TAKE_PROFIT_RATIO)
            sl_price = price * (1 - LOSS_CUTOFF_RATIO)
        elif action == 'SHORT':
            tp_price = price * (1 - TAKE_PROFIT_RATIO)
            sl_price = price * (1 + LOSS_CUTOFF_RATIO)
        else:
            tp_price = sl_price = price

        potential_profit = balance * RISK_PERCENT / 100 * leverage * TAKE_PROFIT_RATIO
        potential_loss = balance * RISK_PERCENT / 100 * leverage * LOSS_CUTOFF_RATIO

        current_pos = current_position if current_position else "NONE"
        log(f"\n=== SIGNAL: {action} ===")
        log(f"  Entry:  ${price:.2f}")
        log(f"  TP:     ${tp_price:.2f} (+{TAKE_PROFIT_RATIO*100:.2f}%) -> +${potential_profit:.2f}")
        log(f"  SL:     ${sl_price:.2f} (-{LOSS_CUTOFF_RATIO*100:.2f}%) -> -${potential_loss:.2f}")
        log(f"  Conf:   {confidence:.2f} | Leverage: {leverage}x | Balance: ${balance:.2f}")
        log(f"  Position: {current_pos}")

        # Save signal status for external monitoring
        save_signal_status(action, price, confidence, tp_price, sl_price, leverage, balance, current_pos)

        BREAKOUT_MARGIN_PERCENT = 0.01
        breakout_margin = price * BREAKOUT_MARGIN_PERCENT

        recent_high = df[f'high_{INTERVALS[0]}'].iloc[-2] if f'high_{INTERVALS[0]}' in df.columns else None
        recent_low = df[f'low_{INTERVALS[0]}'].iloc[-2] if f'low_{INTERVALS[0]}' in df.columns else None

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

            rsi_now = df[f'rsi_{INTERVALS[0]}'].iloc[-1] if f'rsi_{INTERVALS[0]}' in df.columns else None
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
                log(f"[AUTO-CLOSE] Closing LONG -> SHORT @ {confidence:.2f}")
                # Log exit before closing
                current_price = get_realtime_price(SYMBOL)
                if current_price:
                    log_trade_exit(current_price, 'SWITCH_TO_SHORT')
                close_position(SYMBOL, 'LONG')
                last_switch_time = time.time()
                action_to_take = 'SHORT'
            elif current_position == 'SHORT' and action == 'LONG' and confidence >= SWITCH_THRESHOLD:
                log(f"[AUTO-CLOSE] Closing SHORT -> LONG @ {confidence:.2f}")
                # Log exit before closing
                current_price = get_realtime_price(SYMBOL)
                if current_price:
                    log_trade_exit(current_price, 'SWITCH_TO_LONG')
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
                    log("[ERROR] Khong lay duoc entry price sau khi dat lenh!")
                    continue

                # Log trade entry for continuous learning
                trade_id = log_trade_entry(SYMBOL, action_to_take, entry_price, qty, confidence)
                if trade_id:
                    log(f"[CL] Trade logged: {trade_id}")

                sl, tp = place_sl_tp_order(SYMBOL, action_to_take, qty, entry_price, TAKE_PROFIT_RATIO, LOSS_CUTOFF_RATIO)
                log(f"[TRADE] {action_to_take} {qty} {SYMBOL} @ {entry_price:.4f} | Confidence: {confidence:.2f} | Leverage: {leverage}x | TP: {tp} | SL: {sl}")
            else:
                log("[INFO] Signal not strong enough to act.")
        else:
            log("[INFO] No trade signal.")

    except Exception as e:
        log(f"[ERROR] {e}")

    # Sleep based on candle interval (no need to check every minute for 1h candles)
    INTERVAL_SLEEP = {
        '1m': 60,      # 1 minute
        '5m': 180,     # 3 minutes
        '15m': 300,    # 5 minutes
        '30m': 600,    # 10 minutes
        '1h': 900,     # 15 minutes
        '4h': 1800,    # 30 minutes
        '1d': 3600,    # 1 hour
    }
    sleep_time = INTERVAL_SLEEP.get(INTERVALS[0], 300)
    time.sleep(sleep_time)
