#trader.py
from binance.client import Client
from binance.enums import *
from config import *
import numpy as np
import time

client = Client(API_KEY, API_SECRET)

def safe_api_call(api_func, *args, signed=False, **kwargs):
    for attempt in range(3):
        try:
            if signed:
                # Do NOT add timestamp manually. Let the Client sign it.
                pass
            result = api_func(*args, **kwargs)
            if isinstance(result, str) and "<html" in result.lower():
                raise ValueError("Received HTML instead of JSON")
            return result
        except Exception as e:
            print(f"[WARN] API call failed ({attempt+1}/3): {e}")
            time.sleep(2)
    print("[ERROR] All retries failed.")
    return None


def get_realtime_price(symbol):
    try:
        ticker = safe_api_call(client.futures_symbol_ticker, symbol=symbol)
        return float(ticker['price']) if ticker else None
    except Exception as e:
        print(f"[ERROR] Failed to get realtime price: {e}")
        return None

def get_step_size(symbol):
    exchange_info = safe_api_call(client.futures_exchange_info)
    if not exchange_info:
        return 0.001
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['stepSize'])
    return 0.001

def round_step_size(quantity, step_size):
    try:
        precision = int(round(-np.log10(step_size)))
        return round(quantity, precision)
    except:
        return 0.0

def get_qty_limits(symbol):
    exchange_info = safe_api_call(client.futures_exchange_info)
    if not exchange_info:
        return {"minQty": 0.001, "maxQty": 999999.0, "stepSize": 0.001}
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return {
                        "minQty": float(f['minQty']),
                        "maxQty": float(f['maxQty']),
                        "stepSize": float(f['stepSize'])
                    }
    return {"minQty": 0.001, "maxQty": 999999.0, "stepSize": 0.001}

def calculate_qty(balance, price, leverage, risk_percent, symbol=SYMBOL):
    if price <= 0 or balance <= 0 or leverage <= 0:
        print(f"[ERROR] Tham số không hợp lệ: balance={balance}, price={price}, leverage={leverage}")
        return 0.0

    capital = balance * (risk_percent / 100)
    limits = get_qty_limits(symbol)
    for attempt in range(5):
        raw_qty = (capital * leverage) / price
        qty = round_step_size(raw_qty, limits['stepSize'])
        if qty > limits['maxQty']:
            print(f"[WARN] Qty {qty} > maxQty {limits['maxQty']} → giảm leverage (hiện tại: {leverage}x)")
            leverage -= 1
            if leverage < 1:
                print(f"[ERROR] Leverage đã xuống dưới 1x → dừng.")
                return 0.0
        elif qty < limits['minQty']:
            print(f"[WARN] Qty {qty} < minQty {limits['minQty']} → tăng leverage nhẹ")
            leverage += 1
        else:
            return qty
    print("[ERROR] Không thể tìm được leverage phù hợp để có qty hợp lệ.")
    return 0.0

def get_entry_price(symbol):
    positions = client.futures_position_information(symbol=symbol)
    for pos in positions:
        if float(pos["positionAmt"]) != 0:
            return float(pos["entryPrice"])
    return None


def get_open_position_qty(symbol, side):
    positions = safe_api_call(client.futures_position_information, symbol=symbol, signed=True)
    if not positions:
        return 0.0
    side_upper = side.upper()
    for pos in positions:
        try:
            if pos['symbol'] == symbol and pos['positionSide'] == side_upper:
                return abs(float(pos['positionAmt']))
        except Exception as e:
            print(f"[WARN] Lỗi khi đọc vị trí: {e}")
    return 0.0

def close_partial_position(symbol, side, qty):
    if qty <= 0:
        print(f"[INFO] Invalid partial qty ({qty}) for closing {side}")
        return
    order = safe_api_call(
        client.futures_create_order,
        symbol=symbol,
        side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=round(qty, 3),
        reduceOnly=True,
        positionSide=side
    )
    if order:
        print(f"[PARTIAL-CLOSE] Closed {qty} {symbol} from {side} position")
    else:
        print(f"[ERROR] Failed to partially close {side} position of {qty}")

def close_position(symbol, side):
    qty = get_open_position_qty(symbol, side)
    if qty <= 0:
        print(f"[INFO] No open {side} position to close.")
        return
    order = safe_api_call(
        client.futures_create_order,
        symbol=symbol,
        side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=qty,
        positionSide=side
    )
    if order:
        print(f"[CLOSE] Closed {side} position of qty {qty}")
    else:
        print(f"[ERROR] Failed to close position {side}")

def get_balance(asset="USDT"):
    balance = safe_api_call(client.futures_account_balance, signed=True)
    if not balance:
        return 0.0
    for b in balance:
        if b['asset'] == asset:
            return float(b['balance'])
    return 0.0

def get_unrealized_pnl(symbol):
    positions = safe_api_call(client.futures_position_information, symbol=symbol, signed=True)
    for pos in positions:
        if float(pos['positionAmt']) != 0:
            return float(pos['unRealizedProfit'])
    return 0.0

def cancel_open_orders(symbol, position_side='BOTH'):
    open_orders = safe_api_call(client.futures_get_open_orders, symbol=symbol)
    if not open_orders:
        print(f"[WARN] Không thể lấy danh sách lệnh để huỷ.")
        return

    count = 0
    for o in open_orders:
        o_pos_side = o.get('positionSide', 'BOTH')
        if o['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and (position_side == 'BOTH' or o_pos_side == position_side):
            result = safe_api_call(client.futures_cancel_order, symbol=symbol, orderId=o['orderId'])
            if result:
                print(f"[INFO] Đã huỷ lệnh {o['type']} - ID {o['orderId']} - Side {o_pos_side}")
                count += 1

    print(f"[INFO] Tổng số lệnh STOP/TP đã huỷ cho {position_side}: {count}")


def get_current_position_side(symbol):
    positions = safe_api_call(client.futures_position_information, symbol=symbol, signed=True)
    if not positions:
        return None
    for pos in positions:
        if pos['symbol'] == symbol:
            amt = float(pos['positionAmt'])
            if amt > 0:
                return 'LONG'
            elif amt < 0:
                return 'SHORT'
    return None

def place_market_order(symbol, side, qty, leverage):
    if qty is None or qty <= 0 or np.isnan(qty):
        print(f"[ERROR] Invalid quantity: {qty}")
        return None, None
    if leverage is None or np.isnan(leverage) or leverage <= 0:
        print(f"[ERROR] Invalid leverage: {leverage}")
        return None, None

    leverage = int(leverage)

    # Time-based front-running: execute before other bots at round minute marks
    wait_for_optimal_execution_time()

    safe_api_call(client.futures_change_leverage, symbol=symbol, leverage=leverage)

    position_mode = safe_api_call(client.futures_get_position_mode)
    order_args = {
        "symbol": symbol,
        "side": SIDE_BUY if side == 'LONG' else SIDE_SELL,
        "type": ORDER_TYPE_MARKET,
        "quantity": qty
    }
    if position_mode and position_mode.get('dualSidePosition'):
        order_args["positionSide"] = 'LONG' if side == 'LONG' else 'SHORT'

    order = safe_api_call(client.futures_create_order, **order_args)

    # Đợi một chút để Binance cập nhật
    time.sleep(1.0)

    # Lấy lại entry price thực tế
    positions = safe_api_call(client.futures_position_information, symbol=symbol)
    entry_price = None
    for p in positions:
        if p['positionSide'] == ('LONG' if side == 'LONG' else 'SHORT') and float(p['positionAmt']) != 0:
            entry_price = float(p['entryPrice'])
            break

    return order, entry_price

def wait_for_optimal_execution_time():
    """
    Wait for optimal order execution time to front-run other bots.

    Most bots execute at exact minute marks (00, 15, 30, 45 seconds).
    We execute a few seconds BEFORE to get better fills.

    Returns:
        True if waited, False if time-based front-running is disabled
    """
    try:
        from config import USE_TIME_FRONT_RUNNING, FRONT_RUN_SECONDS_BEFORE
    except ImportError:
        return False

    if not USE_TIME_FRONT_RUNNING:
        return False

    import datetime

    now = datetime.datetime.now()
    current_second = now.second

    # Find next round minute mark (00, 15, 30, 45)
    round_marks = [0, 15, 30, 45]
    next_mark = None
    for mark in round_marks:
        if current_second < mark:
            next_mark = mark
            break
    if next_mark is None:
        next_mark = 60  # Next minute's :00

    # Calculate optimal execution time (FRONT_RUN_SECONDS_BEFORE before next mark)
    optimal_second = next_mark - FRONT_RUN_SECONDS_BEFORE
    if optimal_second < 0:
        optimal_second += 60

    # If we're already past optimal time for this window, check if we should wait
    if current_second < optimal_second and (optimal_second - current_second) < 10:
        wait_time = optimal_second - current_second
        print(f"[FRONT-RUN] Waiting {wait_time}s to execute before round minute mark :{next_mark:02d}")
        time.sleep(wait_time)
        return True
    elif current_second > optimal_second and current_second < next_mark:
        # We're in the optimal window already
        print(f"[FRONT-RUN] Executing in optimal window before :{next_mark:02d}")
        return True

    return False


def apply_bot_front_running(entry_price, tp_price, sl_price, side,
                            bot_tp_levels, bot_sl_levels,
                            tp_offset, sl_offset,
                            round_intervals, price_offset):
    """
    Adjust TP/SL prices to front-run other trading bots.

    Strategy:
    - TP: Sell BEFORE other bots hit their TP (avoid price dump from mass selling)
    - SL: Place SL slightly AFTER round levels (avoid bot cascade stop-hunts)
    - Avoid round price numbers where bots cluster orders

    Args:
        entry_price: Entry price of the position
        tp_price: Original TP price
        sl_price: Original SL price
        side: 'LONG' or 'SHORT'
        bot_tp_levels: Common bot TP percentages [0.5, 1.0, 1.5, ...]
        bot_sl_levels: Common bot SL percentages [0.5, 1.0, 1.5, ...]
        tp_offset: Percentage to front-run TP (e.g., 0.03 = 0.03%)
        sl_offset: Percentage to offset SL (e.g., 0.05 = 0.05%)
        round_intervals: Price intervals to avoid [$100, $500, $1000, ...]
        price_offset: Dollar offset from round prices

    Returns:
        (adjusted_tp_price, adjusted_sl_price)
    """
    adjusted_tp = tp_price
    adjusted_sl = sl_price

    # Calculate TP/SL percentages from entry
    if side == 'LONG':
        tp_pct = (tp_price - entry_price) / entry_price * 100
        sl_pct = (entry_price - sl_price) / entry_price * 100
    else:
        tp_pct = (entry_price - tp_price) / entry_price * 100
        sl_pct = (sl_price - entry_price) / entry_price * 100

    # Check if TP is near a common bot level
    for bot_level in bot_tp_levels:
        if abs(tp_pct - bot_level) < 0.1:  # Within 0.1% of bot level
            # Front-run: set TP slightly BEFORE the bot level
            if side == 'LONG':
                adjusted_tp = entry_price * (1 + (bot_level - tp_offset) / 100)
            else:
                adjusted_tp = entry_price * (1 - (bot_level - tp_offset) / 100)
            print(f"[FRONT-RUN] TP near bot level {bot_level}% -> adjusted to {bot_level - tp_offset}%")
            break

    # Check if SL is near a common bot level
    for bot_level in bot_sl_levels:
        if abs(sl_pct - bot_level) < 0.1:  # Within 0.1% of bot level
            # Offset SL: place slightly AFTER the bot level (wider)
            if side == 'LONG':
                adjusted_sl = entry_price * (1 - (bot_level + sl_offset) / 100)
            else:
                adjusted_sl = entry_price * (1 + (bot_level + sl_offset) / 100)
            print(f"[FRONT-RUN] SL near bot level {bot_level}% -> adjusted to {bot_level + sl_offset}%")
            break

    # Avoid round price numbers (where bots cluster orders)
    for interval in sorted(round_intervals, reverse=True):
        # Check TP
        tp_remainder = adjusted_tp % interval
        if tp_remainder < price_offset or (interval - tp_remainder) < price_offset:
            # TP is near a round number - offset it
            if side == 'LONG':
                # For LONG TP, offset DOWN (sell before round number)
                adjusted_tp = (adjusted_tp // interval) * interval - price_offset
            else:
                # For SHORT TP, offset UP (buy before round number)
                adjusted_tp = (adjusted_tp // interval) * interval + price_offset
            print(f"[FRONT-RUN] TP near round ${interval} level -> offset by ${price_offset}")
            break

    for interval in sorted(round_intervals, reverse=True):
        # Check SL
        sl_remainder = adjusted_sl % interval
        if sl_remainder < price_offset or (interval - sl_remainder) < price_offset:
            # SL is near a round number - offset it away
            if side == 'LONG':
                # For LONG SL, offset DOWN (place SL below round number)
                adjusted_sl = (adjusted_sl // interval) * interval - price_offset
            else:
                # For SHORT SL, offset UP (place SL above round number)
                adjusted_sl = (adjusted_sl // interval) * interval + price_offset
            print(f"[FRONT-RUN] SL near round ${interval} level -> offset by ${price_offset}")
            break

    return adjusted_tp, adjusted_sl


def place_sl_tp_order(symbol, side, qty, entry_price, tp_ratio, sl_ratio):
    """
    Place Stop Loss and Take Profit orders with slippage buffer, entropy, and bot front-running.

    Slippage adjustments:
    - TP: Set slightly closer to entry (easier to hit before reversal)
    - SL: Set slightly further from entry (avoid wick stop-outs)

    Entropy (randomness):
    - Adds random variation to avoid bot detection
    - Offsets prices from round numbers

    Bot Front-Running:
    - TP: Sell BEFORE other bots dump at round levels
    - SL: Place AFTER round levels to avoid bot cascade stop-hunts
    """
    import random

    # Import slippage settings
    try:
        from config import (USE_SLIPPAGE_BUFFER, SLIPPAGE_TP_BUFFER, SLIPPAGE_SL_BUFFER,
                           USE_SLIPPAGE_ENTROPY, ENTROPY_TP_RANGE, ENTROPY_SL_RANGE,
                           ENTROPY_PRICE_OFFSET)
    except ImportError:
        USE_SLIPPAGE_BUFFER = False
        SLIPPAGE_TP_BUFFER = 0
        SLIPPAGE_SL_BUFFER = 0
        USE_SLIPPAGE_ENTROPY = False
        ENTROPY_TP_RANGE = 0
        ENTROPY_SL_RANGE = 0
        ENTROPY_PRICE_OFFSET = False

    # Import bot front-running settings
    try:
        from config import (USE_BOT_FRONT_RUNNING, BOT_COMMON_TP_LEVELS, BOT_COMMON_SL_LEVELS,
                           FRONT_RUN_TP_OFFSET, FRONT_RUN_SL_OFFSET,
                           BOT_ROUND_PRICE_INTERVALS, FRONT_RUN_PRICE_OFFSET)
    except ImportError:
        USE_BOT_FRONT_RUNNING = False
        BOT_COMMON_TP_LEVELS = []
        BOT_COMMON_SL_LEVELS = []
        FRONT_RUN_TP_OFFSET = 0
        FRONT_RUN_SL_OFFSET = 0
        BOT_ROUND_PRICE_INTERVALS = []
        FRONT_RUN_PRICE_OFFSET = 0

    # Start with base ratios
    adjusted_tp = tp_ratio
    adjusted_sl = sl_ratio

    # Apply slippage buffer if enabled
    if USE_SLIPPAGE_BUFFER:
        adjusted_tp = tp_ratio - (SLIPPAGE_TP_BUFFER / 100)
        adjusted_sl = sl_ratio + (SLIPPAGE_SL_BUFFER / 100)

    # Apply entropy (randomness) if enabled
    if USE_SLIPPAGE_ENTROPY:
        # Random variation for TP/SL percentages
        tp_entropy = random.uniform(-ENTROPY_TP_RANGE, ENTROPY_TP_RANGE) / 100
        sl_entropy = random.uniform(-ENTROPY_SL_RANGE, ENTROPY_SL_RANGE) / 100
        adjusted_tp += tp_entropy
        adjusted_sl += sl_entropy
        print(f"[ENTROPY] TP entropy: {tp_entropy*100:+.3f}% | SL entropy: {sl_entropy*100:+.3f}%")

    print(f"[SLIPPAGE] TP: {tp_ratio*100:.2f}% -> {adjusted_tp*100:.3f}% | SL: {sl_ratio*100:.2f}% -> {adjusted_sl*100:.3f}%")

    # Calculate base prices
    if side == 'LONG':
        sl_price = entry_price * (1 - adjusted_sl)
        tp_price = entry_price * (1 + adjusted_tp)
    else:
        sl_price = entry_price * (1 + adjusted_sl)
        tp_price = entry_price * (1 - adjusted_tp)

    # Apply bot front-running logic
    if USE_BOT_FRONT_RUNNING:
        tp_price, sl_price = apply_bot_front_running(
            entry_price, tp_price, sl_price, side,
            BOT_COMMON_TP_LEVELS, BOT_COMMON_SL_LEVELS,
            FRONT_RUN_TP_OFFSET, FRONT_RUN_SL_OFFSET,
            BOT_ROUND_PRICE_INTERVALS, FRONT_RUN_PRICE_OFFSET
        )

    # Apply price offset entropy (avoid round numbers) - final adjustment
    if USE_SLIPPAGE_ENTROPY and ENTROPY_PRICE_OFFSET:
        # Add random offset to avoid round numbers like $80,000
        tp_offset = random.uniform(1, 50) * (1 if random.random() > 0.5 else -1)
        sl_offset = random.uniform(1, 50) * (1 if random.random() > 0.5 else -1)
        tp_price += tp_offset
        sl_price += sl_offset
        print(f"[ENTROPY] Price offsets: TP {tp_offset:+.0f} | SL {sl_offset:+.0f}")

    position_side = 'LONG' if side == 'LONG' else 'SHORT'

    sl_order = safe_api_call(client.futures_create_order,
                             symbol=symbol,
                             side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
                             type=ORDER_TYPE_STOP_MARKET,
                             stopPrice=round(sl_price, 2),
                             closePosition=True,
                             positionSide=position_side,
                             timeInForce=TIME_IN_FORCE_GTC,
                             priceProtect=True,
                             workingType='MARK_PRICE'
                             )

    tp_order = safe_api_call(client.futures_create_order,
                             symbol=symbol,
                             side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
                             type=ORDER_TYPE_TAKE_PROFIT_MARKET,
                             stopPrice=round(tp_price, 2),
                             closePosition=True,
                             positionSide=position_side,
                             timeInForce=TIME_IN_FORCE_GTC,
                             priceProtect=True,
                             workingType='MARK_PRICE'
                             )

    print(f"[TP/SL] Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} ({adjusted_tp*100:.3f}%) | SL: ${sl_price:.2f} ({adjusted_sl*100:.3f}%)")

    return sl_order, tp_order


