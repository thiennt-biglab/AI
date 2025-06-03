#trader.py
from binance.client import Client
from binance.enums import *
from config import *
import numpy as np
import time

ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
ORDER_TYPE_MARKET = "MARKET"
TIME_IN_FORCE_GTC = "GTC"
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

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

def place_sl_tp_order(symbol, side, qty, entry_price, tp_ratio, sl_ratio):
    sl_price = entry_price * (1 - sl_ratio) if side == 'LONG' else entry_price * (1 + sl_ratio)
    tp_price = entry_price * (1 + tp_ratio) if side == 'LONG' else entry_price * (1 - tp_ratio)
    position_side = 'LONG' if side == 'LONG' else 'SHORT'

    sl_order = safe_api_call(client.futures_create_order,
                             symbol=symbol,
                             side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
                             type=ORDER_TYPE_STOP_MARKET,
                             stopPrice=round(sl_price, 5),
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
                             stopPrice=round(tp_price, 5),
                             closePosition=True,
                             positionSide=position_side,
                             timeInForce=TIME_IN_FORCE_GTC,
                             priceProtect=True,
                             workingType='MARK_PRICE'
                             )

    return sl_order, tp_order


