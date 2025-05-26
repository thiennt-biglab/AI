# trader.py
from binance.client import Client
from binance.enums import *
from config import *
import pandas as pd
import numpy as np
ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
ORDER_TYPE_MARKET = "MARKET"
TIME_IN_FORCE_GTC = "GTC"
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"

client = Client(API_KEY, API_SECRET)

def get_latest_klines(symbol, interval, limit=200):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_asset_volume','num_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def get_step_size(symbol):
    exchange_info = client.futures_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    return float(f['stepSize'])  # ví dụ: 0.001
    return 0.001  # fallback

def round_step_size(quantity, step_size):
    try:
        precision = int(round(-np.log10(step_size)))
        return round(quantity, precision)
    except:
        return 0.0  # fallback if step_size is invalid

def calculate_qty(balance, price, leverage, risk_percent):
    capital = balance * (risk_percent / 100)
    raw_qty = (capital * leverage) / price
    step_size = get_step_size(SYMBOL)
    qty = round_step_size(raw_qty, step_size)
    return qty

    
def get_balance(asset="USDT"):
    balance = client.futures_account_balance()
    for b in balance:
        if b['asset'] == asset:
            return float(b['balance'])
    return 0.0

def cancel_open_orders(symbol, position_side):
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        count = 0
        for o in open_orders:
            if o['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET'] and o.get('positionSide', '') == position_side:
                client.futures_cancel_order(symbol=symbol, orderId=o['orderId'])
                count += 1
        print(f"[INFO] Canceled {count} open stop/TP orders for {position_side}")
    except Exception as e:
        print(f"[WARN] Cannot cancel open orders for {position_side}: {e}")

def get_current_position_side(symbol):
    positions = client.futures_position_information(symbol=symbol)
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
        return

    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    position_mode = client.futures_get_position_mode()
    order_args = {
        "symbol": symbol,
        "side": SIDE_BUY if side == 'LONG' else SIDE_SELL,
        "type": ORDER_TYPE_MARKET,
        "quantity": qty
    }
    if position_mode['dualSidePosition']:
        order_args["positionSide"] = 'LONG' if side == 'LONG' else 'SHORT'

    return client.futures_create_order(**order_args)


def place_sl_tp_order(symbol, side, qty, entry_price):
    sl_price = entry_price * (1 - SL_PERCENT/100) if side == 'LONG' else entry_price * (1 + SL_PERCENT/100)
    tp_price = entry_price * (1 + TP_PERCENT/100) if side == 'LONG' else entry_price * (1 - TP_PERCENT/100)
    position_side = 'LONG' if side == 'LONG' else 'SHORT'

    # ❗ KHÔNG GỬI quantity nếu closePosition = True
    sl_order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
        type=ORDER_TYPE_STOP_MARKET,
        stopPrice=round(sl_price, 2),
        closePosition=True,
        positionSide=position_side,
        timeInForce=TIME_IN_FORCE_GTC
    )

    tp_order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_SELL if side == 'LONG' else SIDE_BUY,
        type=ORDER_TYPE_TAKE_PROFIT_MARKET,
        stopPrice=round(tp_price, 2),
        closePosition=True,
        positionSide=position_side,
        timeInForce=TIME_IN_FORCE_GTC
    )

    return sl_order


