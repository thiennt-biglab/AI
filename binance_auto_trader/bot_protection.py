"""
Bot Protection Module - Defensive strategies against other trading bots.

Features:
1. Detect manipulation periods (avoid trading)
2. Randomize execution to avoid pattern detection
3. Avoid stop-hunt zones
4. Detect fake breakouts
5. Smart order placement

These strategies PROTECT your win rate by avoiding bot traps.
"""
import numpy as np
import random
import time
from datetime import datetime


# ============================================================
# 1. MANIPULATION DETECTION (Improves win rate by avoiding traps)
# ============================================================

def detect_stop_hunt(high, low, close, lookback=20):
    """
    Detect stop-hunt wicks (bots hunting stop losses).

    Pattern: Long wick that quickly reverses = manipulation

    Returns: (is_manipulation, direction)
    - direction: 'UP' = upward wick (short squeeze), 'DOWN' = downward wick (long squeeze)
    """
    if len(close) < lookback + 1:
        return False, None

    current_close = close[-1]
    current_high = high[-1]
    current_low = low[-1]
    current_open = close[-2]  # Approximate open

    # Calculate candle body and wicks
    body = abs(current_close - current_open)
    upper_wick = current_high - max(current_close, current_open)
    lower_wick = min(current_close, current_open) - current_low
    total_range = current_high - current_low

    if total_range == 0:
        return False, None

    # Recent volatility
    recent_ranges = high[-lookback:-1] - low[-lookback:-1]
    avg_range = np.mean(recent_ranges)

    # Stop hunt detection criteria:
    # 1. Large wick relative to body (>2x)
    # 2. Wick size > 1.5x average range (unusual)
    # 3. Price closed back inside normal range

    # Upward stop hunt (short squeeze then dump)
    if upper_wick > body * 2 and upper_wick > avg_range * 1.5:
        if current_close < current_high - upper_wick * 0.7:  # Closed back down
            return True, 'UP'

    # Downward stop hunt (long squeeze then pump)
    if lower_wick > body * 2 and lower_wick > avg_range * 1.5:
        if current_close > current_low + lower_wick * 0.7:  # Closed back up
            return True, 'DOWN'

    return False, None


def detect_fake_breakout(close, high, low, lookback=50, threshold=0.8):
    """
    Detect fake breakouts (bots triggering breakout traders then reversing).

    Pattern: Price breaks key level but quickly reverses back.

    Returns: (is_fake_breakout, direction)
    """
    if len(close) < lookback + 5:
        return False, None

    # Find recent support/resistance
    recent_high = np.max(high[-lookback:-1])
    recent_low = np.min(low[-lookback:-1])

    current_high = high[-1]
    current_low = low[-1]
    current_close = close[-1]

    # Fake breakout UP: broke above resistance but closed below
    if current_high > recent_high:
        break_amount = (current_high - recent_high) / recent_high
        if current_close < recent_high and break_amount > 0.002:  # >0.2% break
            return True, 'UP'

    # Fake breakout DOWN: broke below support but closed above
    if current_low < recent_low:
        break_amount = (recent_low - current_low) / recent_low
        if current_close > recent_low and break_amount > 0.002:
            return True, 'DOWN'

    return False, None


def detect_volume_manipulation(volume, lookback=20, spike_threshold=3.0):
    """
    Detect unusual volume spikes that indicate bot activity.

    High volume spikes followed by quick reversals = manipulation.

    Returns: (is_manipulation, spike_ratio)
    """
    if len(volume) < lookback + 1:
        return False, 1.0

    avg_volume = np.mean(volume[-lookback:-1])
    current_volume = volume[-1]

    if avg_volume == 0:
        return False, 1.0

    spike_ratio = current_volume / avg_volume

    # Extreme volume spike = likely manipulation
    if spike_ratio > spike_threshold:
        return True, spike_ratio

    return False, spike_ratio


def is_safe_to_trade(high, low, close, volume, lookback=20):
    """
    Combined check: Is it safe to trade right now?

    Returns: (is_safe, reasons)
    """
    reasons = []

    # Check stop hunt
    is_stop_hunt, sh_dir = detect_stop_hunt(high, low, close, lookback)
    if is_stop_hunt:
        reasons.append(f"Stop hunt detected ({sh_dir})")

    # Check fake breakout
    is_fake_bo, bo_dir = detect_fake_breakout(close, high, low, lookback * 2)
    if is_fake_bo:
        reasons.append(f"Fake breakout detected ({bo_dir})")

    # Check volume manipulation
    is_vol_manip, spike = detect_volume_manipulation(volume, lookback)
    if is_vol_manip:
        reasons.append(f"Volume spike ({spike:.1f}x normal)")

    is_safe = len(reasons) == 0
    return is_safe, reasons


# ============================================================
# 2. SMART TP/SL PLACEMENT (Avoid stop hunts)
# ============================================================

def get_safe_stop_loss(entry_price, base_sl_pct, high, low, lookback=20, buffer_pct=0.001):
    """
    Calculate stop loss that avoids obvious stop-hunt levels.

    Strategy:
    1. Don't place SL at round numbers
    2. Don't place SL at recent swing highs/lows
    3. Add small random buffer

    Returns: safe_sl_price
    """
    # Base SL level
    base_sl = entry_price * (1 - base_sl_pct)

    # Find recent swing lows (where bots might hunt)
    recent_lows = low[-lookback:]
    swing_lows = []
    for i in range(2, len(recent_lows) - 2):
        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2]:
            if recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]:
                swing_lows.append(recent_lows[i])

    # Check if base SL is too close to swing low
    for swing in swing_lows:
        if abs(base_sl - swing) / entry_price < 0.003:  # Within 0.3%
            # Move SL slightly below the swing low
            base_sl = swing * (1 - buffer_pct - random.uniform(0.0005, 0.002))
            break

    # Avoid round numbers
    base_sl = avoid_round_number(base_sl)

    return base_sl


def get_safe_take_profit(entry_price, base_tp_pct, high, low, lookback=20, buffer_pct=0.001):
    """
    Calculate take profit that avoids obvious resistance levels.

    Returns: safe_tp_price
    """
    # Base TP level
    base_tp = entry_price * (1 + base_tp_pct)

    # Find recent swing highs
    recent_highs = high[-lookback:]
    swing_highs = []
    for i in range(2, len(recent_highs) - 2):
        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2]:
            if recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]:
                swing_highs.append(recent_highs[i])

    # If TP is just below resistance, move it slightly below
    for swing in swing_highs:
        if 0 < (swing - base_tp) / entry_price < 0.003:  # TP just below resistance
            # Take profit slightly before resistance
            base_tp = swing * (1 - buffer_pct - random.uniform(0.0005, 0.002))
            break

    # Avoid round numbers
    base_tp = avoid_round_number(base_tp)

    return base_tp


def avoid_round_number(price, variance=0.0003):
    """
    Slightly adjust price to avoid round numbers where bots cluster orders.

    Bots often place orders at:
    - Round numbers (100, 50000, etc.)
    - Psychological levels (.00, .50)
    """
    # Add small random offset
    offset = price * random.uniform(-variance, variance)
    adjusted = price + offset

    # Check if close to round number
    str_price = f"{adjusted:.2f}"

    # Avoid .00 endings
    if str_price.endswith('00'):
        adjusted += price * random.uniform(0.0001, 0.0005)
    # Avoid .50 endings
    elif str_price.endswith('50'):
        adjusted += price * random.uniform(0.0001, 0.0005)

    return adjusted


# ============================================================
# 3. EXECUTION RANDOMIZATION (Avoid pattern detection)
# ============================================================

def get_random_delay(min_ms=100, max_ms=2000):
    """
    Get random delay before execution to avoid detection.
    Small delays don't affect win rate but make patterns undetectable.
    """
    return random.randint(min_ms, max_ms) / 1000  # Return seconds


def should_skip_candle_start(seconds_into_candle=None, skip_seconds=30):
    """
    Avoid trading right at candle open (where most bots execute).

    Returns: True if should wait
    """
    if seconds_into_candle is None:
        # Calculate seconds into current 5-min candle
        now = datetime.now()
        seconds_into_candle = (now.minute % 5) * 60 + now.second

    return seconds_into_candle < skip_seconds


def randomize_order_size(base_size, variance_pct=0.05):
    """
    Slightly randomize order size to avoid pattern detection.
    ±5% variance doesn't meaningfully affect profits.
    """
    multiplier = 1 + random.uniform(-variance_pct, variance_pct)
    return base_size * multiplier


# ============================================================
# 4. ANTI-FRONT-RUNNING
# ============================================================

def split_order(total_size, num_parts=3, time_between_ms=500):
    """
    Split large orders into smaller parts to avoid front-running.

    Returns: list of (size, delay_ms) tuples
    """
    if num_parts <= 1:
        return [(total_size, 0)]

    # Random split (not equal parts to avoid detection)
    parts = []
    remaining = total_size

    for i in range(num_parts - 1):
        # Random portion between 20-40% of remaining
        portion = remaining * random.uniform(0.2, 0.4)
        parts.append(portion)
        remaining -= portion

    parts.append(remaining)  # Last part gets the rest

    # Shuffle parts
    random.shuffle(parts)

    # Add random delays
    result = []
    for i, size in enumerate(parts):
        delay = 0 if i == 0 else random.randint(time_between_ms // 2, time_between_ms * 2)
        result.append((size, delay))

    return result


def get_limit_price_offset(current_price, side='BUY', offset_pct=0.0002):
    """
    Get slightly better limit price to avoid market order detection.

    Using limit orders slightly inside the spread:
    - Avoids slippage
    - Harder for bots to front-run
    - Might get better fill
    """
    if side == 'BUY':
        # Place buy limit slightly below current price
        return current_price * (1 - offset_pct)
    else:
        # Place sell limit slightly above current price
        return current_price * (1 + offset_pct)


# ============================================================
# 5. BOT ACTIVITY DETECTION
# ============================================================

def calculate_bot_activity_score(high, low, close, volume, lookback=20):
    """
    Calculate overall bot activity score (0-1).
    Higher score = more bot activity = riskier to trade.

    Components:
    - Wick ratio (stop hunts)
    - Volume spikes
    - Price reversals
    - Order book imbalance proxy
    """
    if len(close) < lookback:
        return 0.5

    score = 0

    # 1. Wick ratio analysis
    wicks = []
    for i in range(-lookback, 0):
        body = abs(close[i] - close[i-1])
        total = high[i] - low[i]
        if total > 0:
            wick_ratio = 1 - (body / total)
            wicks.append(wick_ratio)

    avg_wick_ratio = np.mean(wicks) if wicks else 0.5
    if avg_wick_ratio > 0.7:  # High wick ratio = manipulation
        score += 0.3

    # 2. Volume irregularity
    vol_std = np.std(volume[-lookback:])
    vol_mean = np.mean(volume[-lookback:])
    vol_cv = vol_std / vol_mean if vol_mean > 0 else 0

    if vol_cv > 1.0:  # High volume variance = bot activity
        score += 0.3

    # 3. Price reversal frequency
    reversals = 0
    for i in range(-lookback + 1, 0):
        prev_dir = close[i-1] - close[i-2]
        curr_dir = close[i] - close[i-1]
        if prev_dir * curr_dir < 0:  # Direction changed
            reversals += 1

    reversal_rate = reversals / lookback
    if reversal_rate > 0.6:  # High reversal rate = manipulation
        score += 0.4

    return min(1.0, score)


# ============================================================
# MAIN PROTECTION WRAPPER
# ============================================================

class BotProtection:
    """Main class for bot protection features."""

    def __init__(self,
                 enable_manipulation_detection=True,
                 enable_safe_tpsl=True,
                 enable_random_delays=True,
                 enable_order_splitting=False,
                 max_bot_activity_score=0.7):

        self.enable_manipulation_detection = enable_manipulation_detection
        self.enable_safe_tpsl = enable_safe_tpsl
        self.enable_random_delays = enable_random_delays
        self.enable_order_splitting = enable_order_splitting
        self.max_bot_activity_score = max_bot_activity_score

    def should_trade(self, high, low, close, volume):
        """
        Check if it's safe to trade right now.
        Returns: (should_trade, reason)
        """
        if not self.enable_manipulation_detection:
            return True, "Protection disabled"

        # Check manipulation
        is_safe, reasons = is_safe_to_trade(high, low, close, volume)
        if not is_safe:
            return False, f"Manipulation detected: {reasons}"

        # Check bot activity score
        bot_score = calculate_bot_activity_score(high, low, close, volume)
        if bot_score > self.max_bot_activity_score:
            return False, f"High bot activity ({bot_score:.2f})"

        # Check candle timing
        if should_skip_candle_start():
            return False, "Too close to candle open"

        return True, "Safe to trade"

    def get_protected_tpsl(self, entry_price, base_tp_pct, base_sl_pct, high, low):
        """
        Get TP/SL levels that avoid bot traps.
        """
        if not self.enable_safe_tpsl:
            tp = entry_price * (1 + base_tp_pct)
            sl = entry_price * (1 - base_sl_pct)
            return tp, sl

        tp = get_safe_take_profit(entry_price, base_tp_pct, high, low)
        sl = get_safe_stop_loss(entry_price, base_sl_pct, high, low)

        return tp, sl

    def prepare_execution(self, order_size, current_price, side='BUY'):
        """
        Prepare order execution with protection.
        Returns: dict with execution parameters
        """
        result = {
            'delay': 0,
            'orders': [(order_size, 0)],
            'use_limit': False,
            'limit_price': current_price
        }

        if self.enable_random_delays:
            result['delay'] = get_random_delay()

        if self.enable_order_splitting and order_size > 100:  # Only split large orders
            result['orders'] = split_order(order_size)

        # Use limit orders for better fills
        result['use_limit'] = True
        result['limit_price'] = get_limit_price_offset(current_price, side)

        # Randomize size slightly
        result['orders'] = [(randomize_order_size(size), delay)
                          for size, delay in result['orders']]

        return result


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("BOT PROTECTION MODULE - Test")
    print("=" * 60)

    # Generate test data with some manipulation patterns
    np.random.seed(42)
    n = 100

    close = 50000 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 30)
    low = close - np.abs(np.random.randn(n) * 30)
    volume = np.random.randint(100, 1000, n).astype(float)

    # Add stop hunt pattern
    high[50] = close[50] + 200  # Large upper wick
    close[50] = close[49] - 10  # Closed lower

    # Add volume spike
    volume[70] = volume[69] * 5

    protection = BotProtection()

    # Test manipulation detection
    print("\n1. Manipulation Detection:")
    is_safe, reasons = is_safe_to_trade(high[:51], low[:51], close[:51], volume[:51])
    print(f"   After stop hunt candle: Safe={is_safe}, Reasons={reasons}")

    is_safe, reasons = is_safe_to_trade(high[:71], low[:71], close[:71], volume[:71])
    print(f"   After volume spike: Safe={is_safe}, Reasons={reasons}")

    # Test safe TP/SL
    print("\n2. Safe TP/SL Placement:")
    entry = 50000
    base_tp = 0.015
    base_sl = 0.01

    safe_tp, safe_sl = protection.get_protected_tpsl(entry, base_tp, base_sl, high, low)
    naive_tp = entry * (1 + base_tp)
    naive_sl = entry * (1 - base_sl)

    print(f"   Entry: ${entry}")
    print(f"   Naive TP: ${naive_tp:.2f} -> Safe TP: ${safe_tp:.2f}")
    print(f"   Naive SL: ${naive_sl:.2f} -> Safe SL: ${safe_sl:.2f}")

    # Test bot activity score
    print("\n3. Bot Activity Score:")
    score = calculate_bot_activity_score(high, low, close, volume)
    print(f"   Current bot activity: {score:.2f} (0=low, 1=high)")

    # Test execution preparation
    print("\n4. Protected Execution:")
    exec_params = protection.prepare_execution(1000, 50000, 'BUY')
    print(f"   Delay: {exec_params['delay']*1000:.0f}ms")
    print(f"   Limit price: ${exec_params['limit_price']:.2f}")
    print(f"   Order parts: {len(exec_params['orders'])}")

    print("\n" + "=" * 60)
    print("Bot protection ready!")
    print("=" * 60)
