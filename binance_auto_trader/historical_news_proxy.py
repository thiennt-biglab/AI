"""
Historical News Proxy for Training.

Since we don't have historical news data, we create proxy features:
1. Price anomaly detection (unusual moves indicate news)
2. Volatility regime detection
3. Known event calendar (FED meetings, halvings, etc.)
4. Sentiment proxy from price action

This allows the model to learn patterns around news-like events.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# KNOWN HISTORICAL EVENTS (Crypto-relevant)
# ============================================================
# Format: (date, event_type, impact_direction, magnitude)
# impact_direction: 1 = bullish, -1 = bearish, 0 = uncertain
# magnitude: 0-1 scale of importance

KNOWN_EVENTS = [
    # 2024 Events
    ("2024-01-10", "ETF", 1, 1.0),      # Bitcoin ETF Approved
    ("2024-03-20", "FED", 0, 0.8),      # FOMC Meeting
    ("2024-04-20", "HALVING", 1, 1.0),  # Bitcoin Halving
    ("2024-05-01", "FED", 0, 0.8),      # FOMC Meeting
    ("2024-06-12", "FED", 0, 0.8),      # FOMC Meeting
    ("2024-07-31", "FED", 0, 0.8),      # FOMC Meeting
    ("2024-09-18", "FED", 1, 0.9),      # Rate Cut
    ("2024-11-05", "ELECTION", 1, 1.0), # Trump Election
    ("2024-11-07", "FED", 0, 0.8),      # FOMC Meeting
    ("2024-12-18", "FED", -1, 0.8),     # Hawkish FED

    # 2025 Events
    ("2025-01-29", "FED", 0, 0.8),      # FOMC Meeting
    ("2025-03-19", "FED", 0, 0.8),      # FOMC Meeting
    ("2025-05-07", "FED", 0, 0.8),      # FOMC Meeting
    ("2025-06-18", "FED", 0, 0.8),      # FOMC Meeting

    # Major Crisis Events (examples)
    ("2024-08-05", "CRASH", -1, 0.9),   # Japan carry trade unwind
]


def detect_price_anomalies(close, high, low, window=20, threshold=2.5):
    """
    Detect unusual price movements that likely indicate news events.
    Uses z-score of returns to identify anomalies.

    Returns: array of anomaly scores (-1 to 1, negative=bearish, positive=bullish)
    """
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])

    # Rolling mean and std
    anomaly_scores = np.zeros(len(close))

    for i in range(window, len(close)):
        window_returns = returns[i-window:i]
        mean = np.mean(window_returns)
        std = np.std(window_returns)

        if std > 0:
            z_score = (returns[i] - mean) / std

            # Convert z-score to anomaly score
            if abs(z_score) > threshold:
                # Significant anomaly detected
                anomaly_scores[i] = np.clip(z_score / 5, -1, 1)

    return anomaly_scores


def detect_volatility_regime(close, atr, window=20):
    """
    Detect high volatility periods (often news-driven).
    Returns: array of volatility regime (0=low, 0.5=normal, 1=high)
    """
    atr_pct = atr / close

    regime = np.zeros(len(close))

    for i in range(window, len(close)):
        window_atr = atr_pct[i-window:i]
        mean_atr = np.mean(window_atr)
        std_atr = np.std(window_atr)

        if std_atr > 0:
            z = (atr_pct[i] - mean_atr) / std_atr

            if z > 1.5:
                regime[i] = 1.0  # High volatility
            elif z > 0.5:
                regime[i] = 0.7
            elif z < -0.5:
                regime[i] = 0.3  # Low volatility
            else:
                regime[i] = 0.5  # Normal

    return regime


def get_event_impact(timestamps, known_events=KNOWN_EVENTS, hours_before=24, hours_after=48):
    """
    Create event impact features based on known historical events.

    Returns: dict with multiple event-related features
    """
    n = len(timestamps)
    event_impact = np.zeros(n)
    event_direction = np.zeros(n)
    event_type = np.zeros(n)  # 0=none, 1=FED, 2=ETF, 3=HALVING, 4=ELECTION, 5=CRASH

    event_type_map = {'FED': 1, 'ETF': 2, 'HALVING': 3, 'ELECTION': 4, 'CRASH': 5}

    for event_date_str, etype, direction, magnitude in known_events:
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")

        for i, ts in enumerate(timestamps):
            # Convert timestamp to datetime
            if isinstance(ts, (int, float)):
                candle_time = datetime.fromtimestamp(ts / 1000)
            else:
                candle_time = pd.to_datetime(ts)

            # Calculate hours from event
            hours_diff = (candle_time - event_date).total_seconds() / 3600

            # Before event (anticipation)
            if -hours_before <= hours_diff < 0:
                proximity = 1 - abs(hours_diff) / hours_before
                event_impact[i] = max(event_impact[i], magnitude * proximity * 0.5)
                event_direction[i] = direction
                event_type[i] = event_type_map.get(etype, 0)

            # After event (reaction)
            elif 0 <= hours_diff <= hours_after:
                proximity = 1 - hours_diff / hours_after
                event_impact[i] = max(event_impact[i], magnitude * proximity)
                event_direction[i] = direction
                event_type[i] = event_type_map.get(etype, 0)

    return {
        'event_impact': event_impact,
        'event_direction': event_direction,
        'event_type': event_type
    }


def calculate_sentiment_proxy(close, volume, window=10):
    """
    Calculate sentiment proxy from price action.
    Based on: price momentum + volume confirmation

    Returns: sentiment score (-1 to 1)
    """
    sentiment = np.zeros(len(close))

    for i in range(window, len(close)):
        # Price momentum
        price_change = (close[i] - close[i-window]) / close[i-window]

        # Volume trend
        vol_now = np.mean(volume[i-3:i+1])
        vol_before = np.mean(volume[i-window:i-3])
        vol_change = (vol_now - vol_before) / vol_before if vol_before > 0 else 0

        # High volume + price up = bullish sentiment
        # High volume + price down = bearish sentiment
        if vol_change > 0.2:  # Above average volume
            sentiment[i] = np.clip(price_change * 10, -1, 1)
        else:
            sentiment[i] = np.clip(price_change * 5, -0.5, 0.5)

    return sentiment


def calculate_fear_greed_proxy(close, high, low, volume, window=14):
    """
    Calculate Fear & Greed proxy index.
    Components:
    - Price momentum
    - Volatility (high vol = fear)
    - Volume (high vol on down = fear)
    - Distance from highs

    Returns: index 0-100 (0=extreme fear, 100=extreme greed)
    """
    fear_greed = np.full(len(close), 50.0)

    for i in range(window, len(close)):
        # 1. Price momentum (0-25 points)
        momentum = (close[i] - close[i-window]) / close[i-window]
        momentum_score = np.clip((momentum + 0.1) / 0.2 * 25, 0, 25)

        # 2. Volatility (0-25 points, inverse)
        atr = np.mean(high[i-window:i] - low[i-window:i])
        atr_pct = atr / close[i]
        vol_score = np.clip((0.03 - atr_pct) / 0.03 * 25, 0, 25)

        # 3. Volume trend (0-25 points)
        vol_ma = np.mean(volume[i-window:i])
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 1
        if close[i] > close[i-1]:  # Up day with volume = greed
            vol_score_2 = np.clip(vol_ratio / 2 * 25, 0, 25)
        else:  # Down day with volume = fear
            vol_score_2 = np.clip((2 - vol_ratio) / 2 * 25, 0, 25)

        # 4. Distance from recent high (0-25 points)
        recent_high = np.max(high[i-window:i])
        distance = (close[i] - recent_high) / recent_high
        distance_score = np.clip((distance + 0.1) / 0.1 * 25, 0, 25)

        fear_greed[i] = momentum_score + vol_score + vol_score_2 + distance_score

    return fear_greed


def create_news_proxy_features(df, timestamp_col='timestamp'):
    """
    Create all news proxy features for training.

    Input: DataFrame with OHLCV data
    Output: DataFrame with news proxy features added
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Calculate ATR if not present
    if 'atr' in df.columns:
        atr = df['atr'].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values

    # 1. Price anomaly detection
    df['news_anomaly'] = detect_price_anomalies(close, high, low)

    # 2. Volatility regime
    df['news_vol_regime'] = detect_volatility_regime(close, atr)

    # 3. Sentiment proxy
    df['news_sentiment_proxy'] = calculate_sentiment_proxy(close, volume)

    # 4. Fear & Greed proxy
    df['news_fear_greed'] = calculate_fear_greed_proxy(close, high, low, volume) / 100

    # 5. Event impact (if timestamps available)
    if timestamp_col in df.columns:
        events = get_event_impact(df[timestamp_col].values)
        df['news_event_impact'] = events['event_impact']
        df['news_event_direction'] = events['event_direction']
        df['news_event_type'] = events['event_type']
    else:
        df['news_event_impact'] = 0
        df['news_event_direction'] = 0
        df['news_event_type'] = 0

    # 6. Combined news score
    df['news_score'] = (
        df['news_anomaly'] * 0.3 +
        df['news_sentiment_proxy'] * 0.3 +
        (df['news_fear_greed'] - 0.5) * 0.2 +
        df['news_event_direction'] * df['news_event_impact'] * 0.2
    )

    return df


def get_news_proxy_feature_names():
    """Get list of news proxy feature names for model input."""
    return [
        'news_anomaly',
        'news_vol_regime',
        'news_sentiment_proxy',
        'news_fear_greed',
        'news_event_impact',
        'news_event_direction',
        'news_event_type',
        'news_score'
    ]


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HISTORICAL NEWS PROXY - Feature Generation Test")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 1000

    # Simulate price data with some "news events"
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.002)

    # Add some "news spikes"
    close[200:210] *= 1.05  # Bullish news
    close[500:510] *= 0.95  # Bearish news
    close[800:810] *= 1.03  # Moderate bullish

    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    volume = np.random.randint(1000, 10000, n).astype(float)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    })

    # Add news proxy features
    df = create_news_proxy_features(df)

    print("\nFeatures created:")
    for col in get_news_proxy_feature_names():
        print(f"  {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")

    # Show detected anomalies
    anomalies = df[abs(df['news_anomaly']) > 0.3]
    print(f"\nDetected {len(anomalies)} news-like anomalies")

    print("\nTop 5 bullish anomalies:")
    top_bullish = df.nlargest(5, 'news_anomaly')
    for idx, row in top_bullish.iterrows():
        print(f"  Index {idx}: score={row['news_anomaly']:.3f}, price={row['close']:.2f}")

    print("\nTop 5 bearish anomalies:")
    top_bearish = df.nsmallest(5, 'news_anomaly')
    for idx, row in top_bearish.iterrows():
        print(f"  Index {idx}: score={row['news_anomaly']:.3f}, price={row['close']:.2f}")
