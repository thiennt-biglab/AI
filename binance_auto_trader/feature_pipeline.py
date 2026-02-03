from binance.client import Client
import pandas as pd
import ta
import requests
import numpy as np
from config import API_KEY, API_SECRET, SYMBOL, INTERVALS
from trader import safe_api_call

# Import historical news proxy for training
try:
    from historical_news_proxy import create_news_proxy_features, get_news_proxy_feature_names
    HAS_NEWS_PROXY = True
except ImportError:
    HAS_NEWS_PROXY = False
    print("[WARN] historical_news_proxy not available")

# Import whale tracker for smart money features
try:
    from whale_tracker import (
        get_top_trader_long_short_ratio,
        get_top_trader_account_ratio,
        get_global_long_short_ratio,
        get_open_interest_history,
        get_order_book_imbalance,
        WhaleTracker
    )
    HAS_WHALE_TRACKER = True
except ImportError:
    HAS_WHALE_TRACKER = False
    print("[WARN] whale_tracker not available")

client = Client(API_KEY, API_SECRET)

def fetch_full_klines(symbol, interval, total_limit=50000):
    all_klines = []
    last_time = None

    while len(all_klines) < total_limit:
        fetch_limit = min(1000, total_limit - len(all_klines))
        klines = safe_api_call(
            client.futures_klines,
            symbol=symbol,
            interval=interval,
            limit=fetch_limit,
            endTime=last_time
        )

        if not klines:
            break
        all_klines = klines + all_klines
        last_time = klines[0][0] - 1
        if len(klines) < fetch_limit:
            break
    return all_klines

def get_btc_dominance():
    try:
        r = requests.get('https://api.coingecko.com/api/v3/global').json()
        return r['data']['market_cap_percentage']['btc']
    except:
        return 0.0

def get_dxy():
    """Get DXY (US Dollar Index) - optional feature."""
    try:
        import yfinance as yf
        dxy = yf.download("DX=F", period="7d", interval="1m", progress=False)
        return float(dxy['Close'].iloc[-1])
    except:
        return 0.0  # Return 0 if yfinance not available

def get_funding_rate(symbol="PEOPLEUSDT"):
    try:
        url = f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1'
        r = requests.get(url).json()
        return float(r[0]['fundingRate'])
    except:
        return 0.0

def fetch_features_multi_timeframe():
    all_dfs = []

    for interval in INTERVALS:
        klines = fetch_full_klines(SYMBOL, interval, total_limit=50000)

        if not klines or len(klines) < 1000:
            print(f"[ERROR] Klines for {interval} is too short or empty.")
            continue

        df = pd.DataFrame(klines, columns=[
            'timestamp','open','high','low','close','volume',
            'close_time','quote_asset_volume','num_trades',
            'taker_buy_base','taker_buy_quote','ignore'])

        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        try:
            # === MOMENTUM INDICATORS ===
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_diff'] = df['rsi'].diff()
            df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()  # Fast RSI
            macd = ta.trend.MACD(df['close'])
            df['macd_diff'] = macd.macd_diff()
            df['macd_line'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['mom'] = ta.momentum.ROCIndicator(df['close']).roc()
            df['mom_5'] = ta.momentum.ROCIndicator(df['close'], window=5).roc()  # Short momentum

            # === TREND INDICATORS ===
            df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()

            # ADX - Trend Strength (very important for accuracy!)
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()

            # CCI
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

            # === VOLATILITY INDICATORS ===
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['atr_percent'] = df['atr'] / df['close'] * 100  # ATR as percentage

            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            # Price position within Bollinger Bands (0-1 scale)
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low']).replace(0, np.nan)

            # Keltner Channel
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_high'] = kc.keltner_channel_hband()
            df['kc_low'] = kc.keltner_channel_lband()

            # === VOLUME INDICATORS ===
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ema'] = ta.trend.EMAIndicator(df['volume'], window=20).ema_indicator()
            df['volume_ratio'] = df['volume'] / df['volume_ema']

            # On-Balance Volume
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = ta.trend.EMAIndicator(df['obv'], window=20).ema_indicator()

            # Volume Force Index
            df['fi'] = ta.volume.ForceIndexIndicator(df['close'], df['volume']).force_index()

            # === PRICE ACTION FEATURES ===
            df['candle_body'] = abs(df['close'] - df['open'])
            df['candle_range'] = df['high'] - df['low']
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

            # Normalize candle features
            df['body_to_range'] = df['candle_body'] / df['candle_range'].replace(0, np.nan)
            df['upper_to_range'] = df['upper_shadow'] / df['candle_range'].replace(0, np.nan)
            df['lower_to_range'] = df['lower_shadow'] / df['candle_range'].replace(0, np.nan)

            # === TREND STRENGTH FEATURES (important for accuracy) ===
            # EMA crossover signals
            df['ema_cross'] = (df['ema_9'] - df['ema_20']) / df['close'] * 100
            df['ema_trend'] = (df['ema_20'] - df['ema_50']) / df['close'] * 100

            # Price vs EMAs
            df['price_vs_ema9'] = (df['close'] - df['ema_9']) / df['close'] * 100
            df['price_vs_ema20'] = (df['close'] - df['ema_20']) / df['close'] * 100
            df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['close'] * 100

            # === LAG FEATURES (helps model see patterns) ===
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].pct_change(lag) * 100
                df[f'volume_lag_{lag}'] = df['volume'].pct_change(lag) * 100

            # === ROLLING STATISTICS ===
            df['close_std_10'] = df['close'].rolling(10).std() / df['close'] * 100
            df['close_std_20'] = df['close'].rolling(20).std() / df['close'] * 100
            df['high_low_range_10'] = (df['high'].rolling(10).max() - df['low'].rolling(10).min()) / df['close'] * 100

            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)

        except Exception as e:
            print(f"[ERROR] Failed to compute indicators for {interval}: {e}")
            continue

        df.dropna(inplace=True)

        required_cols = [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',
            # Momentum
            'rsi', 'rsi_diff', 'rsi_6', 'macd_diff', 'macd_line', 'macd_signal',
            'stoch_k', 'stoch_d', 'mom', 'mom_5',
            # Trend
            'ema_9', 'ema_20', 'ema_50', 'sma_20', 'adx', 'adx_pos', 'adx_neg', 'cci',
            # Volatility
            'atr', 'atr_percent', 'bb_width', 'bb_high', 'bb_low', 'bb_mid', 'bb_position',
            'kc_high', 'kc_low',
            # Volume
            'volume_change', 'volume_ema', 'volume_ratio', 'obv', 'obv_ema', 'fi',
            # Price action
            'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
            'body_to_range', 'upper_to_range', 'lower_to_range',
            # Trend strength
            'ema_cross', 'ema_trend', 'price_vs_ema9', 'price_vs_ema20', 'price_vs_ema50',
            # Lag features
            'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5',
            # Rolling stats
            'close_std_10', 'close_std_20', 'high_low_range_10'
        ]

        # Filter to only existing columns (in case some fail)
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = set(required_cols) - set(available_cols)
        if missing_cols:
            print(f"[WARN] Missing columns for {interval}: {missing_cols}")

        if len(available_cols) < 20:  # Need at least 20 features
            print(f"[ERROR] Too few features for {interval}: {len(available_cols)}")
            continue

        df = df[available_cols]
        df.columns = [f"{col}_{interval}" for col in df.columns]
        all_dfs.append(df.reset_index(drop=True))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    if not all_dfs:
        raise ValueError("[CRITICAL] All intervals failed. No data available for training or prediction.")

    # Giữ số dòng đồng nhất
    min_len = min(len(df) for df in all_dfs)
    all_dfs = [df.iloc[-min_len:].reset_index(drop=True) for df in all_dfs]

    combined = pd.concat(all_dfs, axis=1)

    if combined.empty:
        raise ValueError("[CRITICAL] Combined DataFrame is empty after concat and dropna.")

    # === HISTORICAL NEWS PROXY FEATURES (for training) ===
    # These features learn from price action patterns that typically occur around news events
    if HAS_NEWS_PROXY:
        try:
            print("[INFO] Adding historical news proxy features...")
            # Create a temporary df with required columns for news proxy
            primary_tf = INTERVALS[0]
            temp_df = pd.DataFrame({
                'close': combined[f'close_{primary_tf}'],
                'high': combined[f'high_{primary_tf}'],
                'low': combined[f'low_{primary_tf}'],
                'volume': combined[f'volume_{primary_tf}'],
            })

            # Add news proxy features
            temp_df = create_news_proxy_features(temp_df)

            # Copy news features to combined df
            for feature in get_news_proxy_feature_names():
                if feature in temp_df.columns:
                    combined[feature] = temp_df[feature].values

            print(f"[INFO] Added {len(get_news_proxy_feature_names())} news proxy features")

        except Exception as e:
            print(f"[WARN] Failed to create news proxy features: {e}")
            # Add placeholder features
            for feature in ['news_anomaly', 'news_vol_regime', 'news_sentiment_proxy',
                           'news_fear_greed', 'news_event_impact', 'news_score']:
                combined[feature] = 0.0
    else:
        # Fallback: add placeholder news features
        combined['news_anomaly'] = 0.0
        combined['news_vol_regime'] = 0.5
        combined['news_sentiment_proxy'] = 0.0
        combined['news_fear_greed'] = 0.5
        combined['news_event_impact'] = 0.0
        combined['news_score'] = 0.0

    # === EXTERNAL FEATURES (current values for reference) ===
    try:
        combined['btc_dominance'] = get_btc_dominance()
        combined['funding_rate'] = get_funding_rate(SYMBOL)
    except Exception as e:
        print(f"[WARN] Failed to fetch external features: {e}")
        combined['btc_dominance'] = 0.0
        combined['funding_rate'] = 0.0

    # === WHALE/SMART MONEY FEATURES (for higher win rate) ===
    if HAS_WHALE_TRACKER:
        try:
            print("[INFO] Adding whale/smart money features...")

            # Get current whale data
            top_position = get_top_trader_long_short_ratio(SYMBOL)
            top_account = get_top_trader_account_ratio(SYMBOL)
            global_ratio = get_global_long_short_ratio(SYMBOL)
            oi_data = get_open_interest_history(SYMBOL)
            order_book = get_order_book_imbalance(SYMBOL)

            # Add whale features to dataframe
            # Top trader position ratio (smart money direction)
            combined['whale_top_position_ratio'] = top_position['long_short_ratio']
            combined['whale_top_account_ratio'] = top_account['long_short_ratio']

            # Retail sentiment (contrarian indicator)
            combined['whale_retail_ratio'] = global_ratio['long_short_ratio']
            # Contrarian score: when retail is very long, smart money often shorts
            retail_ratio = global_ratio['long_short_ratio']
            combined['whale_contrarian_score'] = np.where(
                retail_ratio > 1.5, -1.0,  # Retail very long -> bearish signal
                np.where(retail_ratio < 0.67, 1.0, 0.0)  # Retail very short -> bullish signal
            )

            # Open interest features
            combined['whale_oi_change'] = oi_data['oi_change_5'] / 10  # Normalized
            combined['whale_oi_trend'] = 1.0 if oi_data['oi_trend'] == 'UP' else (
                -1.0 if oi_data['oi_trend'] == 'DOWN' else 0.0
            )

            # Order book imbalance
            combined['whale_orderbook_imbalance'] = order_book['imbalance']

            # Combined whale score (-1 to +1)
            whale_score = 0.0
            if top_position['long_short_ratio'] > 1.1:
                whale_score += 0.3
            elif top_position['long_short_ratio'] < 0.9:
                whale_score -= 0.3

            if retail_ratio > 1.5:  # Contrarian
                whale_score -= 0.2
            elif retail_ratio < 0.67:
                whale_score += 0.2

            if order_book['imbalance'] > 0.1:
                whale_score += 0.2
            elif order_book['imbalance'] < -0.1:
                whale_score -= 0.2

            combined['whale_combined_score'] = whale_score

            print(f"[INFO] Added 8 whale features. Current whale score: {whale_score:.2f}")

        except Exception as e:
            print(f"[WARN] Failed to create whale features: {e}")
            # Add placeholder features
            combined['whale_top_position_ratio'] = 1.0
            combined['whale_top_account_ratio'] = 1.0
            combined['whale_retail_ratio'] = 1.0
            combined['whale_contrarian_score'] = 0.0
            combined['whale_oi_change'] = 0.0
            combined['whale_oi_trend'] = 0.0
            combined['whale_orderbook_imbalance'] = 0.0
            combined['whale_combined_score'] = 0.0
    else:
        # Fallback: add placeholder whale features
        print("[INFO] Adding placeholder whale features (tracker not available)")
        combined['whale_top_position_ratio'] = 1.0
        combined['whale_top_account_ratio'] = 1.0
        combined['whale_retail_ratio'] = 1.0
        combined['whale_contrarian_score'] = 0.0
        combined['whale_oi_change'] = 0.0
        combined['whale_oi_trend'] = 0.0
        combined['whale_orderbook_imbalance'] = 0.0
        combined['whale_combined_score'] = 0.0

    # Clean up any NaN values
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.fillna(0, inplace=True)

    return combined
