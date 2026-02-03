# historical_whale.py
# Fetch historical whale/sentiment data from Binance and align with training data

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

try:
    from config import API_KEY, API_SECRET, SYMBOL
except ImportError:
    API_KEY = ""
    API_SECRET = ""
    SYMBOL = "BTCUSDT"

BASE_URL = "https://fapi.binance.com"


def fetch_long_short_ratio_history(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    """
    Fetch global long/short account ratio history.
    Shows retail sentiment (contrarian indicator).
    """
    url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(500, limit - len(all_data)),
            "endTime": current_end
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data = data + all_data

            if len(data) < 500:
                break

            current_end = int(data[0]["timestamp"]) - 1
            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Failed to fetch long/short ratio: {e}")
            break

    return all_data


def fetch_top_trader_ratio_history(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    """
    Fetch top trader long/short position ratio history.
    Shows smart money positioning.
    """
    url = f"{BASE_URL}/futures/data/topLongShortPositionRatio"

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(500, limit - len(all_data)),
            "endTime": current_end
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data = data + all_data

            if len(data) < 500:
                break

            current_end = int(data[0]["timestamp"]) - 1
            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Failed to fetch top trader ratio: {e}")
            break

    return all_data


def fetch_top_trader_account_ratio_history(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    """
    Fetch top trader long/short account ratio history.
    """
    url = f"{BASE_URL}/futures/data/topLongShortAccountRatio"

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(500, limit - len(all_data)),
            "endTime": current_end
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data = data + all_data

            if len(data) < 500:
                break

            current_end = int(data[0]["timestamp"]) - 1
            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Failed to fetch top account ratio: {e}")
            break

    return all_data


def fetch_open_interest_history(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    """
    Fetch open interest history.
    Rising OI + rising price = strong trend
    Rising OI + falling price = potential reversal
    """
    url = f"{BASE_URL}/futures/data/openInterestHist"

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(500, limit - len(all_data)),
            "endTime": current_end
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data = data + all_data

            if len(data) < 500:
                break

            current_end = int(data[0]["timestamp"]) - 1
            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Failed to fetch open interest: {e}")
            break

    return all_data


def fetch_taker_buy_sell_ratio_history(symbol="BTCUSDT", period="1h", limit=500, start_time=None, end_time=None):
    """
    Fetch taker buy/sell volume ratio history.
    > 1 = more aggressive buyers
    < 1 = more aggressive sellers
    """
    url = f"{BASE_URL}/futures/data/takerlongshortRatio"

    all_data = []
    current_end = end_time or int(datetime.now().timestamp() * 1000)

    while len(all_data) < limit:
        params = {
            "symbol": symbol,
            "period": period,
            "limit": min(500, limit - len(all_data)),
            "endTime": current_end
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data = data + all_data

            if len(data) < 500:
                break

            current_end = int(data[0]["timestamp"]) - 1
            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Failed to fetch taker ratio: {e}")
            break

    return all_data


def get_historical_whale_features(symbol="BTCUSDT", period="1h", limit=5000):
    """
    Fetch all historical whale/sentiment features and combine into DataFrame.
    Returns DataFrame indexed by timestamp with whale features.
    """
    print(f"[INFO] Fetching historical whale data ({limit} candles)...")

    # Fetch all data types
    print("[INFO] Fetching global long/short ratio...")
    ls_ratio = fetch_long_short_ratio_history(symbol, period, limit)

    print("[INFO] Fetching top trader position ratio...")
    top_position = fetch_top_trader_ratio_history(symbol, period, limit)

    print("[INFO] Fetching top trader account ratio...")
    top_account = fetch_top_trader_account_ratio_history(symbol, period, limit)

    print("[INFO] Fetching open interest history...")
    oi_history = fetch_open_interest_history(symbol, period, limit)

    print("[INFO] Fetching taker buy/sell ratio...")
    taker_ratio = fetch_taker_buy_sell_ratio_history(symbol, period, limit)

    # Convert to DataFrames
    dfs = {}

    if ls_ratio:
        df_ls = pd.DataFrame(ls_ratio)
        df_ls["timestamp"] = pd.to_datetime(df_ls["timestamp"], unit="ms")
        df_ls["whale_retail_long_ratio"] = df_ls["longShortRatio"].astype(float)
        df_ls["whale_retail_long_account"] = df_ls["longAccount"].astype(float)
        df_ls["whale_retail_short_account"] = df_ls["shortAccount"].astype(float)
        dfs["ls"] = df_ls[["timestamp", "whale_retail_long_ratio", "whale_retail_long_account", "whale_retail_short_account"]]
        print(f"[INFO] Global long/short ratio: {len(df_ls)} records")

    if top_position:
        df_top = pd.DataFrame(top_position)
        df_top["timestamp"] = pd.to_datetime(df_top["timestamp"], unit="ms")
        df_top["whale_top_position_ratio"] = df_top["longShortRatio"].astype(float)
        df_top["whale_top_long_account"] = df_top["longAccount"].astype(float)
        df_top["whale_top_short_account"] = df_top["shortAccount"].astype(float)
        dfs["top_pos"] = df_top[["timestamp", "whale_top_position_ratio", "whale_top_long_account", "whale_top_short_account"]]
        print(f"[INFO] Top trader position ratio: {len(df_top)} records")

    if top_account:
        df_acc = pd.DataFrame(top_account)
        df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"], unit="ms")
        df_acc["whale_top_account_ratio"] = df_acc["longShortRatio"].astype(float)
        dfs["top_acc"] = df_acc[["timestamp", "whale_top_account_ratio"]]
        print(f"[INFO] Top trader account ratio: {len(df_acc)} records")

    if oi_history:
        df_oi = pd.DataFrame(oi_history)
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")
        df_oi["whale_open_interest"] = df_oi["sumOpenInterest"].astype(float)
        df_oi["whale_open_interest_value"] = df_oi["sumOpenInterestValue"].astype(float)
        dfs["oi"] = df_oi[["timestamp", "whale_open_interest", "whale_open_interest_value"]]
        print(f"[INFO] Open interest history: {len(df_oi)} records")

    if taker_ratio:
        df_taker = pd.DataFrame(taker_ratio)
        df_taker["timestamp"] = pd.to_datetime(df_taker["timestamp"], unit="ms")
        df_taker["whale_taker_buy_ratio"] = df_taker["buyVol"].astype(float) / (df_taker["buyVol"].astype(float) + df_taker["sellVol"].astype(float))
        df_taker["whale_taker_sell_ratio"] = df_taker["sellVol"].astype(float) / (df_taker["buyVol"].astype(float) + df_taker["sellVol"].astype(float))
        dfs["taker"] = df_taker[["timestamp", "whale_taker_buy_ratio", "whale_taker_sell_ratio"]]
        print(f"[INFO] Taker buy/sell ratio: {len(df_taker)} records")

    if not dfs:
        print("[WARN] No whale data fetched")
        return pd.DataFrame()

    # Merge all DataFrames on timestamp
    result = None
    for name, df in dfs.items():
        if result is None:
            result = df
        else:
            result = pd.merge(result, df, on="timestamp", how="outer")

    # Sort by timestamp
    result = result.sort_values("timestamp").reset_index(drop=True)

    # Calculate derived features
    if "whale_retail_long_ratio" in result.columns:
        # Contrarian score: when retail is very long, often bearish
        result["whale_contrarian_score"] = np.where(
            result["whale_retail_long_ratio"] > 1.5, -1.0,
            np.where(result["whale_retail_long_ratio"] < 0.67, 1.0, 0.0)
        )

    if "whale_open_interest" in result.columns:
        # OI change (momentum)
        result["whale_oi_change"] = result["whale_open_interest"].pct_change(5) * 100
        result["whale_oi_change"] = result["whale_oi_change"].clip(-50, 50)

    if "whale_top_position_ratio" in result.columns and "whale_retail_long_ratio" in result.columns:
        # Smart money vs retail divergence
        result["whale_smart_retail_divergence"] = result["whale_top_position_ratio"] - result["whale_retail_long_ratio"]

    # Combined whale score
    score_cols = []
    if "whale_top_position_ratio" in result.columns:
        score_cols.append(("whale_top_position_ratio", 1.0, 0.3))  # col, neutral, weight
    if "whale_contrarian_score" in result.columns:
        score_cols.append(("whale_contrarian_score", 0.0, 0.2))
    if "whale_taker_buy_ratio" in result.columns:
        score_cols.append(("whale_taker_buy_ratio", 0.5, 0.2))

    if score_cols:
        result["whale_combined_score"] = 0.0
        for col, neutral, weight in score_cols:
            if col in result.columns:
                normalized = (result[col] - neutral).clip(-1, 1)
                result["whale_combined_score"] += normalized * weight
        result["whale_combined_score"] = result["whale_combined_score"].clip(-1, 1)

    # Fill NaN
    result = result.fillna(method="ffill").fillna(0)

    # Set timestamp as index
    result = result.set_index("timestamp")

    print(f"[INFO] Total whale features: {len(result)} records, {len(result.columns)} features")

    return result


# Feature names for training
WHALE_FEATURE_NAMES = [
    "whale_retail_long_ratio",
    "whale_retail_long_account",
    "whale_retail_short_account",
    "whale_top_position_ratio",
    "whale_top_long_account",
    "whale_top_short_account",
    "whale_top_account_ratio",
    "whale_open_interest",
    "whale_open_interest_value",
    "whale_taker_buy_ratio",
    "whale_taker_sell_ratio",
    "whale_contrarian_score",
    "whale_oi_change",
    "whale_smart_retail_divergence",
    "whale_combined_score",
]


if __name__ == "__main__":
    # Test the module
    print("Fetching historical whale data...")
    whale_features = get_historical_whale_features(limit=1000)
    print(f"\nWhale features shape: {whale_features.shape}")
    if not whale_features.empty:
        print(f"\nColumns: {whale_features.columns.tolist()}")
        print(f"\nSample data:\n{whale_features.head()}")
        print(f"\nFeature stats:\n{whale_features.describe()}")
