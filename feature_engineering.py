"""
Microstructure Feature Engineering from L2 Order Book Data

Demonstrates:
- Vectorised NumPy operations (no Python loops on large arrays)
- Financial microstructure metrics (OBI, VPIN, Kyle's Lambda)
- Rolling statistics with cumsum trick (O(n) instead of O(n*window))
- Production feature pipeline design

Features extracted per tick:
- Order Book Imbalance (OBI) — buying vs selling pressure
- Volume-synchronised PIN (VPIN) — informed trading probability
- Kyle's Lambda — price impact per unit of order flow
- Amihud illiquidity — absolute return per unit volume
- Depth-weighted spread — cost of walking the book
"""

import numpy as np
import pandas as pd


def compute_obi(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    levels: int = 5,
) -> np.ndarray:
    """
    Order Book Imbalance across top N levels.

    OBI = sum(bid_vol) / (sum(bid_vol) + sum(ask_vol))

    Values:
        > 0.5 → buying pressure (bullish signal)
        < 0.5 → selling pressure (bearish signal)
        = 0.5 → balanced book
    """
    # bid_volumes shape: (n_ticks, n_levels)
    bid_sum = bid_volumes[:, :levels].sum(axis=1)
    ask_sum = ask_volumes[:, :levels].sum(axis=1)
    total = bid_sum + ask_sum

    # Avoid division by zero
    obi = np.where(total > 0, bid_sum / total, 0.5)
    return obi


def compute_exponential_obi(
    bid_volumes: np.ndarray,
    ask_volumes: np.ndarray,
    levels: int = 10,
    decay: float = 0.85,
) -> np.ndarray:
    """
    Exponentially-weighted OBI — closer levels matter more.

    Decay of 0.85 means level 1 has weight 1.0, level 2 has 0.85,
    level 3 has 0.72, etc. Better captures near-touch liquidity dynamics.
    """
    weights = np.array([decay**i for i in range(levels)])
    weights /= weights.sum()  # normalise

    bid_weighted = (bid_volumes[:, :levels] * weights).sum(axis=1)
    ask_weighted = (ask_volumes[:, :levels] * weights).sum(axis=1)
    total = bid_weighted + ask_weighted

    return np.where(total > 0, bid_weighted / total, 0.5)


def compute_vpin(
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    bucket_size: int = 50,
    n_buckets: int = 20,
) -> np.ndarray:
    """
    Volume-synchronised Probability of Informed Trading (VPIN).

    Groups trades into fixed-volume buckets, then measures the
    imbalance between buy and sell volume. High VPIN indicates
    informed traders are active → potential adverse selection.

    Returns array same length as input with forward-filled VPIN values.
    """
    n = len(buy_volume)
    vpin = np.full(n, np.nan)

    # Accumulate volume into buckets
    cum_buy = np.cumsum(buy_volume)
    cum_sell = np.cumsum(sell_volume)
    cum_total = cum_buy + cum_sell

    # Find bucket boundaries
    bucket_boundaries = np.arange(bucket_size, cum_total[-1], bucket_size)
    bucket_indices = np.searchsorted(cum_total, bucket_boundaries)

    if len(bucket_indices) < n_buckets:
        return np.full(n, 0.5)  # Not enough data

    for i in range(n_buckets, len(bucket_indices)):
        start_idx = bucket_indices[i - n_buckets]
        end_idx = bucket_indices[i]

        if end_idx >= n:
            break

        # Volume imbalance across last n_buckets
        buy_vol = cum_buy[end_idx] - cum_buy[start_idx]
        sell_vol = cum_sell[end_idx] - cum_sell[start_idx]
        total_vol = buy_vol + sell_vol

        if total_vol > 0:
            vpin[end_idx] = abs(buy_vol - sell_vol) / total_vol

    # Forward-fill
    mask = ~np.isnan(vpin)
    if mask.any():
        indices = np.arange(n)
        vpin = np.interp(indices, indices[mask], vpin[mask])

    return vpin


def compute_kyles_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Kyle's Lambda — price impact coefficient.

    Measures how much prices move per unit of signed order flow.
    High lambda = illiquid market (small trades move price).
    Low lambda = liquid market (large trades needed to move price).

    Uses rolling OLS regression: delta_price = lambda * signed_volume + epsilon
    Vectorised with cumsum trick for O(n) performance.
    """
    n = len(prices)
    kyles_lambda = np.full(n, np.nan)

    # Price changes
    dp = np.diff(prices, prepend=prices[0])

    # Signed volume (positive = buy, negative = sell using tick rule)
    tick_sign = np.sign(dp)
    tick_sign[tick_sign == 0] = 1  # neutral → buy
    signed_vol = volumes * tick_sign

    # Rolling regression using cumsum trick
    # For y = a + bx, b = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    cum_xy = np.cumsum(signed_vol * dp)
    cum_x = np.cumsum(signed_vol)
    cum_y = np.cumsum(dp)
    cum_x2 = np.cumsum(signed_vol**2)

    for i in range(window, n):
        j = i - window
        n_w = window

        sum_xy = cum_xy[i] - cum_xy[j]
        sum_x = cum_x[i] - cum_x[j]
        sum_y = cum_y[i] - cum_y[j]
        sum_x2 = cum_x2[i] - cum_x2[j]

        denom = n_w * sum_x2 - sum_x**2
        if abs(denom) > 1e-10:
            kyles_lambda[i] = (n_w * sum_xy - sum_x * sum_y) / denom

    return kyles_lambda


def compute_amihud_illiquidity(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Amihud Illiquidity Ratio — |return| / volume.

    High values = illiquid (small volume causes big price moves).
    Computed as rolling average over the window.

    Vectorised with cumsum for O(n) performance.
    """
    returns = np.abs(np.diff(np.log(prices), prepend=0))

    # Avoid division by zero
    safe_vol = np.maximum(volumes, 1e-10)
    ratio = returns / safe_vol

    # Rolling mean via cumsum
    cum = np.cumsum(ratio)
    rolling_mean = np.full(len(prices), np.nan)
    rolling_mean[window:] = (cum[window:] - cum[:-window]) / window

    return rolling_mean


def compute_depth_weighted_spread(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    target_volume: float = 1.0,
) -> np.ndarray:
    """
    Depth-weighted spread — cost to execute a target volume.

    Unlike simple bid-ask spread, this accounts for walking the book.
    If you need to buy 1 BTC but best ask only has 0.3, you'll
    consume deeper levels at worse prices.

    Returns effective spread in basis points.
    """
    n = bid_prices.shape[0]
    spreads = np.full(n, np.nan)

    for i in range(n):
        # Walk the ask side (buying)
        remaining = target_volume
        cost = 0.0
        for j in range(bid_prices.shape[1]):
            if remaining <= 0:
                break
            fill = min(remaining, ask_sizes[i, j])
            cost += fill * ask_prices[i, j]
            remaining -= fill

        if remaining > 0:
            continue  # Not enough depth

        avg_buy = cost / target_volume

        # Walk the bid side (selling)
        remaining = target_volume
        proceeds = 0.0
        for j in range(bid_prices.shape[1]):
            if remaining <= 0:
                break
            fill = min(remaining, bid_sizes[i, j])
            proceeds += fill * bid_prices[i, j]
            remaining -= fill

        if remaining > 0:
            continue

        avg_sell = proceeds / target_volume
        mid = (bid_prices[i, 0] + ask_prices[i, 0]) / 2

        if mid > 0:
            spreads[i] = (avg_buy - avg_sell) / mid * 10_000  # basis points

    return spreads


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build complete feature matrix from OHLCV + order book data.

    Input columns: open, high, low, close, volume,
                   bid_vol_1..10, ask_vol_1..10,
                   bid_price_1..10, ask_price_1..10

    Output: Original columns + ~20 engineered features.
    """
    features = df.copy()

    # --- Price features ---
    features["returns_1m"] = df["close"].pct_change()
    features["returns_5m"] = df["close"].pct_change(5)
    features["returns_15m"] = df["close"].pct_change(15)
    features["log_volume"] = np.log1p(df["volume"])

    # --- Volatility (rolling std of returns) ---
    features["volatility_20"] = features["returns_1m"].rolling(20).std()
    features["volatility_60"] = features["returns_1m"].rolling(60).std()

    # --- Order book features ---
    bid_cols = [f"bid_vol_{i}" for i in range(1, 11)]
    ask_cols = [f"ask_vol_{i}" for i in range(1, 11)]

    if all(c in df.columns for c in bid_cols + ask_cols):
        bid_vols = df[bid_cols].values
        ask_vols = df[ask_cols].values

        features["obi_5"] = compute_obi(bid_vols, ask_vols, levels=5)
        features["obi_10"] = compute_obi(bid_vols, ask_vols, levels=10)
        features["exp_obi"] = compute_exponential_obi(bid_vols, ask_vols)

    # --- Microstructure ---
    features["kyles_lambda"] = compute_kyles_lambda(
        df["close"].values, df["volume"].values
    )
    features["amihud"] = compute_amihud_illiquidity(
        df["close"].values, df["volume"].values
    )

    # --- RSI (vectorised) ---
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # --- ADX (simplified) ---
    features["atr_14"] = (
        pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .ewm(span=14, adjust=False)
        .mean()
    )

    # Drop NaN rows from rolling calculations
    features = features.dropna()

    return features
