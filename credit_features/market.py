from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf

from .utils import max_drawdown
from .config import MarketHorizon

def _returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()

def _ann_vol(daily_returns: pd.Series) -> float:
    if daily_returns is None or daily_returns.empty: return np.nan
    return float(daily_returns.std() * np.sqrt(252))

def _beta(asset_ret: pd.Series, bench_ret: Optional[pd.Series]) -> float:
    if asset_ret is None or asset_ret.empty or bench_ret is None or bench_ret.empty: return np.nan
    df = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    if df.shape[0] < 60: return np.nan
    cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
    var = np.var(df.iloc[:,1])
    if var == 0: return np.nan
    return float(cov / var)

def risk_free_rate_est() -> float:
    for sym in ["^IRX", "^TNX"]:
        try:
            h = yf.download(sym, period="1y", interval="1d", auto_adjust=False, progress=False)
            if h is not None and not h.empty:
                return float(h["Close"].dropna().iloc[-1]) / 100.0
        except Exception:
            continue
    return 0.03

def market_features(symbol: str, benchmark: str, period: str, interval: str, horizons: MarketHorizon) -> Dict[str, Any]:
    hist = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if hist is None or hist.empty: raise ValueError(f"No price history for {symbol}")
    close = hist["Close"].dropna()
    volm  = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)

    bench = yf.download(benchmark, period=period, interval=interval, auto_adjust=True, progress=False)
    bench_close = bench["Close"].dropna() if bench is not None and not bench.empty else None

    r = _returns(close)
    rb = _returns(bench_close) if bench_close is not None else None

    def trailing(series: pd.Series, n: int):
        if series is None or series.empty: return np.nan
        if len(series) < n: n = len(series)
        return float(series.tail(1).values[0] / series.tail(n).values[0] - 1.0)

    adv_3m = np.nan
    if not volm.empty:
        n = min(len(volm), horizons.days_63)
        adv_3m = float((volm.tail(n) * close.tail(n)).mean())

    feats = {
        "last_price": float(close.iloc[-1]),
        "ret_1m": trailing(close, horizons.days_21),
        "ret_3m": trailing(close, horizons.days_63),
        "ret_6m": trailing(close, horizons.days_126),
        "ret_1y": trailing(close, horizons.days_252),
        "vol_1y": _ann_vol(r.tail(horizons.days_252)),
        "beta_1y": _beta(r.tail(horizons.days_252), rb.tail(horizons.days_252) if rb is not None else None),
        "mdd_1y": max_drawdown(close.tail(horizons.days_252)),
        "adv_value_3m": adv_3m,
        "equity_vol_1y": _ann_vol(r.tail(horizons.days_252)),
        "rf_estimate": risk_free_rate_est(),
    }
    return feats
