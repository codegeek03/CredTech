import time, math, logging
import numpy as np
import pandas as pd

logger = logging.getLogger("credit_features")
logger.setLevel(logging.INFO)
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(_sh)

def retry(n=3, backoff=1.6):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    time.sleep(backoff ** i)
            raise last
        return wrapper
    return deco

def safe_div(a, b):
    try:
        b = float(b)
        if b == 0.0: return np.nan
        return float(a) / b
    except Exception:
        return np.nan

def pct_change(a, b):
    try:
        b = float(b)
        if b == 0.0: return np.nan
        return (float(a) - b) / b
    except Exception:
        return np.nan

def first_not_nan(*vals):
    for v in vals:
        try:
            f = float(v)
            if not (isinstance(f, float) and math.isnan(f)):
                return f
        except Exception:
            continue
    return np.nan

def latest_col(df: pd.DataFrame):
    if df is None or df.empty: return None
    try:
        cols = sorted(df.columns, key=lambda c: pd.to_datetime(c), reverse=True)
        return df[cols[0]]
    except Exception:
        return df.iloc[:, 0]

def prev_col(df: pd.DataFrame):
    if df is None or df.empty or df.shape[1] < 2: return None
    try:
        cols = sorted(df.columns, key=lambda c: pd.to_datetime(c), reverse=True)
        return df[cols[1]]
    except Exception:
        return df.iloc[:, 1]

def max_drawdown(series: pd.Series):
    if series is None or series.empty: return np.nan
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan
