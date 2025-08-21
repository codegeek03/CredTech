from typing import Dict, Any, Optional
import pandas as pd
import yfinance as yf
from .utils import retry

class YFClient:
    def __init__(self, session: Optional[Any] = None):
        self.session = session
        self._tickers: Dict[str, yf.Ticker] = {}

    def ticker(self, symbol: str) -> yf.Ticker:
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol, session=self.session)
        return self._tickers[symbol]

    @retry(n=3, backoff=1.7)
    def download_history(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise ValueError(f"No price history for {symbol}")
        return df

    def statements(self, symbol: str) -> Dict[str, Any]:
        t = self.ticker(symbol)
        def as_df(x):
            try:
                return x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
            except Exception:
                return pd.DataFrame()
        data = {
            "financials_annual": as_df(getattr(t, "financials", pd.DataFrame())),
            "financials_quarterly": as_df(getattr(t, "quarterly_financials", pd.DataFrame())),
            "balance_annual": as_df(getattr(t, "balance_sheet", pd.DataFrame())),
            "balance_quarterly": as_df(getattr(t, "quarterly_balance_sheet", pd.DataFrame())),
            "cashflow_annual": as_df(getattr(t, "cashflow", pd.DataFrame())),
            "cashflow_quarterly": as_df(getattr(t, "quarterly_cashflow", pd.DataFrame())),
        }
        try:
            data["info"] = getattr(t, "info", {}) or {}
        except Exception:
            data["info"] = {}
        try:
            data["fast_info"] = getattr(t, "fast_info", {}) or {}
        except Exception:
            data["fast_info"] = {}
        return data
