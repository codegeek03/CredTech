from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import time, logging, math


@dataclass
class MarketHorizon:
    days_21: int = 21      # ~1m trading days
    days_63: int = 63      # ~3m
    days_126: int = 126    # ~6m
    days_252: int = 252    # ~1y

def infer_benchmark(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".NS"): return "^NSEI"     # Nifty 50
    if t.endswith(".BO"): return "^BSESN"    # Sensex
    if t.endswith(".L"):  return "^FTSE"     # FTSE 100
    if t.endswith(".TO"): return "^GSPTSE"   # S&P/TSX
    if t.endswith(".HK"): return "^HSI"      # Hang Seng
    if t.endswith(".AX"): return "^AXJO"     # ASX 200
    if t.endswith(".TW"): return "^TWII"     # TAIEX
    return "^GSPC"                           # S&P 500 fallback




logger = logging.getLogger("credtech")
if not logger.handlers:
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_sh)
logger.setLevel(logging.INFO)

def retry(n=3, backoff=1.6):
    def deco(fn):
        def wrapper(*args, **kwargs):
            last = None
            for i in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last = e
                    wait = backoff ** i
                    logger.warning(f"{fn.__name__} failed (attempt {i+1}/{n}): {e} — retrying in {wait:.1f}s")
                    time.sleep(wait)
            raise last
        return wrapper
    return deco

def safe_div(a, b):
    try:
        if a is None or b is None: return np.nan
        b = float(b)
        if b == 0.0: return np.nan
        return float(a) / b
    except Exception:
        return np.nan

def pct_change(a, b):
    try:
        if b is None: return np.nan
        b = float(b)
        if b == 0.0: return np.nan
        return (float(a) - b) / b
    except Exception:
        return np.nan

def first_not_nan(*vals):
    for v in vals:
        try:
            if v is None: 
                continue
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


class MarketFeatures:
    @staticmethod
    def _returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    @staticmethod
    def _ann_vol(daily_returns: pd.Series) -> float:
        if daily_returns is None or daily_returns.empty: return np.nan
        return float(daily_returns.std() * np.sqrt(252))

    @staticmethod
    def _beta(asset_ret: pd.Series, bench_ret: Optional[pd.Series]) -> float:
        if asset_ret is None or asset_ret.empty or bench_ret is None or bench_ret.empty: return np.nan
        df = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
        if df.shape[0] < 60: return np.nan
        cov = np.cov(df.iloc[:,0], df.iloc[:,1])[0,1]
        var = np.var(df.iloc[:,1])
        if var == 0: return np.nan
        return float(cov / var)

    @staticmethod
    def risk_free_rate_est() -> float:
        for sym in ["^IRX", "^TNX"]:
            try:
                h = yf.download(sym, period="1y", interval="1d", auto_adjust=False, progress=False)
                if h is not None and not h.empty:
                    return float(h["Close"].dropna().iloc[-1]) / 100.0
            except Exception:
                continue
        return 0.03

    @classmethod
    def extract(cls, symbol: str, benchmark: str, period: str, interval: str, horizons: MarketHorizon) -> Dict[str, Any]:
        hist = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
        if hist is None or hist.empty: raise ValueError(f"No price history for {symbol}")
        close = hist["Close"].dropna()
        volm  = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)

        bench = yf.download(benchmark, period=period, interval=interval, auto_adjust=True, progress=False)
        bench_close = bench["Close"].dropna() if bench is not None and not bench.empty else None

        r = cls._returns(close)
        rb = cls._returns(bench_close) if bench_close is not None else None

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
            "vol_1y": cls._ann_vol(r.tail(horizons.days_252)),
            "beta_1y": cls._beta(r.tail(horizons.days_252), rb.tail(horizons.days_252) if rb is not None else None),
            "mdd_1y": max_drawdown(close.tail(horizons.days_252)),
            "adv_value_3m": adv_3m,
            "equity_vol_1y": cls._ann_vol(r.tail(horizons.days_252)),
            "rf_estimate": cls.risk_free_rate_est(),
        }
        return feats

# ----------------- Fundamentals (class-based) -----------------
class FundamentalFeatures:
    # ---- Helpers to fetch common line items with loose matching ----
    @staticmethod
    def _get_row(series: Optional[pd.Series], key: str):
        if series is None or series.empty: return np.nan
        for k in [key, key.title(), key.upper(), key.lower()]:
            try:
                if k in series.index: return float(series.loc[k])
            except Exception:
                pass
        try:
            matches = [idx for idx in series.index if key.lower() in str(idx).lower()]
            if matches: return float(series.loc[matches[0]])
        except Exception:
            pass
        return np.nan

    @classmethod
    def _sum_debt(cls, bs: pd.Series) -> float:
        keys = ["Total Debt","Short Long Term Debt","Short Term Debt","Current Debt","Long Term Debt"]
        vals = [cls._get_row(bs,k) for k in keys]
        if not all(np.isnan(vals)):
            if not np.isnan(vals[0]): return vals[0]
            parts = [v for v in vals[1:] if not np.isnan(v)]
            return float(np.nansum(parts)) if parts else np.nan
        try:
            debtish = [float(bs.loc[i]) for i in bs.index if "debt" in str(i).lower()]
            return float(np.nansum(debtish)) if debtish else np.nan
        except Exception:
            return np.nan

    @classmethod
    def _cash_equivalents(cls, bs: pd.Series) -> float:
        ca = cls._get_row(bs, "Cash And Cash Equivalents")
        if np.isnan(ca): ca = cls._get_row(bs, "Cash")
        return ca

    @classmethod
    def _ebit(cls, fin: pd.Series) -> float:
        return first_not_nan(cls._get_row(fin,"Ebit"), cls._get_row(fin,"EBIT"), cls._get_row(fin,"Operating Income"))

    @classmethod
    def _ebitda(cls, fin: pd.Series) -> float:
        return first_not_nan(cls._get_row(fin,"Ebitda"), cls._get_row(fin,"EBITDA"))

    @classmethod
    def _interest_expense(cls, fin: pd.Series) -> float:
        ie = cls._get_row(fin, "Interest Expense")
        if not np.isnan(ie): return abs(ie)
        return np.nan

    @classmethod
    def _sales(cls, fin: pd.Series) -> float:
        return first_not_nan(cls._get_row(fin,"Total Revenue"), cls._get_row(fin,"Revenue"), cls._get_row(fin,"Sales"))

    @classmethod
    def _net_income(cls, fin: pd.Series) -> float:
        return cls._get_row(fin, "Net Income")

    @classmethod
    def _gross_profit(cls, fin: pd.Series) -> float:
        return cls._get_row(fin, "Gross Profit")

    @classmethod
    def _current_assets(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Total Current Assets")

    @classmethod
    def _current_liabilities(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Total Current Liabilities")

    @classmethod
    def _inventory(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Inventory")

    @classmethod
    def _total_assets(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Total Assets")

    @classmethod
    def _total_liab(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Total Liab")

    @classmethod
    def _retained_earnings(cls, bs: pd.Series) -> float:
        return cls._get_row(bs, "Retained Earnings")

    @classmethod
    def _total_equity(cls, bs: pd.Series) -> float:
        return first_not_nan(cls._get_row(bs,"Total Stockholder Equity"), cls._get_row(bs,"Total Shareholder Equity"))

    @classmethod
    def _operating_cf(cls, cf: pd.Series) -> float:
        return first_not_nan(cls._get_row(cf,"Operating Cash Flow"), cls._get_row(cf,"Total Cash From Operating Activities"))

    @classmethod
    def _capex(cls, cf: pd.Series) -> float:
        return cls._get_row(cf,"Capital Expenditures")

    @classmethod
    def _interest_paid_cf(cls, cf: pd.Series) -> float:
        return abs(cls._get_row(cf,"Interest Paid"))

    @classmethod
    def _free_cash_flow(cls, cf: pd.Series) -> float:
        f = cls._get_row(cf,"Free Cash Flow")
        if not np.isnan(f): return f
        cfo = cls._operating_cf(cf); capex = cls._capex(cf)
        if np.isnan(cfo) or np.isnan(capex): return np.nan
        return float(cfo + capex)

    @classmethod
    def extract(cls, stmts: Dict[str, Any], market_cap: Optional[float]) -> Dict[str, Any]:
        fin_y = latest_col(stmts.get("financials_annual")); fin_y_prev = prev_col(stmts.get("financials_annual"))
        bs_y  = latest_col(stmts.get("balance_annual"));    bs_y_prev  = prev_col(stmts.get("balance_annual"))
        cf_y  = latest_col(stmts.get("cashflow_annual"))
        fin_q = latest_col(stmts.get("financials_quarterly"))
        bs_q  = latest_col(stmts.get("balance_quarterly"))
        cf_q  = latest_col(stmts.get("cashflow_quarterly"))

        fin = fin_y if fin_y is not None else fin_q
        bs  = bs_y  if bs_y  is not None else bs_q
        cf  = cf_y  if cf_y  is not None else cf_q

        debt = cls._sum_debt(bs) if bs is not None else np.nan
        cash = cls._cash_equivalents(bs) if bs is not None else np.nan
        ebit = cls._ebit(fin) if fin is not None else np.nan
        ebitda = cls._ebitda(fin) if fin is not None else np.nan
        interest = cls._interest_expense(fin) if fin is not None else np.nan
        sales = cls._sales(fin) if fin is not None else np.nan
        gross_profit = cls._gross_profit(fin) if fin is not None else np.nan
        net_income = cls._net_income(fin) if fin is not None else np.nan
        curr_assets = cls._current_assets(bs) if bs is not None else np.nan
        curr_liab   = cls._current_liabilities(bs) if bs is not None else np.nan
        inventory   = cls._inventory(bs) if bs is not None else np.nan
        tot_assets  = cls._total_assets(bs) if bs is not None else np.nan
        tot_liab    = cls._total_liab(bs) if bs is not None else np.nan
        retained    = cls._retained_earnings(bs) if bs is not None else np.nan
        equity      = cls._total_equity(bs) if bs is not None else np.nan
        cfo         = cls._operating_cf(cf) if cf is not None else np.nan
        capex       = cls._capex(cf) if cf is not None else np.nan
        fcf         = cls._free_cash_flow(cf) if cf is not None else np.nan

        net_debt = (debt - cash) if (not np.isnan(debt) and not np.isnan(cash)) else np.nan
        current_ratio = safe_div(curr_assets, curr_liab)
        quick_ratio   = safe_div((curr_assets - inventory) if (not np.isnan(curr_assets) and not np.isnan(inventory)) else np.nan, curr_liab)
        cash_ratio    = safe_div(cash, curr_liab)
        debt_to_equity = safe_div(debt, equity)
        net_debt_to_ebitda = safe_div(net_debt, ebitda)
        interest_cov  = safe_div(ebit, interest)
        gross_margin  = safe_div(gross_profit, sales)
        ebit_margin   = safe_div(ebit, sales)
        net_margin    = safe_div(net_income, sales)
        roa = safe_div(net_income, tot_assets)
        roe = safe_div(net_income, equity)
        fcf_to_debt = safe_div(fcf, debt)
        cfo_to_interest = safe_div(cfo, cls._interest_paid_cf(cf) if cf is not None else np.nan)
        cfo_to_net_income = safe_div(cfo, net_income)

        sales_prev = cls._sales(fin_y_prev) if fin_y_prev is not None else np.nan
        ebit_prev  = cls._ebit(fin_y_prev) if fin_y_prev is not None else np.nan
        ni_prev    = cls._net_income(fin_y_prev) if fin_y_prev is not None else np.nan
        revenue_yoy = pct_change(sales, sales_prev)
        ebit_yoy    = pct_change(ebit, ebit_prev)
        ni_yoy      = pct_change(net_income, ni_prev)

        feats = {
            "market_cap": float(market_cap) if market_cap is not None else np.nan,
            "total_debt": debt, "cash_and_equiv": cash, "net_debt": net_debt,
            "ebit": ebit, "ebitda": ebitda, "interest_expense_abs": interest,
            "sales": sales, "gross_profit": gross_profit, "net_income": net_income,
            "total_assets": tot_assets, "total_liab": tot_liab, "equity": equity,
            "current_ratio": current_ratio, "quick_ratio": quick_ratio, "cash_ratio": cash_ratio,
            "debt_to_equity": debt_to_equity, "net_debt_to_ebitda": net_debt_to_ebitda, "interest_coverage": interest_cov,
            "gross_margin": gross_margin, "ebit_margin": ebit_margin, "net_margin": net_margin,
            "roa": roa, "roe": roe,
            "fcf": fcf, "cfo": cfo, "capex": capex,
            "fcf_to_debt": fcf_to_debt, "cfo_to_interest": cfo_to_interest, "cfo_to_net_income": cfo_to_net_income,
            "revenue_yoy": revenue_yoy, "ebit_yoy": ebit_yoy, "net_income_yoy": ni_yoy,
        }

        wc = (curr_assets - curr_liab) if (not np.isnan(curr_assets) and not np.isnan(curr_liab)) else np.nan
        re = retained; mve = market_cap
        z = np.nan
        if not any(np.isnan(x) for x in [wc, re, ebit, mve, tot_liab, sales, tot_assets]):
            z = 1.2*(wc/tot_assets) + 1.4*(re/tot_assets) + 3.3*(ebit/tot_assets) + 0.6*(mve/tot_liab) + 1.0*(sales/tot_assets)
        feats["altman_z"] = z

        f = 0
        ta_prev = cls._total_assets(bs_y_prev) if bs_y_prev is not None else np.nan
        roa_prev = safe_div(ni_prev, ta_prev)
        f += 1 if (not np.isnan(feats["roa"]) and feats["roa"] > 0) else 0
        f += 1 if (not np.isnan(cfo) and cfo > 0) else 0
        f += 1 if (not np.isnan(roa_prev) and not np.isnan(feats["roa"]) and feats["roa"] > roa_prev) else 0
        f += 1 if (not np.isnan(cfo) and not np.isnan(net_income) and cfo > net_income) else 0
        lt_debt = cls._get_row(bs,"Long Term Debt"); lt_debt_prev = cls._get_row(bs_y_prev,"Long Term Debt") if bs_y_prev is not None else np.nan
        lev = safe_div(lt_debt, tot_assets); lev_prev = safe_div(lt_debt_prev, ta_prev)
        f += 1 if (not np.isnan(lev_prev) and not np.isnan(lev) and lev < lev_prev) else 0
        cr_prev = safe_div(cls._current_assets(bs_y_prev) if bs_y_prev is not None else np.nan,
                           cls._current_liabilities(bs_y_prev) if bs_y_prev is not None else np.nan)
        f += 1 if (not np.isnan(cr_prev) and not np.isnan(current_ratio) and current_ratio > cr_prev) else 0
        gm = safe_div(gross_profit, sales); gm_prev = safe_div(cls._gross_profit(fin_y_prev) if fin_y_prev is not None else np.nan, sales_prev)
        at = safe_div(sales, tot_assets);   at_prev = safe_div(sales_prev, ta_prev)
        f += 1 if (not np.isnan(gm) and not np.isnan(gm_prev) and gm > gm_prev) else 0
        f += 1 if (not np.isnan(at) and not np.isnan(at_prev) and at > at_prev) else 0
        feats["piotroski_f"] = float(f) if f != 0 else np.nan

        return feats
    
class StructuralRisk:
    @staticmethod
    def merton_distance_to_default(equity_value: Optional[float], equity_vol: Optional[float], debt_value: Optional[float],
                                   risk_free: float = 0.03, horizon_years: float = 1.0) -> Dict[str, Any]:
        try:
            if any(x is None or np.isnan(x) or x <= 0 for x in [equity_value, equity_vol, debt_value]):
                return {"dd_proxy": np.nan, "asset_value": np.nan, "asset_vol": np.nan}
            V = float(equity_value) + float(debt_value)
            sigma_E = float(equity_vol)
            asset_vol = sigma_E * (float(equity_value) / V)
            D = float(debt_value); T = float(horizon_years); mu = float(risk_free)
            dd = (np.log(V / D) + (mu - 0.5 * asset_vol**2) * T) / (asset_vol * np.sqrt(T))
            return {"dd_proxy": float(dd), "asset_value": float(V), "asset_vol": float(asset_vol)}
        except Exception:
            return {"dd_proxy": np.nan, "asset_value": np.nan, "asset_vol": np.nan}



class CreditFeaturePipeline:
    def __init__(self, period: str = "5y", interval: str = "1d"):
        self.client = YFClient()
        self.horizons = MarketHorizon()
        self.period = period
        self.interval = interval

    def _market_cap(self, tkr: yf.Ticker) -> Optional[float]:
        try:
            return getattr(tkr, "fast_info", {}).get("market_cap", None) or getattr(tkr, "info", {}).get("marketCap", None)
        except Exception:
            return None

    def build_for_universe(self, tickers: List[str], benchmark_map: Optional[Dict[str,str]] = None) -> pd.DataFrame:
        rows = []
        for t in tickers:
            try:
                bm = benchmark_map.get(t) if benchmark_map else infer_benchmark(t)
                logger.info(f"Processing {t} (benchmark {bm})")
                # Market
                mkt = MarketFeatures.extract(t, bm, self.period, self.interval, self.horizons)
                # Fundamentals
                stmts = self.client.statements(t)
                tkr = self.client.ticker(t)
                mcap = self._market_cap(tkr)
                fnd = FundamentalFeatures.extract(stmts, mcap)
                # Structural
                dd = StructuralRisk.merton_distance_to_default(
                    equity_value=fnd.get("market_cap"),
                    equity_vol=mkt.get("equity_vol_1y"),
                    debt_value=fnd.get("total_debt"),
                    risk_free=mkt.get("rf_estimate", 0.03),
                    horizon_years=1.0
                )
                # Liquidity turnover
                if not np.isnan(mkt.get("adv_value_3m", np.nan)) and not np.isnan(fnd.get("market_cap", np.nan)):
                    fnd["turnover_3m"] = float(mkt["adv_value_3m"]) / float(fnd["market_cap"])
                else:
                    fnd["turnover_3m"] = np.nan

                row = {"ticker": t, "benchmark": bm}
                row.update(mkt); row.update(fnd); row.update(dd)
                rows.append(row)
            except Exception as e:
                logger.warning(f"Failed {t}: {e}")
                rows.append({"ticker": t, "benchmark": None, "error": str(e)})
        df = pd.DataFrame(rows)

        id_cols = ["ticker","benchmark"]
        market_cols = ["last_price","ret_1m","ret_3m","ret_6m","ret_1y","vol_1y","beta_1y","mdd_1y","adv_value_3m","equity_vol_1y","rf_estimate"]
        struct_cols = ["dd_proxy","asset_value","asset_vol"]
        ordered = id_cols + [c for c in market_cols if c in df.columns] + [c for c in df.columns if c not in id_cols + market_cols + struct_cols + ["error"]] + [c for c in struct_cols if c in df.columns] + (["error"] if "error" in df.columns else [])
        return df[ordered]
