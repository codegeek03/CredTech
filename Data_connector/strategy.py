# credit_features/strategy.py
from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from math import erf

# --- small helpers -----------------------------------------------------------
def _safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def _z_by_sector(df: pd.DataFrame, col: str) -> pd.Series:
    if "sector" in df.columns:
        return df.groupby("sector")[col].transform(_zscore)
    return _zscore(df[col])

def _norm_cdf(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 0.5 * (1.0 + (s / np.sqrt(2.0)).apply(lambda v: erf(v) if np.isfinite(v) else np.nan))

# --- main API ----------------------------------------------------------------
def add_credit_strategy_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector-aware, credit-relevant indicators + composite credit score to the pipeline output.
    Expects columns your core pipeline already produces (any missing will be handled as NaN).
    """
    out = df.copy()

    # ----- Market signals
    # Sharpe-like proxy = (ret_1y - rf) / vol
    if "ret_1y" in out.columns:
        out["sharpe_proxy"] = pd.to_numeric(out["ret_1y"], errors="coerce")
        if "rf_estimate" in out.columns and "vol_1y" in out.columns:
            out["sharpe_proxy"] = _safe_div(out["ret_1y"] - out["rf_estimate"], out["vol_1y"])
    else:
        out["sharpe_proxy"] = np.nan

    out["drawdown_severity"] = -pd.to_numeric(out.get("mdd_1y"), errors="coerce")  # bigger is worse -> invert later
    out["beta_gap_sector"] = (
        pd.to_numeric(out.get("beta_1y"), errors="coerce") -
        pd.to_numeric(out.get("beta_vs_sector"), errors="coerce")
    )
    out["rel_perf_sector"] = pd.to_numeric(out.get("relative_perf_sector"), errors="coerce")

    # ----- Liquidity (market)
    if "turnover_3m" in out.columns:
        out["liq_turnover"] = pd.to_numeric(out["turnover_3m"], errors="coerce")
    else:
        out["liq_turnover"] = _safe_div(out.get("adv_value_3m"), out.get("market_cap"))

    # ----- Leverage / Coverage / Cash quality
    out["lev_nd_ebitda"] = pd.to_numeric(out.get("net_debt_to_ebitda"), errors="coerce")
    out["cov_interest"]  = pd.to_numeric(out.get("interest_coverage"), errors="coerce")
    out["fcf_debt"]      = pd.to_numeric(out.get("fcf_to_debt"), errors="coerce")
    out["earn_quality"]  = pd.to_numeric(out.get("cfo_to_net_income"), errors="coerce")
    out["profitability"] = pd.to_numeric(out.get("ebit_margin"), errors="coerce")
    out["growth"]        = pd.to_numeric(out.get("revenue_yoy"), errors="coerce")

    # ----- Structural -> PD proxy from DD
    if "dd_proxy" in out.columns:
        out["pd_proxy_1y"] = 1.0 - _norm_cdf(out["dd_proxy"])
    else:
        out["pd_proxy_1y"] = np.nan

    # ----- Sector-neutral z-scores (higher z = safer unless inverted)
    out["z_momentum"]     = _z_by_sector(out, "ret_6m") if "ret_6m" in out else np.nan
    out["z_sharpe"]       = _z_by_sector(out, "sharpe_proxy")
    out["z_drawdown"]     = -_z_by_sector(out, "drawdown_severity")     # invert
    out["z_beta_gap"]     = -_z_by_sector(out, "beta_gap_sector")       # invert
    out["z_rel_perf"]     = _z_by_sector(out, "rel_perf_sector")

    out["z_liquidity"]    = _z_by_sector(out, "liq_turnover")

    out["z_leverage"]     = -_z_by_sector(out, "lev_nd_ebitda")         # invert
    out["z_coverage"]     = _z_by_sector(out, "cov_interest")
    out["z_cashflow"]     = _z_by_sector(out, "fcf_debt")
    out["z_quality"]      = _z_by_sector(out, "earn_quality")
    out["z_profitability"]= _z_by_sector(out, "profitability")
    out["z_growth"]       = _z_by_sector(out, "growth")

    out["z_structural"]   = _z_by_sector(out, "dd_proxy") if "dd_proxy" in out else np.nan

    # ----- Composite (weights sum to 1.0). Tweak per sector if you like.
    weights: Dict[str, float] = {
        "z_momentum": 0.08,
        "z_sharpe": 0.08,
        "z_drawdown": 0.08,
        "z_beta_gap": 0.04,
        "z_rel_perf": 0.04,
        "z_liquidity": 0.10,
        "z_leverage": 0.14,
        "z_coverage": 0.12,
        "z_cashflow": 0.10,
        "z_quality": 0.06,
        "z_profitability": 0.06,
        "z_growth": 0.04,
        "z_structural": 0.06,
    }

    cs = np.zeros(len(out), dtype=float)
    for k, w in weights.items():
        if k in out.columns:
            cs += w * pd.to_numeric(out[k], errors="coerce").fillna(0.0).values
    out["credit_score_raw"] = cs

    # Scale to 0–100 within the current universe
    mn, mx = out["credit_score_raw"].min(), out["credit_score_raw"].max()
    if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
        out["credit_score_0_100"] = (out["credit_score_raw"] - mn) / (mx - mn) * 100.0
    else:
        out["credit_score_0_100"] = np.nan

    # Simple quantile buckets -> implied rating
    def bucketize(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.isna().all(): return pd.Series(np.nan, index=s.index)
        q = s.quantile([0.2, 0.4, 0.6, 0.8])
        def b(v):
            if pd.isna(v): return np.nan
            if v >= q[0.8]: return "A"
            if v >= q[0.6]: return "BBB"
            if v >= q[0.4]: return "BB"
            if v >= q[0.2]: return "B"
            return "CCC"
        return s.apply(b)

    out["implied_rating"] = bucketize(out["credit_score_0_100"])
    return out


def explain_row(df_scored: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Convenience: return all z-components and the final score for one ticker,
    so you can explain *why* its score is what it is.
    """
    row = df_scored[df_scored["ticker"] == ticker].head(1).T
    if row.empty: 
        raise ValueError(f"{ticker} not found in df_scored")
    return row.squeeze()
