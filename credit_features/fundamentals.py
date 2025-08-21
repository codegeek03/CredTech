from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from .utils import latest_col, prev_col, safe_div, pct_change, first_not_nan

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

def _sum_debt(bs: pd.Series) -> float:
    keys = ["Total Debt","Short Long Term Debt","Short Term Debt","Current Debt","Long Term Debt"]
    vals = [_get_row(bs,k) for k in keys]
    if not all(np.isnan(vals)):
        if not np.isnan(vals[0]): return vals[0]
        parts = [v for v in vals[1:] if not np.isnan(v)]
        return float(np.nansum(parts)) if parts else np.nan
    try:
        debtish = [float(bs.loc[i]) for i in bs.index if "debt" in str(i).lower()]
        return float(np.nansum(debtish)) if debtish else np.nan
    except Exception:
        return np.nan

def _cash_equivalents(bs: pd.Series) -> float:
    ca = _get_row(bs, "Cash And Cash Equivalents")
    if np.isnan(ca): ca = _get_row(bs, "Cash")
    return ca

def _ebit(fin: pd.Series) -> float:
    return first_not_nan(_get_row(fin,"Ebit"), _get_row(fin,"EBIT"), _get_row(fin,"Operating Income"))

def _ebitda(fin: pd.Series) -> float:
    return first_not_nan(_get_row(fin,"Ebitda"), _get_row(fin,"EBITDA"))

def _interest_expense(fin: pd.Series) -> float:
    ie = _get_row(fin, "Interest Expense")
    if not np.isnan(ie): return abs(ie)
    return np.nan

def _sales(fin: pd.Series) -> float:
    return first_not_nan(_get_row(fin,"Total Revenue"), _get_row(fin,"Revenue"), _get_row(fin,"Sales"))

def _net_income(fin: pd.Series) -> float:
    return _get_row(fin, "Net Income")

def _gross_profit(fin: pd.Series) -> float:
    return _get_row(fin, "Gross Profit")

def _current_assets(bs: pd.Series) -> float:
    return _get_row(bs, "Total Current Assets")

def _current_liabilities(bs: pd.Series) -> float:
    return _get_row(bs, "Total Current Liabilities")

def _inventory(bs: pd.Series) -> float:
    return _get_row(bs, "Inventory")

def _total_assets(bs: pd.Series) -> float:
    return _get_row(bs, "Total Assets")

def _total_liab(bs: pd.Series) -> float:
    return _get_row(bs, "Total Liab")

def _retained_earnings(bs: pd.Series) -> float:
    return _get_row(bs, "Retained Earnings")

def _total_equity(bs: pd.Series) -> float:
    return first_not_nan(_get_row(bs,"Total Stockholder Equity"), _get_row(bs,"Total Shareholder Equity"))

def _operating_cf(cf: pd.Series) -> float:
    return first_not_nan(_get_row(cf,"Operating Cash Flow"), _get_row(cf,"Total Cash From Operating Activities"))

def _capex(cf: pd.Series) -> float:
    return _get_row(cf,"Capital Expenditures")

def _interest_paid_cf(cf: pd.Series) -> float:
    return abs(_get_row(cf,"Interest Paid"))

def _free_cash_flow(cf: pd.Series) -> float:
    f = _get_row(cf,"Free Cash Flow")
    if not np.isnan(f): return f
    cfo = _operating_cf(cf); capex = _capex(cf)
    if np.isnan(cfo) or np.isnan(capex): return np.nan
    return float(cfo + capex)

def core_fundamental_features(stmts: Dict[str, Any], market_cap: Optional[float]) -> Dict[str, Any]:
    fin_y = latest_col(stmts.get("financials_annual")); fin_y_prev = prev_col(stmts.get("financials_annual"))
    bs_y  = latest_col(stmts.get("balance_annual"));    bs_y_prev  = prev_col(stmts.get("balance_annual"))
    cf_y  = latest_col(stmts.get("cashflow_annual"))
    fin_q = latest_col(stmts.get("financials_quarterly"))
    bs_q  = latest_col(stmts.get("balance_quarterly"))
    cf_q  = latest_col(stmts.get("cashflow_quarterly"))

    fin = fin_y if fin_y is not None else fin_q
    bs  = bs_y  if bs_y  is not None else bs_q
    cf  = cf_y  if cf_y  is not None else cf_q

    debt = _sum_debt(bs) if bs is not None else np.nan
    cash = _cash_equivalents(bs) if bs is not None else np.nan
    ebit = _ebit(fin) if fin is not None else np.nan
    ebitda = _ebitda(fin) if fin is not None else np.nan
    interest = _interest_expense(fin) if fin is not None else np.nan
    sales = _sales(fin) if fin is not None else np.nan
    gross_profit = _gross_profit(fin) if fin is not None else np.nan
    net_income = _net_income(fin) if fin is not None else np.nan
    curr_assets = _current_assets(bs) if bs is not None else np.nan
    curr_liab   = _current_liabilities(bs) if bs is not None else np.nan
    inventory   = _inventory(bs) if bs is not None else np.nan
    tot_assets  = _total_assets(bs) if bs is not None else np.nan
    tot_liab    = _total_liab(bs) if bs is not None else np.nan
    retained    = _retained_earnings(bs) if bs is not None else np.nan
    equity      = _total_equity(bs) if bs is not None else np.nan
    cfo         = _operating_cf(cf) if cf is not None else np.nan
    capex       = _capex(cf) if cf is not None else np.nan
    fcf         = _free_cash_flow(cf) if cf is not None else np.nan

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
    cfo_to_interest = safe_div(cfo, _interest_paid_cf(cf) if cf is not None else np.nan)
    cfo_to_net_income = safe_div(cfo, net_income)

    sales_prev = _sales(fin_y_prev) if fin_y_prev is not None else np.nan
    ebit_prev  = _ebit(fin_y_prev) if fin_y_prev is not None else np.nan
    ni_prev    = _net_income(fin_y_prev) if fin_y_prev is not None else np.nan
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
    ta_prev = _total_assets(bs_y_prev) if bs_y_prev is not None else np.nan
    roa_prev = safe_div(ni_prev, ta_prev)
    f += 1 if (not np.isnan(feats["roa"]) and feats["roa"] > 0) else 0
    f += 1 if (not np.isnan(cfo) and cfo > 0) else 0
    f += 1 if (not np.isnan(roa_prev) and not np.isnan(feats["roa"]) and feats["roa"] > roa_prev) else 0
    f += 1 if (not np.isnan(cfo) and not np.isnan(net_income) and cfo > net_income) else 0
    lt_debt = _get_row(bs,"Long Term Debt"); lt_debt_prev = _get_row(bs_y_prev,"Long Term Debt") if bs_y_prev is not None else np.nan
    lev = safe_div(lt_debt, tot_assets); lev_prev = safe_div(lt_debt_prev, ta_prev)
    f += 1 if (not np.isnan(lev_prev) and not np.isnan(lev) and lev < lev_prev) else 0
    cr_prev = safe_div(_current_assets(bs_y_prev) if bs_y_prev is not None else np.nan,
                       _current_liabilities(bs_y_prev) if bs_y_prev is not None else np.nan)
    f += 1 if (not np.isnan(cr_prev) and not np.isnan(current_ratio) and current_ratio > cr_prev) else 0
    gm = safe_div(gross_profit, sales); gm_prev = safe_div(_gross_profit(fin_y_prev) if fin_y_prev is not None else np.nan, sales_prev)
    at = safe_div(sales, tot_assets);   at_prev = safe_div(sales_prev, ta_prev)
    f += 1 if (not np.isnan(gm) and not np.isnan(gm_prev) and gm > gm_prev) else 0
    f += 1 if (not np.isnan(at) and not np.isnan(at_prev) and at > at_prev) else 0
    feats["piotroski_f"] = float(f) if f != 0 else np.nan

    return feats
