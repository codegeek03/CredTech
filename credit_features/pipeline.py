from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import infer_benchmark, MarketHorizon
from .utils import logger
from .yf_client import YFClient
from .market import market_features
from .fundamentals import core_fundamental_features
from .structural import merton_distance_to_default

def build_features_for_universe(tickers: List[str], period: str = "5y", interval: str = "1d",
                                benchmark_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    client = YFClient(); horizons = MarketHorizon()
    rows = []
    for t in tickers:
        try:
            bm = benchmark_map.get(t) if benchmark_map else infer_benchmark(t)
            logger.info(f"Processing {t} (benchmark {bm})")
            mkt = market_features(t, bm, period, interval, horizons)
            tkr = client.ticker(t)
            try:
                mcap = getattr(tkr, "fast_info", {}).get("market_cap", None) or getattr(tkr, "info", {}).get("marketCap", None)
            except Exception:
                mcap = None
            stmts = client.statements(t)
            fnd = core_fundamental_features(stmts, mcap)
            fnd["turnover_3m"] = (mkt.get("adv_value_3m") / fnd.get("market_cap")) if (not np.isnan(mkt.get("adv_value_3m", np.nan)) and not np.isnan(fnd.get("market_cap", np.nan))) else np.nan
            dd = merton_distance_to_default(
                equity_value=fnd.get("market_cap"),
                equity_vol=mkt.get("equity_vol_1y"),
                debt_value=fnd.get("total_debt"),
                risk_free=mkt.get("rf_estimate", 0.03),
                horizon_years=1.0
            )
            row = {"ticker": t, "benchmark": bm}; row.update(mkt); row.update(fnd); row.update(dd)
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
