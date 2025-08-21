from typing import Optional, Dict, Any
import numpy as np

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
