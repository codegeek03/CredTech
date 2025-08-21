from dataclasses import dataclass

def infer_benchmark(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".NS"): return "^NSEI"
    if t.endswith(".BO"): return "^BSESN"
    if t.endswith(".L"):  return "^FTSE"
    if t.endswith(".TO"): return "^GSPTSE"
    if t.endswith(".HK"): return "^HSI"
    if t.endswith(".AX"): return "^AXJO"
    if t.endswith(".TW"): return "^TWII"
    return "^GSPC"

@dataclass
class MarketHorizon:
    days_21: int = 21
    days_63: int = 63
    days_126: int = 126
    days_252: int = 252
