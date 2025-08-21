# examples/run_pipeline.py
import argparse
import pandas as pd
from credit_features.core import CreditFeaturePipeline

# --- Default US-20 universe (no args needed) ---
DEFAULT_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META",
    "NVDA","TSLA","JPM","JNJ","V",
    "MA","PG","UNH","HD","PEP",
    "KO","XOM","BAC","COST","ADBE"
]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "PG": "Procter & Gamble Co.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "Home Depot Inc.",
    "PEP": "PepsiCo Inc.",
    "KO": "Coca-Cola Co.",
    "XOM": "Exxon Mobil Corp.",
    "BAC": "Bank of America Corp.",
    "COST": "Costco Wholesale Corp.",
    "ADBE": "Adobe Inc."
}

def main():
    p = argparse.ArgumentParser(
        description="Compact yfinance credit features pipeline (defaults to 20 large US tickers if --tickers is omitted)"
    )
    # Make --tickers optional; if omitted, we use DEFAULT_TICKERS
    p.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated list (e.g., AAPL,MSFT,RELIANCE.NS). If omitted, uses a default US-20 universe."
    )
    p.add_argument("--period", type=str, default="5y", help="History period for prices (e.g., 3y, 5y, max)")
    p.add_argument("--interval", type=str, default="1d", help="Price interval (e.g., 1d, 1wk)")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional CSV output path. If omitted, defaults to features_us20.csv when using defaults, else features.csv"
    )
    args = p.parse_args()

    # Decide ticker set
    if args.tickers.strip():
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        out_path = args.out or "features.csv"
        using_defaults = False
    else:
        tickers = DEFAULT_TICKERS
        out_path = args.out or "features_us20.csv"
        using_defaults = True
        print("No --tickers provided → using built-in US-20 universe.")

    # Build features
    pipe = CreditFeaturePipeline(period=args.period, interval=args.interval)
    df = pipe.build_for_universe(tickers)

    # Add friendly company names when available
    if "ticker" in df.columns:
        df.insert(1, "company", df["ticker"].map(COMPANY_NAMES))

    # Preview + save
    preview_cols = [c for c in ["ticker","company","ret_1y","vol_1y","beta_1y","altman_z","piotroski_f","dd_proxy"] if c in df.columns]
    if preview_cols:
        print(df[preview_cols].head(10).to_string(index=False))
    else:
        print(df.head().to_string(index=False))

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
