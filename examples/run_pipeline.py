import argparse, sys
from credit_features.pipeline import build_features_for_universe

def main():
    p = argparse.ArgumentParser(description="Build yfinance credit features for tickers")
    p.add_argument("--tickers", type=str, required=True, help="Comma-separated list, e.g., AAPL,MSFT,RELIANCE.NS")
    p.add_argument("--period", type=str, default="5y")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        print("No tickers parsed", file=sys.stderr); sys.exit(2)
    df = build_features_for_universe(tickers, period=args.period, interval=args.interval)
    print(df.head().to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
