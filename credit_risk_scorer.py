import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


from yahooquery import search

def company_to_ticker(name):
    result = search(name)
    if "quotes" in result and len(result["quotes"]) > 0:
        return result["quotes"][0]["symbol"]
    return None



class RiskScorer:
    def __init__(self, ticker="AAPL", start="2020-01-01", end="2023-01-01", fallback_csv=None):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.fallback_csv = fallback_csv
        self.df = None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            df = yf.download(self.ticker, start=self.start, end=self.end, group_by="ticker")
            if df.empty:
                raise ValueError("Empty dataframe from yfinance.")
        except Exception as e:
            if self.fallback_csv:
                df = pd.read_csv(self.fallback_csv)
            else:
                raise RuntimeError(f"Failed to fetch data for {self.ticker}: {e}")

        df.columns = [col if isinstance(col, str) else col[1] for col in df.columns]
        df = df.reset_index()
        if df.empty:
            raise ValueError("No data available for processing.")

        df["Returns"] = df["Close"].pct_change()
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility_20d"] = df["Returns"].rolling(window=20).std()
        delta = df["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI_14"] = 100 - (100 / (1 + rs))
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["BB_Upper"] = df["MA_20"] + 2*df["Close"].rolling(window=20).std()
        df["BB_Lower"] = df["MA_20"] - 2*df["Close"].rolling(window=20).std()
        df["OBV"] = (np.sign(df["Returns"]) * df["Volume"]).fillna(0).cumsum()
        df["Cumulative_Returns"] = (1 + df["Returns"]).cumprod()
        df["Downside_Returns"] = np.where(df["Returns"] < 0, df["Returns"], 0)
        df["Downside_Deviation"] = df["Downside_Returns"].rolling(20).std()
        df["Sharpe_Ratio_20d"] = df["Returns"].rolling(20).mean() / df["Returns"].rolling(20).std()
        df = df.fillna(0)
        df["Risk_Flag"] = np.where(
            (df["Volatility_20d"] > df["Volatility_20d"].median()) & (df["Returns"] < 0),
            1, 0
        )
        self.df = df

    def train_model(self):
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty. Load data before training.")

        features = [
            'Log_Returns','Volatility_20d','SMA_20','SMA_50','SMA_200',
            'RSI_14','MACD','Signal_Line','Returns','MA_20',
            'BB_Upper','BB_Lower','OBV',
            'Cumulative_Returns','Downside_Deviation','Sharpe_Ratio_20d'
        ]
        X = self.df[features]
        y = self.df["Risk_Flag"]

        if len(X) < 50:
            raise ValueError(f"Not enough samples ({len(X)} rows) to train model.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train_scaled = self.scaler.fit_transform(X_train)

        base_model = HistGradientBoostingClassifier(max_iter=500, max_depth=6, learning_rate=0.05, random_state=42)
        self.model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        self.model.fit(X_train_scaled, y_train)

    def get_today_risk_score(self):
        features = [
            'Log_Returns','Volatility_20d','SMA_20','SMA_50','SMA_200',
            'RSI_14','MACD','Signal_Line','Returns','MA_20',
            'BB_Upper','BB_Lower','OBV',
            'Cumulative_Returns','Downside_Deviation','Sharpe_Ratio_20d'
        ]
        today_features = self.df[features].iloc[[-1]]
        today_scaled = self.scaler.transform(today_features)
        prob = self.model.predict_proba(today_scaled)[:,1][0]
        return (1 - prob) * 100

if __name__ == "__main__":
    scorer = RiskScorer()
    scorer.load_data()
    scorer.train_model()
    today_risk = scorer.get_today_risk_score()
    print("Today's Risk Score:", today_risk)
    company = "Microsoft"
    ticker = company_to_ticker(company)
    if ticker:
        print(f"The ticker for {company} is {ticker}.")
    else:
        print(f"Could not find ticker for {company}.")