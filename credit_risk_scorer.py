import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from yahooquery import search
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


def company_to_ticker(name):
    result = search(name)
    if "quotes" in result and len(result["quotes"]) > 0:
        return result["quotes"][0]["symbol"]
    return None


class RiskScorer:
    def __init__(self, ticker="AAPL", start="2020-01-01", end="2023-01-01", fallback_csv=None, shap_folder="shap_images"):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.fallback_csv = fallback_csv
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.shap_folder = shap_folder

        if not os.path.exists(shap_folder):
            os.makedirs(shap_folder)

        self.features = [
            'Log_Returns','Volatility_20d','SMA_20','SMA_50','SMA_200',
            'RSI_14','MACD','Signal_Line','Returns','MA_20',
            'BB_Upper','BB_Lower','OBV',
            'Cumulative_Returns','Downside_Deviation','Sharpe_Ratio_20d'
        ]

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

        # Feature engineering
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
        X = self.df[self.features]
        y = self.df["Risk_Flag"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_train_scaled, self.y_train)

    def get_metrics(self):
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        auc = roc_auc_score(self.y_test, y_prob)

        return (
            f"Accuracy: {acc:.4f}, "
            f"Precision: {prec:.4f}, "
            f"Recall: {rec:.4f}, "
            f"F1: {f1:.4f}, "
            f"AUC: {auc:.4f}"
        )

    def get_today_risk_score(self):
        today_features = self.df[self.features].iloc[[-1]]
        today_scaled = self.scaler.transform(today_features)
        prob = self.model.predict_proba(today_scaled)[:, 1][0]
        return (1 - prob) * 100

    def get_feature_importance(self):
        """Return feature importance as dict"""
        importance = self.model.feature_importances_
        return dict(zip(self.features, importance))

    def explain_model_global(self, num_features=10, filename="global_shap.png"):
        X_scaled = self.scaler.transform(self.df[self.features])
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        plt.figure()
        shap.summary_plot(shap_values, features=self.df[self.features],
                          feature_names=self.features, max_display=num_features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.shap_folder, filename))
        plt.close()
        print(f"Global SHAP plot saved to {self.shap_folder}/{filename}")

    def explain_today(self, filename="today_shap.png"):
        X_scaled = self.scaler.transform(self.df[self.features])
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        today_idx = -1
        plt.figure()
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[today_idx],
            base_values=explainer.expected_value,
            data=X_scaled[today_idx],
            feature_names=self.features
        ))
        plt.tight_layout()
        plt.savefig(os.path.join(self.shap_folder, filename))
        plt.close()
        print(f"Today's SHAP plot saved to {self.shap_folder}/{filename}")


if __name__ == "__main__":
    scorer = RiskScorer(shap_folder="shap_outputs")  # Save images here
    scorer.load_data()
    scorer.train_model()

    print("Model Performance:", scorer.get_metrics())
    print("Today's Risk Score:", scorer.get_today_risk_score())
    print("Feature Importances:", scorer.get_feature_importance())

    scorer.explain_model_global()
    scorer.explain_today()

    company = "Microsoft"
    ticker = company_to_ticker(company)
    if ticker:
        print(f"The ticker for {company} is {ticker}.")
    else:
        print(f"Could not find ticker for {company}.")
