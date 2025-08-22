import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from sklearn.linear_model import LinearRegression


class StockVisualizer:
    def __init__(self, ticker: str):
        """
        Initialize with a stock ticker.
        Example: StockVisualizer("AAPL")
        """
        self.ticker = ticker
        self.data = None

    def download_data(self, period: str = "30d", interval: str = "1d"):
        """
        Download stock data from yfinance.
        """
        self.data = yf.download(
            self.ticker, period=period, interval=interval, auto_adjust=True
        )
        if self.data.empty:
            raise ValueError("No data downloaded. Check ticker or internet connection.")
        self.data.reset_index(inplace=True)  # reset for regression use
        return self.data

    def add_indicators(self):
        """
        Add indicators like SMA50, SMA200, and daily returns.
        """
        self.data["SMA50"] = self.data["Close"].rolling(window=50).mean()
        self.data["SMA200"] = self.data["Close"].rolling(window=200).mean()
        self.data["Daily Return"] = self.data["Close"].pct_change()
        self.data["Volatility"] = self.data["Daily Return"].rolling(
            window=10
        ).std() * np.sqrt(252)

    def plot_price_with_sma(self):
        """
        Plot stock price along with SMA50 and SMA200.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data["Date"],
            self.data["Close"],
            label=f"{self.ticker} Close Price",
            linewidth=2,
        )
        plt.plot(self.data["Date"], self.data["SMA50"], label="SMA50", linestyle="--")
        plt.plot(self.data["Date"], self.data["SMA200"], label="SMA200", linestyle="--")
        plt.title(f"{self.ticker} Price with SMA50 & SMA200")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_volatility(self):
        """
        Plot stock volatility trend.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data["Date"],
            self.data["Volatility"],
            color="red",
            label="Volatility (10-day Rolling)",
        )
        plt.title(f"{self.ticker} Volatility Trend")
        plt.xlabel("Date")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_daily_returns(self):
        """
        Plot daily returns as bar chart.
        """
        plt.figure(figsize=(12, 6))
        plt.bar(self.data["Date"], self.data["Daily Return"], alpha=0.6)
        plt.title(f"{self.ticker} Daily Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.grid(True)
        plt.show()

    def plot_bollinger_bands(self, window=20):
        """
        Plot Bollinger Bands (SMA ± 2 * std dev).
        """
        self.data["Middle Band"] = self.data["Close"].rolling(window=window).mean()
        self.data["Upper Band"] = (
            self.data["Middle Band"]
            + 2 * self.data["Close"].rolling(window=window).std()
        )
        self.data["Lower Band"] = (
            self.data["Middle Band"]
            - 2 * self.data["Close"].rolling(window=window).std()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data["Date"], self.data["Close"], label="Close Price", linewidth=1.5
        )
        plt.plot(
            self.data["Date"],
            self.data["Upper Band"],
            label="Upper Band",
            linestyle="--",
        )
        plt.plot(
            self.data["Date"],
            self.data["Middle Band"],
            label="SMA (Middle)",
            linestyle="--",
        )
        plt.plot(
            self.data["Date"],
            self.data["Lower Band"],
            label="Lower Band",
            linestyle="--",
        )
        plt.fill_between(
            self.data["Date"],
            self.data["Lower Band"],
            self.data["Upper Band"],
            alpha=0.2,
            color="gray",
        )
        plt.title(f"{self.ticker} Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cumulative_returns(self):
        """
        Plot cumulative returns from daily returns.
        """
        self.data["Cumulative Return"] = (1 + self.data["Daily Return"]).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.data["Date"],
            self.data["Cumulative Return"],
            label="Cumulative Return",
            color="green",
        )
        plt.title(f"{self.ticker} Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1 investment")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_returns_histogram(self):
        """
        Plot histogram of daily returns.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(
            self.data["Daily Return"].dropna(),
            bins=20,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        plt.title(f"{self.ticker} Distribution of Daily Returns")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

    def plot_candlestick(self):
        """
        Plot candlestick chart using OHLC data.
        """
        df = self.data.set_index("Date")
        mpf.plot(
            df,
            type="candle",
            mav=(20, 50),
            volume=True,
            title=f"{self.ticker} Candlestick Chart",
        )

    def plot_regression_trendline(self, days=30):
        """
        Fit and plot a regression trendline over the last 'days'.
        """
        recent_data = self.data.tail(days)
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data["Close"].values

        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)

        plt.figure(figsize=(12, 6))
        plt.plot(recent_data["Date"], y, label="Close Price", linewidth=2)
        plt.plot(
            recent_data["Date"],
            trend,
            color="orange",
            linestyle="--",
            label=f"{days}-Day Regression Trendline",
        )
        plt.title(f"{self.ticker} Price with Regression Trendline ({days} Days)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def full_visualization(self):
        """
        Run all plots together for a complete analysis.
        """
        self.download_data()
        self.add_indicators()
        self.plot_price_with_sma()
        self.plot_volatility()
        self.plot_daily_returns()
        self.plot_regression_trendline(days=30)  # full-period regression
        self.plot_regression_trendline(days=10)  # short-term regression


# Example usage:
if __name__ == "__main__":
    stock_viz = StockVisualizer("AAPL")
    stock_viz.full_visualization()
