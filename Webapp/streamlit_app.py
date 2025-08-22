import streamlit as st
import sys
import os

# Add the parent directory to the path to import from clustering folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from clustering.stock_visualizer import StockVisualizer
from yahooquery import search
import matplotlib.pyplot as plt
import pandas as pd


def company_to_ticker(name):
    """Convert company name to ticker symbol"""
    try:
        result = search(name)
        if "quotes" in result and len(result["quotes"]) > 0:
            return result["quotes"][0]["symbol"]
        return None
    except Exception as e:
        st.error(f"Error searching for company: {e}")
        return None


class StreamlitStockVisualizer(StockVisualizer):
    """Extended StockVisualizer for Streamlit compatibility"""

    def plot_price_with_sma(self):
        """Plot stock price with SMA - Streamlit version"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.data["Date"],
            self.data["Close"],
            label=f"{self.ticker} Close Price",
            linewidth=2,
        )
        ax.plot(self.data["Date"], self.data["SMA50"], label="SMA50", linestyle="--")
        ax.plot(self.data["Date"], self.data["SMA200"], label="SMA200", linestyle="--")
        ax.set_title(f"{self.ticker} Price with SMA50 & SMA200")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_volatility(self):
        """Plot volatility - Streamlit version"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.data["Date"],
            self.data["Volatility"],
            color="red",
            label="Volatility (10-day Rolling)",
        )
        ax.set_title(f"{self.ticker} Volatility Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_daily_returns(self):
        """Plot daily returns - Streamlit version"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(self.data["Date"], self.data["Daily Return"], alpha=0.6)
        ax.set_title(f"{self.ticker} Daily Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_bollinger_bands(self, window=20):
        """Plot Bollinger Bands - Streamlit version"""
        # Calculate bands properly to avoid multi-column assignment
        close_prices = self.data["Close"]
        middle_band = close_prices.rolling(window=window).mean()
        std_dev = close_prices.rolling(window=window).std()

        self.data["Middle Band"] = middle_band
        self.data["Upper Band"] = middle_band + 2 * std_dev
        self.data["Lower Band"] = middle_band - 2 * std_dev

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.data["Date"], self.data["Close"], label="Close Price", linewidth=1.5
        )
        ax.plot(
            self.data["Date"],
            self.data["Upper Band"],
            label="Upper Band",
            linestyle="--",
        )
        ax.plot(
            self.data["Date"],
            self.data["Middle Band"],
            label="SMA (Middle)",
            linestyle="--",
        )
        ax.plot(
            self.data["Date"],
            self.data["Lower Band"],
            label="Lower Band",
            linestyle="--",
        )
        ax.fill_between(
            self.data["Date"],
            self.data["Lower Band"],
            self.data["Upper Band"],
            alpha=0.2,
            color="gray",
        )
        ax.set_title(f"{self.ticker} Bollinger Bands")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_cumulative_returns(self):
        """Plot cumulative returns - Streamlit version"""
        self.data["Cumulative Return"] = (1 + self.data["Daily Return"]).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            self.data["Date"],
            self.data["Cumulative Return"],
            label="Cumulative Return",
            color="green",
        )
        ax.set_title(f"{self.ticker} Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1 investment")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_returns_histogram(self):
        """Plot histogram of returns - Streamlit version"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            self.data["Daily Return"].dropna(),
            bins=20,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        ax.set_title(f"{self.ticker} Distribution of Daily Returns")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        st.pyplot(fig)
        plt.close()

    def plot_regression_trendline(self, days=30):
        """Plot regression trendline - Streamlit version"""
        from sklearn.linear_model import LinearRegression
        import numpy as np

        recent_data = self.data.tail(days)
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data["Close"].values

        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(recent_data["Date"], y, label="Close Price", linewidth=2)
        ax.plot(
            recent_data["Date"],
            trend,
            color="orange",
            linestyle="--",
            label=f"{days}-Day Regression Trendline",
        )
        ax.set_title(f"{self.ticker} Price with Regression Trendline ({days} Days)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close()


def main():
    st.set_page_config(page_title="Stock Visualizer", page_icon="📈", layout="wide")

    st.title("📈 Stock Market Analyzer")
    st.markdown(
        "Enter a company name to analyze its stock performance with various technical indicators."
    )

    # Sidebar for inputs
    st.sidebar.header("Settings")

    # Company name input
    company_name = st.sidebar.text_input(
        "Enter Company Name", placeholder="e.g., Apple, Microsoft, Tesla"
    )

    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period", ["30d", "60d", "90d", "6mo", "1y", "2y", "5y"]
    )

    # Chart selection
    st.sidebar.header("Select Charts to Display")
    show_price_sma = st.sidebar.checkbox("Price with Moving Averages", True)
    show_bollinger = st.sidebar.checkbox("Bollinger Bands", True)
    show_volatility = st.sidebar.checkbox("Volatility", False)
    show_returns = st.sidebar.checkbox("Daily Returns", False)
    show_cumulative = st.sidebar.checkbox("Cumulative Returns", True)
    show_histogram = st.sidebar.checkbox("Returns Histogram", False)
    show_regression = st.sidebar.checkbox("Regression Trendline", True)

    if st.sidebar.button("Analyze Stock", type="primary"):
        if company_name:
            with st.spinner("Searching for company..."):
                ticker = company_to_ticker(company_name)

            if ticker:
                st.success(f"Found ticker: {ticker}")

                try:
                    with st.spinner("Downloading stock data..."):
                        viz = StreamlitStockVisualizer(ticker)
                        viz.download_data(period=period)
                        viz.add_indicators()

                    # Display basic info - Convert to Python scalars to avoid Series formatting issues
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        current_price = viz.data["Close"].iloc[-1].item()
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        daily_change = viz.data["Daily Return"].iloc[-1].item() * 100
                        if pd.isna(daily_change):
                            st.metric("Daily Change", "N/A")
                        else:
                            st.metric("Daily Change", f"{daily_change:.2f}%")
                    with col3:
                        max_price = viz.data["Close"].max().item()
                        st.metric("52W High", f"${max_price:.2f}")
                    with col4:
                        min_price = viz.data["Close"].min().item()
                        st.metric("52W Low", f"${min_price:.2f}")

                    # Display selected charts
                    if show_price_sma:
                        st.header("📊 Price with Moving Averages")
                        viz.plot_price_with_sma()

                    if show_bollinger:
                        st.header("📈 Bollinger Bands")
                        viz.plot_bollinger_bands()

                    if show_volatility:
                        st.header("📉 Volatility Analysis")
                        viz.plot_volatility()

                    if show_returns:
                        st.header("📊 Daily Returns")
                        viz.plot_daily_returns()

                    if show_cumulative:
                        st.header("📈 Cumulative Returns")
                        viz.plot_cumulative_returns()

                    if show_histogram:
                        st.header("📊 Returns Distribution")
                        viz.plot_returns_histogram()

                    if show_regression:
                        st.header("📈 Regression Trendline")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("30-Day Trend")
                            viz.plot_regression_trendline(30)
                        with col2:
                            st.subheader("10-Day Trend")
                            viz.plot_regression_trendline(10)

                    # Data table
                    with st.expander("View Raw Data"):
                        st.dataframe(viz.data.tail(20))

                except Exception as e:
                    st.error(f"Error analyzing stock: {e}")
            else:
                st.error("Company not found. Please check the spelling and try again.")
        else:
            st.warning("Please enter a company name.")


if __name__ == "__main__":
    main()
