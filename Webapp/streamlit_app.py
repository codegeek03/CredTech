import streamlit as st
import sys
import os

# Add the parent directory to the path to import from clustering folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from clustering.stock_visualizer import StockVisualizer
from yahooquery import search
import matplotlib.pyplot as plt
import pandas as pd

# Import credit scoring functionality
try:
    from main import credit_score

    CREDIT_SCORE_AVAILABLE = True
except ImportError as e:
    st.error(f"Error importing credit score functionality: {e}")
    CREDIT_SCORE_AVAILABLE = False

# Import Insight Agent functionality
try:
    from Insight_agent import StockInsightAgent

    INSIGHT_AGENT_AVAILABLE = True
except ImportError as e:
    st.warning(f"Insight Agent not available: {e}")
    INSIGHT_AGENT_AVAILABLE = False


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
    st.set_page_config(page_title="CredTech Analytics", page_icon="📈", layout="wide")

    st.title("🏦 CredTech Analytics Platform")
    st.markdown(
        "Comprehensive financial analysis with stock market and credit scoring capabilities."
    )

    # Create tabs
    tab1, tab2 = st.tabs(["📈 Stock Market Analyzer", "💳 Credit Score"])

    with tab1:
        stock_analyzer_tab()

    with tab2:
        credit_score_tab()


def stock_analyzer_tab():
    """Stock Market Analyzer functionality"""
    st.header("📈 Stock Market Analyzer")
    st.markdown(
        "Enter a company name to analyze its stock performance with various technical indicators."
    )

    # Sidebar for inputs
    st.sidebar.header("Stock Analysis Settings")

    # Company name input
    company_name = st.sidebar.text_input(
        "Enter Company Name",
        placeholder="e.g., Apple, Microsoft, Tesla",
        key="stock_company",
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

    if st.sidebar.button("Analyze Stock", type="primary", key="analyze_stock"):
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


def credit_score_tab():
    """Credit Score functionality"""
    st.header("💳 Credit Score Analysis")
    st.markdown(
        "Enter a company name to analyze its creditworthiness using news sentiment, risk factors, and financial metrics."
    )

    if not CREDIT_SCORE_AVAILABLE:
        st.error(
            "Credit scoring functionality is not available. Please check the import dependencies."
        )
        return

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        credit_company_name = st.text_input(
            "Enter Company Name for Credit Analysis",
            placeholder="e.g., Microsoft, Apple, Tesla",
            key="credit_company",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        analyze_credit = st.button(
            "🔍 Analyze Credit Score", type="primary", key="analyze_credit"
        )

    # Weight configuration
    with st.expander("⚙️ Advanced Settings - Score Weights"):
        st.markdown("Adjust the weights for different scoring components:")
        col1, col2, col3 = st.columns(3)

        with col1:
            financial_weight = st.slider("Financial Score Weight", 0.0, 1.0, 0.5, 0.1)
        with col2:
            risk_weight = st.slider("Risk Score Weight", 0.0, 1.0, 0.3, 0.1)
        with col3:
            news_weight = st.slider("News Sentiment Weight", 0.0, 1.0, 0.2, 0.1)

        # Normalize weights to sum to 1
        total_weight = financial_weight + risk_weight + news_weight
        if total_weight > 0:
            financial_weight_norm = financial_weight / total_weight
            risk_weight_norm = risk_weight / total_weight
            news_weight_norm = news_weight / total_weight
        else:
            financial_weight_norm = risk_weight_norm = news_weight_norm = 0.33

    if analyze_credit and credit_company_name:
        try:
            with st.spinner(f"Analyzing credit score for {credit_company_name}..."):
                # Call the credit_score function from main.py
                scores = credit_score(credit_company_name)

                # Validate the returned scores
                if scores is None:
                    st.error(
                        "Failed to calculate credit scores. The function returned None."
                    )
                    return

                if not isinstance(scores, (list, tuple)) or len(scores) < 4:
                    st.error(
                        f"Invalid score format returned. Expected 4 values, got: {scores}"
                    )
                    return

                # Extract and validate individual scores
                try:
                    news_score = float(scores[0]) if scores[0] is not None else 50.0
                    risk_score = float(scores[1]) if scores[1] is not None else 50.0
                    financial_score = (
                        float(scores[2]) if scores[2] is not None else 50.0
                    )
                    feature_importance = scores[3] if scores[3] is not None else {}
                except (ValueError, TypeError, IndexError) as e:
                    st.error(f"Error processing scores: {e}")
                    st.info(f"Raw scores received: {scores}")
                    return

                # Ensure scores are within reasonable bounds
                news_score = max(0, min(100, news_score))
                risk_score = max(0, min(100, risk_score))
                financial_score = max(0, min(100, financial_score))

                # Calculate composite score
                composite_score = (
                    financial_weight_norm * financial_score
                    + risk_weight_norm * risk_score
                    + news_weight_norm * news_score
                )

                # Display main metrics
                st.success(f"✅ Credit analysis completed for {credit_company_name}")

                # Main score display
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "📰 News Sentiment Score",
                        f"{news_score:.2f}",
                        help="Score based on recent news sentiment analysis",
                    )

                with col2:
                    st.metric(
                        "⚠️ Risk Score",
                        f"{risk_score:.2f}",
                        help="Machine learning-based risk assessment",
                    )

                with col3:
                    st.metric(
                        "💰 Financial Score",
                        f"{financial_score:.2f}",
                        help="Financial metrics and ratios analysis",
                    )

                with col4:
                    # Color code the composite score
                    if composite_score >= 70:
                        score_emoji = "🟢"
                    elif composite_score >= 50:
                        score_emoji = "🟡"
                    else:
                        score_emoji = "🔴"

                    st.metric(
                        f"{score_emoji} Composite Credit Score",
                        f"{composite_score:.2f}",
                        help="Weighted combination of all scoring components",
                    )

                # Score breakdown visualization
                st.subheader("📊 Score Breakdown")

                # Create a DataFrame for the score breakdown
                score_data = {
                    "Component": [
                        "News Sentiment",
                        "Risk Assessment",
                        "Financial Metrics",
                    ],
                    "Score": [news_score, risk_score, financial_score],
                    "Weight": [
                        news_weight_norm,
                        risk_weight_norm,
                        financial_weight_norm,
                    ],
                    "Weighted Score": [
                        news_score * news_weight_norm,
                        risk_score * risk_weight_norm,
                        financial_score * financial_weight_norm,
                    ],
                }

                score_df = pd.DataFrame(score_data)

                # Display as a bar chart
                col1, col2 = st.columns(2)

                with col1:
                    st.bar_chart(score_df.set_index("Component")["Score"])
                    st.caption("Individual Component Scores")

                with col2:
                    st.bar_chart(score_df.set_index("Component")["Weighted Score"])
                    st.caption("Weighted Component Contributions")

                # Feature importance
                if (
                    feature_importance
                    and isinstance(feature_importance, dict)
                    and len(feature_importance) > 0
                ):
                    st.subheader("🔍 Key Risk Factors")

                    # Convert feature importance to DataFrame
                    importance_df = pd.DataFrame(
                        list(feature_importance.items()),
                        columns=["Feature", "Importance"],
                    ).sort_values("Importance", ascending=False)

                    # Display top 10 features
                    top_features = importance_df.head(10)
                    st.dataframe(top_features, hide_index=True)

                    # Feature importance chart
                    if not top_features.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(
                            top_features["Feature"][::-1],
                            top_features["Importance"][::-1],
                        )
                        ax.set_xlabel("Feature Importance")
                        ax.set_title("Top 10 Risk Factors")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.info("No feature importance data available for this analysis.")

                # Credit rating interpretation
                st.subheader("📋 Credit Rating Interpretation")

                if composite_score >= 80:
                    rating = "AAA - Excellent"
                    color = "green"
                    description = (
                        "Exceptionally strong capacity to meet financial commitments."
                    )
                elif composite_score >= 70:
                    rating = "A - Good"
                    color = "lightgreen"
                    description = "Strong capacity to meet financial commitments."
                elif composite_score >= 60:
                    rating = "BBB - Moderate"
                    color = "yellow"
                    description = "Adequate capacity to meet financial commitments, but more susceptible to adverse economic conditions."
                elif composite_score >= 50:
                    rating = "BB - Speculative"
                    color = "orange"
                    description = "Less vulnerable in the near term, but faces ongoing uncertainties."
                else:
                    rating = "B - Highly Speculative"
                    color = "red"
                    description = "More vulnerable to adverse business, financial, and economic conditions."

                st.markdown(
                    f"""
                <div style="padding: 1rem; border-left: 4px solid {color}; background-color: rgba(255,255,255,0.1);">
                    <h4 style="color: {color}; margin: 0;">{rating}</h4>
                    <p style="margin: 0.5rem 0 0 0;">{description}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # AI-Powered Insights Section
                st.divider()
                st.subheader("🤖 AI-Powered Financial Insights")

                if INSIGHT_AGENT_AVAILABLE:
                    with st.spinner("Generating AI insights..."):
                        try:
                            # Get ticker for the company
                            ticker = company_to_ticker(credit_company_name)

                            # Prepare financial data for the agent
                            financial_data = {
                                "Corporation": credit_company_name,
                                "Ticker": ticker if ticker else "N/A",
                                "Credit Rating": rating,
                                "Composite Credit Score": f"{composite_score:.2f}",
                                "News Sentiment Score": f"{news_score:.2f}",
                                "Risk Assessment Score": f"{risk_score:.2f}",
                                "Financial Metrics Score": f"{financial_score:.2f}",
                                "News Weight": f"{news_weight_norm:.2f}",
                                "Risk Weight": f"{risk_weight_norm:.2f}",
                                "Financial Weight": f"{financial_weight_norm:.2f}",
                                "Analysis Date": pd.Timestamp.now().strftime(
                                    "%Y-%m-%d"
                                ),
                                "Key Risk Factors": (
                                    ", ".join(list(feature_importance.keys())[:5])
                                    if feature_importance
                                    else "N/A"
                                ),
                            }

                            # Add top risk factors as individual entries
                            if feature_importance:
                                for i, (factor, importance) in enumerate(
                                    list(feature_importance.items())[:3]
                                ):
                                    financial_data[f"Risk Factor {i+1}"] = (
                                        f"{factor} ({importance:.3f})"
                                    )

                            # Initialize and run the insight agent
                            if ticker:
                                insight_agent = StockInsightAgent(ticker)
                            else:
                                insight_agent = StockInsightAgent(
                                    "SPY"
                                )  # Fallback to market index

                            insights = insight_agent.get_insights(financial_data)

                            # Display the insights
                            if insights:
                                st.markdown(insights)
                            else:
                                st.info("No insights generated by the AI agent.")

                        except Exception as e:
                            st.error(f"Error generating AI insights: {str(e)}")
                            st.info(
                                "AI insights are temporarily unavailable. Please check your API configuration."
                            )

                            # Show error details in expander
                            with st.expander("🔧 AI Error Details"):
                                import traceback

                                st.code(traceback.format_exc())
                else:
                    st.warning(
                        "🤖 AI Insight Agent is not available. Please check the dependencies."
                    )
                    st.info("To enable AI insights, ensure you have:")
                    st.markdown(
                        """
                    - Google Gemini API key configured
                    - `agno` library installed
                    - `Insight_agent.py` properly set up
                    """
                    )

                # Debug information (can be removed in production)
                with st.expander("🔧 Debug Information"):
                    st.write("Raw scores returned:", scores)
                    st.write("Processing details:")
                    st.json(
                        {
                            "news_score": news_score,
                            "risk_score": risk_score,
                            "financial_score": financial_score,
                            "composite_score": composite_score,
                            "weights": {
                                "financial": financial_weight_norm,
                                "risk": risk_weight_norm,
                                "news": news_weight_norm,
                            },
                        }
                    )

                    if INSIGHT_AGENT_AVAILABLE:
                        st.write("Financial data sent to AI agent:")
                        if "financial_data" in locals():
                            st.json(financial_data)

        except Exception as e:
            st.error(f"Error calculating credit score: {str(e)}")
            st.info(
                "Please ensure all required dependencies are installed and the company name is valid."
            )

            # Show detailed error for debugging
            import traceback

            with st.expander("🔧 Error Details"):
                st.code(traceback.format_exc())

    elif analyze_credit and not credit_company_name:
        st.warning("Please enter a company name to analyze.")


if __name__ == "__main__":
    main()
