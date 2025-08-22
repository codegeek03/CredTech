import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools

load_dotenv()


class StockInsightAgent:
    """
    A class-based implementation of an AI-powered stock insight generator.
    Leverages Gemini + YFinanceTools to analyze financial metrics and
    provide human-readable insights.
    """

    def __init__(self, ticker: str, period: str = "30d", interval: str = "1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval

        # Initialize the agent with YFinanceTools
        self.agent = Agent(
            name="Insight Agent",
            description=(
                "An agent that provides actionable insights on stock data, "
                "including price trends, volatility, and key financial metrics."
            ),
            model=Gemini(id="gemini-2.0-flash"),
            tools=[YFinanceTools()],
            markdown=True,
        )

    def build_prompt(self, financial_data: dict) -> str:
        """
        Build a structured prompt using the provided financial dataset.

        Can handle both traditional financial data and credit scoring data.
        """

        # Convert financial_data dict to readable format
        formatted_data = "\n".join([f"{k}: {v}" for k, v in financial_data.items()])

        company_name = financial_data.get("Corporation", self.ticker)
        ticker = financial_data.get("Ticker", self.ticker)

        # Check if this is credit scoring data
        if "Composite Credit Score" in financial_data:
            prompt = f"""
            You are a senior financial analyst and credit risk expert. 
            Analyze the comprehensive credit profile for {company_name} ({ticker}).

            Credit Analysis Data:
            {formatted_data}

            Please provide a detailed professional analysis covering:

            **EXECUTIVE SUMMARY:**
            - Overall creditworthiness assessment
            - Key strengths and concerns
            - Investment recommendation (Buy/Hold/Sell)

            **CREDIT RISK ANALYSIS:**
            1. Credit score interpretation and rating implications
            2. News sentiment impact on credit profile
            3. Risk factor analysis and mitigation strategies
            4. Comparative analysis vs industry peers

            **FINANCIAL HEALTH INDICATORS:**
            1. Score component breakdown analysis
            2. Weight allocation effectiveness
            3. Risk-adjusted return potential
            4. Liquidity and solvency indicators

            **MARKET CONTEXT & OUTLOOK:**
            1. Sector-specific risks and opportunities
            2. Macroeconomic impact on credit profile
            3. Forward-looking credit trajectory
            4. Potential rating migration scenarios

            **ACTIONABLE RECOMMENDATIONS:**
            1. For investors: portfolio allocation suggestions
            2. For lenders: credit terms and monitoring
            3. For management: areas for improvement
            4. Risk monitoring key metrics

            Please provide specific, actionable insights based on the data provided.
            """
        else:
            # Traditional financial data analysis
            prompt = f"""
            You are a financial insights assistant. 
            Analyze the stock and credit profile for {company_name}.

            Provided Financial & Credit Data:
            {formatted_data}

            Please generate a professional report covering:
            1. Creditworthiness & rating implications (based on Rating & Binary Rating).
            2. Profitability analysis using Gross Margin, Operating Margin, EBIT, and Net Profit Margin.
            3. Leverage and capital structure insights using Debt/Equity and Long-term Debt/Capital.
            4. Liquidity analysis with Current Ratio and cash flow per share metrics.
            5. Returns analysis (ROE, ROA, ROI, Return on Tangible Equity).
            6. Sector and SIC Code benchmarking insights.
            7. Risks and forward-looking signals for investors & rating agencies.
            """

        return prompt.strip()

    def get_insights(self, financial_data: dict):
        """
        Run the agent with the constructed prompt and return insights.
        """
        prompt = self.build_prompt(financial_data)
        response = self.agent.run(prompt)
        return response.content


# Example usage
if __name__ == "__main__":
    sample_data = {
        "Rating Agency": "Moody's",
        "Corporation": "Apple Inc.",
        "Rating": "Aa1",
        "Rating Date": "2025-06-30",
        "CIK": "0000320193",
        "Binary Rating": "Investment Grade",
        "SIC Code": "3571",
        "Sector": "Technology",
        "Ticker": "AAPL",
        "Current Ratio": "0.88",
        "Long-term Debt / Capital": "0.35",
        "Debt/Equity Ratio": "1.5",
        "Gross Margin": "43.0%",
        "Operating Margin": "28.0%",
        "EBIT Margin": "27.5%",
        "EBITDA Margin": "31.2%",
        "Pre-Tax Profit Margin": "26.8%",
        "Net Profit Margin": "24.1%",
        "Asset Turnover": "0.88",
        "ROE - Return On Equity": "75.0%",
        "Return On Tangible Equity": "95.0%",
        "ROA - Return On Assets": "19.0%",
        "ROI - Return On Investment": "21.5%",
        "Operating Cash Flow Per Share": "6.15",
        "Free Cash Flow Per Share": "5.80",
    }

    stock_agent = StockInsightAgent("AAPL")
    insights = stock_agent.get_insights(sample_data)

    print("\n=== Financial Insights Report ===\n")
    print(insights)
