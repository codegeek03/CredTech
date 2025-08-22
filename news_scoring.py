import finnhub
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from collections import defaultdict
from textblob import TextBlob

load_dotenv()

class AdvancedNewsMetricGenerator:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.api_key = os.getenv("FINNHUB_API_KEY")
        if not self.api_key: raise ValueError("Error: FINNHUB_API_KEY environment variable not set.")
        self.client = finnhub.Client(api_key=self.api_key)
        self.news_categories = {
            'earnings_mentions': ['earnings', 'revenue', 'eps', 'profit', 'margin'],
            'm&a_mentions': ['merger', 'acquisition', 'acquire', 'partner', 'stake', 'deal'],
            'analyst_ratings': ['upgrade', 'downgrade', 'outperform', 'underperform', 'price target', 'rating'],
            'forward_looking': ['expects', 'guidance', 'outlook', 'forecast', 'future', 'plans to'],
            'risk_factors': ['risk', 'warning', 'uncertain', 'volatile', 'concern', 'investigation', 'lawsuit']
        }
    def fetch_company_news(self, days_history: int) -> list:
        print(f"--- Fetching news for {self.ticker} for the last {days_history} days ---")
        end_date = datetime.now(); start_date = end_date - timedelta(days=days_history)
        news_list = self.client.company_news(self.ticker, _from=start_date.strftime("%Y-%m-%d"), to=end_date.strftime("%Y-%m-%d"))
        print(f"Found {len(news_list)} articles.")
        return news_list
    def generate_news_metrics(self, days_history: int = 30) -> pd.DataFrame:
        articles = self.fetch_company_news(days_history)
        if not articles: return pd.DataFrame()
        daily_data = defaultdict(list)
        for article in articles:
            date = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
            daily_data[date].append(article)
        processed_daily_features = []
        for date, articles_on_day in sorted(daily_data.items()):
            full_day_text = ' '.join([(a['headline'] + ' ' + a['summary']).lower() for a in articles_on_day])
            day_features = {'date': date}
            day_features['num_articles'] = len(articles_on_day)
            day_features['sentiment'] = TextBlob(full_day_text).sentiment.polarity
            for category, keywords in self.news_categories.items():
                day_features[category] = sum(1 for kw in keywords if kw in full_day_text)
            processed_daily_features.append(day_features)
        if not processed_daily_features: return pd.DataFrame()
        df = pd.DataFrame(processed_daily_features).set_index('date')
        df.index = pd.to_datetime(df.index)
        all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(all_days).fillna(0)
        return df

class NewsCreditScorer:
    def __init__(self, metrics_df: pd.DataFrame):
        if not isinstance(metrics_df, pd.DataFrame) or metrics_df.empty: raise ValueError("A non-empty pandas DataFrame must be provided.")
        self.metrics_df = metrics_df.copy()
        self.scorecard = {
            'sentiment': {'bins': [-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf], 'points': [-30, -15, 0, 15, 30]},
            'num_articles': {'bins': [-np.inf, 0, 3, 7, np.inf], 'points': [-5, 0, 5, 10]},
            'earnings_mentions': {'bins': [-np.inf, 0, 2, np.inf], 'points': [0, 10, 20]},
            'm&a_mentions': {'bins': [-np.inf, 0, 1, np.inf], 'points': [0, 15, 25]},
            'forward_looking': {'bins': [-np.inf, 0, 2, np.inf], 'points': [0, 10, 20]},
            'risk_factors': {'bins': [-np.inf, 0, 2, np.inf], 'points': [10, -10, -25]},
            'analyst_ratings': {'bins': [-np.inf, 0, 2, np.inf], 'points': [0, 5, 10]}
        }
    def calculate_daily_credit_score(self):
        print("\n--- Calculating Daily News Credit Score (DNCS) ---")
        df = self.metrics_df
        df['daily_credit_score'] = 0
        for metric, criteria in self.scorecard.items():
            if metric in df.columns:
                points_col = pd.cut(df[metric], bins=criteria['bins'], labels=criteria['points'], right=False)
                df['daily_credit_score'] += pd.to_numeric(points_col, errors='coerce').fillna(0)
        print("Daily News Credit Score calculated.")
        return self
    def get_scored_df(self) -> pd.DataFrame:
        return self.metrics_df


class PredictiveNewsScorer:
    def __init__(self, credit_score_df: pd.DataFrame, lookback_period: int = 7):
        if 'daily_credit_score' not in credit_score_df.columns:
            raise ValueError("Input DataFrame must contain 'daily_credit_score' column.")
        self.df = credit_score_df.copy()
        self.lookback = lookback_period

    def _calculate_historical_features(self):
        """
        Calculates momentum, trend, and volatility features from previous days.
        The .shift(1) is CRITICAL here to prevent data leakage from the current day.
        """
        # EWMA captures momentum, giving more weight to recent days
        self.df['ewma_prev'] = self.df['daily_credit_score'].ewm(span=self.lookback).mean().shift(1)
        
        # SMA is the simple baseline average
        self.df['sma_prev'] = self.df['daily_credit_score'].rolling(window=self.lookback).mean().shift(1)
        
        # Trend is the difference between the last actual score and the recent average
        self.df['trend_prev'] = (self.df['daily_credit_score'].shift(1) - self.df['sma_prev'])
        
        # Volatility is the standard deviation of recent scores
        self.df['volatility_prev'] = self.df['daily_credit_score'].rolling(window=self.lookback).std().shift(1)
        
        # Fill any initial NaN values that result from rolling calculations
        self.df.fillna(0, inplace=True)

    def calculate_predictive_score(self, weights: dict, base_score: int = 50):
        """
        Calculates the Predictive Daily News Score (PDNS).
        Formula: PDNS_t = Base + (w_ewma * EWMA_{t-1}) + (w_trend * TREND_{t-1}) - (w_vol * STD_{t-1})
        """
        print("\n--- Calculating Predictive Daily News Score (PDNS) ---")
        self._calculate_historical_features()
        
        # Apply the weighted formula
        self.df['predictive_daily_score'] = (
            base_score +
            (weights.get('ewma', 0) * self.df['ewma_prev']) +
            (weights.get('trend', 0) * self.df['trend_prev']) -
            (weights.get('volatility', 0) * self.df['volatility_prev'])
        )
        print("Predictive score calculated for each day based on its previous metrics.")
        return self

    def get_scored_df(self) -> pd.DataFrame:
        return self.df


if __name__ == "__main__":
    try:
        # 1. Generate the base news metrics
        metrics_generator = AdvancedNewsMetricGenerator(ticker='NVDA')
        daily_features_df = metrics_generator.generate_news_metrics(days_history=90)
        
        if not daily_features_df.empty:
            # 2. Calculate the descriptive daily credit score
            credit_scorer = NewsCreditScorer(daily_features_df)
            credit_score_df = (credit_scorer.calculate_daily_credit_score()
                                            .get_scored_df())

            # 3. Define weights for the predictive model
            # These weights determine the influence of each historical feature.
            predictive_weights = {
                'ewma': 1.2,       # Momentum is the most important factor
                'trend': 0.8,      # Recent trend direction is also important
                'volatility': 0.5  # Penalize for instability in news scores
            }

            # 4. Calculate the predictive daily score
            predictive_scorer = PredictiveNewsScorer(credit_score_df)
            final_df = (predictive_scorer.calculate_predictive_score(weights=predictive_weights, base_score=50)
                                         .get_scored_df())

            # 5. Display the results
            print("\n--- Final DataFrame with Predictive Scores ---")
            
            # Select columns for a clean final output
            display_cols = [
                'daily_credit_score',     # The actual score for the day
                'ewma_prev',              # Historical feature: momentum
                'trend_prev',             # Historical feature: trend
                'volatility_prev',        # Historical feature: volatility
                'predictive_daily_score'  # The final predictive score for the day
            ]
            
            print("Showing last 15 days of data:")
            # Round the output for better readability
            print(final_df[display_cols].tail(15).round(2))

        else:
            print("No news data available to generate features.")

    except (ValueError, ImportError) as e:
        print(f"\nAn error occurred: {e}")