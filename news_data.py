import finnhub
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class FinnhubNewsFetcher:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

    def fetch_company_news(self, days_history: int = 7) -> list:
        print(f"--- Fetching news for {self.ticker} via Finnhub API ---")
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_history)
            
            date_format = "%Y-%m-%d"
            
            news_list = self.client.company_news(
                self.ticker,
                _from=start_date.strftime(date_format),
                to=end_date.strftime(date_format)
            )

            if not news_list:
                print("No news found for this period.")
                return []
            
            print(f"Successfully fetched {len(news_list)} articles.")
            return news_list

        except Exception as e:
            print(f"An error occurred with Finnhub API: {e}")
            return []

if __name__ == "__main__":
    if not os.getenv("FINNHUB_API_KEY"):
        print("Error: FINNHUB_API_KEY environment variable not set.")
    else:
        fetcher = FinnhubNewsFetcher(ticker='TSLA')
        articles = fetcher.fetch_company_news()
        
        for i, article in enumerate(articles[:10], 1): # Print top 5
            print(f"  {i}. Headline: {article['headline']}")
            print(f"     Source: {article['source']}")
            print(f"     URL: {article['url']}\n")
