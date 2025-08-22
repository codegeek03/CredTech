from credit_risk_scorer import RiskScorer, company_to_ticker
from clustering.score_gen import get_company_scores, main
from news_scoring import AdvancedNewsMetricGenerator, NewsCreditScorer, PredictiveNewsScorer

def credit_score(company_name: str) -> tuple:
    news_score = 0
    metrics_generator = AdvancedNewsMetricGenerator(ticker='NVDA')
    daily_features_df = metrics_generator.generate_news_metrics(days_history=90)
    all_features = main()
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
            
        news_score = final_df.tail(1)['predictive_daily_score'].values[0]

    else:
        print("No news data available to generate features.")
        
    ticker = company_to_ticker(company_name)
    if ticker:
        risk_scorer = RiskScorer(shap_folder="shap_outputs")
        risk_scorer.load_data()
        risk_scorer.train_model()

        print("Model Performance:", risk_scorer.get_metrics())
        print("Today's Risk Score:", risk_scorer.get_today_risk_score())
        risk_score = risk_scorer.get_today_risk_score()
        feature_importance = risk_scorer.get_feature_importance()   
        print("Feature Importances:", risk_scorer.get_feature_importance())

        risk_scorer.explain_model_global()
        risk_scorer.explain_today()

        company_scores = get_company_scores(company_name, all_features)
        financial_score = company_scores.iloc[0]['Composite_Score']
        print("\n=== Company Scores ===\n", company_scores)
    else:
        print(f"Ticker not found for {company_name}")
    
    return [news_score, risk_score, financial_score, feature_importance]

if __name__ == "__main__":
    company_name = "Microsoft"
    scores = credit_score(company_name)
    print(f"\nScores for {company_name}:")
    print(f"News Score: {scores[0]}")
    print(f"Risk Score: {scores[1]}")
    print(f"Financial Score: {scores[2]}")
    print(f"Feature Importance: {scores[3]}")
    financial_weight = 0.5
    risk_weight = 0.3
    news_weight = 0.2
    composite_score = (financial_weight * scores[2] +
                       risk_weight * scores[1] +
                       news_weight * scores[0])
    print(f"Composite Score: {composite_score:.2f}")
    