from credit_risk_scorer import RiskScorer, company_to_ticker
from clustering.score_gen import get_company_scores, main
from news_scoring import (
    AdvancedNewsMetricGenerator,
    NewsCreditScorer,
    PredictiveNewsScorer,
)


def credit_score(company_name: str) -> tuple:
    """
    Calculate comprehensive credit score for a company.
    Returns: [news_score, risk_score, financial_score, feature_importance]
    """
    # Initialize default values
    news_score = 0.0
    risk_score = 0.0
    financial_score = 0.0
    feature_importance = {}

    try:
        # 1. Calculate News Score
        print(f"Calculating news score for {company_name}...")
        ticker = company_to_ticker(company_name)

        if ticker:
            metrics_generator = AdvancedNewsMetricGenerator(ticker=ticker)
            daily_features_df = metrics_generator.generate_news_metrics(days_history=90)

            if not daily_features_df.empty:
                # 2. Calculate the descriptive daily credit score
                credit_scorer = NewsCreditScorer(daily_features_df)
                credit_score_df = (
                    credit_scorer.calculate_daily_credit_score().get_scored_df()
                )

                # 3. Define weights for the predictive model
                predictive_weights = {
                    "ewma": 1.2,  # Momentum is the most important factor
                    "trend": 0.8,  # Recent trend direction is also important
                    "volatility": 0.5,  # Penalize for instability in news scores
                }

                # 4. Calculate the predictive daily score
                predictive_scorer = PredictiveNewsScorer(credit_score_df)
                final_df = predictive_scorer.calculate_predictive_score(
                    weights=predictive_weights, base_score=50
                ).get_scored_df()

                if not final_df.empty and "predictive_daily_score" in final_df.columns:
                    news_score = float(
                        final_df.tail(1)["predictive_daily_score"].values[0]
                    )
                else:
                    print("Warning: No predictive daily score available")
                    news_score = 50.0  # Default neutral score
            else:
                print("No news data available to generate features.")
                news_score = 50.0  # Default neutral score
        else:
            print(f"Warning: Could not find ticker for {company_name}")
            news_score = 50.0  # Default neutral score

        # 2. Calculate Risk Score
        print(f"Calculating risk score for {company_name}...")
        if ticker:
            try:
                risk_scorer = RiskScorer(shap_folder="shap_outputs")
                risk_scorer.load_data()
                risk_scorer.train_model()

                print("Model Performance:", risk_scorer.get_metrics())
                risk_score_value = risk_scorer.get_today_risk_score()
                print("Today's Risk Score:", risk_score_value)

                if risk_score_value is not None:
                    risk_score = float(risk_score_value)
                else:
                    risk_score = 50.0  # Default neutral score

                feature_importance_raw = risk_scorer.get_feature_importance()
                if feature_importance_raw:
                    feature_importance = dict(feature_importance_raw)
                    print("Feature Importances:", feature_importance)
                else:
                    feature_importance = {}

                # Generate SHAP explanations
                try:
                    risk_scorer.explain_model_global()
                    risk_scorer.explain_today()
                except Exception as e:
                    print(f"Warning: Could not generate SHAP explanations: {e}")

            except Exception as e:
                print(f"Error calculating risk score: {e}")
                risk_score = 50.0  # Default neutral score
                feature_importance = {}
        else:
            print(f"Risk scoring skipped: no ticker found for {company_name}")
            risk_score = 50.0

        # 3. Calculate Financial Score
        print(f"Calculating financial score for {company_name}...")
        try:
            all_features = main()
            company_scores = get_company_scores(company_name, all_features)

            if (
                company_scores is not None
                and not company_scores.empty
                and "Composite_Score" in company_scores.columns
            ):
                financial_score = float(company_scores.iloc[0]["Composite_Score"])
                print("\n=== Company Scores ===\n", company_scores)
            else:
                print(
                    f"Warning: Could not calculate financial score for {company_name}"
                )
                financial_score = 50.0  # Default neutral score

        except Exception as e:
            print(f"Error calculating financial score: {e}")
            financial_score = 50.0  # Default neutral score

    except Exception as e:
        print(f"Error in credit_score function: {e}")
        # Return default values
        return [50.0, 50.0, 50.0, {}]

    # Ensure all scores are valid numbers
    news_score = float(news_score) if news_score is not None else 50.0
    risk_score = float(risk_score) if risk_score is not None else 50.0
    financial_score = float(financial_score) if financial_score is not None else 50.0

    print(f"\nFinal scores for {company_name}:")
    print(f"News Score: {news_score}")
    print(f"Risk Score: {risk_score}")
    print(f"Financial Score: {financial_score}")

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
    composite_score = (
        financial_weight * scores[2] + risk_weight * scores[1] + news_weight * scores[0]
    )
    print(f"Composite Score: {composite_score:.2f}")
