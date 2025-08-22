import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import logging

# --- 1. Configuration ---
# Centralize all parameters for easy modification and clarity.

CONFIG = {
    "data": {
        "filepath": r"clustering/corporateCreditRatingWithFinancialRatios.csv",
        "required_columns": ['Corporation', 'Rating', 'Binary Rating']
    },
    "features": {
        "rating_map": {
            'AAA': 23, 'AA+': 22, 'AA': 21, 'AA-': 20, 'A+': 19, 'A': 18, 'A-': 17,
            'BBB+': 16, 'BBB': 15, 'BBB-': 14, 'BB+': 13, 'BB': 12, 'BB-': 11,
            'B+': 10, 'B': 9, 'B-': 8, 'CCC+': 7, 'CCC': 6, 'CCC-': 5,
            'CC+': 4, 'CC': 3, 'C': 2, 'D': 1
        },
        "financial_ratios": [
            'Current Ratio', 'Long-term Debt / Capital', 'Debt/Equity Ratio',
            'Gross Margin', 'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
            'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
            'ROE - Return On Equity', 'Return On Tangible Equity',
            'ROA - Return On Assets', 'ROI - Return On Investment',
            'Operating Cash Flow Per Share', 'Free Cash Flow Per Share'
        ]
    },
    "composite_score": {
        "alpha": 0.6,  # 60% weight on agency rating, 40% on financial ratios
        # Define weights for financial ratios. Must sum to 1.0.
        # Giving higher weight to leverage and profitability.
        "ratio_weights": {
            'Debt/Equity Ratio': 0.2,
            'Long-term Debt / Capital': 0.2,
            'Net Profit Margin': 0.15,
            'ROA - Return On Assets': 0.15,
            'Current Ratio': 0.1,
            'Asset Turnover': 0.1,
            'EBITDA Margin': 0.1,
        } # Other ratios not listed will get a weight of 0
    },
    "logit_score": {
        "regularization_strength": 1.0  # Corresponds to C in LogisticRegression
    },
    "scaling": {
        "min_score": 0,
        "max_score": 1000
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filepath: str, required_columns: list) -> pd.DataFrame:
    """
    Loads data from a CSV file and validates its structure.

    Args:
        filepath: Path to the CSV file.
        required_columns: A list of column names that must be present.

    Returns:
        A pandas DataFrame with the loaded data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the DataFrame is empty or missing required columns.
    """
    try:
        logging.info(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found at the specified path: {filepath}")
        raise

    if df.empty:
        raise ValueError("The loaded DataFrame is empty.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
    
    logging.info("Data loaded successfully.")
    return df


def preprocess_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepares the DataFrame for modeling by mapping ratings, imputing missing values,
    and normalizing financial ratios.

    Args:
        df: The input DataFrame.
        config: The configuration dictionary.

    Returns:
        A new DataFrame with preprocessed features.
    """
    logging.info("Starting feature preprocessing...")
    processed_df = df.copy()
    
    # 1. Map categorical ratings to a numeric scale (R_j)
    rating_map = config['features']['rating_map']
    processed_df['R_j'] = processed_df['Rating'].map(rating_map)
    if processed_df['R_j'].isnull().any():
        unmapped_ratings = processed_df[processed_df['R_j'].isnull()]['Rating'].unique()
        logging.warning(f"Unmapped ratings found: {unmapped_ratings}. Imputing with median.")
        processed_df['R_j'].fillna(processed_df['R_j'].median(), inplace=True)
    
    # Normalize R_j to a [0, 1] scale for consistent weighting in composite score
    scaler_r_j = MinMaxScaler()
    processed_df['R_j_norm'] = scaler_r_j.fit_transform(processed_df[['R_j']])
    
    # 2. Impute and normalize financial ratios (X_j_star)
    ratios = [r for r in config['features']['financial_ratios'] if r in processed_df.columns]
    for col in ratios:
        if processed_df[col].isnull().any():
            median_val = processed_df[col].median()
            logging.warning(f"NaNs found in '{col}'. Imputing with median ({median_val:.2f}).")
            processed_df[col].fillna(median_val, inplace=True)
            
    scaler_ratios = MinMaxScaler()
    # Normalize and store with a '_norm' suffix
    norm_ratio_cols = [f"{col}_norm" for col in ratios]
    processed_df[norm_ratio_cols] = scaler_ratios.fit_transform(processed_df[ratios])
    
    logging.info("Feature preprocessing completed.")
    return processed_df


def calculate_composite_score(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Calculates a composite score based on a weighted average of the numeric
    rating and financial ratios.

    Args:
        df: The preprocessed DataFrame.
        config: The configuration dictionary.

    Returns:
        A pandas Series containing the unscaled composite score.
    """
    logging.info("Calculating composite score...")
    alpha = config['composite_score']['alpha']
    ratio_weights = config['composite_score']['ratio_weights']

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(ratio_weights.values())
    if total_weight == 0:
        raise ValueError("Sum of ratio_weights cannot be zero.")
    normalized_weights = {key: value / total_weight for key, value in ratio_weights.items()}

    # Calculate the weighted sum of financial ratios
    weighted_financial_score = pd.Series(0.0, index=df.index)
    for ratio, weight in normalized_weights.items():
        norm_col = f"{ratio}_norm"
        if norm_col in df.columns:
            weighted_financial_score += df[norm_col] * weight
        else:
            logging.warning(f"Weighted ratio '{ratio}' not found in DataFrame columns.")

    # Calculate final composite score
    unscaled_score = alpha * df['R_j_norm'] + (1 - alpha) * weighted_financial_score
    logging.info("Composite score calculation complete.")
    return unscaled_score


def calculate_logit_score(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Calculates a score based on the log-odds from a logistic regression model.
    A lower log-odds (eta) indicates lower default risk and thus a better score.

    Args:
        df: The preprocessed DataFrame.
        config: The configuration dictionary.

    Returns:
        A pandas Series containing the negative log-odds (-eta_j) as the score.
    """
    logging.info("Calculating logit-based score...")
    
    # Define target variable Y_j (1 = default/speculative, 0 = no-default/investment grade)
    df['Y_j'] = 1 - df['Binary Rating']
    
    # Prepare features for the model
    ratio_cols_norm = [f"{col}_norm" for col in config['features']['financial_ratios'] if f"{col}_norm" in df.columns]
    feature_cols = ['R_j'] + ratio_cols_norm
    X = df[feature_cols]
    y = df['Y_j']

    # Fit Logistic Regression model
    model = LogisticRegression(
        solver='liblinear',
        random_state=42,
        C=config['logit_score']['regularization_strength']
    )
    model.fit(X, y)
    
    # Calculate log-odds (eta_j)
    eta_j = model.decision_function(X)
    
    logging.info("Logit-based score calculation complete.")
    # Return negative eta, so that higher values mean better creditworthiness
    return pd.Series(-eta_j, index=df.index, name='neg_log_odds_score')


def scale_score_to_range(score_series: pd.Series, min_target: int, max_target: int) -> pd.Series:
    """
    Linearly scales a pandas Series of scores to a specified target range.

    Args:
        score_series: A pandas Series containing the raw scores.
        min_target: The minimum value of the target scale.
        max_target: The maximum value of the target scale.

    Returns:
        A pandas Series containing the scaled and rounded scores.
    """
    raw_min, raw_max = score_series.min(), score_series.max()
    
    # Edge case: if all scores are the same, avoid division by zero
    if raw_max - raw_min == 0:
        logging.warning("All raw scores are identical. Scaling will result in a single value.")
        return pd.Series(np.full(len(score_series), (min_target + max_target) / 2), index=score_series.index)
        
    # Linear scaling formula
    scaled_score = (
        (score_series - raw_min) / (raw_max - raw_min)
    ) * (max_target - min_target) + min_target
    
    # Clip to ensure scores are strictly within the target range
    return scaled_score.clip(min_target, max_target).round(0)


# --- NEW FUNCTION TO GET SCORE FOR A SINGLE COMPANY ---
def get_company_scores(company_name: str, scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves the credit scores for a specific company from the DataFrame.

    Args:
        company_name: The name of the company to search for.
        scores_df: The DataFrame containing the calculated scores.

    Returns:
        A pandas DataFrame with the scores for the specified company, 
        or an empty DataFrame if the company is not found.
    """
    logging.info(f"Searching for scores for: {company_name}")
    
    # Filter the DataFrame to find the company
    # The .str.contains() method allows for partial matches.
    # The `case=False` argument makes the search case-insensitive.
    company_data = scores_df[scores_df['Corporation'].str.contains(company_name, case=False, na=False)]

    if company_data.empty:
        logging.warning(f"No data found for a company with the name: {company_name}")
        return pd.DataFrame() # Return an empty DataFrame if no match is found

    # Return only the relevant columns
    return company_data[['Corporation', 'Rating', 'Composite_Score', 'Logit_Score']]


def main() -> pd.DataFrame | None:
    """
    Main function to execute the credit scoring pipeline.
    
    Returns:
        A DataFrame with the final scores, or None if an error occurs.
    """
    try:
        # 1. Load and Preprocess Data
        raw_df = load_data(CONFIG['data']['filepath'], CONFIG['data']['required_columns'])
        processed_df = preprocess_features(raw_df, CONFIG)

        # 2. Calculate Scores using both methods
        # Method A: Simple Composite Score
        composite_unscaled = calculate_composite_score(processed_df, CONFIG)
        processed_df['Composite_Score'] = scale_score_to_range(
            composite_unscaled, 
            CONFIG['scaling']['min_score'], 
            CONFIG['scaling']['max_score']
        )
        
        # Method B: Probabilistic/Logit Score
        logit_unscaled = calculate_logit_score(processed_df, CONFIG)
        processed_df['Logit_Score'] = scale_score_to_range(
            logit_unscaled,
            CONFIG['scaling']['min_score'],
            CONFIG['scaling']['max_score']
        )
        
        # 3. Prepare and Display Final Results
        logging.info("\n--- Final Credit Scores (0-1000) ---")
        final_results = processed_df[['Corporation', 'Rating', 'Composite_Score', 'Logit_Score']]
        print(final_results.sort_values(by='Logit_Score', ascending=False).to_string(index=False))
        
        return final_results

    except (FileNotFoundError, ValueError) as e:
        logging.error(f"A critical error occurred: {e}")
        logging.error("Pipeline execution aborted.")
        return None


if __name__ == "__main__":
    # Execute the main pipeline to process all data and calculate scores
    all_company_scores_df = main()
    
    # Check if the main pipeline ran successfully before trying to fetch scores
    if all_company_scores_df is not None:
        print("\n" + "="*50)
        print("--- FETCHING SCORES FOR SPECIFIC COMPANIES ---")
        print("="*50)

        # Example 1: A company that is likely in the dataset
        company_to_find = "Microsoft" 
        microsoft_scores = get_company_scores(company_to_find, all_company_scores_df)
        print("===============================")
        print(microsoft_scores.iloc[0]['Composite_Score'])
        print("===============================")
        if not microsoft_scores.empty:
            print(f"\nScores for '{company_to_find}':\n", microsoft_scores.to_string(index=False))

        # Example 2: Another company
        company_to_find = "Apple"
        apple_scores = get_company_scores(company_to_find, all_company_scores_df)
        if not apple_scores.empty:
            print(f"\nScores for '{company_to_find}':\n", apple_scores.to_string(index=False))
            
        # Example 3: A company that is NOT in the dataset
        company_to_find = "NonExistent Company Inc"
        non_existent_scores = get_company_scores(company_to_find, all_company_scores_df)
        if non_existent_scores.empty:
            print(f"\nCould not find scores for '{company_to_find}'. This is expected.")