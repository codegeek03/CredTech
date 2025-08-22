import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CreditScoreModel:
    def __init__(self, rating_weights='ordinal', alpha=0.3, score_range=(300, 900)):
        """
        rating_weights: 'ordinal', 'learned', or dict
        alpha: weight for rating vs ratios (0=pure ratios, 1=pure rating)
        score_range: tuple (min_score, max_score)
        """
        self.rating_weights = rating_weights
        self.alpha = alpha
        self.score_range = score_range
        self.rating_map = None
        self.scaler = StandardScaler()
        self.logit_model = None
        self.calibration_params = None
        
    def _create_rating_map(self, ratings):
        """Create ordinal mapping: higher score = better rating"""
        # Standard S&P/Moody's ordering (best to worst)
        rating_order = [
            'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
            'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC+', 'CC', 'C', 'D'
        ]
        
        # Create mapping (higher numeric = better rating)
        rating_map = {}
        for rating in ratings.unique():
            if rating in rating_order:
                # Invert index so AAA=1.0, D=0.0
                rating_map[rating] = 1.0 - (rating_order.index(rating) / (len(rating_order) - 1))
            else:
                rating_map[rating] = 0.5  # Unknown ratings get neutral score
                
        return rating_map
    
    def _create_binary_target(self, ratings):
        """Create binary default target: 1 if rating <= B-, 0 otherwise"""
        default_ratings = ['B-', 'CCC+', 'CCC', 'CCC-', 'CC+', 'CC', 'C', 'D']
        return ratings.isin(default_ratings).astype(int)
    
    def _winsorize_features(self, X, limits=(0.01, 0.99)):
        """Winsorize features to handle outliers"""
        X_win = X.copy()
        for col in X_win.columns:
            if X_win[col].dtype in ['float64', 'int64']:
                lower, upper = X_win[col].quantile([limits[0], limits])
                X_win[col] = np.clip(X_win[col], lower, upper)
        return X_win
    
    def fit(self, df):
        """Fit the complete credit scoring model"""
        
        # 1. Data preprocessing
        print("Step 1: Preprocessing data...")
        
        # Create rating mapping
        if isinstance(self.rating_weights, dict):
            self.rating_map = self.rating_weights
        else:
            self.rating_map = self._create_rating_map(df['Rating'])
        
        # Map ratings to numeric
        df['Rating_Numeric'] = df['Rating'].map(self.rating_map)
        
        # Create binary target
        df['Default'] = self._create_binary_target(df['Rating'])
        
        # Select financial ratio columns (exclude non-numeric)
        ratio_cols = [col for col in df.columns if col not in 
                     ['Rating Agency', 'Corporation', 'Rating', 'Rating Date', 
                      'CIK', 'Binary Rating', 'SIC Code', 'Sector', 'Ticker',
                      'Rating_Numeric', 'Default']]
        
        # Handle missing values and winsorize
        X_ratios = df[ratio_cols].copy()
        X_ratios = X_ratios.fillna(X_ratios.median())  # Simple median imputation
        X_ratios = self._winsorize_features(X_ratios)
        
        # Normalize features
        X_ratios_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_ratios),
            columns=X_ratios.columns,
            index=X_ratios.index
        )
        
        # 2. Fit logistic regression model
        print("Step 2: Fitting logistic regression...")
        
        # Combine rating and ratios
        X_combined = pd.concat([
            df[['Rating_Numeric']],
            X_ratios_scaled
        ], axis=1)
        
        y = df['Default']
        
        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit regularized logistic regression
        self.logit_model = LogisticRegression(
            penalty='elasticnet',
            l1_ratio=0.5,  # Mix of L1 and L2
            C=1.0,
            solver='saga',
            max_iter=1000,
            random_state=42
        )
        
        self.logit_model.fit(X_train, y_train)
        
        # 3. Model validation
        print("Step 3: Model validation...")
        
        y_pred_proba = self.logit_model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        print(f"AUC: {auc:.4f}")
        print(f"Brier Score: {brier:.4f}")
        print(f"Log Loss: {logloss:.4f}")
        
        # 4. Calibrate score scaling
        print("Step 4: Calibrating score scaling...")
        
        # Get log-odds (linear predictor)
        log_odds_train = (X_train @ np.concatenate([[self.logit_model.intercept_[0]], 
                                                   self.logit_model.coef_]))
        
        # Map to score range using percentiles
        eta_05, eta_95 = np.percentile(log_odds_train, [5, 95])
        
        # S = A - B * eta (higher eta = worse, so negative B)
        B = (self.score_range - self.score_range) / (eta_95 - eta_05)
        A = self.score_range + B * eta_05
        
        self.calibration_params = {'A': A, 'B': B}
        
        # Store training data for future reference
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.ratio_cols = ratio_cols
        
        print(f"Model fitted successfully!")
        print(f"Calibration: Score = {A:.2f} - {B:.4f} * log_odds")
        
        return self
    
    def predict_score(self, df):
        """Generate credit scores for new data"""
        
        # Preprocess the same way as training
        df_proc = df.copy()
        df_proc['Rating_Numeric'] = df_proc['Rating'].map(self.rating_map)
        
        # Get ratio features
        X_ratios = df_proc[self.ratio_cols].copy()
        X_ratios = X_ratios.fillna(X_ratios.median())
        X_ratios = self._winsorize_features(X_ratios)
        X_ratios_scaled = pd.DataFrame(
            self.scaler.transform(X_ratios),
            columns=X_ratios.columns,
            index=X_ratios.index
        )
        
        # Combine features
        X_combined = pd.concat([
            df_proc[['Rating_Numeric']],
            X_ratios_scaled
        ], axis=1)
        
        # Get log-odds
        log_odds = (X_combined @ np.concatenate([[self.logit_model.intercept_[0]], 
                                               self.logit_model.coef_]))
        
        # Convert to probability
        probabilities = 1 / (1 + np.exp(log_odds))  # P(default)
        
        # Convert to credit score (higher = better)
        scores = self.calibration_params['A'] - self.calibration_params['B'] * log_odds
        
        return pd.DataFrame({
            'Credit_Score': scores,
            'Default_Probability': probabilities,
            'Log_Odds': log_odds
        })
    
    def feature_importance(self):
        """Get feature importance with confidence intervals"""
        
        coefs = self.logit_model.coef_[0]
        feature_names = ['Rating_Numeric'] + list(self.ratio_cols)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs,
            'Abs_Coefficient': np.abs(coefs)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return importance_df
    
    def calibration_plot(self):
        """Generate calibration plot"""
        
        y_pred_proba = self.logit_model.predict_proba(self.X_test)[:, 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Calibration Plot")
        ax1.legend()
        ax1.grid(True)
        
        # Score distribution by rating
        test_scores = self.predict_score(
            pd.concat([self.X_test.reset_index(drop=True), 
                      pd.DataFrame({'Rating': ['A'] * len(self.X_test)})], axis=1)
        )['Credit_Score']
        
        ax2.hist(test_scores, bins=30, alpha=0.7, density=True)
        ax2.set_xlabel("Credit Score")
        ax2.set_ylabel("Density")
        ax2.set_title("Score Distribution")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Usage Example
def main():
    # Load your data
    df = pd.read_csv(r"C:\Users\shami\OneDrive\Documents\GitHub\CredTech\clustering\corporateCreditRatingWithFinancialRatios.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique ratings: {sorted(df['Rating'].unique())}")
    print(f"Missing values per column:\n{df.isnull().sum().sort_values(ascending=False).head()}")
    
    # Initialize and fit model
    model = CreditScoreModel(alpha=0.3, score_range=(300, 900))
    model.fit(df)
    
    # Generate scores for the dataset
    scores_df = model.predict_score(df)
    
    # Combine with original data
    results = pd.concat([df[['Corporation', 'Rating']], scores_df], axis=1)
    
    print("\nSample results:")
    print(results.head(10))
    
    print(f"\nScore statistics:")
    print(scores_df['Credit_Score'].describe())
    
    # Feature importance
    print(f"\nTop 10 most important features:")
    importance = model.feature_importance()
    print(importance.head(10))
    
    # Generate calibration plot
    model.calibration_plot()
    
    # Score by rating analysis
    score_by_rating = results.groupby('Rating')['Credit_Score'].agg(['mean', 'std', 'count']).round(2)
    score_by_rating = score_by_rating.sort_values('mean', ascending=False)
    print(f"\nAverage score by rating:")
    print(score_by_rating)
    
    return model, results

if __name__ == "__main__":
    model, results = main()
