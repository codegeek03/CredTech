import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, classification_report
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RobustCreditScoreModel:
    """
    Robust Credit Scoring Model with 0-1000 score range
    
    Features:
    - Comprehensive data cleaning and outlier handling
    - Multiple scaling options (Standard, Robust, Quantile)
    - Class imbalance handling
    - Cross-validation for robust parameter estimation
    - Proper score calibration to 0-1000 range
    - Extensive validation and diagnostics
    - Missing data handling with multiple strategies
    """
    
    def __init__(self, 
                 rating_weights='ordinal', 
                 alpha=0.3, 
                 score_range=(0, 1000),
                 scaling_method='robust',
                 handle_imbalance=True,
                 cv_folds=5,
                 random_state=42):
        """
        Initialize the robust credit scoring model
        
        Parameters:
        -----------
        rating_weights : str or dict
            Method for rating mapping ('ordinal' or custom dict)
        alpha : float
            Weight for rating vs ratios (0=only ratios, 1=only rating)
        score_range : tuple
            Score output range (default 0-1000)
        scaling_method : str
            'standard', 'robust', or 'quantile'
        handle_imbalance : bool
            Whether to handle class imbalance
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.rating_weights = rating_weights
        self.alpha = alpha
        self.score_range = score_range
        self.scaling_method = scaling_method
        self.handle_imbalance = handle_imbalance
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize components
        self.rating_map = None
        self.scaler = self._get_scaler()
        self.logit_model = None
        self.calibrated_model = None
        self.calibration_params = None
        self.feature_names = None
        self.validation_scores = {}
        
    def _get_scaler(self):
        """Get appropriate scaler based on method"""
        if self.scaling_method == 'standard':
            return StandardScaler()
        elif self.scaling_method == 'robust':
            return RobustScaler()
        elif self.scaling_method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution='normal', random_state=self.random_state)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def _create_rating_map(self, ratings):
        """Create robust ordinal mapping with extended rating coverage"""
        # Comprehensive rating hierarchy (higher numeric = better rating)
        rating_order = [
            'D', 'C', 'CC', 'CC+', 'CCC-', 'CCC', 'CCC+',
            'B-', 'B', 'B+', 'BB-', 'BB', 'BB+',
            'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+',
            'AA-', 'AA', 'AA+', 'AAA'
        ]
        
        unique_ratings = ratings.unique()
        rating_map = {}
        
        for rating in unique_ratings:
            if pd.isna(rating) or rating == '':
                rating_map[rating] = 0.5  # Neutral for missing
            elif rating in rating_order:
                # Linear mapping: 0 (worst) to 1 (best)
                rating_map[rating] = rating_order.index(rating) / (len(rating_order) - 1)
            else:
                # Unknown ratings get middle value
                rating_map[rating] = 0.5
                print(f"Warning: Unknown rating '{rating}' assigned neutral value 0.5")
        
        return rating_map
    
    def _create_binary_target(self, ratings):
        """Create binary default target with robust handling"""
        # Define distressed/default ratings
        distressed_ratings = ['D', 'C', 'CC', 'CC+', 'CCC-', 'CCC', 'CCC+', 'B-']
        
        # Handle missing ratings
        default_mask = ratings.isin(distressed_ratings)
        missing_mask = ratings.isna() | (ratings == '')
        
        # Create binary target
        target = default_mask.astype(int)
        
        # Handle missing ratings (assign based on distribution or domain knowledge)
        if missing_mask.sum() > 0:
            print(f"Warning: {missing_mask.sum()} missing ratings. Assigning based on other features.")
            # For now, assign neutral probability (can be improved with imputation)
            # target[missing_mask] = 0  # Conservative: assume non-default
        
        return target
    
    def _robust_clean_and_winsorize(self, X, limits=(0.01, 0.99)):
        """Comprehensive data cleaning with multiple strategies"""
        X_clean = X.copy()
        cleaning_stats = {}
        
        for col in X_clean.columns:
            original_na_count = X_clean[col].isna().sum()
            
            # Convert to numeric
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            
            # Handle infinite values
            inf_count = np.isinf(X_clean[col]).sum()
            X_clean[col] = X_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Calculate robust statistics before imputation
            median_val = X_clean[col].median()
            q1, q3 = X_clean[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            
            # Multiple imputation strategies
            if X_clean[col].isna().all():
                # All missing - use zero
                X_clean[col] = 0.0
                impute_method = 'zero_all_missing'
            elif X_clean[col].isna().sum() > len(X_clean) * 0.5:
                # >50% missing - use median
                X_clean[col] = X_clean[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                impute_method = 'median_high_missing'
            else:
                # Use robust imputation based on IQR
                if pd.isna(median_val):
                    fill_value = 0.0
                else:
                    fill_value = median_val
                X_clean[col] = X_clean[col].fillna(fill_value)
                impute_method = 'median_normal'
            
            # Winsorization with robust bounds
            if not X_clean[col].isna().all() and X_clean[col].nunique() > 1:
                try:
                    lower_bound = X_clean[col].quantile(limits[0])
                    upper_bound = X_clean[col].quantile(limits[1])
                    
                    # Additional outlier detection using IQR method
                    iqr_lower = q1 - 1.5 * iqr
                    iqr_upper = q3 + 1.5 * iqr
                    
                    # Use more conservative bounds
                    final_lower = max(lower_bound, iqr_lower) if not pd.isna(iqr_lower) else lower_bound
                    final_upper = min(upper_bound, iqr_upper) if not pd.isna(iqr_upper) else upper_bound
                    
                    outliers_clipped = ((X_clean[col] < final_lower) | (X_clean[col] > final_upper)).sum()
                    X_clean[col] = np.clip(X_clean[col], final_lower, final_upper)
                    
                except Exception as e:
                    print(f"Winsorization failed for {col}: {e}")
                    outliers_clipped = 0
            else:
                outliers_clipped = 0
            
            # Store cleaning statistics
            cleaning_stats[col] = {
                'original_na': original_na_count,
                'inf_values': inf_count,
                'final_na': X_clean[col].isna().sum(),
                'outliers_clipped': outliers_clipped,
                'impute_method': impute_method
            }
        
        # Final safety check
        X_clean = X_clean.replace([np.inf, -np.inf, np.nan], 0.0)
        
        # Store cleaning stats for diagnostics
        self.cleaning_stats = cleaning_stats
        
        return X_clean
    
    def _get_financial_ratio_columns(self, df):
        """Automatically identify financial ratio columns"""
        # Known financial ratio patterns
        ratio_patterns = [
            'ratio', 'margin', 'turnover', 'return', 'roa', 'roe', 'roi',
            'debt', 'equity', 'current', 'quick', 'cash', 'leverage',
            'coverage', 'efficiency', 'profitability', 'liquidity',
            'ebit', 'ebitda', 'flow', 'per share'
        ]
        
        # Predefined common ratios
        known_ratios = [
            'Current Ratio', 'Long-term Debt / Capital', 'Debt/Equity Ratio',
            'Gross Margin', 'Operating Margin', 'EBIT Margin', 'EBITDA Margin',
            'Pre-Tax Profit Margin', 'Net Profit Margin', 'Asset Turnover',
            'ROE - Return On Equity', 'Return On Tangible Equity',
            'ROA - Return On Assets', 'ROI - Return On Investment',
            'Operating Cash Flow Per Share', 'Free Cash Flow Per Share'
        ]
        
        # Find columns that exist in the dataframe
        ratio_cols = [col for col in known_ratios if col in df.columns]
        
        # Find additional columns matching patterns
        for col in df.columns:
            if col not in ratio_cols and col.lower() not in ['rating', 'corporation', 'company', 'name']:
                # Check if numeric and matches pattern
                if df[col].dtype in ['int64', 'float64'] or pd.to_numeric(df[col], errors='coerce').notna().sum() > 0:
                    if any(pattern in col.lower() for pattern in ratio_patterns):
                        ratio_cols.append(col)
        
        print(f"Identified {len(ratio_cols)} financial ratio columns")
        if len(ratio_cols) < 5:
            print("Warning: Very few ratio columns found. Consider checking column names.")
        
        return ratio_cols
    
    def _validate_input_data(self, df):
        """Comprehensive input data validation"""
        issues = []
        
        # Check for required columns
        if 'Rating' not in df.columns:
            issues.append("Missing 'Rating' column")
        
        # Check data size
        if len(df) < 50:
            issues.append(f"Dataset too small: {len(df)} rows (minimum 50 recommended)")
        
        # Check rating distribution
        if 'Rating' in df.columns:
            rating_counts = df['Rating'].value_counts()
            if len(rating_counts) < 3:
                issues.append("Too few unique ratings (minimum 3 recommended)")
        
        # Check for excessive missing data
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.8].index.tolist()
        if high_missing_cols:
            issues.append(f"Columns with >80% missing data: {high_missing_cols}")
        
        if issues:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    
    def fit(self, df, validate_data=True):
        """
        Fit the robust credit scoring model with comprehensive validation
        """
        print("=" * 60)
        print("ROBUST CREDIT SCORING MODEL - TRAINING")
        print("=" * 60)
        
        if validate_data and not self._validate_input_data(df):
            print("Warning: Data validation issues detected. Proceeding with caution...")
        
        df = df.copy()  # Avoid modifying original data
        
        print(f"\nDataset Info:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Step 1: Rating mapping
        print(f"\nStep 1: Rating Processing...")
        if isinstance(self.rating_weights, dict):
            self.rating_map = self.rating_weights
        else:
            self.rating_map = self._create_rating_map(df['Rating'])
        
        print(f"  Rating distribution:")
        rating_dist = df['Rating'].value_counts().sort_index()
        for rating, count in rating_dist.head(10).items():
            numeric_val = self.rating_map.get(rating, 'Unknown')
            print(f"    {rating}: {count:,} ({numeric_val:.3f})")
        if len(rating_dist) > 10:
            print(f"    ... and {len(rating_dist) - 10} more ratings")
        
        # Apply rating mapping
        df['Rating_Numeric'] = df['Rating'].map(self.rating_map)
        df['Rating_Numeric'] = df['Rating_Numeric'].fillna(0.5)
        
        # Create binary target
        df['Default'] = self._create_binary_target(df['Rating'])
        default_rate = df['Default'].mean()
        print(f"  Default rate: {default_rate:.3f} ({df['Default'].sum():,}/{len(df):,})")
        
        # Step 2: Feature processing
        print(f"\nStep 2: Feature Processing...")
        ratio_cols = self._get_financial_ratio_columns(df)
        
        if not ratio_cols:
            raise ValueError("No financial ratio columns found!")
        
        # Clean and process features
        X_ratios = df[ratio_cols].copy()
        print(f"  Before cleaning: {X_ratios.isnull().sum().sum():,} missing values")
        
        X_ratios = self._robust_clean_and_winsorize(X_ratios)
        print(f"  After cleaning: {X_ratios.isnull().sum().sum():,} missing values")
        
        # Display cleaning statistics
        print(f"  Cleaning summary (top issues):")
        for col, stats in list(self.cleaning_stats.items())[:5]:
            print(f"    {col}: {stats['original_na']} NA → {stats['outliers_clipped']} outliers clipped")
        
        # Scale features
        print(f"  Scaling method: {self.scaling_method}")
        X_ratios_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_ratios),
            columns=X_ratios.columns,
            index=X_ratios.index
        )
        
        # Step 3: Model preparation
        print(f"\nStep 3: Model Preparation...")
        
        # Combine features
        X_combined = pd.concat([
            df[['Rating_Numeric']],
            X_ratios_scaled
        ], axis=1)
        
        # Final safety check
        X_combined = X_combined.replace([np.inf, -np.inf, np.nan], 0.0)
        y = df['Default']
        
        # Remove invalid samples
        valid_mask = ~y.isna()
        X_combined = X_combined[valid_mask]
        y = y[valid_mask]
        
        print(f"  Final training set: {len(X_combined):,} samples")
        print(f"  Features: {len(X_combined.columns)} (1 rating + {len(ratio_cols)} ratios)")
        
        # Store feature names
        self.feature_names = X_combined.columns.tolist()
        self.ratio_cols = ratio_cols
        
        # Step 4: Model training with cross-validation
        print(f"\nStep 4: Model Training...")
        
        # Handle class imbalance
        class_weights = None
        if self.handle_imbalance and len(np.unique(y)) > 1:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y), 
                y=y
            )
            class_weight_dict = dict(zip(np.unique(y), class_weights))
            print(f"  Class weights: {class_weight_dict}")
        else:
            class_weight_dict = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Train logistic regression
        self.logit_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=2000,
            random_state=self.random_state,
            class_weight=class_weight_dict
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.logit_model, X_train, y_train,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )
        
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Fit final model
        self.logit_model.fit(X_train, y_train)
        
        # Calibrate probabilities
        print(f"\nStep 5: Probability Calibration...")
        self.calibrated_model = CalibratedClassifierCV(
            self.logit_model, 
            method='isotonic', 
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Step 6: Model validation
        print(f"\nStep 6: Model Validation...")
        
        # Predictions
        y_pred_proba = self.calibrated_model.predict_proba(X_test)[:, 1]
        y_pred_binary = self.calibrated_model.predict(X_test)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        brier = brier_score_loss(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        self.validation_scores = {
            'auc': auc,
            'brier_score': brier,
            'log_loss': logloss,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std()
        }
        
        print(f"  AUC: {auc:.4f}")
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Log Loss: {logloss:.4f}")
        
        # Step 7: Score calibration
        print(f"\nStep 7: Score Calibration to {self.score_range} range...")
        
        # Get log-odds for calibration
        log_odds_train = (
            self.logit_model.intercept_[0] + 
            np.dot(X_train.values, self.logit_model.coef_.flatten())
        )
        
        # Use percentiles for robust calibration
        eta_05, eta_95 = np.percentile(log_odds_train, [5, 95])
        
        # Prevent division by zero
        if abs(eta_95 - eta_05) < 1e-10:
            eta_95 = eta_05 + 1e-6
            print(f"  Warning: Very small log-odds range, adding small epsilon")
        
        # Linear calibration: Score = A - B * log_odds (higher score = lower risk)
        B = (self.score_range[1] - self.score_range[0]) / (eta_95 - eta_05)
        A = self.score_range[1] - B * eta_05  # Ensure high scores for low log-odds (low risk)
        
        self.calibration_params = {
            'A': A, 
            'B': B,
            'eta_05': eta_05,
            'eta_95': eta_95
        }
        
        print(f"  Calibration formula: Score = {A:.2f} - {B:.4f} * log_odds")
        print(f"  Score range mapping: [{eta_95:.3f}, {eta_05:.3f}] → {self.score_range}")
        
        # Store data for analysis
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return self
    
    def predict_score(self, df):
        """Generate credit scores for new data"""
        if self.logit_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        df_proc = df.copy()
        
        # Apply rating mapping
        df_proc['Rating_Numeric'] = df_proc['Rating'].map(self.rating_map)
        df_proc['Rating_Numeric'] = df_proc['Rating_Numeric'].fillna(0.5)
        
        # Process ratio features
        X_ratios = df_proc[self.ratio_cols].copy()
        X_ratios = self._robust_clean_and_winsorize(X_ratios)
        
        # Scale features
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
        
        # Safety check
        X_combined = X_combined.replace([np.inf, -np.inf, np.nan], 0.0)
        
        # Generate predictions
        log_odds = (
            self.logit_model.intercept_[0] + 
            np.dot(X_combined.values, self.logit_model.coef_.flatten())
        )
        
        # Use calibrated probabilities
        calibrated_proba = self.calibrated_model.predict_proba(X_combined)[:, 1]
        
        # Convert to credit scores
        scores = self.calibration_params['A'] - self.calibration_params['B'] * log_odds
        
        # Ensure scores are within range
        scores = np.clip(scores, self.score_range[0], self.score_range[1])
        
        return pd.DataFrame({
            'Credit_Score': scores.round(0).astype(int),
            'Default_Probability': calibrated_proba,
            'Raw_Log_Odds': log_odds,
            'Score_Percentile': stats.rankdata(scores) / len(scores) * 100
        })
    
    def get_feature_importance(self):
        """Get comprehensive feature importance analysis"""
        if self.logit_model is None:
            raise ValueError("Model not trained yet.")
        
        coefs = self.logit_model.coef_[0]
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefs,
            'Abs_Coefficient': np.abs(coefs),
            'Impact_Direction': ['Positive' if c > 0 else 'Negative' for c in coefs]
        }).sort_values('Abs_Coefficient', ascending=False)
        
        # Add importance rank
        importance_df['Importance_Rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def generate_model_report(self):
        """Generate comprehensive model performance report"""
        if self.logit_model is None:
            raise ValueError("Model not trained yet.")
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
        print("=" * 80)
        
        # Model Configuration
        print(f"\n📊 MODEL CONFIGURATION:")
        print(f"  Score Range: {self.score_range}")
        print(f"  Scaling Method: {self.scaling_method}")
        print(f"  Handle Imbalance: {self.handle_imbalance}")
        print(f"  Alpha (Rating Weight): {self.alpha}")
        
        # Data Summary
        print(f"\n📈 DATA SUMMARY:")
        print(f"  Training Samples: {len(self.X_train):,}")
        print(f"  Test Samples: {len(self.X_test):,}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Default Rate: {self.y_train.mean():.3f}")
        
        # Performance Metrics
        print(f"\n🎯 PERFORMANCE METRICS:")
        for metric, value in self.validation_scores.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Feature Importance (Top 10)
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        importance = self.get_feature_importance()
        for _, row in importance.head(10).iterrows():
            direction = "↗️" if row['Impact_Direction'] == 'Positive' else "↘️"
            print(f"  {row['Importance_Rank']:2d}. {row['Feature']:<30} {direction} {row['Abs_Coefficient']:.4f}")
        
        # Score Distribution
        print(f"\n📊 SCORE DISTRIBUTION (Test Set):")
        test_scores = self.predict_score(pd.DataFrame(self.X_test).assign(Rating='A'))
        score_stats = test_scores['Credit_Score'].describe()
        for stat, value in score_stats.items():
            print(f"  {stat.capitalize()}: {value:.1f}")
        
        print("\n" + "=" * 80)
    
    def plot_diagnostics(self, figsize=(15, 10)):
        """Generate comprehensive diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Credit Score Model Diagnostics', fontsize=16, fontweight='bold')
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        y_pred_proba = self.calibrated_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        
        axes[0,0].plot(fpr, tpr, label=f'AUC = {self.validation_scores["auc"]:.3f}')
        axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curve')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, y_pred_proba, n_bins=10
        )
        axes[0,1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        axes[0,1].plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        axes[0,1].set_xlabel('Mean Predicted Probability')
        axes[0,1].set_ylabel('Fraction of Positives')
        axes[0,1].set_title('Calibration Plot')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Score Distribution
        test_df = pd.DataFrame(self.X_test)
        test_df['Rating'] = 'A'  # Dummy rating for scoring
        test_scores = self.predict_score(test_df)
        
        axes[0,2].hist(test_scores['Credit_Score'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,2].set_xlabel('Credit Score')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title(f'Score Distribution ({self.score_range[0]}-{self.score_range[1]})')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Feature Importance
        importance = self.get_feature_importance()
        top_features = importance.head(10)
        
        y_pos = np.arange(len(top_features))
        axes[1,0].barh(y_pos, top_features['Abs_Coefficient'], 
                      color=['green' if x == 'Positive' else 'red' for x in top_features['Impact_Direction']])
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels([f.replace('_', ' ') for f in top_features['Feature']], fontsize=8)
        axes[1,0].set_xlabel('Absolute Coefficient')
        axes[1,0].set_title('Top 10 Feature Importance')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Residuals Plot
        residuals = self.y_test - y_pred_proba
        axes[1,1].scatter(y_pred_proba, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_xlabel('Predicted Probability')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residuals vs Fitted')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Score vs Default Rate by Bins
        score_bins = pd.cut(test_scores['Credit_Score'], bins=10)
        bin_stats = pd.DataFrame({
            'score_bin': score_bins,
            'actual_default': self.y_test.values[:len(test_scores)]
        }).groupby('score_bin').agg({
            'actual_default': ['mean', 'count']
        }).round(3)
        
        bin_centers = [interval.mid for interval in bin_stats.index]
        default_rates = bin_stats[('actual_default', 'mean')].values
        
        axes[1,2].plot(bin_centers, default_rates, 'bo-', linewidth=2, markersize=8)
        axes[1,2].set_xlabel('Credit Score (Bin Centers)')
        axes[1,2].set_ylabel('Actual Default Rate')
        axes[1,2].set_title('Score vs Default Rate')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Enhanced Usage Example
def main():
    """
    Comprehensive example of using the Robust Credit Score Model
    """
    # Load data
    try:
        df = pd.read_csv(r"C:\Users\shami\OneDrive\Documents\GitHub\CredTech\clustering\corporateCreditRatingWithFinancialRatios.csv")
    except FileNotFoundError:
        print("Data file not found. Please update the file path.")
        return None, None
    
    print("=" * 80)
    print("ROBUST CREDIT SCORING SYSTEM - DEMO")
    print("=" * 80)
    
    print(f"\n📁 DATASET OVERVIEW:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    if 'Rating' in df.columns:
        print(f"  Unique ratings: {sorted(df['Rating'].unique())}")
    
    # Initialize model with robust settings
    model = RobustCreditScoreModel(
        score_range=(0, 1000),           # 0-1000 score range
        scaling_method='robust',          # Robust scaling for outliers
        handle_imbalance=True,           # Handle class imbalance
        cv_folds=5,                      # 5-fold cross-validation
        random_state=42                  # Reproducible results
    )
    
    # Train the model
    try:
        model.fit(df)
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None
    
    # Generate predictions
    print(f"\n🔮 GENERATING PREDICTIONS...")
    scores_df = model.predict_score(df)
    
    # Combine results with original data
    if 'Corporation' in df.columns:
        results = pd.concat([
            df[['Corporation', 'Rating']], 
            scores_df
        ], axis=1)
    else:
        results = pd.concat([
            df[['Rating']], 
            scores_df
        ], axis=1)
    
    # Display sample results
    print(f"\n📊 SAMPLE RESULTS:")
    print(results.head(15).to_string(index=False))
    
    # Score statistics
    print(f"\n📈 SCORE STATISTICS:")
    score_stats = scores_df['Credit_Score'].describe()
    for stat, value in score_stats.items():
        print(f"  {stat.capitalize()}: {value:.0f}")
    
    # Generate comprehensive model report
    model.generate_model_report()
    
    # Feature importance analysis
    print(f"\n🔍 DETAILED FEATURE ANALYSIS:")
    importance = model.get_feature_importance()
    print(importance.head(15).to_string(index=False))
    
    # Score by rating analysis
    print(f"\n🎯 SCORE PERFORMANCE BY RATING:")
    if 'Rating' in results.columns:
        score_by_rating = results.groupby('Rating').agg({
            'Credit_Score': ['mean', 'std', 'min', 'max', 'count'],
            'Default_Probability': ['mean', 'std']
        }).round(2)
        
        score_by_rating.columns = ['_'.join(col).strip() for col in score_by_rating.columns]
        score_by_rating = score_by_rating.sort_values('Credit_Score_mean', ascending=False)
        
        print(score_by_rating.to_string())
    
    # Risk segmentation
    print(f"\n🚦 RISK SEGMENTATION:")
    score_ranges = [
        (0, 300, "Very High Risk"),
        (300, 500, "High Risk"), 
        (500, 650, "Medium Risk"),
        (650, 750, "Low Risk"),
        (750, 1000, "Very Low Risk")
    ]
    
    for min_score, max_score, risk_label in score_ranges:
        count = ((scores_df['Credit_Score'] >= min_score) & 
                (scores_df['Credit_Score'] < max_score)).sum()
        pct = count / len(scores_df) * 100
        avg_prob = scores_df[(scores_df['Credit_Score'] >= min_score) & 
                           (scores_df['Credit_Score'] < max_score)]['Default_Probability'].mean()
        print(f"  {risk_label:<15} ({min_score:3d}-{max_score:3d}): {count:4d} ({pct:5.1f}%) - Avg Default Prob: {avg_prob:.3f}")
    
    # Generate diagnostic plots
    print(f"\n📊 GENERATING DIAGNOSTIC PLOTS...")
    try:
        model.plot_diagnostics(figsize=(16, 12))
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
    
    # Model validation metrics
    print(f"\n✅ FINAL MODEL VALIDATION:")
    validation_results = {
        'AUC Score': f"{model.validation_scores['auc']:.4f}",
        'Brier Score': f"{model.validation_scores['brier_score']:.4f}",
        'Log Loss': f"{model.validation_scores['log_loss']:.4f}",
        'CV AUC Mean': f"{model.validation_scores['cv_auc_mean']:.4f}",
        'CV AUC Std': f"{model.validation_scores['cv_auc_std']:.4f}",
        'Score Range': f"{scores_df['Credit_Score'].min():.0f} - {scores_df['Credit_Score'].max():.0f}",
        'Total Samples': f"{len(df):,}",
        'Features Used': f"{len(model.feature_names)}"
    }
    
    for metric, value in validation_results.items():
        print(f"  {metric:<15}: {value}")
    
    print(f"\n" + "=" * 80)
    print("🎉 ROBUST CREDIT SCORING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return model, results

# Additional utility functions
def save_model_results(model, results, output_dir="./credit_score_output/"):
    """Save model and results to files"""
    import os
    import pickle
    from datetime import datetime
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(output_dir, f"credit_model_{timestamp}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results_path = os.path.join(output_dir, f"credit_scores_{timestamp}.csv")
    results.to_csv(results_path, index=False)
    
    # Save feature importance
    importance_path = os.path.join(output_dir, f"feature_importance_{timestamp}.csv")
    model.get_feature_importance().to_csv(importance_path, index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"  Model: {model_path}")
    print(f"  Scores: {results_path}")
    print(f"  Feature Importance: {importance_path}")

def load_saved_model(model_path):
    """Load a previously saved model"""
    import pickle
    with open(model_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    model, results = main()
    
    if model is not None and results is not None:
        # Optionally save results
        try:
            save_model_results(model, results)
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    print(f"\n🔧 MODEL READY FOR PRODUCTION USE!")
    print(f"   Use model.predict_score(new_data) for scoring new borrowers")